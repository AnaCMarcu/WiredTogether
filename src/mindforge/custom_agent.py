from dataclasses import dataclass
import json
import logging
from typing import AsyncGenerator, List, Literal, Sequence

import PIL
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken, Image
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import MultiModalMessage
from pydantic import BaseModel

from agent_modules.action_selection import ActionSelection
from agent_modules.auto_curriculum import AutoCurriculum
from agent_modules.belief_system import BeliefSystem
from agent_modules.critic import Critic
from agent_modules.episodic_memory_manager import EpisodicMemoryManager
from agent_modules.llm_call import llm_call
from agent_modules.craftium_metric import CraftiumMetric as Metric
from agent_modules.skill_manager import SkillManager
from agent_modules.util import AgentResponse, Belief, Observation, autogenImg_to_Pil, create_model_client

try:
    from agent_modules.causal_world_model import CausalWorldModel
except ImportError:
    CausalWorldModel = None
try:
    from agent_modules.surgical_interventions import SurgicalInterventions
except ImportError:
    SurgicalInterventions = None
try:
    from causal_module import CausalBDI
except ImportError:
    CausalBDI = None


class CustomAgent(BaseChatAgent):

    # Initialize the agent with a name, description, and model client
    def __init__(
        self,
        name: str,
        description: str,
        action_selection=None,
        auto_curriculum=None,
        critic=None,
        skill_manager=None,
        episode_manager=None,
        belief_system=None,
        causal_world_model=None,
        surgical_interventions=None,
        number_of_agents: int = 1,
        metric: Metric = None,
        voyager=False,
        causal_predictions=False,
        surgical_action_selection=False,
        causal_bdi=False,
        rl_layer=None,
    ) -> None:
        super().__init__(name, description)
        self.rl_layer = rl_layer

        self.action_selection = (
            action_selection if action_selection else ActionSelection()
        )
        self.auto_curriculum = (
            auto_curriculum if auto_curriculum else AutoCurriculum(agent_name=name)
        )
        self.critic = critic if critic else Critic()
        self.skill_manager = (
            skill_manager if skill_manager else SkillManager(agent_name=name)
        )
        self.episode_manager = (
            episode_manager
            if episode_manager
            else EpisodicMemoryManager(agent_name=name)
        )
        self.belief_system = (
            belief_system
            if belief_system
            else BeliefSystem(number_of_agents=number_of_agents)
        )
        if causal_world_model:
            self.causal_world_model = causal_world_model
        elif CausalWorldModel is not None:
            self.causal_world_model = CausalWorldModel()
        else:
            self.causal_world_model = None
        if surgical_interventions:
            self.surgical_interventions = surgical_interventions
        elif SurgicalInterventions is not None and self.causal_world_model is not None:
            self.surgical_interventions = SurgicalInterventions(causal_world_model=self.causal_world_model)
        else:
            self.surgical_interventions = None
        self.last_response = None
        self.metric = metric
        self.voyager = voyager
        self.causal_predictions = causal_predictions
        self.surgical_action_selection = surgical_action_selection
        self.surgical_action = None
        self.causal_bdi = causal_bdi
        if self.causal_bdi:
            self.causal_module = CausalBDI(
                causal_model_path='/home/arthur/Documents/Repos/CausalForge/causal_model/96_noise0.02_150ep.ckpt',
                pca_model_path='/home/arthur/Documents/Repos/CausalForge/causal_model/contra_pca_model',
                device='cuda',
                separated=True
            )
        else:
            self.causal_module = None
        self._last_reward_text = "N/A"

    # Message types that this agent can produce
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(
        self,
        messages: list[MultiModalMessage],
        cancellation_token: CancellationToken,
        error=None,
        error_count=0,
        communication=None,
        picked_object=None,
        reward_text=None,
    ) -> Response:

        print("Error count: ", error_count)
        # if first round, clear skill manager data
        if self.auto_curriculum.current_task is None:
            await self.skill_manager.clear_data()
            await self.episode_manager.clear_data()
            await self.auto_curriculum.clear_data()

        last_frame = messages[0].content[1]
        last_action = self.last_response["action"] if self.last_response else None
        last_thoughts = self.last_response["thoughts"] if self.last_response else None

        task = self.auto_curriculum.current_task

        self.metric.log(
            f"Agent {self.name}: Error count: {error_count}, error: {error}, held object: {picked_object}"
        )

        # Run critic if not first round
        success = None
        critique = None
        if self.auto_curriculum.current_task is not None:
            success, critique = await self.critic.check_task_success(
                last_frame,
                last_action,
                cancellation_token,
                self.auto_curriculum.current_task,
                self.belief_system.task_beliefs,
                communication,
                error,
                picked_object=picked_object,
            )
            self.metric.log(
                f"Agent {self.name}: Task '{task}' success: {success}, critique: {critique}"
            )
            if success:
                # TODO: add task belief?
                skill_name, skill_description, already_exists = (
                    await self.skill_manager.add_skill(
                        last_action, cancellation_token, last_thoughts
                    )
                )
                if not already_exists and self.metric is not None:
                    self.metric.found_skill(
                        f"Did {last_action}  for task {task}",
                        main=True if self.name == "agent_0" else False,
                    )
            else:
                error_count += 1

            # RL layer: track success for token-opt self-trigger
            if self.rl_layer and self.rl_layer.enabled:
                self.rl_layer.record_success(success)

            # save last episode for both success and failure
            self.episode_manager.add_episode(
                task=task,
                task_beliefs=self.belief_system.task_beliefs,
                last_action=last_action,
                last_thoughts=last_thoughts,
                critique=critique,
                success=success,
            )
        # Generate new task if there is no current task
        # or if the agent has failed more than 5 times in a row (then append last task to failed tasks)
        # or if task was succeded ( then append last task to completed tasks)
        if success or self.auto_curriculum.current_task is None or error_count > 10:
            if success:
                self.auto_curriculum.completed_tasks.append(
                    self.auto_curriculum.current_task
                )
                self.auto_curriculum.save_context(
                    self.belief_system.task_beliefs,
                )
            elif error_count > 10:
                print(f"Failed too many times at {task}, getting new task")
                self.auto_curriculum.failed_tasks.append(
                    self.auto_curriculum.current_task
                )
            task, context = await self.auto_curriculum.get_new_task(
                last_frame,
                last_action,
                last_thoughts,
                communications=communication,
                success=success,
                critique=critique,
                picked_object=picked_object,
                cancellation_token=cancellation_token,
            )
            self.metric.log(f"Agent {self.name}: New task: {task}")
            self.belief_system.task_beliefs = context
            error_count = 0
        task = self.auto_curriculum.current_task
        task_beliefs = self.belief_system.task_beliefs

        # Fetch relevant skills
        # construct query for skill manager
        query = await self.skill_manager.construct_query(
            task, frame=last_frame, cancellation_token=cancellation_token
        )
        # fetch skills
        skill_memory = await self.skill_manager.get_skills(query)

        if not self.voyager:
            # fetch episodes
            episodes = self.episode_manager.retrieve_episodes(query)
            # construct episode summary
            episode_summary = await self.episode_manager.generate_episode_summary(
                episodes, cancellation_token
            )
        else:
            episode_summary = "Not available"

        # generate beliefs
        beliefs = {
            "perception_beliefs": "",
            "interaction_beliefs": "",
            "partner_beliefs": "",
            "task_beliefs": "",
        }


        if not self.voyager:
            perception_beliefs = await self.belief_system.create_perception_beliefs(
                last_frame, communication, error, cancellation_token
            )
            partner_beliefs = await self.belief_system.update_partner_beliefs(
                communication, cancellation_token
            )
            interaction_beliefs = await self.belief_system.update_interaction_beliefs(
                task, communication, cancellation_token
            )
            task_beliefs = await self.belief_system.update_task_beliefs(
                task, cancellation_token
            )
            beliefs = {
                "perception_beliefs": perception_beliefs,
                "interaction_beliefs": interaction_beliefs,
                "partner_beliefs": partner_beliefs,
                "task_beliefs": task_beliefs,
                "reward_text": "",
            }
            self.metric.log(f"Agent {self.name} beliefs: {beliefs}")

        # if surgical intervention was selected last interation, perform second step now
        if self.surgical_action is not None and self.surgical_action_selection:
            content = {
                "thoughts": last_thoughts,
                "action": self.surgical_action,
                "communication": f"Performed surgical action: {self.surgical_action}",
            }
            self.surgical_action = None
            return content, error_count


        if self.causal_bdi:
            # observation to text
            # somehow taking auto encoder outside of the file??


            observation_description = await llm_call(
                model_client=create_model_client(resonse_format=Observation),
                user_prompt=self.causal_module.observation_prompt,
                cancellation_token=cancellation_token,
                frame=last_frame,
                task=task
            )

            observation = observation_description['observation']
            print(f"observation_description: {observation_description}")

            current_belief_description = await llm_call(
                model_client=create_model_client(resonse_format=Belief),
                user_prompt=self.causal_module.belief_prompt,
                cancellation_token=cancellation_token,
                belief=self.episode_manager.retrieve_episodes(task)
                )
 
            if self.causal_module.separated:
                self.causal_module.bdi[0] = current_belief_description['belief']


                self.causal_module.bdi[1] = task

                print(f"Current BDI: {self.causal_module.bdi}")
                self.causal_module.update_BDI(observation=observation)

                print(f"Updated BDI: {self.causal_module.bdi}")
                self.causal_module.bdi[1] = task
                self.belief_system.perception_beliefs = self.causal_module.bdi[0]



                content = await llm_call(
                    model_client=create_model_client(resonse_format=AgentResponse),
                    user_prompt=self.causal_module.action_prompt,
                    cancellation_token=cancellation_token,
                    observation=observation,
                    action=self.causal_module.bdi[2],
                    belief=self.causal_module.bdi[0],
                    # task=self.causal_module.bdi[1],
                    task=task,
                    critique=critique,
                    error=error,
                    last_action=last_action,
                )
            else:
                print(f"Current BDI: {self.causal_module.bdi}")
                self.causal_module.update_BDI(observation=observation)

                print(f"Updated BDI: {self.causal_module.bdi}")

                content = await llm_call(
                    model_client=create_model_client(resonse_format=AgentResponse),
                    user_prompt=self.causal_module.action_prompt,
                    cancellation_token=cancellation_token,
                    observation=observation,
                    action=self.causal_module.bdi,
                    belief=current_belief_description['belief'],
                    # task=self.causal_module.bdi[1],
                    task=task,
                    critique=critique,
                    error=error,
                    last_action=last_action,
                )
        else:
            # ── RL layer (action-level) ──
            rl_content = None
            if self.rl_layer and self.rl_layer.enabled:
                # Build the same prompt the LLM would see
                rl_prompt = (
                    f"Task: {task}\n"
                    f"Last action: {last_action}\n"
                    f"Critique: {critique}\n"
                    f"Error: {error}\n"
                    f"Skills: {skill_memory}\n"
                    f"Episodes: {episode_summary}\n"
                    f"Task beliefs: {beliefs.get('task_beliefs', '')}\n"
                    f"Perception: {beliefs.get('perception_beliefs', '')}\n"
                    f"Interaction: {beliefs.get('interaction_beliefs', '')}\n"
                    f"Partner: {beliefs.get('partner_beliefs', '')}\n"
                )
                rl_content = self.rl_layer.select_action(rl_prompt)

            if rl_content is not None:
                content = rl_content
            else:
                content = await self.action_selection.select_action(
                    messages,
                    last_frame,
                    cancellation_token,
                    agent_name=self.name,
                    task=task,
                    last_action=last_action,
                    critique=critique,
                    error=error,
                    skill_memory=skill_memory,
                    episode_summary=episode_summary,
                    picked_object=picked_object,
                    beliefs=beliefs,
                )

        self.metric.log(f"Agent {self.name} response: {content}")
        print(f"Agent {self.name} response: {content}")

        if (
            self.causal_predictions
            and not self.surgical_action_selection
            and self.causal_world_model is not None
        ):
            pil_img = autogenImg_to_Pil(last_frame)
            predicted_image, extra_info = self.causal_world_model.predict(
                pil_img,
                content["action"],
                held_item=picked_object.split(" ")[0] if picked_object else None,
            )
            self.metric.save_predictions(
                original_image=pil_img,
                predicted_image=predicted_image,
                enc_dec_image=extra_info["enc_dec_image"],
                action=self.causal_world_model.action_to_grammar(content["action"]),
                held_item=picked_object,
                extra_info=extra_info,
                mapper=self.causal_world_model.idx_to_causal_var,
            )
            print("simple pred")
            content = await self.action_selection.select_action(
                messages,
                last_frame,
                cancellation_token,
                agent_name=self.name,
                task=task,
                last_action=last_action,
                critique=critique,
                error=error,
                skill_memory=skill_memory,
                episode_summary=episode_summary,
                picked_object=picked_object,
                beliefs=beliefs,
                pred_action=content["action"],
                pred_frame=Image.from_pil(PIL.Image.fromarray(predicted_image)),
            )
            self.metric.log(f"Agent {self.name} response with prediction: {content}")
            print(f"Agent {self.name} response with prediction: {content}")

        elif (
            self.causal_predictions
            and self.surgical_action_selection
            and self.causal_world_model is not None
        ):
            # generate candidate actions
            candidate_actions = await self.action_selection.select_candidate_actions(
                messages,
                last_frame,
                cancellation_token,
                agent_name=self.name,
                task=task,
                last_action=last_action,
                critique=critique,
                error=error,
                skill_memory=skill_memory,
                episode_summary=episode_summary,
                picked_object=picked_object,
                beliefs=beliefs,
            )
            self.metric.log(f"Candidate actions: {candidate_actions}")
            print(f"Candidate actions: {candidate_actions}")
            # get set of surgical actions
            # pass metric object to save surgical intervention data
            pil_img = autogenImg_to_Pil(last_frame)
            (
                valid_interventions,
                predicted_image,
                original_extra_info,
            ) = self.surgical_interventions.find_surgical_interventions(
                action=content["action"],
                observation=pil_img,
                candidate_interventions=candidate_actions[
                    "candidate_actions"
                ],  # list of candidate actions
                held_item=picked_object.split(" ")[0] if picked_object else None,
                metric=self.metric,
            )

            self.metric.log(
                f"Identified valid surgical interventions: {[x[0] for x in valid_interventions]}"
            )
            print(
                "Identified valid surgical interventions: ",
                [x[0] for x in valid_interventions],
            )

            # select surgical action to perform
            if len(valid_interventions) == 0:
                print("No valid surgical interventions found, using original action")
                content = await self.action_selection.select_action(
                    messages,
                    last_frame,
                    cancellation_token,
                    agent_name=self.name,
                    task=task,
                    last_action=last_action,
                    critique=critique,
                    error=error,
                    skill_memory=skill_memory,
                    episode_summary=episode_summary,
                    picked_object=picked_object,
                    beliefs=beliefs,
                    pred_action=content["action"],
                    pred_frame=Image.from_pil(PIL.Image.fromarray(predicted_image)),
                )
                self.metric.save_predictions(
                    original_image=pil_img,
                    predicted_image=predicted_image,
                    enc_dec_image=original_extra_info["enc_dec_image"],
                    action=content["action"],
                    held_item=picked_object,
                    extra_info=original_extra_info,
                    mapper=self.causal_world_model.idx_to_causal_var,
                )
                self.metric.log(
                    f"Agent {self.name} response with prediction as no surgical action available: {content}"
                )
                print(
                    f"Agent {self.name} response with prediction as no surgical action available: {content}"
                )
            else:
                # How to select best intervention?
                best_intervention = valid_interventions[0]

                # Force use best surgical intervention, uncomment bottom portion instead to let agent choose
                print(
                    "Using best surgical intervention: ",
                    best_intervention[0],
                    ", before: ",
                    content["action"],
                )
                self.surgical_action = content["action"]
                content = {
                    "thoughts": content["thoughts"],
                    "action": best_intervention[0],
                    "communication": f"Performing surgical action: {best_intervention[0]}",
                }
                # perform second step of surgical intervention next iter
                # content = await self.action_selection.select_action(
                #     messages,
                #     last_frame,
                #     cancellation_token,
                #     agent_name=self.name,
                #     task=task,
                #     last_action=last_action,
                #     critique=critique,
                #     error=error,
                #     skill_memory=skill_memory,
                #     episode_summary=episode_summary,
                #     picked_object=picked_object,
                #     beliefs=beliefs,
                #     pred_action=best_intervention[0],
                #     pred_frame=Image.from_pil(
                #         PIL.Image.fromarray(best_intervention[1])
                #     ),
                # )
                self.metric.log(
                    f"Agent {self.name} response with surgical prediction: {content}"
                )
                print(f"Agent {self.name} response with surgical prediction: {content}")
        # log if a surgical intervention was possible
        possible, descrip = self.metric.check_surgical(
            content["action"],
            picked_object,
            valid_interventions if self.surgical_action_selection else None,
        )
        self.last_response = content
        return content, error_count

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass