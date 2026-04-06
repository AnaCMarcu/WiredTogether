import asyncio
import logging
from typing import Sequence

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from autogen_agentchat.messages import MultiModalMessage

from agent_modules.action_selection import ActionSelection
from agent_modules.auto_curriculum import AutoCurriculum
from agent_modules.belief_system import BeliefSystem
from agent_modules.critic import Critic
from agent_modules.episodic_memory_manager import EpisodicMemoryManager
from agent_modules.craftium_metric import CraftiumMetric as Metric
from agent_modules.skill_manager import SkillManager


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
        number_of_agents: int = 1,
        metric: Metric = None,
        voyager=False,
        rl_layer=None,
        belief_interval: int = 1,
        critic_interval: int = 1,
    ) -> None:
        super().__init__(name, description)
        self.rl_layer = rl_layer
        # How often to run the expensive belief/critic LLM calls.
        # belief_interval=5 means beliefs are refreshed every 5 steps (cached in between).
        # critic_interval=20 means task-success is checked every 20 steps.
        self.belief_interval = belief_interval
        self.critic_interval = critic_interval
        self._call_count = 0  # incremented each on_messages call

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
        self.last_response = None
        self.metric = metric
        self.voyager = voyager
        self._last_reward_text = "N/A"
        self._episode_summary_cache = "There are no past episodes."
        self._episode_summary_dirty = False


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
        social_bonds=None,
        position_text=None,
        player_status_text=None,
    ):

        self._call_count += 1
        run_beliefs = (self._call_count % self.belief_interval == 0)
        run_critic  = (self._call_count % self.critic_interval == 0)

        print("Error count: ", error_count)
        # if first round, clear skill manager data
        if self.auto_curriculum.current_task is None:
            await self.skill_manager.clear_data()
            await self.episode_manager.clear_data()
            await self.auto_curriculum.clear_data()

        last_frame = messages[0].content[1]
        last_action = self.last_response.get("action") if self.last_response else None
        last_thoughts = self.last_response.get("thoughts") if self.last_response else None

        task = self.auto_curriculum.current_task

        self.metric.log(
            f"Agent {self.name}: Error count: {error_count}, error: {error}, held object: {picked_object}"
        )

        # Run critic if not first round and this is a scheduled critic step.
        # Between critic steps, reuse the cached result to avoid a wasted LLM call.
        success = None
        critique = None
        if self.auto_curriculum.current_task is not None:
            if run_critic:
                success, critique = await self.critic.check_task_success(
                    last_frame,
                    last_action,
                    cancellation_token,
                    self.auto_curriculum.current_task,
                    self.belief_system.task_beliefs,
                    communication,
                    error,
                    picked_object=picked_object,
                    reward_text=reward_text,
                    position_text=position_text,
                    player_status_text=player_status_text,
                )
                # Cache the result for skipped steps
                self._cached_success = success
                self._cached_critique = critique
                self.metric.log(
                    f"Agent {self.name}: Task '{task}' success: {success}, critique: {critique}"
                )
            else:
                # Reuse cached critic result from the last evaluation
                success = getattr(self, "_cached_success", None)
                critique = getattr(self, "_cached_critique", None)

            if success:
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
            elif success is not None:
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
            self._episode_summary_dirty = True
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
                position_text=position_text,
                player_status_text=player_status_text,
                do_question_answers=False,
            )
            self.metric.log(f"Agent {self.name}: New task: {task}")
            self.belief_system.task_beliefs = context
            error_count = 0
        task = self.auto_curriculum.current_task
        task_beliefs = self.belief_system.task_beliefs

        # ── Parallel belief + memory fetches ──
        async def _fetch_query_and_summary():
            # Fix 1: use task directly as query — no LLM call needed for discrete actions
            _skill_memory = await self.skill_manager.get_skills(task)
            if not self.voyager:
                # Fix 2: only regenerate episode summary when new episodes were added
                if self._episode_summary_dirty:
                    _episodes = self.episode_manager.retrieve_episodes(task)
                    _episode_summary = await self.episode_manager.generate_episode_summary(
                        _episodes, cancellation_token
                    )
                    self._episode_summary_cache = _episode_summary
                    self._episode_summary_dirty = False
                else:
                    _episode_summary = self._episode_summary_cache
            else:
                _episode_summary = "Not available"
            return _skill_memory, _episode_summary

        async def _fetch_beliefs():
            if self.voyager:
                return {
                    "perception_beliefs": "",
                    "interaction_beliefs": "",
                    "partner_beliefs": "",
                    "task_beliefs": task_beliefs,
                }
            if not run_beliefs:
                # Reuse cached beliefs from the last refresh step
                return getattr(self, "_cached_beliefs", {
                    "perception_beliefs": self.belief_system.perception_beliefs,
                    "partner_beliefs": self.belief_system.partner_beliefs,
                    "interaction_beliefs": self.belief_system.interaction_beliefs,
                    "task_beliefs": self.belief_system.task_beliefs,
                })
            # Fix 4: task_beliefs come from curriculum (set at task assignment) — no need to
            # re-ask the LLM every belief_interval steps since the task doesn't change mid-task.
            perception, partner, interaction = await asyncio.gather(
                self.belief_system.create_perception_beliefs(
                    last_frame, communication, error, cancellation_token
                ),
                self.belief_system.update_partner_beliefs(communication, cancellation_token),
                self.belief_system.update_interaction_beliefs(task, communication, cancellation_token),
            )
            result = {
                "perception_beliefs": perception,
                "partner_beliefs": partner,
                "interaction_beliefs": interaction,
                "task_beliefs": self.belief_system.task_beliefs,
            }
            self._cached_beliefs = result
            return result

        (skill_memory, episode_summary), belief_parts = await asyncio.gather(
            _fetch_query_and_summary(),
            _fetch_beliefs(),
        )

        beliefs = {
            **belief_parts,
            "reward_text": reward_text or "N/A",
            "social_bonds": social_bonds or "N/A",
            "position_text": position_text or "Unknown",
            "player_status_text": player_status_text or "Health: ?/20 | Hunger: ?/20 | Time: Unknown",
        }
        self.belief_system.task_beliefs = belief_parts["task_beliefs"]
        self.metric.log(f"Agent {self.name} beliefs: {beliefs}")

        # ── RL layer (action-level) ──
        rl_content = None
        if self.rl_layer and self.rl_layer.enabled:
            # Build the same prompt the LLM would see
            rl_prompt = (
                f"Task: {task}\n"
                f"Last action: {last_action}\n"
                f"Reward: {reward_text or 'N/A'}\n"
                f"Position: {position_text or 'Unknown'}\n"
                f"Status: {player_status_text or 'Health: ?/20 | Hunger: ?/20 | Time: Unknown'}\n"
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

        self.last_response = content
        return content, error_count

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass
