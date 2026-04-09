import os
from agent_modules.llm_call import llm_call
from agent_modules.util import AgentResponse, CandidateResponse, CommunicationResponse, create_model_client, safe_format

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

# load in prompts
with open(os.path.join(_PROMPT_DIR, "system_prompt.txt"), "r") as f:
    system_prompt_txt = f.read()
with open(os.path.join(_PROMPT_DIR, "environment_prompt.txt"), "r") as f:
    environment_prompt = f.read()
with open(os.path.join(_PROMPT_DIR, "instruction_prompt_p2.txt"), "r") as f:
    instruction_prompt_p2 = f.read()
with open(os.path.join(_PROMPT_DIR, "system_prompt_prediction.txt"), "r") as f:
    system_prompt_prediction_txt = f.read()
with open(os.path.join(_PROMPT_DIR, "prediction_prompt.txt"), "r") as f:
    prediction_prompt = f.read()
with open(os.path.join(_PROMPT_DIR, "system_prompt_candidate_interventions.txt"), "r") as f:
    system_prompt_candidate_interventions = f.read()
with open(os.path.join(_PROMPT_DIR, "instruction_prompt_p2_candidate.txt"), "r") as f:
    instruction_prompt_p2_candidate = f.read()
with open(os.path.join(_PROMPT_DIR, "rl_communication_prompt.txt"), "r") as f:
    rl_communication_prompt = f.read()


class ActionSelection:
    def __init__(
        self,
        system_prompt=None,
        system_prompt_prediction=None,
        action_model_client=None,
        candidate_model_client=None,
    ):
        self.system_prompt = (
            system_prompt if system_prompt else safe_format(system_prompt_txt, environment_prompt=environment_prompt)
        )
        self.system_prompt_prediction = (
            system_prompt_prediction
            if system_prompt_prediction
            else safe_format(system_prompt_prediction_txt, environment_prompt=environment_prompt)
        )
        self.action_model_client = (
            action_model_client
            if action_model_client
            else create_model_client(response_format=AgentResponse)
        )
        self.candidate_model_client = (
            candidate_model_client
            if candidate_model_client
            else create_model_client(response_format=CandidateResponse)
        )

    async def select_action(
        self,
        messages,
        last_frame,
        cancellation_token,
        agent_name,
        task,
        last_action,
        critique,
        error,
        skill_memory,
        episode_summary,
        picked_object,
        beliefs,
        pred_action=None,
        pred_frame=None,
    ):
        if not pred_action or not pred_frame:
            content = await llm_call(
                self.action_model_client,
                system_prompt=self.system_prompt,
                user_prompt=instruction_prompt_p2 + messages[0].content[0],
                frame=last_frame,
                cancellation_token=cancellation_token,
                log_prefix=f"Agent {agent_name} on_messages: ",
                task=task,
                last_action=last_action,
                critique=critique,
                error=error,
                skill_memory=skill_memory,
                episode_summary=episode_summary,
                picked_object=picked_object,
                **beliefs,
            )
            return content
        else:
            content = await llm_call(
                self.action_model_client,
                system_prompt=self.system_prompt_prediction,
                user_prompt=instruction_prompt_p2 + messages[0].content[0],
                frame=last_frame,
                cancellation_token=cancellation_token,
                log_prefix=f"Agent {agent_name} on_messages: ",
                pred_prompt=prediction_prompt.format(pred_action=pred_action),
                pred_frame=pred_frame,
                task=task,
                last_action=last_action,
                critique=critique,
                error=error,
                skill_memory=skill_memory,
                episode_summary=episode_summary,
                picked_object=picked_object,
                **beliefs,
            )
            return content

    async def generate_communication(
        self,
        action,
        task,
        last_action,
        picked_object,
        last_frame,
        cancellation_token,
        agent_name,
    ):
        """Generate a natural language communication message for an RL-selected action."""
        comm_client = create_model_client(response_format=CommunicationResponse)
        content = await llm_call(
            comm_client,
            system_prompt=self.system_prompt,
            user_prompt=rl_communication_prompt,
            frame=last_frame,
            cancellation_token=cancellation_token,
            log_prefix=f"Agent {agent_name} rl_comm: ",
            task=task,
            last_action=last_action,
            action=action,
            picked_object=picked_object or "empty",
        )
        return content.get("communication", "")

    async def select_candidate_actions(
        self,
        messages,
        last_frame,
        cancellation_token,
        agent_name,
        task,
        last_action,
        critique,
        error,
        skill_memory,
        episode_summary,
        picked_object,
        beliefs,
    ):
        content = await llm_call(
            self.candidate_model_client,
            system_prompt=safe_format(system_prompt_candidate_interventions, environment_prompt=environment_prompt),
            user_prompt=instruction_prompt_p2_candidate + messages[0].content[0],
            frame=last_frame,
            cancellation_token=cancellation_token,
            log_prefix=f"Agent {agent_name} on_messages: ",
            task=task,
            last_action=last_action,
            critique=critique,
            error=error,
            skill_memory=skill_memory,
            episode_summary=episode_summary,
            picked_object=picked_object,
            **beliefs,
        )
        return content
