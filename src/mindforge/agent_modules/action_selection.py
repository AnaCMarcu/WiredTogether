import os
from agent_modules.llm_call import llm_call
from agent_modules.util import AgentResponse, CommunicationResponse, TargetedCommunicationResponse, create_model_client, safe_format

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

# load in prompts
with open(os.path.join(_PROMPT_DIR, "system_prompt.txt"), "r") as f:
    system_prompt_txt = f.read()
with open(os.path.join(_PROMPT_DIR, "environment_prompt.txt"), "r") as f:
    environment_prompt = f.read()
with open(os.path.join(_PROMPT_DIR, "instruction_prompt_p2.txt"), "r") as f:
    instruction_prompt_p2 = f.read()
with open(os.path.join(_PROMPT_DIR, "rl_communication_prompt.txt"), "r") as f:
    rl_communication_prompt = f.read()


class ActionSelection:
    def __init__(
        self,
        system_prompt=None,
        action_model_client=None,
    ):
        self.system_prompt = (
            system_prompt if system_prompt else safe_format(system_prompt_txt, environment_prompt=environment_prompt)
        )
        self.action_model_client = (
            action_model_client
            if action_model_client
            else create_model_client(response_format=AgentResponse)
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
    ):
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

    async def generate_communication(
        self,
        action,
        task,
        last_action,
        picked_object,
        last_frame,
        cancellation_token,
        agent_name,
        targeted_communication: bool = False,
        num_agents: int = 1,
    ):
        """Generate a natural language communication message for an RL-selected action."""
        if targeted_communication:
            # TargetedCommunicationResponse has communication_target: str (required, not Optional).
            # This forces the JSON schema enforcer to always emit the field — prompt alone is
            # not enough because the model treats Optional fields as skippable.
            comm_client = create_model_client(response_format=TargetedCommunicationResponse)
            try:
                self_idx = int(agent_name.split("_")[1])
            except (IndexError, ValueError):
                self_idx = -1
            targets = ", ".join(
                f"agent_{i}" for i in range(num_agents) if i != self_idx
            )
            # The prompt file uses Python format-string escaping: {{ and }} produce
            # literal braces after .format() is called.  The search string must
            # match the raw file content (double braces), not the rendered output.
            user_prompt = rl_communication_prompt.replace(
                'Respond in JSON: {{"communication": "<message>"}}',
                f'Also set "communication_target" to the most relevant recipient: one of {targets}.\n'
                f'Respond in JSON: {{"communication": "<message>", "communication_target": "<target>"}}'
            )
        else:
            comm_client = create_model_client(response_format=CommunicationResponse)
            user_prompt = rl_communication_prompt
        content = await llm_call(
            comm_client,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            frame=last_frame,
            cancellation_token=cancellation_token,
            log_prefix=f"Agent {agent_name} rl_comm: ",
            task=task,
            last_action=last_action,
            action=action,
            picked_object=picked_object or "empty",
        )
        return content.get("communication", ""), content.get("communication_target", None)
