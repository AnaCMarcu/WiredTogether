import os
from agent_modules.llm_call import llm_call
from agent_modules.util import (
    AgentResponse,
    TargetedCommunicationResponse,
    create_model_client,
    normalize_agent_target,
    safe_format,
)

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

# load in prompts (explicit UTF-8: the templates carry box-drawing chars,
# which the Windows default cp1252 codec cannot decode)
with open(os.path.join(_PROMPT_DIR, "system_prompt.txt"), "r", encoding="utf-8") as f:
    system_prompt_txt = f.read()
with open(os.path.join(_PROMPT_DIR, "environment_prompt.txt"), "r", encoding="utf-8") as f:
    environment_prompt = f.read()
with open(os.path.join(_PROMPT_DIR, "instruction_prompt_p2.txt"), "r", encoding="utf-8") as f:
    instruction_prompt_p2 = f.read()
with open(os.path.join(_PROMPT_DIR, "instruction_prompt_p2_thoughts.txt"), "r", encoding="utf-8") as f:
    instruction_prompt_p2_thoughts = f.read()


class ActionSelection:
    def __init__(
        self,
        system_prompt=None,
        action_model_client=None,
        user_prompt_template=None,
    ):
        self.system_prompt = (
            system_prompt if system_prompt else safe_format(system_prompt_txt, environment_prompt=environment_prompt)
        )
        # Choice mode (Experiment 2) passes the parallel
        # instruction_prompt_p2_choice template here; legacy leaves it None
        # and keeps the original file byte-for-byte.
        self.user_prompt_template = (
            user_prompt_template if user_prompt_template else instruction_prompt_p2
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
        teammate_names: str = "",
    ):
        content = await llm_call(
            self.action_model_client,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt_template + messages[0].content[0],
            frame=last_frame,
            cancellation_token=cancellation_token,
            log_prefix=f"Agent {agent_name} on_messages: ",
            agent_name=agent_name,
            teammate_names=teammate_names,
            task=task,
            last_action=last_action,
            critique=critique,
            error=error,
            skill_memory=skill_memory,
            episode_summary=episode_summary,
            picked_object=picked_object,
            **beliefs,
        )
        if isinstance(content, dict) and "communication_target" in content:
            content["communication_target"] = normalize_agent_target(
                content["communication_target"]
            )
        if isinstance(content, dict) and "social_target" in content:
            content["social_target"] = normalize_agent_target(
                content["social_target"]
            )
        return content

    async def generate_thoughts_and_comm(
        self,
        messages,
        task,
        last_action,
        critique,
        error,
        skill_memory,
        episode_summary,
        picked_object,
        beliefs,
        last_frame,
        cancellation_token,
        agent_name,
        num_agents: int = 1,
        teammate_names: str = "",
    ):
        """Generate per-step LLM reasoning + targeted message BEFORE action selection.

        In RL mode the action comes from the constrained-generation scorer
        (``RLLayer._score_actions``), which produces no freeform text. This
        call runs FIRST and emits ``{thoughts, communication, communication_target}``
        — the thoughts are then appended to the policy's scoring prompt so
        the action is sampled from ``p(a | prompt, thoughts)``. That ordering
        is chain-of-thought: reasoning drives action, not the reverse.

        Uses ``instruction_prompt_p2_thoughts.txt`` — the SAME context as the
        regular policy prompt (chamber, chamber_state, milestones, beliefs,
        partner beliefs, social directive, propagation summary) minus the
        ``action`` field in the JSON response. Same context as the non-RL
        select_action path, just without asking the LLM to also pick a
        keyword.

        Note: there's no ``action`` parameter here because the action HASN'T
        been chosen yet. The thoughts describe intent; the policy translates
        that into one of the discrete candidate actions.
        """
        comm_client = create_model_client(response_format=TargetedCommunicationResponse)
        if not teammate_names:
            try:
                self_idx = int(str(agent_name).split("_")[-1])
            except (ValueError, IndexError):
                self_idx = -1
            teammate_names = ", ".join(
                f"agent_{j}" for j in range(num_agents) if j != self_idx
            )
        per_step_observation = ""
        if messages and getattr(messages[0], "content", None):
            try:
                per_step_observation = messages[0].content[0]
            except (IndexError, TypeError):
                per_step_observation = ""
        content = await llm_call(
            comm_client,
            system_prompt=self.system_prompt,
            user_prompt=instruction_prompt_p2_thoughts + per_step_observation,
            frame=last_frame,
            cancellation_token=cancellation_token,
            log_prefix=f"Agent {agent_name} rl_thoughts: ",
            agent_name=agent_name,
            teammate_names=teammate_names,
            task=task,
            last_action=last_action,
            critique=critique,
            error=error,
            skill_memory=skill_memory,
            episode_summary=episode_summary,
            picked_object=picked_object,
            **beliefs,
        )
        return (
            content.get("thoughts", ""),
            content.get("communication", ""),
            normalize_agent_target(content.get("communication_target", None)),
        )
