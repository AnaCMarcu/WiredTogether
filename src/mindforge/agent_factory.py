"""Prompt loading and agent construction.

Builds the per-agent cognitive stack described in Appendix C: the prompt
bundle read off disk, the role configuration for a team of N agents, and
the :class:`~mindforge.custom_agent.CustomAgent` instances with their
belief / curriculum / critic / memory / social modules wired up.
"""

from __future__ import annotations

import os

from mindforge.agent_modules.action_selection import ActionSelection
from mindforge.agent_modules.auto_curriculum import AutoCurriculum
from mindforge.agent_modules.belief_system import BeliefSystem
from mindforge.agent_modules.critic import Critic
from mindforge.agent_modules.episodic_memory_manager import EpisodicMemoryManager
from mindforge.agent_modules.skill_manager import SkillManager
from mindforge.agent_modules.social_module import SocialModule
from mindforge.custom_agent import CustomAgent
from mindforge.run_layout import RunPaths
from rl_layer import RLLayer

#: Role prompt files shipped in ``prompts/``; index order is the role id the
#: centralised critic encodes.
ROLE_NAMES = ["agent", "hunter", "harvester", "scouter"]


def load_prompts():
    """Load all prompt files and return them as a dict."""
    prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
    belief_dir = os.path.join(prompt_dir, "belief_system")

    def _read(path):
        with open(path, "r") as f:
            return f.read()

    prompts = {
        "environment": _read(os.path.join(prompt_dir, "environment_prompt.txt")),
        "system_template": _read(os.path.join(prompt_dir, "system_prompt.txt")),
        "critic": _read(os.path.join(prompt_dir, "critic_prompt.txt")),
        "curriculum_questions": _read(os.path.join(prompt_dir, "curriculum_questions.txt")),
        "skill_description": _read(os.path.join(prompt_dir, "skill_description_prompt.txt")),
        "skill_info": _read(os.path.join(prompt_dir, "skill_description_info.txt")),
        "perception": _read(os.path.join(belief_dir, "perception_beliefs.txt")),
        "partner": _read(os.path.join(belief_dir, "partner_beliefs.txt")),
        "interaction": _read(os.path.join(belief_dir, "interaction_belief.txt")),
        "context": _read(os.path.join(belief_dir, "update_context.txt")),
    }

    # Role prompts
    prompts["roles"] = {}
    for role in ROLE_NAMES:
        prompts["roles"][role] = _read(os.path.join(prompt_dir, f"role_{role}.txt"))

    return prompts


def build_role_configs(
    num_agents,
    role_prompts,
    team_mode="homogeneous-agent",
    homogeneous_role="agent",
    roles=None,
):
    """Build ROLE_CONFIGS for num_agents.

    Two modes:
      * homogeneous-agent: every agent gets ``homogeneous_role`` (default
        "agent" — matches all prior runs).
      * heterogeneous: ``roles`` is a list of role names of length
        ``num_agents``; agent_i is assigned roles[i]. This is what makes
        the Hebbian bonds meaningful — symmetric teams give symmetric W.

    ``roles`` may also be a comma-separated string for CLI convenience.
    """
    if team_mode == "heterogeneous":
        if roles is None:
            raise ValueError(
                "team_mode='heterogeneous' requires --roles "
                "(comma-separated list, length == num_agents)."
            )
        if isinstance(roles, str):
            roles = [r.strip() for r in roles.split(",") if r.strip()]
        if len(roles) != num_agents:
            raise ValueError(
                f"--roles has {len(roles)} entries but num_agents is "
                f"{num_agents}. They must match."
            )
        for r in roles:
            if r not in role_prompts:
                raise ValueError(
                    f"Unknown role: {r!r}. Available roles: "
                    f"{sorted(role_prompts.keys())}."
                )
        assigned_roles = list(roles)
    else:
        if homogeneous_role not in role_prompts:
            raise ValueError(
                f"Unknown homogeneous_role: {homogeneous_role!r}. "
                f"Available roles: {sorted(role_prompts.keys())}."
            )
        assigned_roles = [homogeneous_role] * num_agents

    return [
        {
            "name": assigned_roles[i],
            "agent_name": f"agent_{i}",
            "curriculum_prompt": role_prompts[assigned_roles[i]].format(
                num_agents=num_agents
            ),
        }
        for i in range(num_agents)
    ]


def build_agents(role_configs, system_prompt, prompts, num_agents, communication, metric,
                 rl_config=None, belief_interval=5, critic_interval=20,
                 centralized_critic=None, is_resume: bool = False,
                 social_module_mode: str = "none", social_interval: int = 8,
                 social_act_mode: str = "legacy",
                 social_act_channels: tuple = (),
                 orchestrator_plan: bool = False,
                 orchestrator_villager: bool = False):
    """Initialize all Mindforge agents.

    ``centralized_critic`` (when not None) is shared by all agents' RLLayers
    and turns the value-loss off in their PPO updates.

    ``is_resume`` controls whether per-agent persistent stores (skill DB) are
    wiped at construction. Fresh runs reset; chained-checkpoint resumes
    preserve previously-learned skills.

    ``social_act_mode`` / ``social_act_channels`` (Experiment 2): in
    "choice" mode agents are built with the PARALLEL choice-mode prompt
    templates + the SocialAgentResponse schema; "legacy" (default) keeps the
    original templates and AgentResponse byte-for-byte.
    """
    # O-plan orchestrator variant: curriculum USER template with the
    # {team_plan_note} placeholder appended. None in every other
    # configuration → AutoCurriculum falls back to its module-level default,
    # byte-identical to the historical prompt.
    _task_info_override = None
    if orchestrator_plan:
        from mindforge.agent_modules.auto_curriculum import curriculum_info as _cur_info
        from orchestrator.curriculum_hook import apply_plan_suffix
        _task_info_override = apply_plan_suffix(_cur_info, True)
    elif orchestrator_villager:
        # Villager: HARD assignment block ({assigned_objective}) instead of
        # the advisory plan-note block.
        from mindforge.agent_modules.auto_curriculum import curriculum_info as _cur_info
        from orchestrator.curriculum_hook import apply_villager_suffix
        _task_info_override = apply_villager_suffix(_cur_info, True)

    # Choice-mode template/client setup — built once, shared by all agents.
    _choice_action_kwargs = {}
    _choice_sm_prompt = None
    if social_act_mode == "choice":
        from mindforge.agent_modules import social_acts as _sacts
        from mindforge.agent_modules.util import safe_format as _safe_format
        _choice_system_txt, _choice_instruction = _sacts.load_choice_templates(
            social_act_channels
        )
        _choice_action_kwargs = {
            "system_prompt": _safe_format(
                _choice_system_txt, environment_prompt=prompts["environment"]
            ),
            "user_prompt_template": _choice_instruction,
        }
        if social_module_mode != "none":
            _choice_sm_prompt = _sacts.load_social_module_choice_prompt(
                social_act_channels
            )

    agents = []
    for i, role_cfg in enumerate(role_configs):
        # Build per-agent RL layer (no-op when rl_config.enabled is False)
        rl_layer = None
        if rl_config and rl_config.enabled:
            rl_layer = RLLayer(
                config=rl_config, role=role_cfg["name"], agent_id=i,
                centralized_critic=centralized_critic,
            )

        # Targeted communication policy lives in the static prompts. The LLM
        # uses its own agent name (passed in via the user message) to exclude
        # itself from the recipient list.
        agent_system_prompt = system_prompt

        # Optional Hebbian-driven social-reasoning module. Stays None when
        # --social-module=none so the agent loop falls back to the legacy
        # raw bond text in the action prompt.
        agent_social_module = None
        if social_module_mode != "none":
            if social_act_mode == "choice":
                from mindforge.agent_modules.util import (
                    SocialThoughtChoice as _STC,
                    create_model_client as _cmc,
                )
                agent_social_module = SocialModule(
                    agent_name=role_cfg["agent_name"],
                    num_agents=num_agents,
                    social_interval=social_interval,
                    social_model_client=_cmc(response_format=_STC),
                    override_prompt=_choice_sm_prompt,
                )
            else:
                agent_social_module = SocialModule(
                    agent_name=role_cfg["agent_name"],
                    num_agents=num_agents,
                    social_interval=social_interval,
                )

        if social_act_mode == "choice":
            from mindforge.agent_modules.util import (
                SocialAgentResponse as _SAR,
                create_model_client as _cmc2,
            )
            _action_selection = ActionSelection(
                action_model_client=_cmc2(response_format=_SAR),
                **_choice_action_kwargs,
            )
        else:
            _action_selection = ActionSelection(system_prompt=agent_system_prompt)

        agent = CustomAgent(
            name=role_cfg["agent_name"],
            description=f"{role_cfg['name']} agent in Craftium open world",
            action_selection=_action_selection,
            auto_curriculum=AutoCurriculum(
                override_curriculum_prompt=role_cfg["curriculum_prompt"],
                override_questions_prompt=prompts["curriculum_questions"],
                override_task_info_prompt=_task_info_override,
                agent_name=role_cfg["agent_name"],
            ),
            critic=Critic(override_critic_prompt=prompts["critic"]),
            skill_manager=SkillManager(
                override_skill_prompt=prompts["skill_description"],
                override_skill_info_prompt=prompts["skill_info"],
                agent_name=role_cfg["agent_name"],
                # On resume from a checkpoint, preserve the per-agent
                # skill DB so skills learned in earlier chained runs
                # remain available. Fresh runs wipe (default).
                reset=not is_resume,
            ),
            episode_manager=EpisodicMemoryManager(
                agent_name=role_cfg["agent_name"],
            ),
            belief_system=BeliefSystem(
                number_of_agents=num_agents,
                override_perception_prompt=prompts["perception"],
                override_partner_prompt=prompts["partner"],
                override_interaction_prompt=prompts["interaction"],
                override_context_prompt=prompts["context"],
            ),
            number_of_agents=num_agents,
            metric=metric,
            voyager=False,
            rl_layer=rl_layer,
            belief_interval=belief_interval,
            critic_interval=critic_interval,
            num_agents=num_agents,
            social_module=agent_social_module,
        )
        agents.append(agent)
        rl_status = " [RL enabled]" if rl_layer else ""
        print(f"Initialized agent {i}: {role_cfg['agent_name']} ({role_cfg['name']}){rl_status}")
    return agents


def _resume_run_paths(run_id: str, group: str | None = None) -> RunPaths:
    """Reconstruct a ``RunPaths`` from a saved ``run_id`` regardless of layout.

    Two cases:
      * ``"<tag>/seed_<N>"`` (Phase B++ tagged layout) → lives under
        ``runs/<group>/<tag>/seed_<N>/``.
      * Anything else (legacy timestamp-based id) → lives under
        ``runs/<run_id>/``.

    Detected by the presence of ``/seed_`` in the run_id, which can only
    come from ``RunPaths.create_tagged``. The dataclass stores both
    forms identically — only the on-disk root differs.

    ``group`` must match the run being resumed (the run_id does not carry
    it); passing ``None`` resolves it from the environment exactly as the
    original run did.
    """
    if "/seed_" in run_id and run_id.count("/") == 1:
        tag, seed_part = run_id.split("/", 1)
        try:
            seed = int(seed_part.removeprefix("seed_"))
        except ValueError:
            # Malformed id — fall back to the untagged factory.
            return RunPaths.create(run_id=run_id, root="runs")
        return RunPaths.create_tagged(tag=tag, seed=seed, group=group)
    return RunPaths.create(run_id=run_id, root="runs")
