"""GRPO + RLVR training entry point.

Parallel to ``multi_agent_craftium.py`` (legacy MAPPO/IPPO/token-PPO entry
point). The legacy file is unchanged — this is an additive sibling.

Launch from the project root with the project convention
(``PYTHONPATH=src``, ``cd src/mindforge``):

    cd src/mindforge
    PYTHONPATH=../ python multi_agent_craftium_grpo.py \\
        --config ../../configs/rlvr/grpo_single_agent_ch3.yaml

See ``docs/rlvr_grpo_plan.md`` §5.2 for the full design. Stage 2 is
single-agent (train one of N agents; the others act under the same LLM
but their outputs don't feed gradient). Stage 3 generalises to N agents
trained simultaneously.

Local-env caveat: this file imports torch / transformers / peft / gymnasium
eagerly. Running it requires the full HPC environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from rlvr.action_parser import _DISCRETE_ACTIONS
from rlvr.grpo_trainer import GRPOConfig, GRPOTrainer
from rlvr.hebbian_grpo_bridge import HebbianGRPOBridge
from rlvr.reference_policy import (
    GRPOLanguageModel,
    GRPOModelConfig,
    LLMPolicy,
    ReferencePolicy,
    default_prompt,
)
from rlvr.rollout_sampler import (
    MultiAgentRolloutSampler,
    MultiAgentSamplerConfig,
    RolloutSampler,
    SamplerConfig,
)
from rlvr.verifier import FiveChambersVerifier, VerifierConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ──── config ───────────────────────────────────────────────────────────


@dataclass
class EnvConfig:
    num_agents: int = 3
    max_episode_steps: int = 1000
    obs_width: int = 320
    obs_height: int = 180
    chamber_subset: list[str] = field(default_factory=lambda: ["ch3"])
    """Stage-2 default: only collect trajectories starting in Ch3 (via
    position-bucket grouping; rollouts in other chambers still happen but
    their groups won't fill quickly)."""

    extra_kwargs: dict = field(default_factory=dict)
    """Anything else to pass to the OpenWorldMultiAgentEnv constructor.
    Set in YAML per your HPC env layout (game paths, server timeouts)."""


@dataclass
class PolicyConfig:
    trained_agent: int = 0
    """Stage-2 single-agent restriction. Ignored if ``trained_agents`` is set."""

    trained_agents: list[int] | None = None
    """Stage-3 multi-agent training. When provided, the entry point uses
    ``MultiAgentRolloutSampler`` and the trainer's multi-agent dispatch.
    Single-element list collapses to single-agent (Stage 2) for symmetry."""

    checkpoint_path: str | None = None
    """If set, load this LoRA checkpoint into the ``grpo_policy`` adapter
    after init. ``None`` = base LLM (Stage-2 plan default, clean comparison)."""

    def effective_trained_agents(self) -> list[int]:
        """Canonical list of trained agent ids. Falls back to ``[trained_agent]``
        when ``trained_agents`` is unset."""
        if self.trained_agents is not None:
            return list(self.trained_agents)
        return [self.trained_agent]


@dataclass
class FullConfig:
    """Top-level YAML schema."""

    env: EnvConfig = field(default_factory=EnvConfig)
    llm: GRPOModelConfig = field(default_factory=lambda: GRPOModelConfig(
        base_model_name="MUST_BE_SET_IN_YAML"))
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    hebbian: dict = field(default_factory=dict)   # raw dict, parsed by hebbian.HebbianConfig
    seed: int | None = None
    checkpoint_dir: str | None = None
    log_dir: str | None = None


def load_config(path: str | Path, overrides: list[str] | None = None) -> FullConfig:
    """Load a YAML config and optionally apply ``--set key.subkey=value`` overrides.

    Overrides are applied to the *raw* dict before constructing dataclasses, so
    they can introduce new keys the YAML didn't have. Values are parsed loosely
    (int / float / bool / str / null).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data = _apply_overrides(data, overrides)
    return _build_full_config(data)


def _apply_overrides(data: dict, overrides: list[str]) -> dict:
    """Apply a list of ``key.subkey=value`` strings to a nested dict in place.

    Uses dotted-key notation. Missing intermediate dicts are created. Values
    are parsed via ``_parse_override_value``.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"--set requires key=value, got {override!r}"
            )
        key, _, raw_value = override.partition("=")
        keys = [k.strip() for k in key.split(".") if k.strip()]
        if not keys:
            raise ValueError(f"--set has empty key: {override!r}")
        target = data
        for k in keys[:-1]:
            existing = target.get(k)
            if not isinstance(existing, dict):
                existing = {}
                target[k] = existing
            target = existing
        target[keys[-1]] = _parse_override_value(raw_value.strip())
    return data


def _parse_override_value(value: str):
    """Convert a CLI value string to bool / int / float / None / list / str.

    The lossy heuristic is fine for hyperparameters — explicit YAML stays the
    source of truth, ``--set`` is for one-line tweaks (seeds, model paths,
    output dirs).
    """
    lowered = value.lower()
    if lowered in ("null", "none", "~"):
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        # Lightweight list parsing — comma-separated, recurse per element.
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_override_value(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _build_full_config(data: dict) -> FullConfig:
    return FullConfig(
        env=EnvConfig(**(data.get("env") or {})),
        llm=GRPOModelConfig(**(data.get("llm") or {})),
        policy=PolicyConfig(**(data.get("policy") or {})),
        sampler=SamplerConfig(**(data.get("sampler") or {})),
        grpo=GRPOConfig(**(data.get("grpo") or {})),
        verifier=VerifierConfig(**(data.get("verifier") or {})),
        hebbian=data.get("hebbian") or {},
        seed=data.get("seed"),
        checkpoint_dir=data.get("checkpoint_dir"),
        log_dir=data.get("log_dir"),
    )


# ──── env adapter ──────────────────────────────────────────────────────


class SingleAgentEnvAdapter:
    """Wrap a PettingZoo ParallelEnv to look single-agent to ``RolloutSampler``.

    Steps **all** agents per env step. Scenery agents (non-trained) use
    the same ``LLMPolicy`` for action generation — their actions are
    discarded for gradient purposes (the trainer only stores tensors for
    the trained agent's trajectory).

    Compatible with both the old (4-tuple) and new (5-tuple) PettingZoo
    step return shapes.
    """

    def __init__(self, multi_env, trained_agent_id: int, scenery_policy: LLMPolicy):
        self.env = multi_env
        self.trained_agent_name = f"agent_{trained_agent_id}"
        self.scenery_policy = scenery_policy
        self._scenery_obs: dict[str, Any] = {}
        self._scenery_info: dict[str, dict] = {}

    def reset(self) -> tuple[Any, dict]:
        result = self.env.reset()
        # PettingZoo's reset can be 2-tuple (obs, info) or 1-tuple (obs).
        if isinstance(result, tuple) and len(result) == 2:
            obs_dict, info_dict = result
        else:
            obs_dict = result
            info_dict = {}

        for name, obs in obs_dict.items():
            if name != self.trained_agent_name:
                self._scenery_obs[name] = obs
                self._scenery_info[name] = info_dict.get(name, {}) if info_dict else {}

        trained_obs = obs_dict[self.trained_agent_name]
        trained_info = info_dict.get(self.trained_agent_name, {}) if info_dict else {}
        return trained_obs, trained_info

    def step(self, trained_action: dict) -> tuple[Any, float, bool, dict]:
        actions = {self.trained_agent_name: _action_dict_to_env(trained_action)}
        for name in list(self._scenery_obs):
            scen_action, _ = self.scenery_policy.act(
                self._scenery_obs[name], self._scenery_info[name]
            )
            actions[name] = _action_dict_to_env(scen_action)

        step_result = self.env.step(actions)
        if len(step_result) == 5:
            obs_dict, reward_dict, term_dict, trunc_dict, info_dict = step_result
            done_dict = {
                k: bool(term_dict.get(k, False) or trunc_dict.get(k, False))
                for k in obs_dict
            }
        else:
            obs_dict, reward_dict, done_dict, info_dict = step_result

        for name in list(self._scenery_obs):
            if name in obs_dict:
                self._scenery_obs[name] = obs_dict[name]
                self._scenery_info[name] = info_dict.get(name, {})

        return (
            obs_dict[self.trained_agent_name],
            float(reward_dict.get(self.trained_agent_name, 0.0)),
            bool(done_dict.get(self.trained_agent_name, False)),
            info_dict.get(self.trained_agent_name, {}),
        )


def _action_dict_to_env(action_dict: dict) -> int:
    """Convert ``{"action": "dig", ...}`` to the env's integer action
    (Discrete(23): 0 = NOP, 1..22 = ``_DISCRETE_ACTIONS``)."""
    name = action_dict.get("action", "nop")
    if name == "nop":
        return 0
    try:
        return _DISCRETE_ACTIONS.index(name) + 1
    except ValueError:
        return 0


# ──── multi-agent env adapter ─────────────────────────────────────────


class MultiAgentEnvAdapter:
    """Wrap a PettingZoo ParallelEnv to expose the int-keyed
    ``MultiAgentRolloutEnv`` interface that ``MultiAgentRolloutSampler``
    expects.

    No scenery agents — every agent in the env is trained. (Mixed
    trained/scenery in multi-agent mode is a future variant; for Stage 3
    headline runs all N agents are trained.)
    """

    def __init__(self, multi_env):
        self.env = multi_env

    def reset(self) -> tuple[dict[int, object], dict[int, dict]]:
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs_dict, info_dict = result
        else:
            obs_dict, info_dict = result, {}
        return self._intify(obs_dict), self._intify(info_dict)

    def step(
        self, actions: dict[int, dict]
    ) -> tuple[dict[int, object], dict[int, float], dict[int, bool], dict[int, dict]]:
        env_actions = {
            f"agent_{aid}": _action_dict_to_env(action) for aid, action in actions.items()
        }
        result = self.env.step(env_actions)
        if len(result) == 5:
            obs, reward, term, trunc, info = result
            done = {k: bool(term.get(k, False) or trunc.get(k, False))
                    for k in obs}
        else:
            obs, reward, done, info = result
        return (self._intify(obs), self._intify(reward),
                {k: bool(v) for k, v in self._intify(done).items()},
                self._intify(info))

    @staticmethod
    def _intify(d: dict) -> dict:
        """Convert ``{"agent_0": ...}`` to ``{0: ...}``. Drop unparseable keys."""
        out = {}
        if not d:
            return out
        for k, v in d.items():
            try:
                aid = int(str(k).split("_")[-1])
                out[aid] = v
            except (ValueError, AttributeError):
                continue
        return out


# ──── component builders ──────────────────────────────────────────────


def build_multi_agent_env(cfg: EnvConfig):
    """Construct the five-chambers PettingZoo env. See ``EnvConfig.extra_kwargs``
    for HPC-specific options (game paths, server timeouts).

    Defers the import so this module doesn't blow up at import time when
    gymnasium / craftium aren't available (e.g. local dev).
    """
    from marl_craftium.openworld_multi_agents import OpenWorldMultiAgentEnv

    return OpenWorldMultiAgentEnv(
        n_agents=cfg.num_agents,
        max_episode_steps=cfg.max_episode_steps,
        obs_width=cfg.obs_width,
        obs_height=cfg.obs_height,
        **cfg.extra_kwargs,
    )


def build_hebbian(hebbian_cfg: dict):
    """Return a ``HebbianSocialGraph`` if ``enabled`` is True, else ``None``."""
    if not hebbian_cfg or not hebbian_cfg.get("enabled", False):
        return None
    from hebbian import HebbianConfig, HebbianSocialGraph

    hc = HebbianConfig(**{k: v for k, v in hebbian_cfg.items() if k != "enabled"})
    return HebbianSocialGraph(hc)


def set_seeds(seed: int | None) -> None:
    if seed is None:
        return
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ──── main ─────────────────────────────────────────────────────────────


def main(config_path: str, overrides: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = load_config(config_path, overrides=overrides)
    if overrides:
        logger.info("Applied %d config overrides", len(overrides))
    set_seeds(cfg.seed)

    logger.info("Loading model: %s", cfg.llm.base_model_name)
    model = GRPOLanguageModel(cfg.llm)

    if cfg.policy.checkpoint_path:
        logger.info("Loading policy checkpoint from %s", cfg.policy.checkpoint_path)
        model.model.load_adapter(cfg.policy.checkpoint_path, cfg.llm.policy_adapter)

    reference = ReferencePolicy(model)

    llm_policy = LLMPolicy(
        model=model,
        n_agents=cfg.env.num_agents,
        prompt_template=default_prompt,   # entry point can swap for a richer one
    )

    multi_env = build_multi_agent_env(cfg.env)

    # Build Hebbian graph + bridge (None when disabled — passes through cleanly).
    hebbian = build_hebbian(cfg.hebbian)
    hebbian_bridge = HebbianGRPOBridge(hebbian) if hebbian is not None else None
    if hebbian_bridge is not None:
        logger.info("Hebbian enabled: reward_diffusion=%s, group_composition=%s",
                    cfg.verifier.hebbian_reward_diffusion,
                    cfg.grpo.hebbian_group_composition)

    trained_agents = cfg.policy.effective_trained_agents()
    if len(trained_agents) > 1:
        logger.info("Multi-agent mode: training agents %s (team_reward=%s)",
                    trained_agents, cfg.grpo.team_reward)
        env = MultiAgentEnvAdapter(multi_env=multi_env)
        ma_sampler_cfg = MultiAgentSamplerConfig(
            n_per_group=cfg.sampler.n_per_group,
            horizon=cfg.sampler.horizon,
            position_bucket_size=cfg.sampler.position_bucket_size,
            num_agents=cfg.env.num_agents,
            trained_agents=tuple(trained_agents),
            max_resets_per_group=cfg.sampler.max_resets_per_group,
        )
        sampler = MultiAgentRolloutSampler(
            env=env, policy=llm_policy, config=ma_sampler_cfg,
            hebbian_bridge=hebbian_bridge,
        )
    else:
        logger.info("Single-agent mode: training agent %d", trained_agents[0])
        if hebbian_bridge is not None:
            logger.warning(
                "Hebbian is enabled but training is single-agent — graph will "
                "not be updated (Stage 4 wiring is multi-agent only)."
            )
        env = SingleAgentEnvAdapter(
            multi_env=multi_env,
            trained_agent_id=trained_agents[0],
            scenery_policy=llm_policy,
        )
        sampler = RolloutSampler(
            env=env, policy=llm_policy,
            config=cfg.sampler,
            agent_id=trained_agents[0],
        )

    # Keep the verifier's n_agents in sync with the env so action_parser's
    # comm-target validation matches the env's agent count.
    verifier_config = cfg.verifier
    verifier_config.n_agents = cfg.env.num_agents
    verifier = FiveChambersVerifier(verifier_config, hebbian=hebbian)

    checkpoint_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else None
    trainer = GRPOTrainer(
        config=cfg.grpo, model=model, reference=reference,
        verifier=verifier, sampler=sampler, checkpoint_dir=checkpoint_dir,
        hebbian_bridge=hebbian_bridge,
        rng_seed=cfg.seed,
    )

    logger.info(
        "Starting GRPO training: %d steps, G=%d, H=%d, trained_agent=%d",
        cfg.grpo.total_steps, cfg.sampler.n_per_group,
        cfg.sampler.horizon, cfg.policy.trained_agent,
    )
    metrics_path = None
    if cfg.log_dir:
        metrics_path = Path(cfg.log_dir) / "grpo_metrics.jsonl"
        logger.info("Metrics will be appended to %s", metrics_path)
    trainer.train(metrics_path=metrics_path)
    logger.info("Training complete. Final step: %d", trainer.step_idx)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True,
                        help="Path to GRPO YAML config (configs/rlvr/*.yaml)")
    parser.add_argument(
        "--set", action="append", dest="overrides", default=[], metavar="K=V",
        help="Override a config key from the CLI: --set key.subkey=value. "
             "Repeat for multiple. Values parsed loosely (int/float/bool/null/list)."
    )
    args = parser.parse_args()
    sys.exit(main(args.config, overrides=args.overrides))
