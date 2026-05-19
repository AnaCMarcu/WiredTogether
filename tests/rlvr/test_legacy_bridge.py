"""Tests for the legacy → GRPO schema bridge in ``rlvr.legacy_bridge``.

Covers:
    * ``infer_tag`` truth table (M1/L1/L2/M2/M3/M4/M5)
    * ``extract_seed`` resolution priority (config.seed > --seed in args)
    * per-section translators (``translate_time_to_first``,
      ``translate_hebbian_snapshots``, ``translate_metrics_jsonl``,
      ``gather_episode_summaries``)
    * end-to-end ``translate_run`` produces all four sidecars + tag.txt
    * directory mode ``translate_directory`` handles mixed runs
    * downstream consumption: ``RunMetrics`` loads the translated JSONL
      and ``aggregate_seeds`` works on it
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.compare import (
    aggregate_seeds,
    load_runs,
)
from rlvr.legacy_bridge import (
    extract_seed,
    gather_episode_summaries,
    infer_tag,
    translate_directory,
    translate_hebbian_snapshots,
    translate_metrics_jsonl,
    translate_run,
    translate_time_to_first,
)


# ──── infer_tag truth table ─────────────────────────────────────────────


@pytest.mark.parametrize("flags, expected", [
    # M1 — plain LLM, no RL, no Hebbian.
    ([], "M1"),
    (["--num-agents", "3"], "M1"),

    # L1 — LLM + Hebbian-in-prompt, no RL, no reward propagation.
    (["--hebbian"], "L1"),
    (["--num-agents", "3", "--hebbian"], "L1"),

    # L2 — LLM + Hebbian-in-prompt + reward propagation, no RL.
    (["--hebbian", "--reward-propagation"], "L2"),
    (["--reward-propagation", "--hebbian"], "L2"),  # order-insensitive

    # M2 — MAPPO (centralized critic), no Hebbian.
    (["--rl"], "M2"),
    (["--rl", "--rl-critic-mode", "centralized"], "M2"),

    # M3 — MAPPO + Hebbian.
    (["--rl", "--hebbian"], "M3"),

    # M4 — IPPO (independent critic), no Hebbian.
    (["--rl", "--rl-critic-mode", "independent"], "M4"),

    # M5 — IPPO + Hebbian.
    (["--rl", "--rl-critic-mode", "independent", "--hebbian"], "M5"),
])
def test_infer_tag(flags, expected):
    assert infer_tag(flags) == expected


def test_infer_tag_rejects_rl_plus_reward_propagation():
    """The combination isn't defined in Phase B+ — should fail loudly."""
    with pytest.raises(ValueError):
        infer_tag(["--rl", "--reward-propagation"])


def test_infer_tag_handles_string_cli_args():
    """``cli_args`` may be a whole-command string rather than a list."""
    assert infer_tag("python multi_agent_craftium.py --rl --hebbian") == "M3"


def test_infer_tag_none_returns_m1():
    """When cli_args is missing entirely, fall back to the default (M1)."""
    assert infer_tag(None) == "M1"


# ──── extract_seed ──────────────────────────────────────────────────────


def test_extract_seed_from_config_field():
    config = {"seed": 42}
    assert extract_seed(config) == 42


def test_extract_seed_falls_back_to_cli_args():
    config = {"seed": None, "cli_args": ["--seed", "7", "--rl"]}
    assert extract_seed(config) == 7


def test_extract_seed_default_when_missing():
    assert extract_seed({}, default=99) == 99


def test_extract_seed_handles_non_int_cli_value():
    """If --seed has a non-integer value (corruption), fall back to default."""
    config = {"cli_args": ["--seed", "notanint"]}
    assert extract_seed(config, default=5) == 5


# ──── translate_time_to_first ───────────────────────────────────────────


def test_translate_time_to_first_flattens_tracks():
    final_metrics = {
        "steps_to_milestone": {
            "ch1_solo": {"m1_move_5": 10, "m2_dig_3_any": None},
            "ch3_switches": {"m17_switch_pressed": 42},
        },
    }
    out = translate_time_to_first(final_metrics)
    assert out == {
        "m1_move_5": 10,
        "m2_dig_3_any": None,
        "m17_switch_pressed": 42,
    }


def test_translate_time_to_first_handles_missing_section():
    assert translate_time_to_first({}) == {}


def test_translate_time_to_first_handles_malformed():
    """A track whose value isn't a dict → skipped."""
    final_metrics = {
        "steps_to_milestone": {
            "ch1_solo": "not_a_dict",
            "ch3_switches": {"m17_switch_pressed": 1},
        },
    }
    out = translate_time_to_first(final_metrics)
    assert out == {"m17_switch_pressed": 1}


# ──── translate_hebbian_snapshots ──────────────────────────────────────


def test_translate_hebbian_snapshots_adds_enabled_flag():
    final_metrics = {
        "graph_snapshots": [
            {"step": 100, "mean_bond_strength": 0.3, "sparsity": 0.5},
            {"step": 200, "mean_bond_strength": 0.4, "sparsity": 0.4},
        ],
    }
    out = translate_hebbian_snapshots(final_metrics)
    assert len(out) == 2
    for snap in out:
        assert snap["enabled"] is True
    assert out[0]["mean_bond_strength"] == 0.3


def test_translate_hebbian_snapshots_empty():
    assert translate_hebbian_snapshots({}) == []


# ──── translate_metrics_jsonl ──────────────────────────────────────────


def test_translate_metrics_jsonl_produces_one_record_per_snapshot():
    final_metrics = {
        "config": {"num_agents": 2},
        "timestep_data": {
            "timesteps": [10, 20, 30],
            "cumulative_returns": {
                "0": [1.0, 2.0, 3.0],
                "1": [0.5, 1.5, 2.5],
            },
            "milestone_count": {
                "0": [0, 1, 1],
                "1": [0, 0, 1],
            },
            "total_milestones": [0, 1, 2],
        },
    }
    records = translate_metrics_jsonl(final_metrics)
    assert len(records) == 3
    # Step values track timesteps.
    assert [r["step"] for r in records] == [10, 20, 30]
    # Group mean reward = mean of per-agent cumulative at each snapshot.
    assert records[0]["group_mean_reward"] == pytest.approx(0.75)  # (1 + 0.5) / 2
    assert records[1]["group_mean_reward"] == pytest.approx(1.75)  # (2 + 1.5) / 2
    # Milestone fires = delta in total_milestones.
    assert records[0]["milestone_fires"] == 0
    assert records[1]["milestone_fires"] == 1
    assert records[2]["milestone_fires"] == 1


def test_translate_metrics_jsonl_zero_fills_grpo_fields():
    """GRPO-specific fields (kl_loss, surrogate_loss, ...) must be 0.0."""
    final_metrics = {
        "config": {"num_agents": 1},
        "timestep_data": {
            "timesteps": [1],
            "cumulative_returns": {"0": [0.5]},
            "milestone_count": {"0": [0]},
            "total_milestones": [0],
        },
    }
    record = translate_metrics_jsonl(final_metrics)[0]
    for key in ("kl_loss", "surrogate_loss", "fraction_clipped", "grad_norm",
                "borrowed_fraction", "hebbian_mean_bond"):
        assert record[key] == 0.0


def test_translate_metrics_jsonl_includes_chamber_dict():
    """All 6 chambers appear with zero counts (legacy doesn't split by chamber)."""
    final_metrics = {
        "config": {"num_agents": 1},
        "timestep_data": {
            "timesteps": [1], "cumulative_returns": {"0": [0.0]},
            "milestone_count": {"0": [0]}, "total_milestones": [0],
        },
    }
    record = translate_metrics_jsonl(final_metrics)[0]
    from rlvr.metrics_grpo import CHAMBERS
    assert set(record["milestone_fires_by_chamber"].keys()) == set(CHAMBERS)
    assert all(v == 0 for v in record["milestone_fires_by_chamber"].values())


def test_translate_metrics_jsonl_empty_run():
    """A run with no snapshots → no records (consistent with GRPO behaviour)."""
    final_metrics = {
        "config": {"num_agents": 3},
        "timestep_data": {
            "timesteps": [], "cumulative_returns": {},
            "milestone_count": {}, "total_milestones": [],
        },
    }
    assert translate_metrics_jsonl(final_metrics) == []


# ──── gather_episode_summaries ──────────────────────────────────────────


def test_gather_episode_summaries(tmp_path: Path):
    for i in range(3):
        ep_dir = tmp_path / "episodes" / f"ep_{i:04d}"
        ep_dir.mkdir(parents=True)
        (ep_dir / "episode_summary.json").write_text(
            json.dumps({"episode": i, "cooperation_score": 0.5 + 0.1 * i}),
            encoding="utf-8",
        )
    out = gather_episode_summaries(tmp_path)
    assert [r["episode"] for r in out] == [0, 1, 2]


def test_gather_episode_summaries_falls_back_to_summary_json(tmp_path: Path):
    """``EpisodeLogger.finalize`` writes BOTH ``episode_summary.json`` and
    ``summary.json``. If only ``summary.json`` exists, accept it."""
    ep_dir = tmp_path / "episodes" / "ep_0000"
    ep_dir.mkdir(parents=True)
    (ep_dir / "summary.json").write_text(
        json.dumps({"cooperation_score": 0.42}), encoding="utf-8",
    )
    out = gather_episode_summaries(tmp_path)
    assert len(out) == 1


def test_gather_episode_summaries_missing_episodes_dir(tmp_path: Path):
    assert gather_episode_summaries(tmp_path) == []


def test_gather_episode_summaries_skips_corrupt_json(tmp_path: Path):
    """Corrupt JSON → skipped silently, other episodes still loaded."""
    ep0 = tmp_path / "episodes" / "ep_0000"
    ep0.mkdir(parents=True)
    (ep0 / "episode_summary.json").write_text("not json", encoding="utf-8")
    ep1 = tmp_path / "episodes" / "ep_0001"
    ep1.mkdir(parents=True)
    (ep1 / "episode_summary.json").write_text(
        json.dumps({"good": True}), encoding="utf-8",
    )
    out = gather_episode_summaries(tmp_path)
    assert len(out) == 1
    assert out[0]["good"] is True


# ──── translate_run end-to-end ─────────────────────────────────────────


def _write_legacy_run(
    run_dir: Path,
    cli_args: list[str],
    seed: int = 42,
    num_agents: int = 3,
    snapshots: list[int] | None = None,
) -> Path:
    """Build a synthetic legacy run on disk."""
    snapshots = snapshots or [10, 20, 30]
    run_dir.mkdir(parents=True, exist_ok=True)

    cum_returns = {str(a): [(a + 1) * s for s in snapshots]
                   for a in range(num_agents)}
    milestone_count = {str(a): [s // 10 for s in snapshots]
                       for a in range(num_agents)}
    total_milestones = [sum(milestone_count[str(a)][i]
                            for a in range(num_agents))
                        for i in range(len(snapshots))]

    final_metrics = {
        "config": {
            "num_agents": num_agents, "seed": seed,
            "cli_args": cli_args,
        },
        "timestep_data": {
            "timesteps": snapshots,
            "cumulative_returns": cum_returns,
            "milestone_count": milestone_count,
            "total_milestones": total_milestones,
        },
        "steps_to_milestone": {
            "ch1_solo": {"m1_move_5": 5},
            "ch3_switches": {"m17_switch_pressed": 25, "m18_door_opened": None},
        },
        "graph_snapshots": [
            {"step": 10, "mean_bond_strength": 0.2, "sparsity": 0.7},
            {"step": 20, "mean_bond_strength": 0.4, "sparsity": 0.5},
        ],
    }
    (run_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics), encoding="utf-8",
    )

    # One episode summary file.
    ep_dir = run_dir / "episodes" / "ep_0001"
    ep_dir.mkdir(parents=True)
    (ep_dir / "episode_summary.json").write_text(
        json.dumps({
            "cooperation_score": 0.42,
            "communication_efficacy": 0.55,
            "carry_imbalance": 1.2,
            "ch4_damage_gini": 0.3,
            "ch5_damage_gini": 0.25,
        }),
        encoding="utf-8",
    )
    return run_dir


def test_translate_run_produces_all_four_sidecars(tmp_path: Path):
    run_dir = _write_legacy_run(
        tmp_path / "legacy_run",
        cli_args=["--rl", "--hebbian"],
        seed=42,
    )
    summary = translate_run(run_dir, tmp_path / "translated")
    assert summary["tag"] == "M3"
    assert summary["seed"] == 42

    out_dir = tmp_path / "translated" / "M3" / "seed_42"
    assert (out_dir / "grpo_metrics.jsonl").exists()
    assert (out_dir / "time_to_first.json").exists()
    assert (out_dir / "episode_summary.jsonl").exists()
    assert (out_dir / "hebbian_snapshots.jsonl").exists()
    assert (out_dir / "tag.txt").read_text(encoding="utf-8").strip() == "M3"


def test_translate_run_tag_override(tmp_path: Path):
    """Manual tag overrides auto-classification (useful for unusual flag combos)."""
    run_dir = _write_legacy_run(
        tmp_path / "run", cli_args=["--rl", "--hebbian"], seed=7,
    )
    summary = translate_run(
        run_dir, tmp_path / "translated", tag="custom_tag",
    )
    assert summary["tag"] == "custom_tag"
    assert (tmp_path / "translated" / "custom_tag" / "seed_7").exists()


def test_translate_run_raises_when_final_metrics_missing(tmp_path: Path):
    empty = tmp_path / "empty_run"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        translate_run(empty, tmp_path / "out")


# ──── translated output integrates with Phase A aggregator ────────────


def test_translated_run_is_consumable_by_aggregate_seeds(tmp_path: Path):
    """End-to-end: translate → ``load_runs`` → ``aggregate_seeds``."""
    # Build 3 seeds of a synthetic M3 run.
    for seed_idx in range(3):
        run_dir = _write_legacy_run(
            tmp_path / "legacy" / f"run_{seed_idx}",
            cli_args=["--rl", "--hebbian"],
            seed=seed_idx,
        )
    translate_directory(tmp_path / "legacy", tmp_path / "translated")

    # Load via Phase A's RunMetrics + aggregate.
    seed_paths = sorted(
        (tmp_path / "translated" / "M3").glob("seed_*/grpo_metrics.jsonl")
    )
    assert len(seed_paths) == 3
    runs = load_runs(seed_paths, labels=[p.parent.name for p in seed_paths])

    summary = aggregate_seeds(runs, label="M3", window=2, bootstrap=200)
    assert summary.label == "M3"
    assert summary.n_seeds == 3
    assert "milestone_fire_rate" in summary.final_metrics
    # Cooperation block populated from the episode_summary sidecar.
    assert "cooperation_score" in summary.cooperation
    # Time-to-first sidecar consumed too.
    assert "m1_move_5" in summary.time_to_first
    assert summary.time_to_first["m1_move_5"]["median"] == 5.0


# ──── translate_directory ──────────────────────────────────────────────


def test_translate_directory_handles_multiple_runs(tmp_path: Path):
    """A directory containing 3 distinct legacy runs → 3 translated trees."""
    _write_legacy_run(tmp_path / "legacy" / "run_a",
                       cli_args=[], seed=0)        # M1
    _write_legacy_run(tmp_path / "legacy" / "run_b",
                       cli_args=["--hebbian"], seed=1)  # L1
    _write_legacy_run(tmp_path / "legacy" / "run_c",
                       cli_args=["--rl"], seed=2)       # M2

    summaries = translate_directory(tmp_path / "legacy", tmp_path / "out")
    tags = sorted(s["tag"] for s in summaries)
    assert tags == ["L1", "M1", "M2"]


def test_translate_directory_skips_non_run_subdirs(tmp_path: Path):
    """Subdirs without ``final_metrics.json`` are silently skipped."""
    (tmp_path / "legacy").mkdir()
    (tmp_path / "legacy" / "not_a_run").mkdir()
    _write_legacy_run(tmp_path / "legacy" / "real_run",
                       cli_args=[], seed=0)
    summaries = translate_directory(tmp_path / "legacy", tmp_path / "out")
    assert len(summaries) == 1
    assert summaries[0]["tag"] == "M1"
