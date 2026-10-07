"""Calibrate the multi-channel three-factor rule for 55-round AgentWorld episodes.

WIRE ran ~1000 Hebbian updates per episode; AgentWorld runs ≤ 55. This replays
synthetic event streams through ``MultiChannelHebbianGraph`` (the deployed
code, not a re-implementation) and reports, per rate setting:

    working pair   DMs every round + a transfer every 4th round, rewarded on transfer
    chatty pair    DMs every round, no transfers, no reward
    near pair      co-located every round, silent
    stranger pair  nothing

Targets (plan §4): the working pair reaches W ≈ 0.6 by round ~20, the chatty
pair stays clearly below it, a silent co-located pair stays low, and after the
pair stops interacting W relaxes back toward its start within one episode.

    python analysis/agentworld/calibrate_rule.py [--rounds 55] [--grid]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hebbian.multichannel import MultiChannelConfig, MultiChannelHebbianGraph  # noqa: E402

# agents: 0–1 working pair, 2–3 chatty pair, 4–5 near pair, 6 stranger
N = 7
WORK, CHAT, NEAR, STRANGER = (0, 1), (2, 3), (4, 5), (0, 6)


def simulate(eta_0, eta_plus, decay, rho, rounds=55, active_until=None, salience="ego"):
    cfg = MultiChannelConfig(enabled=True, num_agents=N, eta_0=eta_0, eta_plus=eta_plus,
                             decay=decay, eligibility_rho=rho, interaction_radius=4.0,
                             reward_norm_R=20.0, salience_mode=salience)
    g = MultiChannelHebbianGraph(cfg)
    active_until = rounds if active_until is None else active_until
    trace = []
    for r in range(1, rounds + 1):
        live = r <= active_until
        pos = [(0, 0), (40, 0), (80, 0), (120, 0), (200, 0), (201, 0), (300, 300)]
        events, bond = [], [0.0] * N
        if live:
            events += [(0, 1, "comm"), (2, 3, "comm")]
            if r % 4 == 0:
                events.append((0, 1, "xfer"))
                bond[0] = 15.0          # 5 relevant items handed over (3 × 5)
                bond[1] = 10.0          # the receiver crafts with them
        else:
            pos[4], pos[5] = (200, 0), (260, 0)
        g.update(pos, social_events=events, bond_rewards=bond, total_rewards=bond)
        trace.append({k: float(g.W[a, b]) for k, (a, b) in
                      {"work": WORK, "chat": CHAT, "near": NEAR, "stranger": STRANGER}.items()})
    return trace


def summarise(eta_0, eta_plus, decay, rho, rounds, salience="ego"):
    full = simulate(eta_0, eta_plus, decay, rho, rounds, salience=salience)
    stop = simulate(eta_0, eta_plus, decay, rho, rounds, active_until=rounds // 2,
                    salience=salience)
    w20 = full[min(19, rounds - 1)]
    end = full[-1]
    relax = stop[-1]["work"] / max(stop[rounds // 2 - 1]["work"], 1e-6)
    return {"work@20": w20["work"], "chat@20": w20["chat"], "near@20": w20["near"],
            "work@end": end["work"], "stranger@end": end["stranger"],
            "relax_ratio": relax}


def score(s) -> float:
    """Lower is better: distance to the targets."""
    return (abs(s["work@20"] - 0.6)
            + max(0.0, s["chat@20"] - 0.6 * s["work@20"])
            + max(0.0, s["near@20"] - 0.25)
            + max(0.0, s["relax_ratio"] - 0.7))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=55)
    ap.add_argument("--grid", action="store_true", help="search a small grid of rates")
    ap.add_argument("--salience", choices=("ego", "joint"), default="ego")
    args = ap.parse_args(argv)
    settings = [("WIRE defaults", 0.001, 0.05, 0.001, 0.9),
                ("first guess (plan)", 0.02, 0.3, 0.005, 0.8),
                ("AgentWorld defaults (cli.py)", 0.01, 0.1, 0.02, 0.7)]
    if args.grid:
        for e0, ep, lam, rho in itertools.product((0.005, 0.01, 0.02), (0.05, 0.08, 0.1, 0.15),
                                                  (0.01, 0.02, 0.03), (0.7, 0.8)):
            settings.append((f"grid e0={e0} e+={ep} lam={lam} rho={rho}", e0, ep, lam, rho))
    rows = []
    for name, e0, ep, lam, rho in settings:
        s = summarise(e0, ep, lam, rho, args.rounds, args.salience)
        rows.append((score(s), name, s))
    shown = rows[:3] + sorted(rows[3:], key=lambda r: r[0])[:8]
    print(f"{'setting':42s} work@20 chat@20 near@20 work@end relax  score")
    for sc, name, s in shown:
        print(f"{name:42s} {s['work@20']:7.2f} {s['chat@20']:7.2f} {s['near@20']:7.2f} "
              f"{s['work@end']:8.2f} {s['relax_ratio']:5.2f}  {sc:5.2f}")


if __name__ == "__main__":
    main()
