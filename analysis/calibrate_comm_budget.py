"""Calibrate the communication-budget ladder against the served tokenizer.

The ladder in hpc/daic/experiments/submit_comm_budget.sh is defined in
MESSAGES (50 / 200 / 800 per agent per 1000-step episode) but charged in
TOKENS. The token values (800 / 3200 / 12800) assume ~16 tokens per short
message. This script measures the real cost of the messages agents actually
wrote (any messages.jsonl on disk) under the model's tokenizer and prints the
ladder to pin before the first submission.

    # exact, with the served model's tokenizer (login node, no GPU needed)
    python analysis/calibrate_comm_budget.py --model $WORKSPACE/models/gemma-4-E4B-it

    # tokenizer-free estimate (whitespace words x 1.3), e.g. locally
    python analysis/calibrate_comm_budget.py

    # a different reference suite / message-count ladder
    python analysis/calibrate_comm_budget.py --runs-root runs_from_daic/compute/agent_scaling_3f \
        --msg-counts 50 200 800 --json paper_assets/comm_budget/calibration.json

Prints tokens-per-message percentiles, the tokens-per-word ratio, the share
of messages that the per-message cap would cut, and the pinned ladder
(msgs x median tokens per message, rounded to the nearest 100).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from paths import RUNS, group  # noqa: E402  (also puts siblings on sys.path)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindforge.env.comm_budget import (  # noqa: E402
    DEFAULT_MSG_CAP,
    FALLBACK_TOKENS_PER_WORD,
    fallback_token_count,
)


def _percentile(values, p):
    if not values:
        return float("nan")
    v = sorted(values)
    k = (len(v) - 1) * p
    lo = int(math.floor(k))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def load_counter(model: str | None):
    """Return (counter, label). Exact tokenizer when --model is given."""
    if not model:
        return fallback_token_count, f"estimate (words x {FALLBACK_TOKENS_PER_WORD})"
    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover - cluster has it
        sys.exit(f"transformers is required for --model: {e}")
    try:
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception:
        # Vision checkpoints ship an AutoProcessor wrapping the tokenizer.
        from transformers import AutoProcessor
        tok = getattr(AutoProcessor.from_pretrained(model, trust_remote_code=True),
                      "tokenizer")

    def _count(text) -> int:
        if not text:
            return 0
        return len(tok(str(text), add_special_tokens=False)["input_ids"])

    return _count, f"tokenizer of {model}"


def iter_messages(runs_root: Path, pattern: str, limit: int | None):
    n = 0
    for f in sorted(runs_root.glob(pattern)):
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = m.get("text") or ""
                if not text:
                    continue
                yield text
                n += 1
                if limit and n >= limit:
                    return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=None,
                    help="Model dir whose tokenizer charges the budget "
                         "(omit for the whitespace estimate)")
    ap.add_argument("--runs-root", type=Path, default=None,
                    help="Run group holding reference messages.jsonl files "
                         "(default: the agent_scaling_3f group — Gemma-E4B, "
                         "the sweep's backbone)")
    ap.add_argument("--glob", default="**/episodes/ep_*/messages.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many messages")
    ap.add_argument("--msg-counts", type=int, nargs="+", default=[50, 200, 800],
                    help="Ladder in messages per agent per episode")
    ap.add_argument("--msg-cap", type=int, default=DEFAULT_MSG_CAP)
    ap.add_argument("--json", type=Path, default=None,
                    help="Also write the calibration record here")
    args = ap.parse_args()

    runs_root = args.runs_root or group("agent_scaling_3f", RUNS)
    if not runs_root.is_dir():
        sys.exit(f"no such run root: {runs_root}")
    counter, label = load_counter(args.model)

    tokens, words = [], []
    for text in iter_messages(runs_root, args.glob, args.limit):
        tokens.append(counter(text))
        words.append(len(text.split()))
    if not tokens:
        sys.exit(f"no messages found under {runs_root}/{args.glob}")

    p50 = _percentile(tokens, 0.5)
    rec = {
        "counter": label,
        "runs_root": str(runs_root),
        "n_messages": len(tokens),
        "tokens_per_msg": {
            "mean": sum(tokens) / len(tokens),
            "p50": p50, "p90": _percentile(tokens, 0.9),
            "p95": _percentile(tokens, 0.95), "p99": _percentile(tokens, 0.99),
            "max": max(tokens),
        },
        "tokens_per_word": sum(tokens) / max(1, sum(words)),
        "msg_cap": args.msg_cap,
        "share_cut_by_cap": sum(1 for t in tokens if t > args.msg_cap) / len(tokens),
        "ladder": {
            str(n): int(round(n * p50 / 100.0) * 100) for n in args.msg_counts
        },
    }

    print(f"counter        : {rec['counter']}")
    print(f"runs_root      : {rec['runs_root']}")
    print(f"messages       : {rec['n_messages']}")
    t = rec["tokens_per_msg"]
    print(f"tokens/msg     : mean {t['mean']:.1f}  p50 {t['p50']:.0f}  p90 {t['p90']:.0f}"
          f"  p95 {t['p95']:.0f}  p99 {t['p99']:.0f}  max {t['max']}")
    print(f"tokens/word    : {rec['tokens_per_word']:.2f}")
    print(f"cap {args.msg_cap:>3d} cuts    : {100 * rec['share_cut_by_cap']:.1f}% of messages")
    print("ladder (tokens): " + "  ".join(
        f"{n} msgs -> {v}" for n, v in rec["ladder"].items()))
    print("submit with    : BUDGETS=\"0 "
          + " ".join(str(v) for v in rec["ladder"].values())
          + "\" bash hpc/daic/experiments/submit_comm_budget.sh")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
