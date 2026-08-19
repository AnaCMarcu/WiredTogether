"""Retroactive inference-FLOPs accounting for existing runs.

Computes, per run, the compute axis for the Pareto plots:

    FLOPs = 2 * N_eff * (prefill_tokens + decode_tokens)

summed over every LLM call of every agent/module (Kaplan-style 2N-per-token
forward-pass rule; prefill + decode both counted).

Data sources already present in a synced run directory:

  log.txt
      "[LocalModel RAW output] (N tokens): ..." -- EXACT decode token count
      per generate() call (local_model_client._generate logs len(new_tokens)).

      "[LocalModel usage] prompt_tokens=A completion_tokens=B" -- runs made
      after 2026-08-19 also log the EXACT prompt length (image soft tokens and
      chat template included). When these lines are present the char-based
      prefill estimate below is skipped entirely (prefill_source="exact").

  llm_logs/*.log   (one file per module: action_selection, critic, ...)
      "<prefix> call: sys_chars=A user_chars=B frame=True|False ... retry=K"
      -- one line per ATTEMPT (retries are separate calls and separate
      compute). The prompt body is intentionally not logged, so prefill is
      ESTIMATED from character counts:

          prefill_tokens ~= (sys_chars + user_chars) / chars_per_token
                            + overhead_tokens                (template + JSON instr.)
                            + image_tokens * frame            (vision soft tokens)

      chars_per_token is auto-calibrated per run from the same logs: total
      Response-payload characters (llm_logs) / total decode tokens (log.txt).
      Both texts come from the same model + episode distribution, so the
      ratio transfers to the prompt side to within a few percent. Override
      with --chars-per-token if you have a tokenizer-measured value.

Decode tokens are exact; prefill carries the char-ratio approximation. On a
log-scale compute axis the residual error (single-digit percent) is invisible.

Usage:
    python scripts/compute_flops.py runs_from_daic/new_exp_0_gemma
    python scripts/compute_flops.py runs_from_daic/new_exp_0_gemma \
        --n-eff 4.5e9 --image-tokens 280 --csv flops_summary.csv

Defaults are Gemma-4-E4B: N_eff = 4.5e9 (effective params -- PLE tables are
looked up, not multiplied, so raw 8B is wrong for FLOPs) and 280 soft tokens
per image (Gemma 4 processor default; verify against the staged checkpoint's
processor_config.json mm_tokens_per_image). For other backbones pass --n-eff:
E2B 2.3e9, 12B 12e9, 31B 30.7e9, 26B-A4B 3.8e9 (ACTIVE params -- on a FLOPs
axis the MoE sits at its active-compute cost, which resolves the
total-vs-active ambiguity the parameter axis suffers from).

Stdlib only; no tokenizer or torch needed.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

RAW_OUTPUT = re.compile(r"\[LocalModel RAW output\] \((\d+) tokens\)")
USAGE_LINE = re.compile(
    r"\[LocalModel usage\] prompt_tokens=(\d+) completion_tokens=(\d+)"
)
CALL_LINE = re.compile(
    r"call:\s+sys_chars=(\d+) user_chars=(\d+) frame=(True|False)"
)
# llm_logs line header: "2026-08-03 20:06:33 INFO <message>"
LOG_HEADER = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} (?:INFO|ERROR|WARNING) ")
RESPONSE_MARK = "Response: "


def parse_log_txt(path: Path):
    """Token totals from log.txt.

    Returns (decode_total, generate_calls, exact_prefill_total, usage_calls).
    Runs recorded after the "[LocalModel usage]" line shipped carry exact
    prompt_tokens per call (usage_calls > 0); older runs only have the
    RAW-output decode counts and need the char-based prefill estimate.
    """
    decode_total, generate_calls = 0, 0
    prefill_total, usage_calls = 0, 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RAW_OUTPUT.search(line)
            if m:
                decode_total += int(m.group(1))
                generate_calls += 1
                continue
            m = USAGE_LINE.search(line)
            if m:
                prefill_total += int(m.group(1))
                usage_calls += 1
    return decode_total, generate_calls, prefill_total, usage_calls


def parse_llm_log(path: Path):
    """Per-module stats from one llm_logs/*.log file.

    Returns dict with: calls, frames, sys_chars, user_chars, response_chars.
    Response payloads may span multiple lines; any line that does not start
    with a timestamp header continues the previous record.
    """
    stats = dict(calls=0, frames=0, sys_chars=0, user_chars=0, response_chars=0)
    in_response = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if LOG_HEADER.match(line):
                in_response = False
                m = CALL_LINE.search(line)
                if m:
                    stats["calls"] += 1
                    stats["sys_chars"] += int(m.group(1))
                    stats["user_chars"] += int(m.group(2))
                    stats["frames"] += m.group(3) == "True"
                    continue
                idx = line.find(RESPONSE_MARK)
                if idx != -1:
                    stats["response_chars"] += len(line) - idx - len(RESPONSE_MARK)
                    in_response = True
            elif in_response:
                stats["response_chars"] += len(line)
    return stats


def analyze_run(run_dir: Path, args):
    """One row per run; None if the run lacks the needed artifacts."""
    log_txt = run_dir / "log.txt"
    llm_logs = sorted((run_dir / "llm_logs").glob("*.log"))
    if not log_txt.is_file() or not llm_logs:
        return None

    decode_tokens, generate_calls, exact_prefill, usage_calls = \
        parse_log_txt(log_txt)
    modules = {p.stem: parse_llm_log(p) for p in llm_logs}

    calls = sum(m["calls"] for m in modules.values())
    frames = sum(m["frames"] for m in modules.values())
    prompt_chars = sum(m["sys_chars"] + m["user_chars"] for m in modules.values())
    response_chars = sum(m["response_chars"] for m in modules.values())

    if usage_calls > 0:
        # Exact prompt_tokens per call are in log.txt — image soft tokens and
        # template overhead already included, no estimation needed.
        prefill_tokens, prefill_src = exact_prefill, "exact"
        cpt, cpt_src = 0.0, "n/a"
        if usage_calls < generate_calls:  # run predates the line partway? be loud
            print(f"  !! {run_dir}: only {usage_calls}/{generate_calls} calls "
                  f"have usage lines — exact prefill is an undercount",
                  file=sys.stderr)
    else:
        prefill_src = "estimate"
        # chars/token: calibrate on this run's own responses unless overridden.
        if args.chars_per_token:
            cpt, cpt_src = args.chars_per_token, "cli"
        elif decode_tokens > 0 and response_chars > 0:
            cpt = response_chars / decode_tokens
            cpt_src = "auto"
            if not (2.0 <= cpt <= 8.0):  # implausible -> logs disagree; be loud
                print(f"  !! {run_dir}: calibrated chars/token={cpt:.2f} out "
                      f"of range, falling back to 4.0", file=sys.stderr)
                cpt, cpt_src = 4.0, "fallback"
        else:
            cpt, cpt_src = 4.0, "fallback"
        prefill_tokens = int(
            prompt_chars / cpt
            + calls * args.overhead_tokens + frames * args.image_tokens
        )
    total_tokens = prefill_tokens + decode_tokens
    flops = 2.0 * args.n_eff * total_tokens

    return {
        "run": str(run_dir),
        "exp": run_dir.parent.name,
        "seed": run_dir.name,
        "llm_calls": calls,
        "generate_calls": generate_calls,   # sanity: should ~equal llm_calls
        "frames": frames,
        "prompt_chars": prompt_chars,
        "chars_per_token": round(cpt, 3),
        "cpt_source": cpt_src,
        "prefill_source": prefill_src,
        "prefill_tokens": prefill_tokens,
        "decode_tokens_exact": decode_tokens,
        "total_tokens": total_tokens,
        "flops": flops,
        "modules": modules,
    }


def fmt_flops(x: float) -> str:
    if x <= 0:
        return "0"
    exp = int(math.floor(math.log10(x)))
    return f"{x / 10**exp:.2f}e{exp}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path,
                    help="run-group dir(s); every <exp>/<seed>/ underneath "
                         "holding log.txt + llm_logs/ is analyzed")
    ap.add_argument("--n-eff", type=float, default=4.5e9,
                    help="effective (active) parameter count for the FLOPs "
                         "rule 2*N*tokens [default: 4.5e9, Gemma-4-E4B]")
    ap.add_argument("--image-tokens", type=int, default=280,
                    help="vision soft tokens per frame=True call "
                         "[default: 280, Gemma 4 processor default]")
    ap.add_argument("--overhead-tokens", type=int, default=60,
                    help="per-call constant for chat template + injected "
                         "JSON instruction [default: 60]")
    ap.add_argument("--chars-per-token", type=float, default=None,
                    help="override the per-run auto calibration")
    ap.add_argument("--csv", type=Path, default=None,
                    help="write per-run rows to this CSV")
    ap.add_argument("--per-module", action="store_true",
                    help="also print the per-module call/char breakdown")
    args = ap.parse_args()

    rows = []
    for root in args.roots:
        for run_dir in sorted(p.parent for p in root.glob("*/*/log.txt")):
            row = analyze_run(run_dir, args)
            if row:
                rows.append(row)
    if not rows:
        sys.exit("no runs with log.txt + llm_logs/ found under the given roots")

    hdr = (f"{'exp':<28} {'seed':<10} {'calls':>6} {'frames':>6} "
           f"{'prefill':>12} {'(src)':>10} {'decode(exact)':>13} "
           f"{'total tok':>10} {'FLOPs':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mismatch = "" if abs(r["llm_calls"] - r["generate_calls"]) <= max(
            5, 0.02 * r["llm_calls"]) else f"  !! generate_calls={r['generate_calls']}"
        print(f"{r['exp']:<28} {r['seed']:<10} {r['llm_calls']:>6} "
              f"{r['frames']:>6} {r['prefill_tokens']:>12,} "
              f"{r['prefill_source']:>10} "
              f"{r['decode_tokens_exact']:>13,} {r['total_tokens']:>10,} "
              f"{fmt_flops(r['flops']):>9}{mismatch}")
        if args.per_module:
            for name, m in sorted(r["modules"].items()):
                print(f"    {name:<24} calls={m['calls']:>6} "
                      f"frames={m['frames']:>6} "
                      f"prompt_chars={m['sys_chars'] + m['user_chars']:>12,}")

    # per-experiment aggregate (mean +/- sd over seeds)
    by_exp = defaultdict(list)
    for r in rows:
        by_exp[r["exp"]].append(r)
    print()
    print(f"{'experiment':<28} {'n':>3} {'mean total tok':>15} "
          f"{'mean FLOPs':>11} {'sd FLOPs':>10}")
    print("-" * 72)
    for exp, rs in sorted(by_exp.items()):
        fl = [r["flops"] for r in rs]
        mean = sum(fl) / len(fl)
        sd = (sum((x - mean) ** 2 for x in fl) / (len(fl) - 1)) ** 0.5 \
            if len(fl) > 1 else 0.0
        mean_tok = sum(r["total_tokens"] for r in rs) / len(rs)
        print(f"{exp:<28} {len(rs):>3} {mean_tok:>15,.0f} "
              f"{fmt_flops(mean):>11} {fmt_flops(sd):>10}")
    print(f"\nFLOPs = 2 * N_eff * total_tokens with N_eff = {args.n_eff:.3g}; "
          f"image_tokens={args.image_tokens}/frame, "
          f"overhead={args.overhead_tokens}/call.")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            fields = [k for k in rows[0] if k != "modules"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in fields})
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
