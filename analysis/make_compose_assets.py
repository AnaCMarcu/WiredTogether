"""Figma-friendly component export for the qualitative figures.

Instead of one baked composite, this writes every piece separately so the
figure can be assembled (and edited) by hand:

  <arm>/final_compose/
    timeline.svg|pdf|png        the graph on its own (SVG text stays editable)
    scenarios/NN_<slug>/
      contact_sheet.png         every exported frame, labelled, for picking
      frames/aN_tXXXX.png       one PNG per agent per step (native pixels)
      chat.svg                  the message block as editable vector text
      text.md                   title, verdict, messages, outcome (copy/paste)
    README.md

Run:  python analysis/make_compose_assets.py [arm ...]
"""
import re
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import ASSETS  # noqa: F401  (puts siblings on sys.path)
import make_final_figures as M
import make_iclr_figure as I
from make_final_figures import (AGENT_C, INK, ARMS, chamber_spans_from_steps,
                                load_bond_series, load_messages, load_run)

ROOT = ASSETS / "timelines"
PAD = 12           # steps exported either side of the scenario window
UPSCALE = 2        # nearest-neighbour, keeps the pixel art crisp


def slug(text):
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return re.sub(r"_+", "_", s)


def local_window(ex):
    steps = []
    for row in ex["frames"]:
        if row[0] == "F":
            steps += [slot[1] for slot in (row[1], row[2]) if slot]
        else:
            steps.append(row[1])
    return max(min(steps) - PAD, 0), max(steps) + PAD


def export_frames(arm, ep, lo, hi, outdir):
    """One PNG per agent per step; returns {(agent, t): rgb array}."""
    import cv2
    outdir.mkdir(parents=True, exist_ok=True)
    wanted = [(a, ep, t) for a in (0, 1, 2) for t in range(lo, hi + 1)]
    frames = M.grab_frames(arm, wanted)
    for (a, _, t), fr in frames.items():
        big = cv2.resize(fr, (fr.shape[1] * UPSCALE, fr.shape[0] * UPSCALE),
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(outdir / f"a{a}_t{t:04d}.png"),
                    cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    return frames


def contact_sheet(frames, lo, hi, ep, path, cols=10):
    """One labelled sheet per scenario, agents in blocks, for picking."""
    import cv2
    tile_w, tile_h = 256, 144
    blocks = []
    for a in (0, 1, 2):
        ts = [t for t in range(lo, hi + 1) if (a, ep, t) in frames]
        if not ts:
            continue
        tiles = []
        for t in ts:
            fr = cv2.cvtColor(frames[(a, ep, t)], cv2.COLOR_RGB2BGR)
            fr = cv2.resize(fr, (tile_w, tile_h),
                            interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(fr, (0, 0), (tile_w - 1, tile_h - 1),
                          (200, 200, 200), 1)
            cv2.putText(fr, f"a{a} t{t}", (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(fr, f"a{a} t{t}", (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(fr)
        rows = []
        for i in range(0, len(tiles), cols):
            row = tiles[i:i + cols]
            while len(row) < cols:
                row.append(np.full((tile_h, tile_w, 3), 255, np.uint8))
            rows.append(np.hstack(row))
        header = np.full((26, cols * tile_w, 3), 255, np.uint8)
        cv2.putText(header, f"agent {a}", (6, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1,
                    cv2.LINE_AA)
        blocks.append(np.vstack([header] + rows))
    if blocks:
        cv2.imwrite(str(path), np.vstack(blocks))


def chat_svg(ex, arm_name, path):
    """The message block as vector text (Figma keeps it editable)."""
    msgs = [r for r in ex["frames"] if r[0] == "M"]
    line_h, width = 0.20, 6.6
    n = sum(max(len(textwrap.wrap(m[4], 76)), 1) for m in msgs)
    H = 0.30 + n * line_h + 0.34
    with plt.rc_context(I.RC):
        fig = plt.figure(figsize=(width, H))
        y = 1 - 0.22 / H
        tag, tcol = I.VERDICT.get((arm_name, ex["key"]), ("", INK))
        fig.text(0.012, y, ex["title"], fontsize=11, color=INK,
                 fontweight="bold", va="center")
        if tag:
            fig.text(0.988, y, tag, fontsize=8.5, color=tcol, ha="right",
                     va="center", fontweight="bold")
        y -= 0.30 / H
        for _, t, src, dst, txt in msgs:
            for i, ln in enumerate(textwrap.wrap(txt, 76) or [""]):
                if i == 0:
                    fig.text(0.012, y, f"<a{src}→a{dst}>", fontsize=8,
                             color=AGENT_C[src], family="monospace",
                             fontweight="bold", va="top")
                fig.text(0.105, y, ln, fontsize=8, color=INK,
                         family="monospace", va="top")
                y -= line_h / H
        y -= 0.06 / H
        for ln in textwrap.wrap(ex["outcome"], 92):
            fig.text(0.012, y, ln, fontsize=8, color=INK, style="italic",
                     va="top")
            y -= line_h / H
        fig.savefig(path, facecolor="none", transparent=True)
        plt.close(fig)


def text_md(ex, arm_name, ep, lo, hi, path):
    tag, _ = I.VERDICT.get((arm_name, ex["key"]), ("", ""))
    out = [f"# {ex['title']}", ""]
    out += [f"- verdict: **{tag}**" if tag else ""]
    out += [f"- episode {ep}, exported steps {lo}-{hi} "
            f"(figure window {ex['x'][0]}-{ex['x'][1]} cumulative)",
            f"- columns: agent {ex['agents'][0]} (left), "
            f"agent {ex['agents'][1]} (right)", "",
            "## frames used in the built figure", ""]
    for row in ex["frames"]:
        if row[0] != "F":
            continue
        for slot in (row[1], row[2]):
            if slot:
                out.append(f"- `a{slot[0]}_t{slot[1]:04d}.png`")
    out += ["", "## messages (verbatim)", ""]
    for _, t, src, dst, txt in [r for r in ex["frames"] if r[0] == "M"]:
        out.append(f"- t={t}  **a{src} -> a{dst}**: {txt}")
    out += ["", "## outcome", "", ex["outcome"], ""]
    path.write_text("\n".join(out), encoding="utf-8")


def build_timeline(arm_name, arm, outdir):
    run = load_run(arm["run"])
    messages = load_messages(run)
    spans = chamber_spans_from_steps(arm, run)
    Wr = load_bond_series(arm)
    examples = sorted(arm["examples"], key=lambda e: e["x"][0])
    H = 0.10 + I.STRIP_H + I.BOND_H + I.XLAB_H + 0.12
    with plt.rc_context(I.RC):
        fig = plt.figure(figsize=(I.FIG_W, H))
        I.draw_timeline(fig, arm, run, Wr, messages, spans, H, examples)
        for ext in ("svg", "pdf", "png"):
            fig.savefig(outdir / f"timeline.{ext}",
                        dpi=300 if ext == "png" else None,
                        facecolor="white")
        plt.close(fig)
    return run


README = """# Compose kit — {arm}

Everything the figure is made of, exported separately so it can be laid
out (and edited) in Figma.

## timeline.svg / .pdf / .png
The graph on its own, at ICLR single-column width (5.5 in).  Import the
**SVG**: text stays live text and the curves stay vectors, so fonts,
colours and labels are all editable.  The numbered badges mark the zoom
window of each scenario.

## scenarios/NN_<name>/
- `contact_sheet.png` — every exported frame for all three agents, labelled
  `aN tSTEP`.  Browse this first and note the frames you want.
- `frames/aN_tXXXX.png` — the individual frames, {scale}x native resolution,
  upscaled with nearest-neighbour so the pixel art stays sharp.  Scale them
  in Figma with smoothing off.
- `chat.svg` — the message block as editable vector text (transparent
  background), already colour-coded per sender.
- `text.md` — title, verdict, the verbatim messages and the outcome line,
  for copy/paste if you would rather set the type yourself.

## suggested assembly
timeline across the top, then one row per scenario: 2 frames per agent
(before / after), agent columns kept pure, chat block underneath.
Frames listed under "frames used in the built figure" in each `text.md`
are the ones in the current version, if you want a starting point.
"""


def build(arm_name, arm):
    outdir = ROOT / arm_name / "final_compose"
    outdir.mkdir(parents=True, exist_ok=True)
    build_timeline(arm_name, arm, outdir)
    print(f"  timeline.svg/pdf/png")
    examples = sorted(arm["examples"], key=lambda e: e["x"][0])
    total = 0
    for n, ex in enumerate(examples, 1):
        sdir = outdir / "scenarios" / f"{n:02d}_{slug(ex['title'])}"
        sdir.mkdir(parents=True, exist_ok=True)
        lo, hi = local_window(ex)
        frames = export_frames(arm, ex["ep"], lo, hi, sdir / "frames")
        contact_sheet(frames, lo, hi, ex["ep"], sdir / "contact_sheet.png")
        chat_svg(ex, arm_name, sdir / "chat.svg")
        text_md(ex, arm_name, ex["ep"], lo, hi, sdir / "text.md")
        total += len(frames)
        print(f"  {sdir.name}: {len(frames)} frames (ep{ex['ep']} "
              f"t{lo}-{hi}) + contact sheet + chat.svg + text.md")
    (outdir / "README.md").write_text(
        README.format(arm=arm_name, scale=UPSCALE), encoding="utf-8")
    print(f"  -> {total} frames total")


def main(only=None):
    ARMS["gemma3f_seed123"]["delib"] = M.GEMMA_DELIB
    for arm_name, arm in ARMS.items():
        if only and arm_name not in only:
            continue
        print(f"== {arm_name} ==")
        build(arm_name, arm)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
