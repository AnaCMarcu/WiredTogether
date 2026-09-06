"""Qualitative environment gallery (ICLR-style): five rows, one per chamber
of the five_chambers curriculum, each showing hand-picked first-person
frames sampled around real milestone events across many runs (not a single
episode) so each row shows different agents/seeds/moments.

Frame sources: real gameplay mp4s under runs_from_daic/ (1 frame per env
step). Step numbers are the per-episode milestone step from
episode_summary.json plus a small fixed offset, resolved directly against
that episode's own video (no cumulative-episode-offset arithmetic needed).
A few chamber-4/5 frames instead reference PNGs already extracted (and
hand-verified) by the make_*_timelines*.py scripts under
paper_assets/timelines/ -- reusing those avoids re-deriving their step
convention, which is cumulative-episode-offset based, unlike the
per-episode-local steps used everywhere else in this file. Every frame was
checked against the per-step chamber column in step_log.csv (the
authoritative source -- episode_summary.json's chamber_entry_steps and
milestone steps can lag the real transition by up to ~10 steps).

Output: paper_assets/chamber_gallery/chamber_gallery.{png,pdf}
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import ASSETS, RUNS  # noqa: E402

OUTDIR = ASSETS / "chamber_gallery"
DPI = 300
NCOLS = 5

# ── row definitions ──────────────────────────────────────────────────────
# label, banner colour, list of (run_dir relative to RUNS, seed, agent, ep, step)
ROWS = [
    dict(
        label="1) Solo Exploration",
        banner="#2E5F8A",
        frames=[
            ("agent_scaling/scale_gemma_hebbian_n2/seed_123", 123, 0, 2, 15),
            ("cofiring_bidi/exp22_cofire_pri/seed_42", 42, 0, 2, 170),
            ("agent_scaling/scale_gemma_hebbian_n2/seed_123", 123, 0, 2, 81),
            ("medium_2k/exp09_llm_9b_allied_all/seed_123", 123, 0, 2, 127),
            ("cofiring_bidi/exp23_cofire_prcoi/seed_123", 123, 2, 2, 50),
        ],
    ),
    dict(
        label="2) Cooperative Puzzle",
        banner="#C1571E",
        frames=[
            ("medium_2k/exp11_llm_9b_allied_none/seed_123", 123, 1, 1, 312),
            ("agent_scaling/scale_gemma_base_n9/seed_123", 123, 6, 2, 132),
            ("agent_scaling/scale_gemma_hebbian_n6/seed_123", 123, 3, 3, 151),
            ("medium_2k/exp09_llm_9b_allied_all/seed_123", 123, 1, 1, 491),
            "timelines/qwen9b_heb42/frames/anvilA1_ep2_t301/03_agent0_t301.png",
        ],
    ),
    dict(
        label="3) Communication Puzzle",
        banner="#5B2C8C",
        frames=[
            ("agent_scaling/scale_gemma_base_n2/seed_123", 123, 0, 3, 216),
            ("agent_scaling/scale_gemma_base_n2/seed_123", 123, 1, 3, 220),
            ("agent_scaling/scale_gemma_base_n2/seed_123", 123, 0, 3, 215),
            ("medium_2k/exp09_llm_9b_allied_all/seed_42", 42, 0, 1, 1109),
            ("agent_scaling/scale_gemma_base_n2/seed_42", 42, 1, 2, 215),
        ],
    ),
    dict(
        label="4) Combat Practice",
        banner="#A32020",
        frames=[
            "timelines/gemma3f_seed123/frames/kill_ep1_t722/01_agent1_t716.png",
            ("agent_scaling/scale_gemma_base_n2/seed_123", 123, 0, 3, 344),
            ("agent_scaling/scale_gemma_base_n2/seed_42", 42, 1, 1, 355),
            "timelines/gemma3f_seed123/frames/kill_ep1_t722/05_agent0_t720.png",
            "timelines/gemma3f_seed123/frames/kill_ep3_t642/03_agent2_t640.png",
        ],
    ),
    dict(
        label="5) Boss Encounter",
        banner="#D98C0F",
        frames=[
            "timelines/gemma3f_seed123/frames/failure_bosswipe_ep1_t799/02_agent0_t801.png",
            "timelines/gemma3f_seed123/frames/failure_bosswipe_ep1_t799/05_agent2_t807.png",
            "timelines/gemma3f_seed123/frames/failure_bosswipe_ep1_t799/07_agent0_t810.png",
            "timelines/gemma3f_seed123/frames/failure_bosswipe_ep1_t799/03_agent0_t804.png",
            "timelines/qwen9b_base789/frames/failure_bosswipe_ep1_t799/06_agent0_t800.png",
        ],
    ),
]


# ── frame grabbing ──────────────────────────────────────────────────────
def resolve_video(run_dir, seed, agent, ep):
    base = RUNS / run_dir
    direct = base / f"seed_{seed}_agent_{agent}_ep{ep}.mp4"
    if direct.exists():
        return direct
    hits = list(base.glob(f"gifs/*/seed_{seed}_agent_{agent}_ep{ep}.mp4"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"no video for {run_dir} seed={seed} agent={agent} ep={ep}")


def grab_frame(run_dir, seed, agent, ep, step):
    import cv2

    path = resolve_video(run_dir, seed, agent, ep)
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(step, 0), n - 1))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {step} from {path}")
    fr = cv2.convertScaleAbs(fr, alpha=1.45, beta=12)
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def load_frame(spec):
    """spec is either a (run_dir, seed, agent, ep, step) tuple (extract from
    the raw mp4) or a string path relative to ASSETS (an already-extracted,
    already-brightened PNG from the timelines pipeline)."""
    if isinstance(spec, str):
        import numpy as np
        from PIL import Image

        return np.asarray(Image.open(ASSETS / spec).convert("RGB"))
    return grab_frame(*spec)


# ── figure assembly ──────────────────────────────────────────────────────
TITLE_FONT = ["Century Gothic", "Verdana", "Segoe UI", "DejaVu Sans"]


def main():
    n_rows = len(ROWS)
    fig_w = 13.2
    img_w_in = fig_w / NCOLS
    img_h_in = img_w_in * 9 / 16
    banner_h_in = 0.34
    row_gap_in = 0.10
    margin_in = 0.06
    row_h_in = banner_h_in + img_h_in
    fig_h = n_rows * (row_h_in + row_gap_in) + 2 * margin_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    banner_h = banner_h_in / fig_h
    row_h = row_h_in / fig_h
    row_gap = row_gap_in / fig_h
    top_margin = margin_in / fig_h
    gap = 0.006
    col_w = 1.0 / NCOLS

    for r, row in enumerate(ROWS):
        y_top = 1.0 - top_margin - r * (row_h + row_gap)
        y_banner = y_top - banner_h
        y_imgs = y_banner - img_h_in / fig_h

        fig.patches.append(
            plt.Rectangle((0.0, y_banner), 1.0, banner_h,
                          transform=fig.transFigure, facecolor=row["banner"],
                          edgecolor="none", zorder=1))
        fig.text(0.5, y_banner + banner_h / 2, row["label"],
                 transform=fig.transFigure, ha="center", va="center",
                 fontsize=15, fontweight="bold", color="white",
                 fontfamily=TITLE_FONT, zorder=2)
        fig.patches.append(
            plt.Rectangle((0.0, y_imgs), 1.0, img_h_in / fig_h,
                          transform=fig.transFigure, facecolor="none",
                          edgecolor=row["banner"], linewidth=1.6, zorder=1))

        for c, spec in enumerate(row["frames"]):
            frame = load_frame(spec)
            x0 = c * col_w
            ax = fig.add_axes([x0 + gap / 2, y_imgs + gap / 2,
                               col_w - gap, img_h_in / fig_h - gap])
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#c3c9d0")
                s.set_linewidth(0.8)
        print(f"row {r + 1}/{n_rows} ({row['label']}) done")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"chamber_gallery.{ext}",
                    dpi=DPI if ext == "png" else None, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUTDIR}/chamber_gallery.(png|pdf)")


if __name__ == "__main__":
    main()
