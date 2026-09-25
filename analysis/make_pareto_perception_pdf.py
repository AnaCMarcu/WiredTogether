#!/usr/bin/env python3
"""make_pareto_perception_pdf.py - the perception Pareto figures as PDFs.

make_pareto_perception_fig.py writes PNG only (fig.savefig(out_png)), but the
paper's \\includegraphics wants figures/pareto_perception.pdf and
figures/pareto_partner.pdf. Rather than touch that working script, this wraps
its main(): every savefig to a .png is mirrored to the same stem with a .pdf
suffix, so the two files are the same draw, not a raster conversion.

Every argument is passed through, so usage mirrors the figure script:

  python analysis/make_pareto_perception_pdf.py \\
      runs_from_daic/compute/_pareto_3f_view \\
      --beliefs paper_assets/perception_3f/beliefs_3f.csv \\
      --x grounding --out-dir paper_assets/perception_3f \\
      --copy-to "C:/Users/marcu/Downloads/WIRED_TOGETHER_revised/figures"

--copy-to copies the PNG (the wrapped script's own behaviour); the PDF is
copied here afterwards, since LaTeX is what needs it.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.figure  # noqa: E402

import make_pareto_perception_fig as fig_mod  # noqa: E402

_written_pdfs: list = []
_orig_savefig = matplotlib.figure.Figure.savefig


def _savefig_with_pdf(self, fname, *a, **kw):
    out = _orig_savefig(self, fname, *a, **kw)
    try:
        p = Path(fname)
    except TypeError:                       # a file object, not a path
        return out
    if p.suffix.lower() == ".png":
        pdf = p.with_suffix(".pdf")
        kw.pop("dpi", None)                 # meaningless for vector output
        _orig_savefig(self, pdf, *a, **kw)
        _written_pdfs.append(pdf)
        print("wrote {}".format(pdf))
    return out


def main() -> int:
    # --arm-label renames the coupled series in the legend ("+Hebbian" in the
    # figure script, "+Social plasticity" in the paper's tables). Consumed
    # here and stripped, since the wrapped parser does not know it.
    argv = sys.argv
    if "--arm-label" in argv:
        i = argv.index("--arm-label")
        fig_mod.ARM_STYLE["hebbian"]["label"] = argv[i + 1]
        del argv[i:i + 2]

    matplotlib.figure.Figure.savefig = _savefig_with_pdf
    try:
        rc = fig_mod.main()
    finally:
        matplotlib.figure.Figure.savefig = _orig_savefig

    # Mirror --copy-to for the PDFs (the wrapped script copies the PNG only).
    argv = sys.argv
    if "--copy-to" in argv:
        dest = Path(argv[argv.index("--copy-to") + 1])
        dest.mkdir(parents=True, exist_ok=True)
        for pdf in _written_pdfs:
            shutil.copy2(pdf, dest / pdf.name)
            print("copied to {}".format(dest / pdf.name))
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
