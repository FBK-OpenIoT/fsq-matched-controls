"""Figure 1: per-seed held-out accuracy (top) and macro-F1 (bottom), logistic probe; masked prediction and reconstruction runs against
rounded PCA, the k-means vocabulary (mean of five fits), raw bands and the majority class (data: figure1_data.py).

    python -m fsq_matched_controls.analysis.figure1 --palette okabe-ito --layout column --au-label Queensland --group-titles

writes outputs/figure1_column.pdf and .png (the letter's Figure 1; the PDF is byte-identical to manuscript/figure1_column.pdf).
Other layouts: --layout grid2x5 (default, 7.16in wide) and --layout singlecol. --out-dir DIR writes elsewhere.
The exports carry no creator, producer or date metadata, so re-renders are byte-reproducible.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fsq_matched_controls import paths
from fsq_matched_controls.analysis.figure1_data import PANELS, LOW_FRANCE_SEEDS

COL_ORDER = ["France, transductive", "France, strict", "Australia, transductive", "Australia, strict", "Estonia"]
METRICS = [("acc", "Accuracy", 0), ("f1", "Macro-F1", 1)]


def main(argv=None):
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--out-dir", default=None, help="output directory (default: outputs/)")
    _ap.add_argument("--palette", choices=["original", "okabe-ito"], default="original")
    _ap.add_argument("--layout", choices=["grid2x5", "singlecol", "column"], default="grid2x5")
    # added 2026-11-13; default unchanged: the name printed for the Australia panels
    # (data keys stay "Australia, ..."; only the displayed title changes)
    _ap.add_argument("--au-label", default="Australia")
    # added 2026-11-13; default off: in --layout column, print each country once,
    # centred over its two panels, with only "transductive" / "strict" per panel
    _ap.add_argument("--group-titles", action="store_true")
    ARGS = _ap.parse_args(argv)


    def shown(region):
        return region.replace("Australia", ARGS.au_label.replace("\\n", "\n"))
    SINGLE = ARGS.layout == "singlecol"
    COLUMN = ARGS.layout == "column"
    FS = 6 if COLUMN else 8  # point size of every text element

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FS,
        "axes.linewidth": 0.5 if COLUMN else 0.6,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "axes.titlesize": FS,
    })

    C_PRED = "#1b4f8a"       # masked prediction dots
    C_PRED_LOW = "#c0392b"   # the three low France runs, called out explicitly
    C_RECON = "#8a6d1b"      # reconstruction dots
    C_PCA = "#7f7f7f"        # rounded PCA control
    C_RAW = "#595959"        # raw monthly bands -- darkened per review (was too faint for the uncompressed reference)
    C_MAJ = "#d9d9d9"        # majority class
    C_KM = "#1a8a4a"         # k-means control -- must be visible at a glance
    if ARGS.palette == "okabe-ito":
        # Okabe & Ito colour-blind-safe set; greys, line styles and marker shapes unchanged
        C_PRED = "#0072B2"       # blue
        C_PRED_LOW = "#D55E00"   # vermillion (black edge kept)
        C_RECON = "#E69F00"      # orange (triangles)
        C_KM = "#009E73"         # bluish green

    LS_PCA, LS_RAW, LS_MAJ, LS_KM = "--", ":", "-.", "-"

    # IEEE double-column width (GRSL: \textwidth for a figure* spanning both
    # columns is 7.16in). Rendered natively at that width so \includegraphics
    # does not rescale it -- fonts above are true point sizes at print size.
    WIDTH = 3.49 if (SINGLE or COLUMN) else 7.16  # 3.49in: LaTeX's columnwidth (3.495in) then scales fonts up, not below 8pt
    if SINGLE:
        # single column: one row per setting, accuracy left, macro-F1 right
        fig, axes = plt.subplots(5, 2, figsize=(WIDTH, 6.3), sharex=False)
    elif COLUMN:
        fig, axes = plt.subplots(2, 5, figsize=(WIDTH, 2.25), sharex=False)
    else:
        fig, axes = plt.subplots(2, 5, figsize=(7.16, 3.55), sharex=False)
    S_DOT, S_LOW, LW_LOW = (5, 8, 0.4) if COLUMN else (14, 20, 0.6)

    for col, region in enumerate(COL_ORDER):
        d = PANELS[region]
        for row, (key, ylabel, idx) in enumerate(METRICS):
            ax = axes[col, row] if SINGLE else axes[row, col]
            has_recon = len(d["reconstruction"]) > 0
            xpos = {"predictive": 0.0, "reconstruction": 1.0}

            # control lines, spanning the panel
            ax.axhline(d["majority"][idx], color=C_MAJ, ls=LS_MAJ, lw=1.1, zorder=1)
            ax.axhline(d["raw"][idx], color=C_RAW, ls=LS_RAW, lw=1.1, zorder=1)
            ax.axhline(d["pca"][idx], color=C_PCA, ls=LS_PCA, lw=1.1, zorder=1)
            ax.axhline(d["kmeans"][idx], color=C_KM, ls=LS_KM, lw=1.6, zorder=2)

            # predictive dots
            seeds = list(d["predictive"].keys())
            jitter = np.linspace(-0.26, 0.26, len(seeds)) if len(seeds) > 1 else [0.0]
            for s, jx in zip(seeds, jitter):
                v = d["predictive"][s][idx]
                low = (region, s) in LOW_FRANCE_SEEDS
                ax.scatter(xpos["predictive"] + jx, v, s=S_DOT if not low else S_LOW,
                           color=C_PRED_LOW if low else C_PRED,
                           edgecolor="black" if low else "none", linewidth=LW_LOW,
                           zorder=4, marker="o")

            # reconstruction dots
            if has_recon:
                seeds_r = list(d["reconstruction"].keys())
                jitter_r = np.linspace(-0.26, 0.26, len(seeds_r)) if len(seeds_r) > 1 else [0.0]
                for s, jx in zip(seeds_r, jitter_r):
                    v = d["reconstruction"][s][idx]
                    ax.scatter(xpos["reconstruction"] + jx, v, s=S_DOT, color=C_RECON,
                               marker="^", zorder=4)

            if has_recon:
                ax.set_xlim(-0.45, 1.45)
                ax.set_xticks([0.0, 1.0])
                if COLUMN:
                    # markers (circles, triangles) and the caption carry Pred./Recon.
                    ax.set_xticklabels([])
                    ax.tick_params(axis="x", length=0)
                else:
                    ax.set_xticklabels(["Pred.", "Recon."])
            else:
                # a panel without reconstruction runs (none since the matched-recipe runs filled the
                # strict panels): show only the masked-prediction column.
                ax.set_xlim(-0.4, 0.4)
                ax.set_xticks([0.0])
                ax.set_xticklabels(["Pred."])

            if SINGLE:
                if col == 0:
                    ax.set_title(ylabel, fontsize=8.2)
                if row == 0:
                    ax.set_ylabel(shown(region).replace(", ", ",\n"))
            elif COLUMN:
                if row == 0 and ARGS.group_titles:
                    ax.set_title(region.split(", ")[1] if ", " in region else "", fontsize=FS)
                elif row == 0:
                    ax.set_title(shown(region).replace(", ", ",\n"), fontsize=FS)
                if col == 0:
                    ax.set_ylabel(ylabel)
            else:
                if row == 0:
                    ax.set_title(shown(region), fontsize=8.2)
                if col == 0:
                    ax.set_ylabel(ylabel)
            if COLUMN:
                ax.tick_params(axis="both", length=1.5, pad=1)
                ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            else:
                ax.tick_params(axis="both", length=2.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    # shared legend -- in-image title removed (the LaTeX \caption in
    # manuscript carries the "Fig. 1..." text; IEEE captions live there, not
    # baked into the graphic). 4+3 over two rows: a true single row at 8pt
    # does not fit 7.16in without matplotlib's tight-bbox expanding the canvas
    # past the IEEE column width (verified: caused a 10.4in-wide export before
    # this fix), which would defeat the point of rendering at native size.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_PRED, markersize=5.5, label="Masked pred. (seed)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_PRED_LOW, markeredgecolor="black", markersize=6, label="Masked pred., low France run"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=C_RECON, markersize=5.5, label="Reconstruction (seed)"),
        Line2D([0], [0], color=C_KM, lw=1.6, ls=LS_KM, label="k-means (mean of 5 fits)"),
        Line2D([0], [0], color=C_PCA, lw=1.0, ls=LS_PCA, label="Rounded PCA"),
        Line2D([0], [0], color=C_RAW, lw=1.0, ls=LS_RAW, label="Raw bands"),
        Line2D([0], [0], color=C_MAJ, lw=1.0, ls=LS_MAJ, label="Majority class"),
    ]
    if SINGLE:
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handletextpad=0.4)
        fig.subplots_adjust(left=0.20, right=0.98, top=0.962, bottom=0.148, wspace=0.42, hspace=0.36)
    elif COLUMN:
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 0.0), columnspacing=0.5, handletextpad=0.25,
                   markerscale=0.7, handlelength=1.3, borderaxespad=0.2)
        # France pair and Australia pair share a y-axis (common limits); Estonia keeps
        # its own: three tick-label columns instead of five, so the panels widen.
        for r in range(2):
            for a, b in ((0, 1), (2, 3)):
                lo = min(axes[r, a].get_ylim()[0], axes[r, b].get_ylim()[0])
                hi = max(axes[r, a].get_ylim()[1], axes[r, b].get_ylim()[1])
                for c in (a, b):
                    axes[r, c].set_ylim(lo, hi)
                    axes[r, c].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
                axes[r, b].tick_params(axis="y", labelleft=False, length=0)
                axes[r, b].spines["left"].set_visible(False)
        # manual placement (inches), fixed canvas: margins, gap within a pair, gap before a labelled axis
        W, H = fig.get_size_inches()
        left, right, gs, gl = 0.34, 0.03, 0.07, 0.30
        w = (W - left - right - 2 * gs - 2 * gl) / 5
        xs = [left, left + w + gs, left + 2 * w + gs + gl, left + 3 * w + 2 * gs + gl, left + 4 * w + 2 * gs + 2 * gl]
        top_m, bottom_m, hgap = 0.30, 0.42, 0.10
        h = (H - top_m - bottom_m - hgap) / 2
        for r in range(2):
            y = H - top_m - h - r * (h + hgap)
            for c in range(5):
                axes[r, c].set_position([xs[c] / W, y / H, w / W, h / H])
        if ARGS.group_titles:
            # country names on the upper title line: centred over each pair, Estonia over its panel
            y_top = (H - top_m + 0.165) / H
            for name, x0, x1 in ((shown("France"), xs[0], xs[1] + w),
                                 (shown("Australia"), xs[2], xs[3] + w),
                                 ("Estonia", xs[4], xs[4] + w)):
                fig.text((x0 + x1) / 2 / W, y_top, name, ha="center", va="bottom", fontsize=FS)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handletextpad=0.4)

        # Fixed-size layout, no tight_layout/bbox_inches="tight": both can grow the
        # canvas past the figsize below when content (esp. the legend) is close to
        # the edge, which would silently defeat the "render at true 7.16in" fix.
        fig.subplots_adjust(left=0.08, right=0.995, top=0.93, bottom=0.20, wspace=0.5, hspace=0.35)

    HERE = Path(ARGS.out_dir).resolve() if ARGS.out_dir else paths.OUTPUT
    HERE.mkdir(parents=True, exist_ok=True)
    STEM = "figure1_singlecol" if SINGLE else "figure1_column" if COLUMN else "figure1"
    # Release hygiene: no Creator/Producer/CreationDate in the PDF Info dict and
    # no Software tEXt chunk in the PNG (None removes the key) -- also makes the
    # exports byte-reproducible across re-renders.
    fig.savefig(HERE / f"{STEM}.pdf", dpi=300,
                metadata={"Creator": None, "Producer": None, "CreationDate": None})
    fig.savefig(HERE / f"{STEM}.png", dpi=200, metadata={"Software": None})

    # sanity check: confirm the export is exactly the intended IEEE width
    actual_w, actual_h = fig.get_size_inches()
    assert abs(actual_w - WIDTH) < 1e-6, f"canvas width drifted to {actual_w}in, expected {WIDTH}in"
    print(f"saved at {actual_w}in x {actual_h}in ->", HERE / f"{STEM}.pdf", "and", HERE / f"{STEM}.png")


if __name__ == "__main__":
    main()
