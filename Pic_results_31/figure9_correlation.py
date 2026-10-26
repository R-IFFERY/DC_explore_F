# -*- coding: utf-8 -*-
"""
Plot selected local correlation maps from existing NPZ caches.

This script reads previously generated NPZ files for specified (k, St)
cases, extracts the designated cycle-pair correlation map for each case,
masks the outer y-boundary bands, and plots three panels in a 1 x 3 layout.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ============================================================
# Figure style
# ============================================================
USE_TEX = True
FONT_SIZE = 10

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": USE_TEX,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
})

if USE_TEX:
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.latex.preamble": (
            r"\usepackage{amsmath}\usepackage{newtxtext,newtxmath}"
        )
    })

# ============================================================
# User settings
# ============================================================
SAVE_ROOT = r"E:/Rensq/DC_explore_F/flow/local_cycle_correlation_maps_all_cases"
PAIR_MODE_CANDIDATES = ["adjacent", "all"]

# 每个元组为: (k, st, pair_label)
SELECTED_CASES = [
    (0.8, 0.6, "19-20"),
    (2.0, 0.3, "19-20"),
    (0.5, 0.6, "14-15"),
]

# Plot window. Display the full window -10 <= y/c <= 10,
# but do not draw the outer boundary bands.
X_MIN = -1.0
X_MAX = 20.0
Y_MIN = -10.0
Y_MAX = 10.0
Y_DRAW_MIN = -9.0
Y_DRAW_MAX = 9.0

FIGURE_BASENAME = os.path.join(
    SAVE_ROOT,
    "selected_cases_rho_mixed_pairs_1x3"
)

# ============================================================
# Utilities
# ============================================================
def fmt_num(value):
    return f"{float(value):g}"


def case_tag(k, st):
    return f"k{fmt_num(k)}_st{fmt_num(st)}"


def get_npz_candidates(k, st):
    tag = case_tag(k, st)
    case_dir = os.path.join(SAVE_ROOT, tag)
    candidates = []
    for pair_mode in PAIR_MODE_CANDIDATES:
        candidates.append(
            os.path.join(
                case_dir,
                f"local_rho_maps_{tag}_{pair_mode}.npz"
            )
        )
    return candidates


def find_existing_npz(k, st):
    for path in get_npz_candidates(k, st):
        if os.path.isfile(path):
            return path
    return None


def load_pair_map_from_npz(npz_path, pair_label_target):
    with np.load(npz_path, allow_pickle=False) as data:
        required = {"x_over_c", "y_over_c", "pair_maps", "pair_labels"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(
                f"NPZ file lacks required arrays: {sorted(missing)}"
            )

        X = np.array(data["x_over_c"], dtype=float)
        Y = np.array(data["y_over_c"], dtype=float)
        pair_maps = np.array(data["pair_maps"], dtype=float)
        pair_labels = [str(item) for item in np.array(data["pair_labels"]).tolist()]

        if pair_label_target not in pair_labels:
            raise KeyError(
                f"Pair label '{pair_label_target}' not found in {npz_path}; "
                f"available labels: {pair_labels}"
            )

        index = pair_labels.index(pair_label_target)
        rho_map = pair_maps[index]

    return X, Y, rho_map


def mask_boundary_rows(Y, field, y_draw_min, y_draw_max):
    field_masked = np.array(field, dtype=float, copy=True)
    keep = (Y >= y_draw_min) & (Y <= y_draw_max)
    field_masked[~keep] = np.nan
    return field_masked


# ============================================================
# Main plotting
# ============================================================
def main():
    loaded_cases = []

    for k, st, pair_label in SELECTED_CASES:
        npz_path = find_existing_npz(k, st)
        if npz_path is None:
            raise FileNotFoundError(
                f"Cannot find NPZ for k={k:g}, St={st:g} under {SAVE_ROOT}"
            )

        X, Y, rho_map = load_pair_map_from_npz(npz_path, pair_label)
        rho_map = mask_boundary_rows(Y, rho_map, Y_DRAW_MIN, Y_DRAW_MAX)

        loaded_cases.append({
            "k": k,
            "st": st,
            "pair_label": pair_label,
            "X": X,
            "Y": Y,
            "rho_map": rho_map,
            "npz_path": npz_path,
        })

    fig, axes = plt.subplots(
        1, 3,
        figsize=(6.5, 2.55),
        dpi=300,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    norm = Normalize(vmin=-1.0, vmax=1.0)

    mesh = None
    for panel_index, case in enumerate(loaded_cases):
        ax = flat_axes[panel_index]

        mesh = ax.pcolormesh(
            case["X"],
            case["Y"],
            np.ma.masked_invalid(case["rho_map"]),
            cmap=cmap,
            norm=norm,
            shading="auto",
            rasterized=True,
        )

        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(top=True, right=True)

        # 单行三图：都显示 x 标签，仅最左侧显示 y 标签
        ax.set_xlabel(r"$x/c$")
        if panel_index == 0:
            ax.set_ylabel(r"$y/c$")

        ax.set_title(
            rf"$k={case['k']:g},\ St={case['st']:g}$",
            pad=3.0,
        )

        # Panel label outside the upper-left corner
        ax.text(
            -0.18,
            1.04,
            rf"$({chr(97 + panel_index)})$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            clip_on=False,
        )

    # Inset colorbar inside panel (c), upper-right corner
    ax_c = flat_axes[2]
    cax = inset_axes(
        ax_c,
        width="40%",
        height="5%",
        loc="upper right",
        borderpad=0.55,
    )
    cbar = fig.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
        ticks=[-1.0, 0.0, 1.0],
    )
    if hasattr(cbar, "solids") and cbar.solids is not None:
        cbar.solids.set_rasterized(False)

    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(length=2.0, width=0.6, pad=1.0)

    fig.subplots_adjust(
        left=0.085,
        right=0.990,
        bottom=0.180,
        top=0.900,
        wspace=0.10,
        hspace=0.0,
    )

    png_path = FIGURE_BASENAME + ".png"
    svg_path = FIGURE_BASENAME + ".svg"
    # pdf_path = FIGURE_BASENAME + ".pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    # fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("Selected cases plotted successfully.")
    for case in loaded_cases:
        print(
            f"  k={case['k']:g}, St={case['st']:g}, "
            f"pair={case['pair_label']} <- {case['npz_path']}"
        )
    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")
    # print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()