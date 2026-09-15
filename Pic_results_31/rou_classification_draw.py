import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

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
    # Keep SVG text editable where possible and preserve short vector paths.
    "svg.fonttype": "none",
    "path.simplify": False,
})

if USE_TEX:
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.latex.preamble":
            r"\usepackage{amsmath}\usepackage{newtxtext,newtxmath}"
    })

# ============================================================
# User settings
# ============================================================
K_LIST = np.array([0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0], dtype=float)
ST_LIST = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=float)

SAVE_ROOT = r"E:/Rensq/DC_explore_F/flow/figure_rho_waketype_v2"

# Root directory produced by local_cycle_correlation_maps_all_cases.py.
# Expected structure for each case:
#   CORRELATION_ROOT/k0.2_st0.2/
#       local_rho_maps_k0.2_st0.2_adjacent.npz
CORRELATION_ROOT = (
    r"E:/Rensq/DC_explore_F/flow/"
    r"local_cycle_correlation_maps_all_cases"
)

# Must match PAIR_MODE used by the batch correlation program.
BATCH_PAIR_MODE = "adjacent"

# ============================================================
# Cycle-pair selection for panel (a)
# ============================================================
# Cases with k = 1.4, 1.7 and 2.0 use cycles 19-20 for all St.
# The individual case (k, St) = (0.8, 0.6) also uses cycles 19-20.
# All remaining cases use cycles 14-15.
DEFAULT_TARGET_PAIR_LABEL = "14-15"
SPECIAL_TARGET_PAIR_LABEL = "19-20"
SPECIAL_PAIR_K_VALUES = np.array([1.4, 1.7, 2.0], dtype=float)
SPECIAL_PAIR_CASES = [(0.8, 0.6)]

FIGURE_BASENAME = os.path.join(
    SAVE_ROOT,
    "rho_selected_cycle_pairs_wake_type_1x2_optimized"
)

# A case is hatched when the spatial mean rho of its selected cycle pair
# is below this value. This replaces the previous rho_star-based criterion.
APERIODIC_RHO_THRESHOLD = 0.90

# ============================================================
# Wake-class definitions
# ============================================================
# English terminology used in panel (b) and its legend.
CLASS_INFO = {
    1: {
        "code": "I-a",
        "name": "Shear layer",
        "color": "#DCE6F1",
    },
    2: {
        "code": "I-b",
        "name": "Roll-up transition",
        "color": "#80B1D3",
    },
    3: {
        "code": "II-a",
        "name": "2P",
        "color": "#66C2A5",
    },
    4: {
        "code": "II-b",
        "name": "BvK transition",
        "color": "#2CA25F",
    },
    5: {
        "code": "II-c",
        "name": "Deflected BvK",
        "color": "#006D77",
    },
    6: {
        "code": "III",
        "name": "Bifurcated dipoles",
        "color": "#F4A261",
    },
    7: {
        "code": "IV-a",
        "name": "Narrow chaotic wake",
        "color": "#CAB2D6",
    },
    8: {
        "code": "IV-b",
        "name": "Broad chaotic wake",
        "color": "#E78AC3",
    },
    # 8: {
    #     "code": "IV-b",
    #     "name": "Broad chaotic wake",
    #     "color": "#E78AC3",
    # },
    # 9: {
    #     "code": "IV-c",
    #     "name": "Broad multi-vortex-pair wake",
    #     "color": "#8E3B8F",
    # },
}

CLASS_ORDER = [1, 2, 3, 4, 5, 6, 7, 8]

# Classification matrix. Rows correspond to ST_LIST from 0.2 to 0.8;
# columns correspond to K_LIST from 0.2 to 2.0.
CLASS_ID_MATRIX = np.array([
    [1, 1, 2, 6, 4, 3, 5],
    [1, 2, 2, 3, 6, 4, 6],
    [2, 2, 6, 3, 6, 4, 7],
    [2, 8, 7, 7, 3, 4, 7],
    [8, 8, 7, 8, 7, 8, 7],
    [8, 8, 8, 7, 7, 7, 7],
    [8, 8, 8, 8, 8, 8, 8],
], dtype=int)

# ============================================================
# Utilities
# ============================================================
def centers_to_edges(centers):
    centers = np.asarray(centers, dtype=float)

    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])

    midpoints = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - 0.5 * (centers[1] - centers[0])
    last = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    return np.concatenate([[first], midpoints, [last]])


def fmt_num(value):
    """Use the same case naming rule as the batch correlation program."""
    return f"{float(value):g}"


def case_tag(k, st):
    return f"k{fmt_num(k)}_st{fmt_num(st)}"


def get_case_npz_path(correlation_root, k, st):
    tag = case_tag(k, st)
    return os.path.join(
        correlation_root,
        tag,
        f"local_rho_maps_{tag}_{BATCH_PAIR_MODE}.npz",
    )


def normalise_pair_label(label):
    """Convert NumPy strings/bytes to a comparable cycle-pair label."""
    if isinstance(label, bytes):
        label = label.decode("utf-8", errors="ignore")
    return str(label).strip().replace(" ", "")


def get_target_pair_label(k, st):
    """Return the cycle-pair label selected for a given (k, St) case."""
    k = float(k)
    st = float(st)

    # All St values at k = 1.4, 1.7 and 2.0 use cycles 19-20.
    if np.any(np.isclose(k, SPECIAL_PAIR_K_VALUES)):
        return SPECIAL_TARGET_PAIR_LABEL

    # Additional individual cases that use cycles 19-20.
    for special_k, special_st in SPECIAL_PAIR_CASES:
        if np.isclose(k, special_k) and np.isclose(st, special_st):
            return SPECIAL_TARGET_PAIR_LABEL

    return DEFAULT_TARGET_PAIR_LABEL


def read_case_target_pair(
    npz_path,
    expected_k,
    expected_st,
    target_pair_label,
):
    """
    Read one case NPZ, locate the requested cycle pair, and return the
    spatial statistics of its pointwise local-correlation map.

    The scalar plotted in the k-St map is the arithmetic mean over all finite
    Eulerian grid points in pair_maps[pair_index]. Solid-occupied phases and
    weak/inactive points have already been excluded by the correlation code
    and therefore appear as NaN here.
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(npz_path)

    with np.load(npz_path, allow_pickle=False) as data:
        required = {"k", "st", "pair_labels", "pair_maps"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(
                f"Missing arrays {sorted(missing)} in {npz_path}"
            )

        saved_k = float(data["k"])
        saved_st = float(data["st"])
        if not np.isclose(saved_k, expected_k):
            raise ValueError(
                f"k mismatch in {npz_path}: {saved_k} != {expected_k}"
            )
        if not np.isclose(saved_st, expected_st):
            raise ValueError(
                f"St mismatch in {npz_path}: {saved_st} != {expected_st}"
            )

        labels = [normalise_pair_label(item) for item in data["pair_labels"]]
        target = normalise_pair_label(target_pair_label)
        matching_indices = [
            index for index, label in enumerate(labels) if label == target
        ]
        if len(matching_indices) != 1:
            raise KeyError(
                f"Pair {target_pair_label!r} was not found uniquely in "
                f"{npz_path}. Available labels: {labels}"
            )

        pair_maps = np.asarray(data["pair_maps"], dtype=float)
        if pair_maps.ndim != 3:
            raise ValueError(
                f"pair_maps must have shape (n_pair, ny, nx), obtained "
                f"{pair_maps.shape} in {npz_path}"
            )
        if pair_maps.shape[0] != len(labels):
            raise ValueError(
                "The number of pair maps does not match pair_labels in "
                f"{npz_path}."
            )

        rho_map = pair_maps[matching_indices[0]]
        finite_values = rho_map[np.isfinite(rho_map)]
        if finite_values.size == 0:
            raise ValueError(
                f"No finite local-correlation values for pair "
                f"{target_pair_label} in {npz_path}."
            )

        return {
            "rho_mean": float(np.mean(finite_values)),
            "rho_median": float(np.median(finite_values)),
            "rho_std": float(np.std(finite_values)),
            "rho_min": float(np.min(finite_values)),
            "rho_max": float(np.max(finite_values)),
            "valid_spatial_points": int(finite_values.size),
            "pair_label": normalise_pair_label(target_pair_label),
        }


def load_metrics(correlation_root):
    """
    Assemble the 7x7 k-St matrix from the per-case NPZ files generated by
    local_cycle_correlation_maps_all_cases.py.
    """
    expected_shape = (len(ST_LIST), len(K_LIST))
    rho_mean = np.full(expected_shape, np.nan, dtype=float)
    rho_median = np.full(expected_shape, np.nan, dtype=float)
    rho_std = np.full(expected_shape, np.nan, dtype=float)
    rho_min = np.full(expected_shape, np.nan, dtype=float)
    rho_max = np.full(expected_shape, np.nan, dtype=float)
    valid_spatial_points = np.zeros(expected_shape, dtype=int)
    valid_case_mask = np.zeros(expected_shape, dtype=bool)
    selected_pair_labels = np.full(expected_shape, "", dtype="U16")

    missing_cases = []
    failed_cases = []

    print("=" * 72)
    print(
        "Reading selected cycle pairs from per-case "
        f"{BATCH_PAIR_MODE} NPZ files"
    )
    print(
        f"Default pair: {DEFAULT_TARGET_PAIR_LABEL}; "
        f"k={SPECIAL_PAIR_K_VALUES.tolist()} use "
        f"{SPECIAL_TARGET_PAIR_LABEL} for all St"
    )
    print(
        f"Additional special cases {SPECIAL_PAIR_CASES} use "
        f"{SPECIAL_TARGET_PAIR_LABEL}"
    )
    print(f"Correlation root: {correlation_root}")
    print("=" * 72)

    for st_index, st in enumerate(ST_LIST):
        for k_index, k in enumerate(K_LIST):
            npz_path = get_case_npz_path(correlation_root, k, st)
            target_pair_label = get_target_pair_label(k, st)
            selected_pair_labels[st_index, k_index] = target_pair_label

            if not os.path.isfile(npz_path):
                missing_cases.append((float(k), float(st), npz_path))
                print(
                    f"[Missing] k={k:g}, St={st:g}, "
                    f"pair={target_pair_label}: {npz_path}"
                )
                continue

            try:
                result = read_case_target_pair(
                    npz_path=npz_path,
                    expected_k=k,
                    expected_st=st,
                    target_pair_label=target_pair_label,
                )
            except Exception as error:
                failed_cases.append(
                    (float(k), float(st), npz_path, repr(error))
                )
                print(
                    f"[Failed] k={k:g}, St={st:g}, "
                    f"pair={target_pair_label}: {error}"
                )
                continue

            rho_mean[st_index, k_index] = result["rho_mean"]
            rho_median[st_index, k_index] = result["rho_median"]
            rho_std[st_index, k_index] = result["rho_std"]
            rho_min[st_index, k_index] = result["rho_min"]
            rho_max[st_index, k_index] = result["rho_max"]
            valid_spatial_points[st_index, k_index] = (
                result["valid_spatial_points"]
            )
            valid_case_mask[st_index, k_index] = True

            print(
                f"[Loaded] k={k:g}, St={st:g}, "
                f"pair={target_pair_label}: "
                f"rho={result['rho_mean']:.6f}, "
                f"valid points={result['valid_spatial_points']}"
            )

    aperiodic_mask = (
        valid_case_mask
        & np.isfinite(rho_mean)
        & (rho_mean < APERIODIC_RHO_THRESHOLD)
    )

    print("=" * 72)
    print(
        f"Loaded cases: {int(np.sum(valid_case_mask))}/"
        f"{rho_mean.size}"
    )
    print(f"Missing files: {len(missing_cases)}")
    print(f"Failed files: {len(failed_cases)}")

    if missing_cases:
        print("Missing case list:")
        for k, st, path in missing_cases:
            print(f"  k={k:g}, St={st:g}: {path}")

    if failed_cases:
        print("Failed case list:")
        for k, st, path, error in failed_cases:
            print(f"  k={k:g}, St={st:g}: {error}; file={path}")

    return {
        "rho_mean": rho_mean,
        "rho_median": rho_median,
        "rho_std": rho_std,
        "rho_min": rho_min,
        "rho_max": rho_max,
        "valid_spatial_points": valid_spatial_points,
        "valid_case_mask": valid_case_mask,
        "selected_pair_labels": selected_pair_labels,
        "aperiodic_mask": aperiodic_mask,
    }

def save_metrics_csv(metrics):
    """
    Save the data used by panels (a) and (b) in one long-format CSV file.

    One row corresponds to one (k, St) case. The file records the selected
    cycle pair, local-correlation statistics, periodicity classification,
    and wake-class information. UTF-8 with BOM is used for convenient
    opening in Microsoft Excel.
    """
    csv_path = FIGURE_BASENAME + "_data.csv"

    fieldnames = [
        "St",
        "k",
        "selected_cycle_pair",
        "rho_mean",
        "rho_median",
        "rho_std",
        "rho_min",
        "rho_max",
        "valid_spatial_points",
        "valid_case",
        "aperiodic_threshold",
        "is_aperiodic",
        "data_status",
        "wake_class_id",
        "wake_class_code",
        "wake_class_name",
        "wake_class_color",
        "source_npz",
    ]

    def finite_or_blank(value):
        value = float(value)
        return f"{value:.10g}" if np.isfinite(value) else ""

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for st_index, st in enumerate(ST_LIST):
            for k_index, k in enumerate(K_LIST):
                valid_case = bool(metrics["valid_case_mask"][st_index, k_index])
                is_aperiodic = bool(metrics["aperiodic_mask"][st_index, k_index])
                class_id = int(CLASS_ID_MATRIX[st_index, k_index])
                class_info = CLASS_INFO[class_id]

                writer.writerow({
                    "St": f"{st:g}",
                    "k": f"{k:g}",
                    "selected_cycle_pair": metrics[
                        "selected_pair_labels"
                    ][st_index, k_index],
                    "rho_mean": finite_or_blank(
                        metrics["rho_mean"][st_index, k_index]
                    ),
                    "rho_median": finite_or_blank(
                        metrics["rho_median"][st_index, k_index]
                    ),
                    "rho_std": finite_or_blank(
                        metrics["rho_std"][st_index, k_index]
                    ),
                    "rho_min": finite_or_blank(
                        metrics["rho_min"][st_index, k_index]
                    ),
                    "rho_max": finite_or_blank(
                        metrics["rho_max"][st_index, k_index]
                    ),
                    "valid_spatial_points": int(
                        metrics["valid_spatial_points"][st_index, k_index]
                    ),
                    "valid_case": int(valid_case),
                    "aperiodic_threshold": f"{APERIODIC_RHO_THRESHOLD:.6g}",
                    "is_aperiodic": int(is_aperiodic) if valid_case else "",
                    "data_status": "loaded" if valid_case else "missing_or_failed",
                    "wake_class_id": class_id,
                    "wake_class_code": class_info["code"],
                    "wake_class_name": class_info["name"],
                    "wake_class_color": class_info["color"],
                    "source_npz": get_case_npz_path(
                        CORRELATION_ROOT, k, st
                    ),
                })

    print(f"Saved CSV: {csv_path}")


def _unit_square_diagonal_segment(offset):
    """
    Return the endpoints of v = u + offset inside the unit square.

    The segment is computed explicitly rather than clipped with an SVG
    pattern. Adobe Illustrator can therefore import every diagonal as a
    normal vector stroke.
    """
    candidates = [
        (0.0, offset),          # left boundary
        (1.0, 1.0 + offset),    # right boundary
        (-offset, 0.0),         # bottom boundary
        (1.0 - offset, 1.0),    # top boundary
    ]

    points = []
    tolerance = 1.0e-12

    for u, v in candidates:
        if (
            -tolerance <= u <= 1.0 + tolerance
            and -tolerance <= v <= 1.0 + tolerance
        ):
            point = (
                float(np.clip(u, 0.0, 1.0)),
                float(np.clip(v, 0.0, 1.0)),
            )

            if not any(
                np.hypot(point[0] - old[0], point[1] - old[1])
                < tolerance
                for old in points
            ):
                points.append(point)

    if len(points) < 2:
        return None

    # At a corner, numerical coincidence may produce more than two
    # candidates. Retain the farthest pair.
    best_pair = None
    best_distance = -np.inf

    for first_index in range(len(points)):
        for second_index in range(first_index + 1, len(points)):
            first = points[first_index]
            second = points[second_index]
            distance = np.hypot(
                second[0] - first[0],
                second[1] - first[1],
            )

            if distance > best_distance:
                best_distance = distance
                best_pair = (first, second)

    return best_pair


def add_aperiodic_diagonal_lines(
    ax,
    mask,
    k_edges,
    st_edges,
    normalized_spacing=0.20,
    line_color="0.35",
    line_width=0.55,
):
    """
    Draw robust vector diagonals in cells identified as aperiodic.

    Matplotlib hatches are saved to SVG as reusable pattern definitions,
    usually combined with clip paths. Some Adobe Illustrator versions can
    discard or misread those patterns during import. Here each slash is
    instead written as an ordinary LineCollection segment whose endpoints
    are already restricted to the corresponding cell. No hatch pattern or
    cell-level clip path is required.
    """
    mask = np.asarray(mask, dtype=bool)

    if mask.shape != (len(st_edges) - 1, len(k_edges) - 1):
        raise ValueError(
            "The aperiodic mask shape must match the k-St cell grid: "
            f"mask={mask.shape}, grid="
            f"{(len(st_edges) - 1, len(k_edges) - 1)}."
        )

    if normalized_spacing <= 0.0 or normalized_spacing > 1.0:
        raise ValueError("normalized_spacing must lie in (0, 1].")

    unit_offsets = np.arange(
        -1.0 + normalized_spacing,
        1.0,
        normalized_spacing,
    )

    segments = []

    for st_index in range(mask.shape[0]):
        for k_index in range(mask.shape[1]):
            if not mask[st_index, k_index]:
                continue

            x0 = float(k_edges[k_index])
            x1 = float(k_edges[k_index + 1])
            y0 = float(st_edges[st_index])
            y1 = float(st_edges[st_index + 1])
            cell_width = x1 - x0
            cell_height = y1 - y0

            for offset in unit_offsets:
                unit_segment = _unit_square_diagonal_segment(offset)
                if unit_segment is None:
                    continue

                (u0, v0), (u1, v1) = unit_segment
                segments.append([
                    (x0 + u0 * cell_width, y0 + v0 * cell_height),
                    (x0 + u1 * cell_width, y0 + v1 * cell_height),
                ])

    if not segments:
        return None

    diagonal_collection = LineCollection(
        segments,
        colors=line_color,
        linewidths=line_width,
        linestyles="solid",
        antialiaseds=True,
        capstyle="butt",
        zorder=6,
    )
    diagonal_collection.set_rasterized(False)
    ax.add_collection(diagonal_collection)

    return diagonal_collection


def mark_missing_cells(ax, valid_case_mask):
    missing_mask = ~np.asarray(valid_case_mask, dtype=bool)

    for st_index, st in enumerate(ST_LIST):
        for k_index, k in enumerate(K_LIST):
            if not missing_mask[st_index, k_index]:
                continue

            ax.text(
                k,
                st,
                r"$\times$",
                ha="center",
                va="center",
                color="0.35",
                fontsize=FONT_SIZE,
                zorder=8,
            )

# ============================================================
# Plotting
# ============================================================
def plot_figure(metrics):
    k_edges = centers_to_edges(K_LIST)
    st_edges = centers_to_edges(ST_LIST)

    # A dedicated GridSpec is used so that the two maps, colour bar,
    # colour bar and categorical legend do not compete for space.
    fig = plt.figure(
        figsize=(6.5, 4.55),
        dpi=300,
    )

    grid = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[1.0, 0.075, 0.42],
        width_ratios=[1.0, 1.0],
        left=0.082,
        right=0.995,
        bottom=0.055,
        top=0.920,
        wspace=0.22,
        hspace=0.34,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1], sharex=ax_a, sharey=ax_a)
    colorbar_ax = fig.add_subplot(grid[1, 0])
    legend_ax = fig.add_subplot(grid[2, :])
    axes = [ax_a, ax_b]

    # --------------------------------------------------------
    # Panel (a): spatially averaged rho map, fixed range 0--1
    # --------------------------------------------------------
    correlation_cmap = plt.get_cmap("viridis").copy()
    correlation_cmap.set_bad("#F2F2F2")
    correlation_norm = Normalize(vmin=0, vmax=1.0)

    mesh_a = ax_a.pcolormesh(
        k_edges,
        st_edges,
        np.ma.masked_invalid(metrics["rho_mean"]),
        cmap=correlation_cmap,
        norm=correlation_norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.35,
        rasterized=True,
    )

    add_aperiodic_diagonal_lines(
        ax_a,
        metrics["aperiodic_mask"],
        k_edges,
        st_edges,
    )

    mark_missing_cells(
        ax_a,
        metrics["valid_case_mask"],
    )

    # --------------------------------------------------------
    # Panel (b): wake classification map
    # --------------------------------------------------------
    class_to_index = {
        class_id: index
        for index, class_id in enumerate(CLASS_ORDER)
    }

    wake_numeric = np.full(
        CLASS_ID_MATRIX.shape,
        np.nan,
        dtype=float,
    )

    for st_index in range(len(ST_LIST)):
        for k_index in range(len(K_LIST)):
            class_id = int(CLASS_ID_MATRIX[st_index, k_index])
            wake_numeric[st_index, k_index] = class_to_index[class_id]

    wake_colors = [
        CLASS_INFO[class_id]["color"]
        for class_id in CLASS_ORDER
    ]

    wake_cmap = ListedColormap(wake_colors)
    wake_norm = BoundaryNorm(
        np.arange(len(CLASS_ORDER) + 1) - 0.5,
        len(CLASS_ORDER),
    )

    ax_b.pcolormesh(
        k_edges,
        st_edges,
        wake_numeric,
        cmap=wake_cmap,
        norm=wake_norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.35,
        rasterized=True,
    )

    # Class codes inside the cells preserve readability at journal size.
    for st_index, st in enumerate(ST_LIST):
        for k_index, k in enumerate(K_LIST):
            class_id = int(CLASS_ID_MATRIX[st_index, k_index])
            ax_b.text(
                k,
                st,
                CLASS_INFO[class_id]["code"],
                ha="center",
                va="center",
                fontsize=FONT_SIZE,
                color="black",
                zorder=4,
            )

    # --------------------------------------------------------
    # Common axis formatting
    # --------------------------------------------------------
    panel_titles = [
        r"$\overline{\rho}_{\omega}$",
        "Wake classification",
    ]

    for panel_index, (ax, title) in enumerate(zip(axes, panel_titles)):
        ax.set_title(title, pad=4.0)
        ax.set_xlabel(r"$k$", labelpad=2.0)
        ax.set_xticks(K_LIST)
        ax.set_yticks(ST_LIST)
        ax.tick_params(top=True, right=True)

        # Required row order from top to bottom: 0.2, 0.3, ..., 0.8.
        ax.set_ylim(st_edges[-1], st_edges[0])

        ax.text(
            -0.16,
            1.04,
            rf"$({chr(97 + panel_index)})$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            clip_on=False,
            fontsize=FONT_SIZE,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(0.7)

    ax_a.set_ylabel(r"$St$", labelpad=3.0)

    # --------------------------------------------------------
    # Colour bar directly beneath panel (a)
    # --------------------------------------------------------
    colorbar = fig.colorbar(
        mesh_a,
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=[0, 1],
    )
    colorbar.set_label(
        r"$\overline{\rho}_{\omega}$",
        labelpad=1.0,
    )
    colorbar.ax.tick_params(
        length=2.5,
        width=0.6,
        pad=1.2,
        labelsize=FONT_SIZE,
    )
    colorbar.outline.set_linewidth(0.6)

    # --------------------------------------------------------
    # Wake-class key: two columns with four entries per column.
    # The legend names are read directly from CLASS_INFO so that
    # they remain identical to the classifications defined above.
    # --------------------------------------------------------
    legend_ax.set_xlim(0.0, 1.0)
    legend_ax.set_ylim(0.0, 1.0)
    legend_ax.axis("off")

    column_x = [0.015, 0.515]
    row_y = [0.875, 0.625, 0.375, 0.125]

    for item_index, class_id in enumerate(CLASS_ORDER):
        # Items 1--4 occupy the left column; items 5--8 occupy
        # the right column.
        column = item_index // 4
        row = item_index % 4
        x0 = column_x[column]
        y0 = row_y[row]

        legend_ax.add_patch(
            Rectangle(
                (x0, y0 - 0.075),
                0.030,
                0.150,
                transform=legend_ax.transAxes,
                facecolor=CLASS_INFO[class_id]["color"],
                edgecolor="0.4",
                linewidth=0.35,
                clip_on=False,
            )
        )

        legend_ax.text(
            x0 + 0.040,
            y0,
            (
                f"{CLASS_INFO[class_id]['code']}  "
                f"{CLASS_INFO[class_id]['name']}"
            ),
            transform=legend_ax.transAxes,
            ha="left",
            va="center",
            fontsize=FONT_SIZE,
        )

    # Force every text object, including legends and colour-bar labels, to 10 pt.
    for text in fig.findobj(match=plt.Text):
        text.set_fontsize(FONT_SIZE)

    png_path = FIGURE_BASENAME + ".png"
    svg_path = FIGURE_BASENAME + ".svg"
    pdf_path = FIGURE_BASENAME + ".pdf"

    # Do not use bbox_inches='tight': this preserves the requested 6.5-inch width.
    fig.savefig(png_path, dpi=600)
    fig.savefig(
        svg_path,
        format="svg",
        metadata={
            "Creator": (
                "Matplotlib; aperiodic cells use explicit vector "
                "LineCollection strokes"
            )
        },
    )
    # fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")
    print(f"Saved PDF: {pdf_path}")

# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    metrics = load_metrics(CORRELATION_ROOT)
    save_metrics_csv(metrics)
    plot_figure(metrics)


if __name__ == "__main__":
    main()