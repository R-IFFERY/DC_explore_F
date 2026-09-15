# -*- coding: utf-8 -*-
"""
Batch pointwise phase-series correlation maps for selected non-zero-pitch cases.

For every explicitly specified (k, St, a) case in CASE_LIST:
1. Read phase-matched vorticity fields from cycles 16-20.
2. Compute pointwise Pearson correlations between adjacent cycles or all pairs.
3. Save one NPZ cache, one per-case CSV, and one multi-panel figure.
4. Merge all case statistics into one all-case summary CSV.
5. Record missing or failed cases without interrupting the remaining batch.

The pitch parameter a is included in input-directory lookup, output names,
NPZ metadata/cache validation, figure titles, and CSV summaries so that cases
with identical k and St but different a values never overwrite one another.
"""

import csv
import gc
import math
import os
import re
import warnings
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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
        "text.latex.preamble":
            r"\usepackage{amsmath}\usepackage{newtxtext,newtxmath}"
    })

# ============================================================
# User parameters: selected (k, St, a) cases
# ============================================================
c = 1.0
dc = 1

# Each tuple is (k, St, a). The order below is also the processing order.
CASE_LIST = [
    # (0.8, 0.3, 19),
    # (0.8, 0.3, 20),
    # (0.8, 0.3, 21),
    #
    # (1.1, 0.3, 19),
    # (1.1, 0.3, 20),
    # (1.1, 0.3, 21),
    #
    # (1.4, 0.3, 16),
    # (1.4, 0.3, 17),
    # (1.4, 0.3, 18),
    #
    (1.7, 0.3, 10),
    # (1.7, 0.3, 11),
    # (1.7, 0.3, 12),
]

# Analyse cycles 16-20. Each cycle contains 32 matched phases.
CYCLE_LIST = np.arange(21, 31, dtype=int)
N_PHASES = 32

# "adjacent": consecutive-cycle pairs determined from CYCLE_LIST.
# "all": all pair combinations among the cycles in CYCLE_LIST.
PAIR_MODE = "adjacent"

# A filename such as F247541.dat denotes t = 24.7541.
FILENAME_TIME_SCALE = 10000.0

# Phase matching tolerance in units of the flapping period T.
PHASE_TOL = 0.020

BASE_ROOT = r"E:/Rensq/DC_explore_F/data_1Tflow/"
SAVE_ROOT = r"E:/Rensq/DC_explore_F/flow/local_cycle_correlation_maps_aneq0"
os.makedirs(SAVE_ROOT, exist_ok=True)

# False enables per-case cache reuse and therefore supports interrupted runs.
FORCE_RECOMPUTE = False

# Phase stacks are large. Correlation maps can be re-plotted from the NPZ
# without them, so False is recommended for this multi-case batch.
SAVE_PHASE_STACKS = False

# Continue processing other cases after a missing directory/read error.
STOP_ON_ERROR = False

# ============================================================
# Analysis window: global coordinates x/c and y/c
# ============================================================
X_MIN = -1.0 * c
X_MAX = 20.0 * c
Y_MIN = -10.0 * c
Y_MAX = 10.0 * c

# Spatial sampling used only for the correlation maps.
CORR_TARGET_DX = 0.10 * c
CORR_TARGET_DY = 0.10 * c

# ============================================================
# Solid-body and local-correlation settings
# ============================================================
EXCLUDE_SOLID_BODY = True
BODY_SOLID_THRESHOLD = 0.5

# A point must contain at least this many mutually valid phase pairs.
MIN_VALID_PHASES = 12

# Weak-flow filtering for pointwise phase-series correlations.
LOCAL_ACTIVITY_FRAC = 0.02
ACTIVITY_PERCENTILE = 99.0

# Pearson correlation is undefined for an almost constant phase series.
CORR_EPS = 1.0e-14

# Used only for diagnostic fractions in CSV files.
LOW_CORRELATION_THRESHOLD = 0.90

# ============================================================
# Global output paths
# ============================================================
ALL_CASES_SUMMARY_CSV = os.path.join(
    SAVE_ROOT,
    f"selected_aneq0_local_rho_{PAIR_MODE}_summary.csv"
)
FAILED_CASES_CSV = os.path.join(
    SAVE_ROOT,
    f"selected_aneq0_failed_cases_{PAIR_MODE}.csv"
)


def fmt_num(value):
    return f"{float(value):g}"


def case_tag(k, st, a_value):
    return (
        f"k{fmt_num(k)}_st{fmt_num(st)}_a{fmt_num(a_value)}"
    )


def get_case_output_paths(k, st, a_value):
    tag = case_tag(k, st, a_value)
    case_output_dir = os.path.join(SAVE_ROOT, tag)
    os.makedirs(case_output_dir, exist_ok=True)

    return {
        "tag": tag,
        "dir": case_output_dir,
        "npz": os.path.join(
            case_output_dir,
            f"local_rho_maps_{tag}_{PAIR_MODE}.npz"
        ),
        "csv": os.path.join(
            case_output_dir,
            f"local_rho_maps_{tag}_{PAIR_MODE}_summary.csv"
        ),
        "figure_basename": os.path.join(
            case_output_dir,
            f"local_rho_maps_{tag}_{PAIR_MODE}"
        ),
    }

# ============================================================
# Path utilities
# ============================================================
def unique_keep_order(sequence):
    output = []
    for item in sequence:
        if item not in output:
            output.append(item)
    return output


def get_case_dir_candidates(base_root, k, st, a_value, dc_value):
    k_names = unique_keep_order([str(k), fmt_num(k)])
    st_names = unique_keep_order([str(st), fmt_num(st)])

    candidates = []
    for ks in k_names:
        for ss in st_names:
            common = os.path.join(
                base_root,
                f"k{ks}",
                f"st{ss}k{ks}",
                f"st{ss}k{ks}a{a_value}",
                f"st{ss}k{ks}a{a_value}dc{int(dc_value)}"
            )
            candidates.extend([
                os.path.join(common, f"k{ks}st{ss}a{a_value}"),
                # Retain the alternative leaf name for compatibility with
                # earlier data exports stored under the same a-value branch.
                os.path.join(common, f"k{ks}st{ss}"),
            ])

    return unique_keep_order(candidates)


def find_existing_case_dir(base_root, k, st, a_value, dc_value):
    for directory in get_case_dir_candidates(
        base_root, k, st, a_value, dc_value
    ):
        if os.path.isdir(directory):
            return directory
    return None

# ============================================================
# File-time utilities
# ============================================================
def parse_time_from_filename(filename):
    base = os.path.basename(filename)
    match = re.match(r"^F\s*(\d+)\.dat$", base, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)) / FILENAME_TIME_SCALE


def is_valid_flow_filename(filename):
    return re.match(
        r"^F\s*\d+\.dat$",
        os.path.basename(filename),
        re.IGNORECASE
    ) is not None


def list_flow_records(case_dir):
    records = []
    for filename in os.listdir(case_dir):
        if not is_valid_flow_filename(filename):
            continue
        time_value = parse_time_from_filename(filename)
        if time_value is None:
            continue
        records.append({
            "filename": filename,
            "path": os.path.join(case_dir, filename),
            "t": time_value,
        })

    records.sort(key=lambda item: item["t"])
    return records


def find_nearest_time_record(records, target_time, tolerance):
    if not records:
        return None

    times = np.fromiter(
        (item["t"] for item in records),
        dtype=float,
        count=len(records)
    )
    index = int(np.argmin(np.abs(times - target_time)))
    best = records[index]

    if abs(best["t"] - target_time) <= tolerance:
        return best
    return None


def build_phase_record_table(records, k, cycles, n_phases):
    """
    Detect either phase-saving convention:
      A: 0, 1/32, ..., 31/32
      B: 1/32, 2/32, ..., 1

    Returns table[phase_index][cycle_index].
    """
    period = np.pi / k
    tolerance = PHASE_TOL * period

    phase_candidates = {
        "zero_to_31_over_32":
            np.arange(n_phases, dtype=float) / n_phases,
        "one_over_32_to_one":
            np.arange(1, n_phases + 1, dtype=float) / n_phases,
    }

    candidate_results = {}

    for name, phases in phase_candidates.items():
        table = []
        errors = []
        valid_count = 0

        for phase in phases:
            phase_records = []
            phase_errors = []

            for cycle in cycles:
                target_time = (cycle - 1 + phase) * period
                record = find_nearest_time_record(
                    records,
                    target_time,
                    tolerance
                )
                phase_records.append(record)

                if record is None:
                    phase_errors.append(np.nan)
                else:
                    valid_count += 1
                    phase_errors.append(
                        abs(record["t"] - target_time) / period
                    )

            table.append(phase_records)
            errors.append(phase_errors)

        candidate_results[name] = {
            "phases": phases,
            "table": table,
            "errors": np.asarray(errors, dtype=float),
            "valid_count": valid_count,
        }

    # In the unlikely event of a tie, choose the convention with smaller
    # mean phase error rather than relying only on dictionary order.
    def candidate_score(name):
        item = candidate_results[name]
        finite_errors = item["errors"][np.isfinite(item["errors"])]
        mean_error = np.mean(finite_errors) if finite_errors.size else np.inf
        return item["valid_count"], -mean_error

    selected_name = max(candidate_results, key=candidate_score)
    selected = candidate_results[selected_name]

    total_targets = len(cycles) * n_phases
    max_error = (
        np.nanmax(selected["errors"])
        if np.any(np.isfinite(selected["errors"]))
        else np.nan
    )

    print(
        f"Phase convention: {selected_name}; "
        f"matched files={selected['valid_count']}/{total_targets}; "
        f"maximum phase error={max_error:.5f}T"
    )

    return (
        selected_name,
        selected["phases"],
        selected["table"],
        selected["errors"],
    )

# ============================================================
# Tecplot ordered-ASCII reader
# ============================================================
VAR_MAP = {
    "X": 0,
    "Y": 1,
    "U": 2,
    "V": 3,
    "Presure": 4,
    "Pressure": 4,
    "Vorticity": 5,
    "body": 6,
    "dvordt": 7,
}


def inspect_tecplot_header(filename, required_columns):
    i_size = None
    j_size = None
    solution_time = None
    data_start_line = None
    header_text = []
    zone_found = False

    with open(filename, "r", encoding="utf-8", errors="ignore") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            if not zone_found and stripped.upper().startswith("ZONE"):
                zone_found = True

            if not zone_found:
                continue

            header_text.append(stripped)
            joined_header = " ".join(header_text)

            if i_size is None:
                match_i = re.search(r"\bI\s*=\s*(\d+)", joined_header, re.I)
                if match_i:
                    i_size = int(match_i.group(1))

            if j_size is None:
                match_j = re.search(r"\bJ\s*=\s*(\d+)", joined_header, re.I)
                if match_j:
                    j_size = int(match_j.group(1))

            if solution_time is None:
                match_time = re.search(
                    r"SOLUTIONTIME\s*=\s*([\d.Ee+\-]+)",
                    joined_header,
                    re.I
                )
                if match_time:
                    try:
                        solution_time = float(match_time.group(1))
                    except ValueError:
                        solution_time = None

            parts = stripped.split()
            if len(parts) < required_columns:
                continue

            try:
                [float(parts[column]) for column in range(required_columns)]
                data_start_line = line_number
                break
            except ValueError:
                continue

    if i_size is None or j_size is None or data_start_line is None:
        raise ValueError(f"Cannot parse Tecplot header: {filename}")

    return i_size, j_size, solution_time, data_start_line


def read_tecplot_ordered_dat(filename, target_vars=("X", "Y", "Vorticity")):
    columns = [VAR_MAP[var] for var in target_vars]
    required_columns = max(columns) + 1

    i_size, j_size, solution_time, data_start_line = inspect_tecplot_header(
        filename,
        required_columns
    )

    data = np.loadtxt(
        filename,
        skiprows=data_start_line - 1,
        max_rows=i_size * j_size,
        usecols=columns,
        dtype=np.float64,
    )

    if data.ndim == 1:
        data = data[:, None]

    if data.shape[0] != i_size * j_size:
        raise ValueError(
            f"Insufficient data points in {filename}: "
            f"expected {i_size * j_size}, obtained {data.shape[0]}"
        )

    result = {
        "I": i_size,
        "J": j_size,
        "solution_time": solution_time,
    }

    for column_index, variable in enumerate(target_vars):
        result[variable] = data[:, column_index].reshape(j_size, i_size)

    return result

# ============================================================
# Grid selection and field loading
# ============================================================
def choose_stride(coordinate_line, target_spacing):
    differences = np.diff(np.asarray(coordinate_line, dtype=float))
    differences = differences[
        np.isfinite(differences) & (differences > 0.0)
    ]

    if differences.size == 0:
        raise ValueError("Cannot determine a valid grid spacing.")

    spacing = float(np.median(differences))
    stride = max(1, int(round(target_spacing / spacing)))
    return stride, spacing


def build_selected_indices(X, Y):
    x_line = np.asarray(X[0, :], dtype=float)
    y_line = np.asarray(Y[:, 0], dtype=float)

    stride_x, raw_dx = choose_stride(x_line, CORR_TARGET_DX)
    stride_y, raw_dy = choose_stride(y_line, CORR_TARGET_DY)

    x_candidates = np.arange(0, x_line.size, stride_x, dtype=int)
    y_candidates = np.arange(0, y_line.size, stride_y, dtype=int)

    x_keep = x_candidates[
        (x_line[x_candidates] >= X_MIN)
        & (x_line[x_candidates] <= X_MAX)
    ]
    y_keep = y_candidates[
        (y_line[y_candidates] >= Y_MIN)
        & (y_line[y_candidates] <= Y_MAX)
    ]

    if x_keep.size < 3 or y_keep.size < 3:
        raise ValueError(
            "The selected coordinate window contains too few grid points."
        )

    diagnostics = {
        "stride_x": stride_x,
        "stride_y": stride_y,
        "raw_dx": raw_dx,
        "raw_dy": raw_dy,
        "effective_dx": stride_x * raw_dx,
        "effective_dy": stride_y * raw_dy,
        "nx": x_keep.size,
        "ny": y_keep.size,
    }
    return x_keep, y_keep, diagnostics


def fluid_mask_from_body(body):
    return np.asarray(body, dtype=float) < BODY_SOLID_THRESHOLD


def find_first_available_record(table):
    for row in table:
        for record in row:
            if record is not None:
                return record
    return None


def load_case_phase_stacks(case_dir, k, cycles, n_phases):
    records = list_flow_records(case_dir)
    if not records:
        raise FileNotFoundError(f"No F*.dat files were found in {case_dir}")

    convention, phases, table, phase_errors = build_phase_record_table(
        records,
        k,
        cycles,
        n_phases
    )

    first_record = find_first_available_record(table)
    if first_record is None:
        raise RuntimeError("No phase-matched flow file could be found.")

    first_data = read_tecplot_ordered_dat(
        first_record["path"],
        target_vars=("X", "Y", "Vorticity", "body")
    )
    full_shape = first_data["Vorticity"].shape

    x_indices, y_indices, sampling_diagnostics = build_selected_indices(
        first_data["X"],
        first_data["Y"]
    )

    X_selected = first_data["X"][np.ix_(y_indices, x_indices)] / c
    Y_selected = first_data["Y"][np.ix_(y_indices, x_indices)] / c

    n_cycles = len(cycles)
    ny, nx = X_selected.shape
    vorticity = np.full(
        (n_cycles, n_phases, ny, nx),
        np.nan,
        dtype=np.float32
    )
    fluid_valid = np.zeros(
        (n_cycles, n_phases, ny, nx),
        dtype=bool
    )
    matched = np.zeros((n_cycles, n_phases), dtype=bool)

    total_files = n_cycles * n_phases
    loaded_files = 0

    for phase_index, phase in enumerate(phases):
        for cycle_index, cycle in enumerate(cycles):
            record = table[phase_index][cycle_index]
            if record is None:
                print(f"[Missing] cycle={cycle}, phase={phase:.5f}T")
                continue

            try:
                data = read_tecplot_ordered_dat(
                    record["path"],
                    target_vars=("Vorticity", "body")
                )
                if data["Vorticity"].shape != full_shape:
                    raise ValueError(
                        f"Grid shape changed: {data['Vorticity'].shape} "
                        f"!= {full_shape}"
                    )

                omega = data["Vorticity"][np.ix_(y_indices, x_indices)]
                body = data["body"][np.ix_(y_indices, x_indices)]

                finite_mask = np.isfinite(omega)
                if EXCLUDE_SOLID_BODY:
                    local_fluid = fluid_mask_from_body(body) & finite_mask
                else:
                    local_fluid = finite_mask

                omega = np.where(local_fluid, omega, np.nan)

                vorticity[cycle_index, phase_index] = omega.astype(
                    np.float32,
                    copy=False
                )
                fluid_valid[cycle_index, phase_index] = local_fluid
                matched[cycle_index, phase_index] = True
                loaded_files += 1

                print(
                    f"Loaded {loaded_files:3d}/{total_files}: "
                    f"cycle={cycle}, phase={phase:.5f}T, "
                    f"file={record['filename']}"
                )

            except Exception as error:
                print(
                    f"[Error] cycle={cycle}, phase={phase:.5f}T, "
                    f"file={record['filename']}: {error}"
                )

    print(
        "Correlation-map sampling: "
        f"stride=({sampling_diagnostics['stride_x']}, "
        f"{sampling_diagnostics['stride_y']}), "
        f"effective spacing=("
        f"{sampling_diagnostics['effective_dx'] / c:.4f}c, "
        f"{sampling_diagnostics['effective_dy'] / c:.4f}c), "
        f"grid={sampling_diagnostics['nx']} x "
        f"{sampling_diagnostics['ny']}"
    )

    metadata = {
        "phase_convention": convention,
        "phases": phases,
        "phase_errors": phase_errors,
        "matched": matched,
        "sampling_diagnostics": sampling_diagnostics,
    }
    return X_selected, Y_selected, vorticity, fluid_valid, metadata

# ============================================================
# Pointwise correlation of phase sequences
# ============================================================
def build_cycle_pairs(cycles, pair_mode):
    cycles = np.asarray(cycles, dtype=int)

    if pair_mode == "adjacent":
        return [
            (index, index + 1)
            for index in range(len(cycles) - 1)
        ]

    if pair_mode == "all":
        return list(combinations(range(len(cycles)), 2))

    raise ValueError("PAIR_MODE must be either 'adjacent' or 'all'.")


def pointwise_pearson_phase_series(stack_1, stack_2):
    """
    At each Eulerian point, calculate Pearson correlation between the two
    phase-resolved vorticity sequences. Solid-occupied phases are removed
    phase by phase instead of excluding the point permanently.
    """
    field_1 = np.asarray(stack_1, dtype=np.float64)
    field_2 = np.asarray(stack_2, dtype=np.float64)

    if field_1.shape != field_2.shape or field_1.ndim != 3:
        raise ValueError(
            "The two cycle stacks must have identical shape "
            "(n_phase, ny, nx)."
        )

    valid = np.isfinite(field_1) & np.isfinite(field_2)
    valid_count = np.sum(valid, axis=0)

    values_1 = np.where(valid, field_1, 0.0)
    values_2 = np.where(valid, field_2, 0.0)

    safe_count = np.maximum(valid_count, 1)
    mean_1 = np.sum(values_1, axis=0) / safe_count
    mean_2 = np.sum(values_2, axis=0) / safe_count

    centered_1 = np.where(valid, field_1 - mean_1[None, :, :], 0.0)
    centered_2 = np.where(valid, field_2 - mean_2[None, :, :], 0.0)

    numerator = np.sum(centered_1 * centered_2, axis=0)
    energy_1 = np.sum(centered_1 ** 2, axis=0)
    energy_2 = np.sum(centered_2 ** 2, axis=0)
    denominator = np.sqrt(energy_1 * energy_2)

    activity_numerator = np.sum(
        np.where(valid, 0.5 * (field_1 ** 2 + field_2 ** 2), 0.0),
        axis=0
    )
    pair_activity = np.sqrt(activity_numerator / safe_count)

    finite_activity = pair_activity[
        np.isfinite(pair_activity) & (valid_count >= MIN_VALID_PHASES)
    ]

    if LOCAL_ACTIVITY_FRAC <= 0.0 or finite_activity.size == 0:
        activity_threshold = 0.0
    else:
        activity_reference = float(
            np.nanpercentile(finite_activity, ACTIVITY_PERCENTILE)
        )
        activity_threshold = LOCAL_ACTIVITY_FRAC * activity_reference

    valid_output = (
        (valid_count >= MIN_VALID_PHASES)
        & np.isfinite(denominator)
        & (denominator > CORR_EPS)
        & np.isfinite(pair_activity)
        & (pair_activity >= activity_threshold)
    )

    rho = np.full(valid_count.shape, np.nan, dtype=np.float64)
    rho[valid_output] = numerator[valid_output] / denominator[valid_output]
    rho[valid_output] = np.clip(rho[valid_output], -1.0, 1.0)

    diagnostics = {
        "activity_threshold": activity_threshold,
        "activity_reference_percentile": ACTIVITY_PERCENTILE,
        "valid_point_count": int(np.sum(np.isfinite(rho))),
        "total_point_count": int(rho.size),
    }
    return rho, valid_count, pair_activity, diagnostics


def compute_local_correlation_maps(vorticity, cycles, pair_mode):
    cycle_pairs = build_cycle_pairs(cycles, pair_mode)
    pair_maps = []
    valid_phase_maps = []
    activity_maps = []
    pair_labels = []
    diagnostics = []

    for cycle_index_1, cycle_index_2 in cycle_pairs:
        cycle_1 = int(cycles[cycle_index_1])
        cycle_2 = int(cycles[cycle_index_2])
        print("=" * 80)
        print(f"Computing pointwise rho: cycle {cycle_1} vs {cycle_2}")

        rho_map, valid_count, activity, pair_diagnostics = (
            pointwise_pearson_phase_series(
                vorticity[cycle_index_1],
                vorticity[cycle_index_2]
            )
        )

        pair_maps.append(rho_map)
        valid_phase_maps.append(valid_count)
        activity_maps.append(activity)
        pair_labels.append(f"{cycle_1}-{cycle_2}")
        diagnostics.append(pair_diagnostics)

        valid_values = rho_map[np.isfinite(rho_map)]
        if valid_values.size:
            print(
                f"  valid points={valid_values.size}/{rho_map.size}; "
                f"mean rho={np.mean(valid_values):.5f}; "
                f"median rho={np.median(valid_values):.5f}; "
                f"activity threshold="
                f"{pair_diagnostics['activity_threshold']:.5e}"
            )
        else:
            print("  No valid local-correlation points were obtained.")

    pair_maps = np.asarray(pair_maps, dtype=np.float64)
    valid_phase_maps = np.asarray(valid_phase_maps, dtype=np.int16)
    activity_maps = np.asarray(activity_maps, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rho_mean_map = np.nanmean(pair_maps, axis=0)
        rho_std_map = np.nanstd(pair_maps, axis=0)
        pair_support_count = np.sum(np.isfinite(pair_maps), axis=0)

    minimum_pair_support = min(2, len(pair_maps))
    summary_valid = pair_support_count >= minimum_pair_support
    rho_mean_map = np.where(summary_valid, rho_mean_map, np.nan)
    rho_std_map = np.where(summary_valid, rho_std_map, np.nan)

    return {
        "pair_maps": pair_maps,
        "valid_phase_maps": valid_phase_maps,
        "activity_maps": activity_maps,
        "pair_labels": np.asarray(pair_labels, dtype="U16"),
        "rho_mean_map": rho_mean_map,
        "rho_std_map": rho_std_map,
        "pair_support_count": pair_support_count,
        "diagnostics": diagnostics,
    }

# ============================================================
# Save and load results
# ============================================================
def save_metrics_npz(
    npz_path,
    k,
    st,
    a_value,
    case_dir,
    X,
    Y,
    vorticity,
    fluid_valid,
    metadata,
    metrics,
):
    sampling = metadata["sampling_diagnostics"]

    arrays = {
        "k": k,
        "st": st,
        "a": a_value,
        "dc": dc,
        "case_directory": np.asarray(case_dir),
        "cycle_list": CYCLE_LIST,
        "n_phases": N_PHASES,
        "pair_mode": np.asarray(PAIR_MODE),
        "phases": metadata["phases"],
        "phase_convention": np.asarray(metadata["phase_convention"]),
        "phase_errors": metadata["phase_errors"],
        "matched_phase_files": metadata["matched"],
        "x_over_c": X,
        "y_over_c": Y,
        "x_range": np.array([X_MIN / c, X_MAX / c], dtype=float),
        "y_range": np.array([Y_MIN / c, Y_MAX / c], dtype=float),
        "corr_target_dx": CORR_TARGET_DX / c,
        "corr_target_dy": CORR_TARGET_DY / c,
        "effective_dx": sampling["effective_dx"] / c,
        "effective_dy": sampling["effective_dy"] / c,
        "phase_tolerance": PHASE_TOL,
        "exclude_solid_body": int(EXCLUDE_SOLID_BODY),
        "body_solid_threshold": BODY_SOLID_THRESHOLD,
        "min_valid_phases": MIN_VALID_PHASES,
        "local_activity_fraction": LOCAL_ACTIVITY_FRAC,
        "activity_percentile": ACTIVITY_PERCENTILE,
        "pair_maps": metrics["pair_maps"],
        "valid_phase_maps": metrics["valid_phase_maps"],
        "activity_maps": metrics["activity_maps"],
        "pair_labels": metrics["pair_labels"],
        "rho_mean_map": metrics["rho_mean_map"],
        "rho_std_map": metrics["rho_std_map"],
        "pair_support_count": metrics["pair_support_count"],
    }

    if SAVE_PHASE_STACKS:
        arrays["vorticity_phase_stacks"] = vorticity
        arrays["fluid_valid_phase_stacks"] = fluid_valid

    np.savez_compressed(npz_path, **arrays)
    print(f"Saved NPZ: {npz_path}")


def existing_npz_matches_current_settings(npz_path, k, st, a_value):
    if not os.path.isfile(npz_path):
        return False

    try:
        with np.load(npz_path, allow_pickle=False) as data:
            checks = [
                np.isclose(float(data["k"]), k),
                np.isclose(float(data["st"]), st),
                np.isclose(float(data["a"]), a_value),
                int(data["dc"]) == int(dc),
                np.array_equal(data["cycle_list"], CYCLE_LIST),
                int(data["n_phases"]) == N_PHASES,
                str(data["pair_mode"].item()) == PAIR_MODE,
                np.allclose(
                    data["x_range"],
                    np.array([X_MIN / c, X_MAX / c], dtype=float)
                ),
                np.allclose(
                    data["y_range"],
                    np.array([Y_MIN / c, Y_MAX / c], dtype=float)
                ),
                np.isclose(float(data["corr_target_dx"]), CORR_TARGET_DX / c),
                np.isclose(float(data["corr_target_dy"]), CORR_TARGET_DY / c),
                np.isclose(float(data["phase_tolerance"]), PHASE_TOL),
                int(data["exclude_solid_body"]) == int(EXCLUDE_SOLID_BODY),
                np.isclose(
                    float(data["body_solid_threshold"]),
                    BODY_SOLID_THRESHOLD
                ),
                int(data["min_valid_phases"]) == MIN_VALID_PHASES,
                np.isclose(
                    float(data["local_activity_fraction"]),
                    LOCAL_ACTIVITY_FRAC
                ),
                np.isclose(
                    float(data["activity_percentile"]),
                    ACTIVITY_PERCENTILE
                ),
            ]
            return bool(all(checks))
    except (KeyError, OSError, ValueError, TypeError):
        return False


def load_metrics_npz(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        required = {
            "x_over_c",
            "y_over_c",
            "pair_maps",
            "pair_labels",
            "rho_mean_map",
            "rho_std_map",
            "valid_phase_maps",
            "activity_maps",
            "pair_support_count",
        }
        missing = required.difference(data.files)
        if missing:
            raise KeyError(
                f"The existing NPZ file lacks required arrays: "
                f"{sorted(missing)}"
            )

        return {
            "X": np.array(data["x_over_c"]),
            "Y": np.array(data["y_over_c"]),
            "metrics": {
                "pair_maps": np.array(data["pair_maps"]),
                "pair_labels": np.array(data["pair_labels"]),
                "rho_mean_map": np.array(data["rho_mean_map"]),
                "rho_std_map": np.array(data["rho_std_map"]),
                "valid_phase_maps": np.array(data["valid_phase_maps"]),
                "activity_maps": np.array(data["activity_maps"]),
                "pair_support_count": np.array(data["pair_support_count"]),
            },
        }


def build_summary_rows(metrics, k, st, a_value):
    rows = []

    for pair_index, pair_label in enumerate(metrics["pair_labels"]):
        rho_map = metrics["pair_maps"][pair_index]
        valid_values = rho_map[np.isfinite(rho_map)]
        valid_phase_values = metrics["valid_phase_maps"][pair_index][
            np.isfinite(rho_map)
        ]

        if valid_values.size == 0:
            row = [
                fmt_num(k),
                fmt_num(st),
                fmt_num(a_value),
                str(pair_label),
                0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        else:
            row = [
                fmt_num(k),
                fmt_num(st),
                fmt_num(a_value),
                str(pair_label),
                int(valid_values.size),
                float(np.mean(valid_values)),
                float(np.median(valid_values)),
                float(np.std(valid_values)),
                float(np.min(valid_values)),
                float(np.max(valid_values)),
                float(np.mean(valid_values < LOW_CORRELATION_THRESHOLD)),
                float(np.mean(valid_phase_values)),
            ]
        rows.append(row)

    mean_values = metrics["rho_mean_map"][
        np.isfinite(metrics["rho_mean_map"])
    ]
    if mean_values.size:
        rows.append([
            fmt_num(k),
            fmt_num(st),
            fmt_num(a_value),
            "pair-mean map",
            int(mean_values.size),
            float(np.mean(mean_values)),
            float(np.median(mean_values)),
            float(np.std(mean_values)),
            float(np.min(mean_values)),
            float(np.max(mean_values)),
            float(np.mean(mean_values < LOW_CORRELATION_THRESHOLD)),
            np.nan,
        ])
    else:
        rows.append([
            fmt_num(k),
            fmt_num(st),
            fmt_num(a_value),
            "pair-mean map",
            0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ])

    return rows


def summary_header():
    return [
        "k",
        "St",
        "a",
        "cycle_pair",
        "valid_spatial_points",
        "mean_local_rho",
        "median_local_rho",
        "std_local_rho",
        "minimum_local_rho",
        "maximum_local_rho",
        f"fraction_rho_below_{LOW_CORRELATION_THRESHOLD:g}",
        "mean_valid_phases_per_spatial_point",
    ]


def write_summary_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(summary_header())
        for row in rows:
            formatted = []
            for value in row:
                if isinstance(value, (float, np.floating)):
                    if np.isfinite(value):
                        formatted.append(f"{float(value):.8e}")
                    else:
                        formatted.append("nan")
                else:
                    formatted.append(value)
            writer.writerow(formatted)

    print(f"Saved CSV: {path}")

# ============================================================
# Plotting
# ============================================================
def plot_correlation_maps(X, Y, metrics, k, st, a_value, figure_basename):
    pair_maps = metrics["pair_maps"]
    pair_labels = metrics["pair_labels"]

    maps = [pair_maps[index] for index in range(pair_maps.shape[0])]
    titles = [
        rf"$\rho_{{{str(label).replace('-', ',')}}}(x,y)$"
        for label in pair_labels
    ]
    maps.extend([
        metrics["rho_mean_map"],
        metrics["rho_std_map"],
    ])
    titles.extend([
        r"$\overline{\rho}(x,y)$",
        r"$\sigma_{\rho}(x,y)$",
    ])

    n_panels = len(maps)
    n_columns = 3 if n_panels <= 6 else 4
    n_rows = int(math.ceil(n_panels / n_columns))

    panel_width = 2.35
    panel_height = 2.30
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(panel_width * n_columns, panel_height * n_rows),
        dpi=300,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    flat_axes = axes.ravel()

    rho_cmap = plt.get_cmap("RdBu_r").copy()
    rho_cmap.set_bad("#E6E6E6")
    rho_norm = Normalize(vmin=-1.0, vmax=1.0)

    finite_std = metrics["rho_std_map"][
        np.isfinite(metrics["rho_std_map"])
    ]
    if finite_std.size:
        std_vmax = max(0.05, float(np.nanpercentile(finite_std, 99.0)))
    else:
        std_vmax = 1.0
    std_norm = Normalize(vmin=0.0, vmax=std_vmax)
    std_cmap = plt.get_cmap("viridis").copy()
    std_cmap.set_bad("#E6E6E6")

    rho_mesh = None
    std_mesh = None

    for panel_index, (field, title) in enumerate(zip(maps, titles)):
        ax = flat_axes[panel_index]
        is_std_panel = panel_index == n_panels - 1

        if is_std_panel:
            mesh = ax.pcolormesh(
                X,
                Y,
                np.ma.masked_invalid(field),
                cmap=std_cmap,
                norm=std_norm,
                shading="auto",
                rasterized=True,
            )
            std_mesh = mesh
        else:
            mesh = ax.pcolormesh(
                X,
                Y,
                np.ma.masked_invalid(field),
                cmap=rho_cmap,
                norm=rho_norm,
                shading="auto",
                rasterized=True,
            )
            rho_mesh = mesh

        ax.set_title(title, pad=3.0)
        ax.set_xlim(X_MIN / c, X_MAX / c)
        ax.set_ylim(Y_MIN / c, Y_MAX / c)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(top=True, right=True)

        row_index = panel_index // n_columns
        column_index = panel_index % n_columns
        if row_index == n_rows - 1:
            ax.set_xlabel(r"$x/c$")
        if column_index == 0:
            ax.set_ylabel(r"$y/c$")

        ax.text(
            0.015,
            0.985,
            rf"$({chr(97 + panel_index)})$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 1.0,
            },
        )

    for panel_index in range(n_panels, flat_axes.size):
        flat_axes[panel_index].set_visible(False)

    fig.suptitle(
        rf"$k={k:g},\ St={st:g},\ a={a_value:g}^\circ$",
        y=0.995,
        fontsize=FONT_SIZE,
    )

    fig.subplots_adjust(
        left=0.080,
        right=0.985,
        bottom=0.145,
        top=0.925,
        wspace=0.16,
        hspace=0.18,
    )

    if rho_mesh is not None:
        rho_cbar_ax = fig.add_axes([0.115, 0.055, 0.500, 0.025])
        rho_cbar = fig.colorbar(
            rho_mesh,
            cax=rho_cbar_ax,
            orientation="horizontal",
            ticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
        )
        rho_cbar.set_label(r"Local phase-series correlation $\rho$")
        rho_cbar.outline.set_linewidth(0.6)

    if std_mesh is not None:
        std_cbar_ax = fig.add_axes([0.700, 0.055, 0.220, 0.025])
        std_cbar = fig.colorbar(
            std_mesh,
            cax=std_cbar_ax,
            orientation="horizontal",
        )
        std_cbar.set_label(r"$\sigma_{\rho}$")
        std_cbar.outline.set_linewidth(0.6)

    png_path = figure_basename + ".png"
    svg_path = figure_basename + ".svg"
    pdf_path = figure_basename + ".pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")
    print(f"Saved PDF: {pdf_path}")

# ============================================================
# Per-case execution
# ============================================================
def process_one_case(k, st, a_value):
    paths = get_case_output_paths(k, st, a_value)
    cache_is_compatible = existing_npz_matches_current_settings(
        paths["npz"],
        k,
        st,
        a_value
    )

    status = "cached"

    if FORCE_RECOMPUTE or not cache_is_compatible:
        status = "computed"
        if os.path.isfile(paths["npz"]) and not FORCE_RECOMPUTE:
            print(
                "Existing NPZ settings differ from the current analysis "
                "settings; recomputing this case."
            )

        case_dir = find_existing_case_dir(BASE_ROOT, k, st, a_value, dc)
        if case_dir is None:
            candidates = get_case_dir_candidates(BASE_ROOT, k, st, a_value, dc)
            candidate_text = "\n".join(f"  {item}" for item in candidates)
            raise FileNotFoundError(
                "Case directory was not found. Checked:\n" + candidate_text
            )

        print(f"Case directory: {case_dir}")
        X, Y, vorticity, fluid_valid, metadata = load_case_phase_stacks(
            case_dir,
            k,
            CYCLE_LIST,
            N_PHASES
        )
        metrics = compute_local_correlation_maps(
            vorticity,
            CYCLE_LIST,
            PAIR_MODE
        )
        save_metrics_npz(
            paths["npz"],
            k,
            st,
            a_value,
            case_dir,
            X,
            Y,
            vorticity,
            fluid_valid,
            metadata,
            metrics
        )
    else:
        print(f"Loading existing metrics: {paths['npz']}")
        loaded = load_metrics_npz(paths["npz"])
        X = loaded["X"]
        Y = loaded["Y"]
        metrics = loaded["metrics"]

    rows = build_summary_rows(metrics, k, st, a_value)
    write_summary_csv(paths["csv"], rows)
    plot_correlation_maps(
        X,
        Y,
        metrics,
        k,
        st,
        a_value,
        paths["figure_basename"]
    )

    return rows, status

# ============================================================
# Main batch loop
# ============================================================
def validate_case_list():
    """Fail early if CASE_LIST contains malformed or duplicate cases."""
    normalized = []
    for index, item in enumerate(CASE_LIST, start=1):
        if len(item) != 3:
            raise ValueError(
                f"CASE_LIST item {index} must be (k, St, a), got: {item}"
            )
        k, st, a_value = item
        normalized.append((float(k), float(st), float(a_value)))

    if len(set(normalized)) != len(normalized):
        raise ValueError("CASE_LIST contains duplicate (k, St, a) cases.")


def main():
    validate_case_list()

    total_cases = len(CASE_LIST)
    all_summary_rows = []
    failed_cases = []
    computed_count = 0
    cached_count = 0

    print("=" * 80)
    print("Batch local cycle-correlation analysis for non-zero pitch cases")
    print(f"Explicit (k, St, a) cases: {total_cases}")
    print(f"Cycles: {CYCLE_LIST.tolist()}")
    print(f"Phases per cycle: {N_PHASES}")
    print(f"Pair mode: {PAIR_MODE}")
    print(f"Duty cycle: dc={dc}")
    print(f"Output root: {SAVE_ROOT}")
    print("=" * 80)

    for case_counter, (k, st, a_value) in enumerate(CASE_LIST, start=1):
        tag = case_tag(k, st, a_value)
        print("\n" + "#" * 80)
        print(f"[{case_counter}/{total_cases}] Processing {tag}")
        print("#" * 80)

        try:
            rows, status = process_one_case(k, st, a_value)
            all_summary_rows.extend(rows)
            if status == "computed":
                computed_count += 1
            else:
                cached_count += 1
        except Exception as error:
            failed_cases.append([
                fmt_num(k),
                fmt_num(st),
                fmt_num(a_value),
                type(error).__name__,
                str(error),
            ])
            print(f"[FAILED] {tag}: {type(error).__name__}: {error}")
            if STOP_ON_ERROR:
                raise
        finally:
            # Explicitly release the arrays of the completed case before
            # reading the next set of large Tecplot files.
            gc.collect()

    write_summary_csv(ALL_CASES_SUMMARY_CSV, all_summary_rows)

    with open(FAILED_CASES_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["k", "St", "a", "error_type", "error_message"])
        writer.writerows(failed_cases)

    print("\n" + "=" * 80)
    print("Batch processing finished")
    print(f"Successfully computed: {computed_count}")
    print(f"Loaded from cache: {cached_count}")
    print(f"Failed or missing: {len(failed_cases)}")
    print(f"All-case summary: {ALL_CASES_SUMMARY_CSV}")
    print(f"Failure log: {FAILED_CASES_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()