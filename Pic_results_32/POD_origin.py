import csv
import os
import re
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch, Rectangle

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
    "hatch.linewidth": 0.45,
})

if USE_TEX:
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.latex.preamble":
            r"\usepackage{amsmath}\usepackage{newtxtext,newtxmath}"
    })

# ============================================================

# ============================================================
# User parameters
# ============================================================
c = 1.0
dc = 1

# 显式指定需要处理的12个非零后翼俯仰角工况。
# 每个元素依次为 (k, St, a)。
CASE_LIST = [
    (0.8, 0.3, 19),
    (0.8, 0.3, 20),
    (0.8, 0.3, 21),
    (1.1, 0.3, 19),
    (1.1, 0.3, 20),
    (1.1, 0.3, 21),
    (1.4, 0.3, 16),
    (1.4, 0.3, 17),
    (1.4, 0.3, 18),
    (1.7, 0.3, 10),
    (1.7, 0.3, 11),
    (1.7, 0.3, 12),
]

CASE_K = np.asarray([case[0] for case in CASE_LIST], dtype=float)
CASE_ST = np.asarray([case[1] for case in CASE_LIST], dtype=float)
CASE_A = np.asarray([case[2] for case in CASE_LIST], dtype=float)

# 不同 k 使用不同的目标周期。
# 由于本代码计算的是相邻周期同相位流场的相关系数，因此：
#   k=0.8、1.1：以第15周期为目标，比较第14--15周期；
#   k=1.4、1.7：以第20周期为目标，比较第19--20周期。
TARGET_CYCLE_BY_K = {
    0.8: 15,
    1.1: 15,
    1.4: 20,
    1.7: 20,
}


def get_cycle_list_for_k(k):
    """返回指定 k 对应的两个相邻周期 [目标周期-1, 目标周期]。"""
    for k_key, target_cycle in TARGET_CYCLE_BY_K.items():
        if np.isclose(float(k), float(k_key)):
            return np.array(
                [target_cycle - 1, target_cycle],
                dtype=int,
            )

    raise ValueError(
        f"未给 k={k:g} 设置目标周期；"
        f"当前可用设置为 {TARGET_CYCLE_BY_K}"
    )


CASE_CYCLE_LISTS = np.asarray(
    [get_cycle_list_for_k(k) for k, _, _ in CASE_LIST],
    dtype=int,
)

N_PHASES = 32

# 文件名 F247541.dat 表示 t=24.7541。
FILENAME_TIME_SCALE = 10000.0

# 相位匹配容差，以周期 T 为单位。
PHASE_TOL = 0.020

BASE_ROOT = r"E:/Rensq/DC_explore_F/data_1Tflow/"
SAVE_ROOT = (
    r"E:/Rensq/DC_explore_F/flow/"
    r"figure_rho_aneq0"
)
os.makedirs(SAVE_ROOT, exist_ok=True)

# 已有 NPZ 时是否强制重新计算。
FORCE_RECOMPUTE = False

# ============================================================
# Coordinate and analysis window
# ============================================================
# 使用计算域的全局无量纲坐标 x/c 和 y/c。
X_MIN = -1.0 * c
X_MAX = 20.0 * c
Y_MIN = -10.0 * c
Y_MAX = 10.0 * c

# 是否排除机翼内部的固体网格点。
# 默认假定 Tecplot 变量 body=0 表示流体，body>=0.5 表示固体区域。
EXCLUDE_SOLID_BODY = True
BODY_SOLID_THRESHOLD = 0.5

# True 表示仅使用两个场中至少一个场具有有效涡量的点，
# 避免大范围共同零背景人为抬高相关系数。
USE_ACTIVE_UNION_MASK = True

# ============================================================
# Correlation settings
# ============================================================
# 弱于各自场最大绝对涡量一定比例的点置零。
RHO_ACTIVITY_FRAC = 0.02
CORR_EPS = 1.0e-14

# 仅在相关性计算时对规则网格做空间抽样。
CORR_TARGET_DX = 0.10 * c
CORR_TARGET_DY = 0.10 * c

# 非周期操作性判据。
APERIODIC_RHO_THRESHOLD = 0.90

# 至少需要多少比例的理论比较数才能给出有效工况结果。
MIN_VALID_COMPARISON_FRACTION = 0.80

# 若分类CSV中的 is_aperiodic 已填写，是否覆盖rho阈值判定。
USE_CSV_APERIODIC_OVERRIDE = True

# ============================================================
# Optional wake classification
# ============================================================
CLASSIFICATION_CSV = os.path.join(
    SAVE_ROOT,
    "wake_classification_aneq0.csv"
)

# ============================================================
# Output paths
# ============================================================
METRICS_NPZ = os.path.join(
    SAVE_ROOT,
    "rho_selected_k_st_a_metrics.npz"
)

SUMMARY_CSV = os.path.join(
    SAVE_ROOT,
    "rho_selected_k_st_a_summary.csv"
)

FIGURE_BASENAME = os.path.join(
    SAVE_ROOT,
    "rho_selected_k_st_a"
)

# ============================================================
# Path utilities
# ============================================================
def fmt_num(value):
    return f"{float(value):g}"


def unique_keep_order(sequence):
    output = []
    for item in sequence:
        if item not in output:
            output.append(item)
    return output


def case_key(k, st, a_value):
    return (
        round(float(k), 8),
        round(float(st), 8),
        round(float(a_value), 8),
    )


def get_case_dir_candidates(base_root, k, st, a_value, dc_value):
    """
    生成可能的流场目录。

    例如 k=0.8, St=0.3, a=20 时会优先检查：
    E:/Rensq/DC_explore_F/data_1Tflow/
      k0.8/st0.3k0.8/st0.3k0.8a20/
      st0.3k0.8a20dc1/k0.8st0.3a20
    """
    k_names = unique_keep_order([str(k), fmt_num(k)])
    st_names = unique_keep_order([str(st), fmt_num(st)])
    a_names = unique_keep_order([str(a_value), fmt_num(a_value)])

    candidates = []

    for ks in k_names:
        for ss in st_names:
            for aa in a_names:
                common = os.path.join(
                    base_root,
                    f"k{ks}",
                    f"st{ss}k{ks}",
                    f"st{ss}k{ks}a{aa}",
                    f"st{ss}k{ks}a{aa}dc{int(dc_value)}",
                )

                candidates.extend([
                    # 用户给出的非零俯仰角目录形式。
                    os.path.join(common, f"k{ks}st{ss}a{aa}"),
                    # 保留旧目录形式作为兼容候选。
                    os.path.join(common, f"k{ks}st{ss}"),
                ])

    return unique_keep_order(candidates)


def find_existing_case_dir(base_root, k, st, a_value, dc_value):
    candidates = get_case_dir_candidates(
        base_root, k, st, a_value, dc_value
    )

    for directory in candidates:
        if os.path.isdir(directory):
            return directory

    print("  checked directory candidates:")
    for directory in candidates:
        print(f"    {directory}")

    return None

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
    自动检测32个相位采用哪种保存约定：
      A: 0, 1/32, ..., 31/32
      B: 1/32, 2/32, ..., 1

    对传入的全部周期目标时刻进行匹配，选择有效文件数更多的约定。
    返回 table[phase_index][cycle_index]。
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

    selected_name = max(
        candidate_results,
        key=lambda key: candidate_results[key]["valid_count"]
    )
    selected = candidate_results[selected_name]

    total_targets = len(cycles) * n_phases
    max_error = np.nanmax(selected["errors"]) \
        if np.any(np.isfinite(selected["errors"])) else np.nan

    print(
        f"  phase convention: {selected_name}; "
        f"matched files={selected['valid_count']}/{total_targets}; "
        f"max phase error={max_error:.5f}T"
    )

    return selected_name, selected["phases"], selected["table"], selected["errors"]

# ============================================================
# Tecplot ordered ASCII reader
# ============================================================
VAR_MAP = {
    "X": 0,
    "Y": 1,
    "U": 2,
    "V": 3,
    "Presure": 4,
    "Vorticity": 5,
    "body": 6,
    "dvordt": 7,
}


def inspect_tecplot_header(filename, required_columns):
    """流式读取文件头，避免把整个大型ASCII文件一次性读入字符串列表。"""
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
        raise ValueError(f"无法解析Tecplot文件头: {filename}")

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
            f"{filename} 数据点不足：期望 {i_size * j_size}，"
            f"实际 {data.shape[0]}"
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
# Correlation calculation
# ============================================================
def choose_stride(coordinate_line, target_spacing):
    differences = np.diff(np.asarray(coordinate_line, dtype=float))
    differences = differences[np.isfinite(differences) & (differences > 0.0)]

    if differences.size == 0:
        raise ValueError("无法确定有效网格间距。")

    spacing = float(np.median(differences))
    stride = max(1, int(round(target_spacing / spacing)))
    return stride, spacing


def suppress_weak_vorticity(field, activity_fraction):
    field = np.nan_to_num(
        np.asarray(field, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    local_max = float(np.max(np.abs(field)))
    if not np.isfinite(local_max) or local_max <= CORR_EPS:
        return np.zeros_like(field)

    threshold = activity_fraction * local_max
    return np.where(np.abs(field) >= threshold, field, 0.0)


def pearson_correlation(reference, candidate, valid_mask=None):
    """
    计算两个同尺寸二维场的 Pearson 空间相关系数。

    valid_mask:
        True 的网格点参与计算；可用于排除固体点、无效点以及共同零背景。
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)

    if reference.shape != candidate.shape:
        raise ValueError("两个相关场的尺寸不一致。")

    if valid_mask is None:
        valid_mask = np.ones(reference.shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    valid_mask = (
        valid_mask
        & np.isfinite(reference)
        & np.isfinite(candidate)
    )

    if np.count_nonzero(valid_mask) < 3:
        return np.nan

    ref = reference[valid_mask]
    cand = candidate[valid_mask]

    ref_centered = ref - np.mean(ref)
    cand_centered = cand - np.mean(cand)

    denominator = np.sqrt(
        np.sum(ref_centered ** 2)
        * np.sum(cand_centered ** 2)
    )

    if not np.isfinite(denominator) or denominator <= CORR_EPS:
        return np.nan

    rho = np.sum(ref_centered * cand_centered) / denominator
    return float(np.clip(rho, -1.0, 1.0))


def compute_global_rho(X, Y, W1, W2, B1=None, B2=None):
    """
    在 -1 <= x/c <= 20、-10 <= y/c <= 10 的统一窗口内，
    计算两个相邻周期同相位涡量场的直接空间相关系数 rho。

    不再进行任何流向平移，也不再计算 rho_star。

    为提高全参数扫描效率，仅在相关性计算阶段按
    CORR_TARGET_DX 和 CORR_TARGET_DY 对规则网格抽样。
    """
    if W1.shape != W2.shape or X.shape != W1.shape or Y.shape != W1.shape:
        raise ValueError("两个周期流场或坐标网格尺寸不一致。")

    if B1 is not None and B1.shape != W1.shape:
        raise ValueError("第一个 body 场与涡量场尺寸不一致。")
    if B2 is not None and B2.shape != W2.shape:
        raise ValueError("第二个 body 场与涡量场尺寸不一致。")

    x_line = np.asarray(X[0, :], dtype=float)
    y_line = np.asarray(Y[:, 0], dtype=float)

    stride_x, raw_dx = choose_stride(x_line, CORR_TARGET_DX)
    stride_y, raw_dy = choose_stride(y_line, CORR_TARGET_DY)

    x_indices = np.arange(0, W1.shape[1], stride_x, dtype=int)
    y_indices = np.arange(0, W1.shape[0], stride_y, dtype=int)

    x_ds = x_line[x_indices]
    y_ds = y_line[y_indices]

    x_keep = np.where((x_ds >= X_MIN) & (x_ds <= X_MAX))[0]
    y_keep = np.where((y_ds >= Y_MIN) & (y_ds <= Y_MAX))[0]

    basic_diagnostics = {
        "stride_x": stride_x,
        "stride_y": stride_y,
        "raw_dx": raw_dx,
        "raw_dy": raw_dy,
        "effective_dx": stride_x * raw_dx,
        "effective_dy": stride_y * raw_dy,
        "n_window_points": 0,
        "n_used_points": 0,
    }

    if x_keep.size < 3 or y_keep.size < 3:
        return np.nan, basic_diagnostics

    selected_x_indices = x_indices[x_keep]
    selected_y_indices = y_indices[y_keep]

    W1_window = W1[np.ix_(selected_y_indices, selected_x_indices)]
    W2_window = W2[np.ix_(selected_y_indices, selected_x_indices)]

    finite_mask = np.isfinite(W1_window) & np.isfinite(W2_window)
    fluid_mask = np.ones(W1_window.shape, dtype=bool)

    if EXCLUDE_SOLID_BODY and B1 is not None and B2 is not None:
        B1_window = B1[np.ix_(selected_y_indices, selected_x_indices)]
        B2_window = B2[np.ix_(selected_y_indices, selected_x_indices)]

        fluid_mask = (
            np.isfinite(B1_window)
            & np.isfinite(B2_window)
            & (B1_window < BODY_SOLID_THRESHOLD)
            & (B2_window < BODY_SOLID_THRESHOLD)
        )

    # 固体点先置零，防止固体内部数值参与最大涡量阈值的确定。
    W1_for_filter = np.where(fluid_mask & finite_mask, W1_window, 0.0)
    W2_for_filter = np.where(fluid_mask & finite_mask, W2_window, 0.0)

    W1_active = suppress_weak_vorticity(
        W1_for_filter,
        RHO_ACTIVITY_FRAC
    )
    W2_active = suppress_weak_vorticity(
        W2_for_filter,
        RHO_ACTIVITY_FRAC
    )

    valid_mask = fluid_mask & finite_mask

    if USE_ACTIVE_UNION_MASK:
        valid_mask &= (
            (np.abs(W1_active) > CORR_EPS)
            | (np.abs(W2_active) > CORR_EPS)
        )

    rho = pearson_correlation(
        W1_active,
        W2_active,
        valid_mask=valid_mask
    )

    diagnostics = {
        **basic_diagnostics,
        "n_window_points": int(W1_window.size),
        "n_used_points": int(np.count_nonzero(valid_mask)),
    }

    return rho, diagnostics

# ============================================================

# ============================================================
# Classification CSV
# ============================================================
def create_classification_template_if_missing():
    """
    创建可选的尾迹分类模板。

    相同的 k 和 St 对应多个 a，因此分类键必须同时包含 k、St 和 a。
    """
    if os.path.isfile(CLASSIFICATION_CSV):
        return False

    with open(CLASSIFICATION_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["k", "St", "a", "wake_type", "is_aperiodic"])

        for k, st, a_value in CASE_LIST:
            writer.writerow([
                f"{k:g}",
                f"{st:g}",
                f"{a_value:g}",
                "Unclassified",
                "",
            ])

    print(f"Created classification template: {CLASSIFICATION_CSV}")
    return True


def parse_optional_bool(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "aperiodic", "nonperiodic"}:
        return True
    if text in {"0", "false", "no", "n", "periodic"}:
        return False
    return None


def load_classification_table():
    create_classification_template_if_missing()

    wake_map = {}
    aperiodic_override = {}

    with open(CLASSIFICATION_CSV, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required = {"k", "St", "a", "wake_type", "is_aperiodic"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"分类文件必须包含列: {sorted(required)}; "
                f"当前列: {reader.fieldnames}"
            )

        for row in reader:
            try:
                key = case_key(
                    float(row["k"]),
                    float(row["St"]),
                    float(row["a"]),
                )
            except (TypeError, ValueError):
                continue

            wake_type = str(row["wake_type"]).strip() or "Unclassified"
            wake_map[key] = wake_type

            flag = parse_optional_bool(row["is_aperiodic"])
            if flag is not None:
                aperiodic_override[key] = flag

    return wake_map, aperiodic_override


# ============================================================
# Compute selected (k, St, a) cases
# ============================================================
def compute_metrics():
    n_cases = len(CASE_LIST)

    # 每个工况均比较两个相邻周期，因此只有一个周期对。
    n_cycles_per_case = CASE_CYCLE_LISTS.shape[1]
    n_pairs = n_cycles_per_case - 1

    # 第一维现在是显式工况编号，而不是原来的 St x k 网格。
    rho_samples = np.full(
        (n_cases, n_pairs, N_PHASES),
        np.nan,
        dtype=float,
    )
    matched_file_counts = np.zeros(n_cases, dtype=int)
    valid_comparison_counts = np.zeros(n_cases, dtype=int)
    phase_conventions = np.full(n_cases, "", dtype="U32")
    case_directories = np.full(n_cases, "", dtype="U1024")

    for case_index, (k, st, a_value) in enumerate(CASE_LIST):
        print("=" * 80)
        print(
            f"Processing case {case_index + 1}/{n_cases}: "
            f"k={k:g}, St={st:g}, a={a_value:g}"
        )

        case_dir = find_existing_case_dir(
            BASE_ROOT,
            k,
            st,
            a_value,
            dc,
        )

        if case_dir is None:
            print("[Warning] Case directory not found.")
            continue

        case_directories[case_index] = case_dir
        print(f"Case directory: {case_dir}")

        records = list_flow_records(case_dir)
        if not records:
            print("[Warning] No F*.dat files found in this directory.")
            continue

        case_cycles = CASE_CYCLE_LISTS[case_index]

        print(
            "Selected cycle pair: "
            f"{int(case_cycles[0])}--{int(case_cycles[1])}"
        )

        convention, phases, table, phase_errors = build_phase_record_table(
            records,
            k,
            case_cycles,
            N_PHASES,
        )
        phase_conventions[case_index] = convention
        matched_file_counts[case_index] = int(
            np.sum([
                [record is not None for record in row]
                for row in table
            ])
        )

        grid_X = None
        grid_Y = None
        expected_shape = None
        diagnostic_printed = False

        # 每个相位一次性读取当前工况的两个相邻周期，
        # 每个文件只读取一次。
        for phase_index, phase in enumerate(phases):
            records_for_phase = table[phase_index]

            if any(record is None for record in records_for_phase):
                missing_cycles = [
                    int(case_cycles[i])
                    for i, record in enumerate(records_for_phase)
                    if record is None
                ]
                print(
                    f"  phase={phase:.5f}T skipped; "
                    f"missing cycles={missing_cycles}"
                )
                continue

            fields = []
            body_fields = []

            try:
                for record in records_for_phase:
                    if grid_X is None:
                        data = read_tecplot_ordered_dat(
                            record["path"],
                            target_vars=("X", "Y", "Vorticity", "body"),
                        )
                        grid_X = data["X"]
                        grid_Y = data["Y"]
                        expected_shape = data["Vorticity"].shape
                    else:
                        data = read_tecplot_ordered_dat(
                            record["path"],
                            target_vars=("Vorticity", "body"),
                        )

                    field = data["Vorticity"]
                    body_field = data["body"]

                    if field.shape != expected_shape:
                        raise ValueError(
                            f"网格尺寸变化: {field.shape} != {expected_shape}"
                        )
                    if body_field.shape != expected_shape:
                        raise ValueError(
                            f"body网格尺寸变化: "
                            f"{body_field.shape} != {expected_shape}"
                        )

                    fields.append(field)
                    body_fields.append(body_field)

                for pair_index in range(n_pairs):
                    rho, diagnostics = compute_global_rho(
                        grid_X,
                        grid_Y,
                        fields[pair_index],
                        fields[pair_index + 1],
                        body_fields[pair_index],
                        body_fields[pair_index + 1],
                    )

                    rho_samples[
                        case_index, pair_index, phase_index
                    ] = rho

                    if not diagnostic_printed:
                        print(
                            "  correlation sampling: "
                            f"stride=({diagnostics.get('stride_x')},"
                            f"{diagnostics.get('stride_y')}), "
                            f"effective spacing="
                            f"({diagnostics.get('effective_dx', np.nan):.4f}c,"
                            f"{diagnostics.get('effective_dy', np.nan):.4f}c), "
                            f"window points="
                            f"{diagnostics.get('n_window_points', 0)}, "
                            f"used points="
                            f"{diagnostics.get('n_used_points', 0)}"
                        )
                        diagnostic_printed = True

                print(
                    f"  phase={phase:.5f}T complete: "
                    + ", ".join(
                        record["filename"] for record in records_for_phase
                    )
                )

            except Exception as error:
                print(
                    f"[Error] k={k:g}, St={st:g}, a={a_value:g}, "
                    f"phase={phase:.5f}T: {error}"
                )

            finally:
                del fields, body_fields

        valid_comparison_counts[case_index] = int(
            np.sum(np.isfinite(rho_samples[case_index]))
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rho_mean = np.nanmean(rho_samples, axis=(1, 2))
        rho_std = np.nanstd(rho_samples, axis=(1, 2))

    theoretical_comparisons = n_pairs * N_PHASES
    valid_fraction = (
        valid_comparison_counts.astype(float)
        / float(theoretical_comparisons)
    )
    valid_case_mask = (
        valid_fraction >= MIN_VALID_COMPARISON_FRACTION
    )

    # 文件不足的工况保持 NaN，不把“缺数据”误判为非周期。
    rho_mean = np.where(valid_case_mask, rho_mean, np.nan)
    rho_std = np.where(valid_case_mask, rho_std, np.nan)

    aperiodic_mask = (
        valid_case_mask
        & np.isfinite(rho_mean)
        & (rho_mean < APERIODIC_RHO_THRESHOLD)
    )

    metrics = {
        "rho_samples": rho_samples,
        "rho_mean": rho_mean,
        "rho_std": rho_std,
        "matched_file_counts": matched_file_counts,
        "valid_comparison_counts": valid_comparison_counts,
        "valid_fraction": valid_fraction,
        "valid_case_mask": valid_case_mask,
        "aperiodic_mask": aperiodic_mask,
        "phase_conventions": phase_conventions,
        "case_directories": case_directories,
    }

    np.savez(
        METRICS_NPZ,
        case_k=CASE_K,
        case_st=CASE_ST,
        case_a=CASE_A,
        case_cycle_lists=CASE_CYCLE_LISTS,
        target_cycle_by_case=CASE_CYCLE_LISTS[:, 1],
        n_phases=N_PHASES,
        dc=dc,
        x_range=np.array([X_MIN, X_MAX], dtype=float),
        y_range=np.array([Y_MIN, Y_MAX], dtype=float),
        exclude_solid_body=EXCLUDE_SOLID_BODY,
        body_solid_threshold=BODY_SOLID_THRESHOLD,
        use_active_union_mask=USE_ACTIVE_UNION_MASK,
        activity_fraction=RHO_ACTIVITY_FRAC,
        corr_target_dx=CORR_TARGET_DX,
        corr_target_dy=CORR_TARGET_DY,
        aperiodic_threshold=APERIODIC_RHO_THRESHOLD,
        min_valid_fraction=MIN_VALID_COMPARISON_FRACTION,
        **metrics,
    )

    print("=" * 80)
    print(f"Saved metrics: {METRICS_NPZ}")
    return metrics


def load_metrics():
    data = np.load(METRICS_NPZ, allow_pickle=False)
    metrics = {key: data[key] for key in data.files}

    required = {
        "case_k",
        "case_st",
        "case_a",
        "case_cycle_lists",
    }
    if not required.issubset(metrics):
        raise ValueError(
            "Existing NPZ does not contain the current case-specific "
            "cycle settings."
        )

    same_cases = (
        np.array_equal(np.asarray(metrics["case_k"], dtype=float), CASE_K)
        and np.array_equal(np.asarray(metrics["case_st"], dtype=float), CASE_ST)
        and np.array_equal(np.asarray(metrics["case_a"], dtype=float), CASE_A)
    )

    same_cycles = np.array_equal(
        np.asarray(metrics["case_cycle_lists"], dtype=int),
        CASE_CYCLE_LISTS,
    )

    if not same_cases:
        raise ValueError(
            "Existing NPZ case list differs from the current CASE_LIST."
        )

    if not same_cycles:
        raise ValueError(
            "Existing NPZ cycle settings differ from the current "
            "case-specific cycle settings."
        )

    return metrics


# ============================================================
# Apply classification and save summary
# ============================================================
def apply_classification(metrics):
    wake_map, aperiodic_override = load_classification_table()

    wake_type = np.full(
        len(CASE_LIST),
        "Unclassified",
        dtype="U128",
    )
    aperiodic_mask = np.asarray(
        metrics["aperiodic_mask"],
        dtype=bool,
    ).copy()

    for case_index, (k, st, a_value) in enumerate(CASE_LIST):
        key = case_key(k, st, a_value)
        wake_type[case_index] = wake_map.get(
            key,
            "Unclassified",
        )

        if USE_CSV_APERIODIC_OVERRIDE and key in aperiodic_override:
            aperiodic_mask[case_index] = aperiodic_override[key]

    metrics["wake_type"] = wake_type
    metrics["aperiodic_mask"] = aperiodic_mask
    return metrics


def save_summary_csv(metrics):
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "case_index",
            "k",
            "St",
            "a",
            "cycle_pair",
            "target_cycle",
            "rho_mean",
            "rho_std_over_adjacent_pairs_and_32_phases",
            "matched_flow_files",
            "valid_comparisons",
            "valid_comparison_fraction",
            "is_valid_case",
            "is_aperiodic",
            "phase_convention",
            "wake_type",
            "case_directory",
        ])

        for case_index, (k, st, a_value) in enumerate(CASE_LIST):
            rho_mean = metrics["rho_mean"][case_index]
            rho_std = metrics["rho_std"][case_index]

            writer.writerow([
                case_index + 1,
                f"{k:g}",
                f"{st:g}",
                f"{a_value:g}",
                (
                    f"{int(CASE_CYCLE_LISTS[case_index, 0])}-"
                    f"{int(CASE_CYCLE_LISTS[case_index, 1])}"
                ),
                int(CASE_CYCLE_LISTS[case_index, 1]),
                "" if not np.isfinite(rho_mean)
                else f"{rho_mean:.8e}",
                "" if not np.isfinite(rho_std)
                else f"{rho_std:.8e}",
                int(metrics["matched_file_counts"][case_index]),
                int(metrics["valid_comparison_counts"][case_index]),
                f"{metrics['valid_fraction'][case_index]:.8e}",
                int(metrics["valid_case_mask"][case_index]),
                int(metrics["aperiodic_mask"][case_index]),
                metrics["phase_conventions"][case_index],
                metrics["wake_type"][case_index],
                metrics["case_directories"][case_index],
            ])

    print(f"Saved summary: {SUMMARY_CSV}")


# ============================================================
# Optional plot: rho versus hindwing pitching angle
# ============================================================
def plot_figure(metrics):
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(4.3, 3.0),
        dpi=300,
    )

    unique_k = unique_keep_order(CASE_K.tolist())

    for k in unique_k:
        indices = np.where(np.isclose(CASE_K, k))[0]
        order = np.argsort(CASE_A[indices])
        indices = indices[order]

        valid = (
            np.asarray(metrics["valid_case_mask"][indices], dtype=bool)
            & np.isfinite(metrics["rho_mean"][indices])
        )

        if not np.any(valid):
            continue

        selected = indices[valid]
        ax.errorbar(
            CASE_A[selected],
            metrics["rho_mean"][selected],
            yerr=metrics["rho_std"][selected],
            marker="o",
            linestyle="-",
            linewidth=1.0,
            markersize=4.0,
            capsize=2.0,
            label=rf"$k={k:g}$",
        )

    ax.axhline(
        APERIODIC_RHO_THRESHOLD,
        color="0.4",
        linestyle="--",
        linewidth=0.8,
        label=rf"$\rho_\omega={APERIODIC_RHO_THRESHOLD:g}$",
    )

    ax.set_xlabel(r"$a\ (^\circ)$")
    ax.set_ylabel(r"$\rho_{\omega}$")
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.tick_params(top=True, right=True)
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()

    for extension in ("png", "svg", "pdf"):
        output_path = f"{FIGURE_BASENAME}.{extension}"
        if extension == "png":
            fig.savefig(output_path, dpi=600, bbox_inches="tight")
        else:
            fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved {extension.upper()}: {output_path}")

    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    if FORCE_RECOMPUTE or not os.path.isfile(METRICS_NPZ):
        metrics = compute_metrics()
    else:
        try:
            print(f"Loading existing metrics: {METRICS_NPZ}")
            metrics = load_metrics()
        except (ValueError, KeyError) as error:
            print(f"[Warning] {error}")
            print(
                "Recomputing metrics with the current CASE_LIST "
                "and cycle settings."
            )
            metrics = compute_metrics()

    metrics = apply_classification(metrics)
    save_summary_csv(metrics)

    # 需要绘制 rho-a 关系图时取消下一行注释。
    # plot_figure(metrics)


if __name__ == "__main__":
    main()