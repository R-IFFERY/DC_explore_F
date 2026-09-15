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
# User parameters
# ============================================================
c = 1.0
a = 0
dc = 1

K_LIST = np.array(
    [0.8],
    dtype=float
)

ST_LIST = np.array(
    [0.6],
    dtype=float
)

# 所有工况均使用第11--15周期。
CYCLE_LIST = np.arange(16, 21, dtype=int)
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
# 这里直接使用计算域的全局无量纲坐标 x/c 和 y/c，
# 不再使用以后翼俯仰轴为原点的 x_H/c。
X_MIN = -1.0 * c
X_MAX = 20.0 * c
Y_MIN = -10.0 * c
Y_MAX = 10.0 * c

# 是否排除机翼内部的固体网格点。
# 默认假定 Tecplot 变量 body=0 表示流体，body>=0.5 表示固体区域。
EXCLUDE_SOLID_BODY = True
BODY_SOLID_THRESHOLD = 0.5

# 扩大窗口后，大量共同的零涡量背景可能人为抬高相关系数。
# True 表示仅使用两个场中至少一个场具有有效涡量的点。
USE_ACTIVE_UNION_MASK = True

# ============================================================
# Correlation settings
# ============================================================
# 先将弱于各自场最大绝对涡量一定比例的点置零，减弱远场数值噪声影响。
RHO_ACTIVITY_FRAC = 0.02
CORR_EPS = 1.0e-14

# 仅在相关性计算时对规则网格做空间抽样；原始流场本身不被修改。
# 0.10c 对全参数扫描更高效；建议另用0.05c做少量工况敏感性检查。
CORR_TARGET_DX = 0.10 * c
CORR_TARGET_DY = 0.10 * c

# 非周期操作性判据：五个周期形成的4组相邻周期比较中，
# 对32个同相位快照及4个周期对平均后，rho低于此值。
# 注意：由 rho_star 改为 rho 且更换空间窗口后，应重新做阈值敏感性检查。
APERIODIC_RHO_THRESHOLD = 0.90

# 至少需要多少比例的理论比较数（4*32=128）才能给出有效工况结果。
MIN_VALID_COMPARISON_FRACTION = 0.80

# 若分类CSV中的 is_aperiodic 已填写，则是否覆盖rho阈值判定。
USE_CSV_APERIODIC_OVERRIDE = True

# ============================================================
# Wake classification settings for panel (b)
# ============================================================
# 尾迹拓扑不应仅由rho自动推断，因此(b)从独立CSV读取。
CLASSIFICATION_CSV = os.path.join(
    SAVE_ROOT,
    "wake_classification.csv"
)

# CSV中建议使用下列英文名称；也可使用自定义名称，程序会自动加入图例。
WAKE_TYPE_ORDER = [
    "Narrow shear-layer wake",
    "Periodic vortex street",
    "Bifurcated wake",
    "Deflected reverse Karman wake",
    "Disordered wake",
    "Unclassified",
]

WAKE_TYPE_COLORS = {
    "Narrow shear-layer wake": "#56B4E9",
    "Periodic vortex street": "#009E73",
    "Bifurcated wake": "#E69F00",
    "Deflected reverse Karman wake": "#CC79A7",
    "Disordered wake": "#D55E00",
    "Unclassified": "#D9D9D9",
}

WAKE_TYPE_DISPLAY_LABELS = {
    "Narrow shear-layer wake": "Narrow wake",
    "Periodic vortex street": "Vortex street",
    "Bifurcated wake": "Bifurcated",
    "Deflected reverse Karman wake": "Deflected wake",
    "Disordered wake": "Disordered",
    "Unclassified": "Unclassified",
}

# ============================================================
# Output paths
# ============================================================
METRICS_NPZ = os.path.join(
    SAVE_ROOT,
    "rho_all_k_st_metrics.npz"
)

SUMMARY_CSV = os.path.join(
    SAVE_ROOT,
    "rho_all_k_st_summary.csv"
)

FIGURE_BASENAME = os.path.join(
    SAVE_ROOT,
    "rho_wake_type_1x2"
)

# ============================================================
# Path utilities
# ============================================================
def fmt_num(value):
    return f"{value:g}"


def unique_keep_order(sequence):
    output = []
    for item in sequence:
        if item not in output:
            output.append(item)
    return output


def case_key(k, st):
    return round(float(k), 8), round(float(st), 8)


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
                os.path.join(common, f"k{ks}st{ss}"),
                os.path.join(common, f"k{ks}st{ss}a{a_value}"),
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
    自动检测32个相位采用哪种保存约定：
      A: 0, 1/32, ..., 31/32
      B: 1/32, 2/32, ..., 1

    对第11--15周期全部目标时刻进行匹配，选择有效文件数更多的约定。
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
# Classification CSV
# ============================================================
def create_classification_template_if_missing():
    if os.path.isfile(CLASSIFICATION_CSV):
        return False

    with open(CLASSIFICATION_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["k", "St", "wake_type", "is_aperiodic"])

        for st in ST_LIST:
            for k in K_LIST:
                writer.writerow([
                    f"{k:g}",
                    f"{st:g}",
                    "Unclassified",
                    "",
                ])

    print(f"Created classification template: {CLASSIFICATION_CSV}")
    print(
        "Fill the wake_type column and rerun to obtain the categorical "
        "wake map in panel (c)."
    )
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

        required = {"k", "St", "wake_type", "is_aperiodic"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"分类文件必须包含列: {sorted(required)}; "
                f"当前列: {reader.fieldnames}"
            )

        for row in reader:
            try:
                key = case_key(float(row["k"]), float(row["St"]))
            except (TypeError, ValueError):
                continue

            wake_type = str(row["wake_type"]).strip() or "Unclassified"
            wake_map[key] = wake_type

            flag = parse_optional_bool(row["is_aperiodic"])
            if flag is not None:
                aperiodic_override[key] = flag

    return wake_map, aperiodic_override

# ============================================================
# Compute all cases
# ============================================================
def compute_metrics():
    n_st = len(ST_LIST)
    n_k = len(K_LIST)
    n_pairs = len(CYCLE_LIST) - 1

    rho_samples = np.full(
        (n_st, n_k, n_pairs, N_PHASES),
        np.nan,
        dtype=float
    )
    matched_file_counts = np.zeros((n_st, n_k), dtype=int)
    valid_comparison_counts = np.zeros((n_st, n_k), dtype=int)
    phase_conventions = np.full((n_st, n_k), "", dtype="U32")

    for st_index, st in enumerate(ST_LIST):
        for k_index, k in enumerate(K_LIST):
            print("=" * 80)
            print(f"Processing k={k:g}, St={st:g}")

            case_dir = find_existing_case_dir(
                BASE_ROOT,
                k,
                st,
                a,
                dc
            )

            if case_dir is None:
                print("[Warning] Case directory not found.")
                continue

            print(f"Case directory: {case_dir}")
            records = list_flow_records(case_dir)

            convention, phases, table, phase_errors = build_phase_record_table(
                records,
                k,
                CYCLE_LIST,
                N_PHASES
            )
            phase_conventions[st_index, k_index] = convention
            matched_file_counts[st_index, k_index] = int(
                np.sum([[record is not None for record in row] for row in table])
            )

            grid_X = None
            grid_Y = None
            expected_shape = None
            diagnostic_printed = False

            # 每个相位一次性读取5个周期；每个文件只读取一次。
            for phase_index, phase in enumerate(phases):
                records_for_phase = table[phase_index]

                if any(record is None for record in records_for_phase):
                    missing_cycles = [
                        int(CYCLE_LIST[i])
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
                    for cycle_index, record in enumerate(records_for_phase):
                        if grid_X is None:
                            data = read_tecplot_ordered_dat(
                                record["path"],
                                target_vars=("X", "Y", "Vorticity", "body")
                            )
                            grid_X = data["X"]
                            grid_Y = data["Y"]
                            expected_shape = data["Vorticity"].shape
                            fields.append(data["Vorticity"])
                            body_fields.append(data["body"])
                        else:
                            data = read_tecplot_ordered_dat(
                                record["path"],
                                target_vars=("Vorticity", "body")
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
                            st_index, k_index, pair_index, phase_index
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
                        + ", ".join(record["filename"] for record in records_for_phase)
                    )

                except Exception as error:
                    print(
                        f"[Error] k={k:g}, St={st:g}, "
                        f"phase={phase:.5f}T: {error}"
                    )

                finally:
                    del fields, body_fields

            valid_comparison_counts[st_index, k_index] = int(
                np.sum(np.isfinite(rho_samples[st_index, k_index]))
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rho_mean = np.nanmean(rho_samples, axis=(2, 3))
        rho_std = np.nanstd(rho_samples, axis=(2, 3))

    theoretical_comparisons = n_pairs * N_PHASES
    valid_fraction = valid_comparison_counts / theoretical_comparisons
    valid_case_mask = valid_fraction >= MIN_VALID_COMPARISON_FRACTION

    # 文件不足的工况保持NaN，不把“缺数据”误判为非周期。
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
    }

    np.savez(
        METRICS_NPZ,
        k_list=K_LIST,
        st_list=ST_LIST,
        cycle_list=CYCLE_LIST,
        n_phases=N_PHASES,
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
    return {key: data[key] for key in data.files}

# ============================================================
# Apply classification and save summary
# ============================================================
def apply_classification(metrics):
    wake_map, aperiodic_override = load_classification_table()

    wake_type_matrix = np.full(
        (len(ST_LIST), len(K_LIST)),
        "Unclassified",
        dtype="U128"
    )

    aperiodic_mask = np.asarray(metrics["aperiodic_mask"], dtype=bool).copy()

    for st_index, st in enumerate(ST_LIST):
        for k_index, k in enumerate(K_LIST):
            key = case_key(k, st)
            wake_type_matrix[st_index, k_index] = wake_map.get(
                key,
                "Unclassified"
            )

            if USE_CSV_APERIODIC_OVERRIDE and key in aperiodic_override:
                aperiodic_mask[st_index, k_index] = aperiodic_override[key]

    metrics["wake_type_matrix"] = wake_type_matrix
    metrics["aperiodic_mask"] = aperiodic_mask
    return metrics


def save_summary_csv(metrics):
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "k",
            "St",
            "rho_mean",
            "rho_std_over_4_pairs_and_32_phases",
            "matched_flow_files",
            "valid_comparisons",
            "valid_comparison_fraction",
            "is_aperiodic",
            "wake_type",
        ])

        for st_index, st in enumerate(ST_LIST):
            for k_index, k in enumerate(K_LIST):
                writer.writerow([
                    f"{k:g}",
                    f"{st:g}",
                    f"{metrics['rho_mean'][st_index, k_index]:.8e}",
                    f"{metrics['rho_std'][st_index, k_index]:.8e}",
                    int(metrics["matched_file_counts"][st_index, k_index]),
                    int(metrics["valid_comparison_counts"][st_index, k_index]),
                    f"{metrics['valid_fraction'][st_index, k_index]:.8e}",
                    int(metrics["aperiodic_mask"][st_index, k_index]),
                    metrics["wake_type_matrix"][st_index, k_index],
                ])

    print(f"Saved summary: {SUMMARY_CSV}")

# ============================================================
# Plotting utilities
# ============================================================
def centers_to_edges(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])

    midpoints = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - 0.5 * (centers[1] - centers[0])
    last = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate([[first], midpoints, [last]])


def add_hatching(ax, mask, k_edges, st_edges):
    for st_index in range(mask.shape[0]):
        for k_index in range(mask.shape[1]):
            if not mask[st_index, k_index]:
                continue

            rectangle = Rectangle(
                (k_edges[k_index], st_edges[st_index]),
                k_edges[k_index + 1] - k_edges[k_index],
                st_edges[st_index + 1] - st_edges[st_index],
                facecolor="none",
                edgecolor="0.45",
                linewidth=0.0,
                hatch="////",
                zorder=5,
            )
            ax.add_patch(rectangle)


def mark_missing_cells(ax, valid_case_mask, k_values, st_values):
    missing = ~np.asarray(valid_case_mask, dtype=bool)

    for st_index, st in enumerate(st_values):
        for k_index, k in enumerate(k_values):
            if missing[st_index, k_index]:
                ax.text(
                    k,
                    st,
                    r"$\times$",
                    ha="center",
                    va="center",
                    color="0.35",
                    fontsize=8,
                    zorder=8,
                )


def ordered_wake_types(wake_type_matrix):
    present = unique_keep_order(wake_type_matrix.ravel().tolist())
    ordered = [item for item in WAKE_TYPE_ORDER if item in present]
    ordered.extend(item for item in present if item not in ordered)
    return ordered


def build_wake_colormap(wake_types):
    fallback = plt.get_cmap("tab20")
    colors = []

    for index, wake_type in enumerate(wake_types):
        colors.append(
            WAKE_TYPE_COLORS.get(
                wake_type,
                fallback(index % fallback.N)
            )
        )

    return ListedColormap(colors)

# ============================================================
# Plot 1 x 2 figure
# ============================================================
def plot_figure(metrics):
    k_edges = centers_to_edges(K_LIST)
    st_edges = centers_to_edges(ST_LIST)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.5, 2.95),
        dpi=300,
        sharex=True,
        sharey=True,
    )

    correlation_cmap = plt.get_cmap("RdBu_r").copy()
    correlation_cmap.set_bad("#F2F2F2")
    correlation_norm = Normalize(vmin=-1.0, vmax=1.0)

    mesh_a = axes[0].pcolormesh(
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

    add_hatching(
        axes[0],
        metrics["aperiodic_mask"],
        k_edges,
        st_edges
    )

    mark_missing_cells(
        axes[0],
        metrics["valid_case_mask"],
        K_LIST,
        ST_LIST
    )

    wake_types = ordered_wake_types(metrics["wake_type_matrix"])
    wake_to_index = {name: index for index, name in enumerate(wake_types)}

    wake_numeric = np.zeros(
        metrics["wake_type_matrix"].shape,
        dtype=int
    )

    for st_index in range(len(ST_LIST)):
        for k_index in range(len(K_LIST)):
            wake_numeric[st_index, k_index] = wake_to_index[
                metrics["wake_type_matrix"][st_index, k_index]
            ]

    wake_cmap = build_wake_colormap(wake_types)
    wake_norm = BoundaryNorm(
        np.arange(len(wake_types) + 1) - 0.5,
        len(wake_types)
    )

    axes[1].pcolormesh(
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

    panel_titles = [
        r"$\rho_{\omega}$",
        r"Wake type",
    ]

    for index, (ax, title) in enumerate(zip(axes, panel_titles)):
        ax.set_title(title, pad=4.0)
        ax.set_xlabel(r"$k$")
        ax.set_xticks(K_LIST)
        ax.set_yticks(ST_LIST)
        ax.tick_params(top=True, right=True)

        ax.text(
            -0.18,
            1.04,
            rf"$({chr(97 + index)})$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            clip_on=False,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(0.7)

    axes[0].set_ylabel(r"$St$")

    # (a) 相关系数色条。
    cbar_ax = fig.add_axes([0.090, 0.095, 0.360, 0.030])
    colorbar = fig.colorbar(
        mesh_a,
        cax=cbar_ax,
        orientation="horizontal",
        ticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
    )
    colorbar.set_label(r"$\rho_{\omega}$", labelpad=1.5)
    colorbar.ax.tick_params(length=2.5, width=0.6, pad=1.5)
    colorbar.outline.set_linewidth(0.6)

    hatch_handle = Patch(
        facecolor="white",
        edgecolor="0.45",
        hatch="////",
        linewidth=0.0,
        label="aperiodic",
    )
    axes[0].legend(
        handles=[hatch_handle],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        handlelength=1.4,
        handleheight=0.9,
        borderaxespad=0.0,
    )

    wake_handles = [
        Patch(
            facecolor=wake_cmap(index),
            edgecolor="none",
            label=WAKE_TYPE_DISPLAY_LABELS.get(wake_type, wake_type),
        )
        for index, wake_type in enumerate(wake_types)
    ]

    axes[1].legend(
        handles=wake_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2 if len(wake_handles) > 3 else 1,
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.35,
        borderaxespad=0.0,
        fontsize=8.2,
    )

    fig.subplots_adjust(
        left=0.082,
        right=0.995,
        bottom=0.355,
        top=0.885,
        wspace=0.23,
    )

    for text in fig.findobj(match=plt.Text):
        if text.get_fontsize() > FONT_SIZE:
            text.set_fontsize(FONT_SIZE)

    png_path = FIGURE_BASENAME + ".png"
    svg_path = FIGURE_BASENAME + ".svg"
    pdf_path = FIGURE_BASENAME + ".pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")
    print(f"Saved PDF: {pdf_path}")

# ============================================================
# Main
# ============================================================
def main():
    if FORCE_RECOMPUTE or not os.path.isfile(METRICS_NPZ):
        metrics = compute_metrics()
    else:
        print(f"Loading existing metrics: {METRICS_NPZ}")
        metrics = load_metrics()

    metrics = apply_classification(metrics)
    save_summary_csv(metrics)
    # plot_figure(metrics)


if __name__ == "__main__":
    main()