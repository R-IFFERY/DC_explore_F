# -*- coding: utf-8 -*-
"""
plot_fig_pod_st03_modes4_coeff_phase.py

绘制 St=0.3、k=0.8/1.1/1.4/1.7 的 POD 代表性结果。

每个 k 工况构成一个主面板，共标注为 (a)–(d)：
    第一行：Mode 1–Mode 4 的能量加权模态涡量；
    第二行：
        (i)   一个周期内 a1–a4 随归一化相位 t/T 的变化；
        (ii)  a1–a2 相图；
        (iii) a3–a4 相图；
        (iv)  a1–a3 相图。

每个主面板内部的八个小图在右上角依次标注 (i)–(viii)。

说明：
1. 脚本只读取已有 *_pod_results.npz，不重新计算 POD。
2. 若 snapshot_labels.csv 存在，默认选取最后一个完整周期；
   可通过 TARGET_CYCLE_NO 指定周期编号。
3. 若标签文件不存在，则默认把最后 DEFAULT_PHASES_PER_CYCLE 个快照
   作为一个周期绘制。
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
from matplotlib.transforms import Bbox


# ============================================================
# 1. 用户配置区
# ============================================================

SAVE_ROOT = r"E:/Rensq/DC_explore_F/POD_results/POD_results_3_1_1/"
FIG_SAVE_DIR = r"E:/Rensq/DC_explore_F/POD_results/paper_figures/"
os.makedirs(FIG_SAVE_DIR, exist_ok=True)

# 仅绘制 St=0.3，k=0.8、1.1、1.4、1.7。
REPRESENTATIVE_CASES = [
    {"k": 0.8, "st": 0.3, "a": 0, "dc": 1},
    {"k": 1.1, "st": 0.3, "a": 0, "dc": 1},
    {"k": 1.4, "st": 0.3, "a": 0, "dc": 1},
    {"k": 1.7, "st": 0.3, "a": 0, "dc": 1},
]

N_MODES = 4

# 模态涡量加权方式："sqrt_lambda"、"sqrt_energy" 或 "none"。
MODAL_VORTICITY_WEIGHT = "sqrt_lambda"
FALLBACK_TO_SQRT_ENERGY = True

# 流场显示范围。
X_LIM = (-1.0, 15.0)
Y_LIM = (-3.0, 3.0)
X_TICKS = [0, 5, 10, 15]
Y_TICKS = [-3, 0, 3]

# 模态涡量归一化方式："global"、"row" 或 "mode"。
# global：四个 k 工况、前四阶模态共用同一色标，便于比较幅值。
NORMALISE_BY = "global"
NORMALISE_PERCENTILE = 99.0

CMAP_VMIN, CMAP_VMAX = -1.0, 1.0
WHITE_SIGMA = 0.16
SHOW_MASK_CONTOUR = False

# 单周期选择：
# None 表示自动选取标签文件中的最后一个周期。
# 若指定整数，例如 15，则优先绘制第 15 周期。
TARGET_CYCLE_NO = None
DEFAULT_PHASES_PER_CYCLE = 32
CLOSE_PHASE_PORTRAIT = True

# 单周期系数曲线设置。
NORMALISE_TIME_COEFFS = False
COEFF_LINE_WIDTH = 0.9
COEFF_MARKER_SIZE = 2.0
COEFF_SHOW_MARKERS = False
COEFF_Y_LIM = (-150.0, 150.0)
COEFF_Y_TICKS = [-150.0, 0.0, 150.0]
COEFF_X_TICKS = [0.0, 0.5, 1.0]
COEFF_X_TICK_LABELS = [r"$0$", r"$0.5$", r"$1$"]

# 时间系数图图例：仅第一组显示，四项水平排列于图内下方。
COEFF_LEGEND_NCOL = 4
COEFF_LEGEND_LOC = "lower center"
COEFF_LEGEND_BBOX = (0.5, -0.03)

# 三组相图统一坐标范围。
NORMALISE_PHASE_COEFFS = False
PHASE_AXIS_MIN = -150.0
PHASE_AXIS_MAX = 150.0
PHASE_TICKS = [-150, 0, 150]
PHASE_LINE_WIDTH = 1
PHASE_SHOW_ZERO_LINES = True

# 图中文字。
FONT_SIZE = 10.0
ENERGY_LABEL_FONT_SIZE = 10
INNER_PANEL_LABEL_FONT_SIZE = 10
MAIN_PANEL_LABEL_FONT_SIZE = 10.0

# 坐标轴标签与坐标轴之间的距离，单位为 point。
# 较小或负的 labelpad 会使标签更靠近坐标轴。
AXIS_LABEL_PAD = -1.3
PHASE_AXIS_LABEL_PAD = -1

# 第二行纵坐标标签的专用水平间距。
# 数值越小，a_i、a_2、a_4、a_3 越靠近对应纵轴。
COEFF_Y_LABEL_PAD = -8.0
PHASE_Y_LABEL_PAD = -8.0

# 图尺寸与手动布局，单位均为 inch。
FIG_WIDTH = 6.5
MARGIN_LEFT_INCH = 0.62
MARGIN_RIGHT_INCH = 0.18
MARGIN_TOP_INCH = 0.50
MARGIN_BOTTOM_INCH = 0.48
COLUMN_GAP_INCH = 0.085
# 第二行四张图之间的统一水平间距，按实际绘图区边界计算。
# 时间系数图与第一个相图、以及三个相图彼此之间均使用该距离。
BOTTOM_ROW_EQUAL_GAP_INCH = 0.5

# 第二行整体相对于可用绘图区域中心的水平微调量。
# 正值向右，负值向左；0 表示严格居中。
BOTTOM_ROW_CENTER_SHIFT_INCH = -0.5

TOP_BOTTOM_ROW_GAP_INCH = 0.30
CASE_BLOCK_GAP_INCH = 0.40

# 下排相图采用正方形；可用该比例适当减小下排高度。
BOTTOM_ROW_HEIGHT_SCALE = 0.4

# 共享色条放在第一个主面板的 Mode 4 内部下方。
SHOW_INSET_COLORBAR = True
INSET_COLORBAR_BOUNDS = [0.57, 0.22, 0.36, 0.065]
COLORBAR_TICKS = [-1.0, 0.0, 1.0]
COLORBAR_TICK_LABEL_SIZE = 7.5

OUT_BASENAME = "Fig_POD_St0.3"
DPI = 600

# 透明背景设置。
# SVG：整张画布、各坐标轴内部及掩膜区域均不填充白色。
TRANSPARENT_SVG_BACKGROUND = True
TRANSPARENT_MASKED_REGIONS = True


# ============================================================
# 2. Matplotlib 风格
# ============================================================

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": 8.0,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{newtxtext,newtxmath}",
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
    "svg.fonttype": "none",
})


def build_white_center_cmap(vmin=-1.0, vmax=1.0, n=512, sigma=0.16):
    """构造中心偏白的蓝白红色图。"""
    base = colormaps["RdBu_r"].resampled(n)
    new_colors = base(np.linspace(0, 1, n))
    vals = np.linspace(vmin, vmax, n)
    weights = np.exp(-0.5 * (vals / sigma) ** 2)

    for i in range(n):
        w = weights[i]
        new_colors[i] = (1.0 - w) * new_colors[i] + w * np.array([1, 1, 1, 1])

    cmap = mcolors.ListedColormap(new_colors)
    if TRANSPARENT_MASKED_REGIONS:
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    else:
        cmap.set_bad("white")
    return cmap


CMAP = build_white_center_cmap(CMAP_VMIN, CMAP_VMAX, sigma=WHITE_SIGMA)


# ============================================================
# 3. 命名与标签函数
# ============================================================

def fmt_num(x: float) -> str:
    if float(x).is_integer():
        return f"{float(x):.1f}"
    return str(float(x)).rstrip("0").rstrip(".")


def fmt_num_short(x: float) -> str:
    return fmt_num(x).replace(".", "p").replace("-", "m")


def case_name_from_values(k: float, st: float, a: float = 0.0, dc: float = 1.0) -> str:
    a_val = int(a) if float(a).is_integer() else a
    dc_val = int(dc) if float(dc).is_integer() else dc
    return f"k{fmt_num_short(k)}_St{fmt_num_short(st)}_a{a_val}_dc{dc_val}"


def get_npz_path(case: dict) -> str:
    cname = case_name_from_values(case["k"], case["st"], case.get("a", 0), case.get("dc", 1))
    return os.path.join(SAVE_ROOT, cname, f"{cname}_pod_results.npz")


def get_labels_path(case: dict) -> str:
    cname = case_name_from_values(case["k"], case["st"], case.get("a", 0), case.get("dc", 1))
    return os.path.join(SAVE_ROOT, cname, f"{cname}_snapshot_labels.csv")


def roman_panel_label(index: int) -> str:
    """返回内部小图标签 (i)–(viii)，采用正体小写罗马数字。"""
    roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
    if not 0 <= index < len(roman):
        raise ValueError("内部小图编号仅支持 0–7。")
    return rf"$\mathrm{{({roman[index]})}}$"


def add_inner_panel_label(ax, index: int):
    ax.text(
        0.97,
        0.94,
        roman_panel_label(index),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=INNER_PANEL_LABEL_FONT_SIZE,
        zorder=20,
        bbox=dict(facecolor="none", edgecolor="none", alpha=0.0, pad=0.8),
    )


# ============================================================
# 4. 数据读取与处理
# ============================================================

def load_snapshot_labels(case: dict, n_snapshots: int) -> list:
    """读取快照所属周期与相位。"""
    labels_path = get_labels_path(case)
    if not os.path.exists(labels_path):
        print(f"警告：未找到快照标签文件：{labels_path}")
        return []

    labels = []
    with open(labels_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                labels.append({
                    "snapshot_index": int(row["snapshot_index"]),
                    "cycle_no": int(row["cycle_no"]),
                    "phase_frac": float(row["phase_frac"]),
                })
            except (KeyError, TypeError, ValueError):
                continue

    labels.sort(key=lambda item: item["snapshot_index"])
    if len(labels) != n_snapshots:
        print(
            f"警告：标签数 {len(labels)} 与 coeffs 快照数 {n_snapshots} 不一致，"
            "将退化为末尾固定快照数选取。"
        )
        return []
    return labels


def load_case_pod_result(case: dict) -> dict:
    """读取单个工况 POD 结果。"""
    npz_path = get_npz_path(case)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"未找到 POD 结果文件：\n{npz_path}\n"
            "请确认该工况已经运行过 POD 批处理程序。"
        )

    with np.load(npz_path) as data:
        required_keys = ["X", "Y", "modes_w", "energy", "coeffs", "fluid_mask"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"{npz_path} 中缺少变量 {key}")

        X = np.asarray(data["X"])
        Y = np.asarray(data["Y"])
        modes_w = np.asarray(data["modes_w"], dtype=float)
        energy = np.asarray(data["energy"], dtype=float)
        eigvals = np.asarray(data["eigvals"], dtype=float) if "eigvals" in data else None
        coeffs = np.asarray(data["coeffs"], dtype=float)
        fluid_mask = np.asarray(data["fluid_mask"], dtype=bool)

    if modes_w.ndim != 3 or modes_w.shape[0] < N_MODES:
        raise ValueError(
            f"{npz_path} 中 modes_w 形状为 {modes_w.shape}，"
            f"至少需要 {N_MODES} 阶模态。"
        )
    if coeffs.ndim != 2 or coeffs.shape[0] < N_MODES:
        raise ValueError(
            f"{npz_path} 中 coeffs 形状为 {coeffs.shape}，"
            f"至少需要 {N_MODES} 阶时间系数。"
        )
    if energy.size < N_MODES:
        raise ValueError(f"{npz_path} 中 energy 数量少于 {N_MODES}。")

    labels = load_snapshot_labels(case, coeffs.shape[1])

    return {
        "case": case,
        "name": case_name_from_values(case["k"], case["st"], case.get("a", 0), case.get("dc", 1)),
        "X": X,
        "Y": Y,
        "modes_w": modes_w[:N_MODES].copy(),
        "energy": energy[:N_MODES].copy(),
        "eigvals": None if eigvals is None else eigvals[:N_MODES].copy(),
        "coeffs": coeffs[:N_MODES].copy(),
        "snapshot_labels": labels,
        "fluid_mask": fluid_mask,
        "npz_path": npz_path,
    }


def apply_modal_vorticity_weighting(modes_w, energy, eigvals=None):
    """对模态涡量进行能量加权。"""
    mode = MODAL_VORTICITY_WEIGHT.lower()

    if mode == "none":
        weights = np.ones(modes_w.shape[0], dtype=float)
        label = "unweighted"
    elif mode == "sqrt_energy":
        weights = np.sqrt(np.asarray(energy[:modes_w.shape[0]], dtype=float))
        label = r"$\sqrt{E_i}\,\omega_i$"
    elif mode == "sqrt_lambda":
        if eigvals is not None:
            weights = np.sqrt(np.asarray(eigvals[:modes_w.shape[0]], dtype=float))
            label = r"$\sqrt{\lambda_i}\,\omega_i$"
        elif FALLBACK_TO_SQRT_ENERGY:
            weights = np.sqrt(np.asarray(energy[:modes_w.shape[0]], dtype=float))
            label = r"$\sqrt{E_i}\,\omega_i$"
            print("警告：npz 中没有 eigvals，已退化为 sqrt(E_i) 加权。")
        else:
            raise KeyError("sqrt_lambda 加权需要 npz 中包含 eigvals。")
    else:
        raise ValueError(
            "MODAL_VORTICITY_WEIGHT 必须为 'sqrt_lambda'、'sqrt_energy' 或 'none'。"
        )

    weights = np.asarray(weights, dtype=float)
    weights[~np.isfinite(weights)] = 0.0
    fields = modes_w * weights[:, None, None]
    return fields, weights, label


def crop_by_xlim_ylim(X, Y, fields, fluid_mask, xlim, ylim):
    """按显示范围裁剪二维场。"""
    display_mask = (
        (X >= xlim[0]) & (X <= xlim[1]) &
        (Y >= ylim[0]) & (Y <= ylim[1])
    )
    rows = np.where(np.any(display_mask, axis=1))[0]
    cols = np.where(np.any(display_mask, axis=0))[0]

    if rows.size == 0 or cols.size == 0:
        raise ValueError("X_LIM/Y_LIM 与 POD 网格没有交集。")

    Xc = X[np.ix_(rows, cols)]
    Yc = Y[np.ix_(rows, cols)]
    Fc = fields[:, rows[:, None], cols]
    Mc = fluid_mask[np.ix_(rows, cols)]
    return Xc, Yc, Fc, Mc


def normalise_fields(case_results, mode="global"):
    """归一化前四阶模态涡量。"""
    if mode not in ["global", "row", "mode"]:
        raise ValueError("NORMALISE_BY 必须为 'global'、'row' 或 'mode'。")

    eps = 1e-12

    if mode == "global":
        values = []
        for res in case_results:
            finite = np.abs(res["fields"][np.isfinite(res["fields"])])
            if finite.size:
                values.append(finite)
        if not values:
            raise ValueError("没有可用于归一化的有限模态涡量数据。")
        scale = max(float(np.nanpercentile(np.concatenate(values), NORMALISE_PERCENTILE)), eps)
        for res in case_results:
            res["fields_plot"] = np.clip(res["fields"] / scale, -1.0, 1.0)
            res["normalise_scale"] = scale

    elif mode == "row":
        for res in case_results:
            values = np.abs(res["fields"][np.isfinite(res["fields"])])
            scale = max(float(np.nanpercentile(values, NORMALISE_PERCENTILE)), eps)
            res["fields_plot"] = np.clip(res["fields"] / scale, -1.0, 1.0)
            res["normalise_scale"] = scale

    else:  # mode
        for res in case_results:
            fields_plot = np.empty_like(res["fields"])
            scales = []
            for m in range(N_MODES):
                values = np.abs(res["fields"][m][np.isfinite(res["fields"][m])])
                scale = max(float(np.nanpercentile(values, NORMALISE_PERCENTILE)), eps)
                fields_plot[m] = np.clip(res["fields"][m] / scale, -1.0, 1.0)
                scales.append(scale)
            res["fields_plot"] = fields_plot
            res["normalise_scale"] = scales

    return case_results


def select_one_cycle(res: dict):
    """
    选取一个周期内的前四阶时间系数。

    返回：phase、coeffs_cycle、cycle_no。
    coeffs_cycle 的形状为 (4, n_phase)。
    """
    coeffs = res["coeffs"]
    labels = res.get("snapshot_labels", [])

    if labels:
        cycle_array = np.array([item["cycle_no"] for item in labels], dtype=int)
        phase_array = np.array([item["phase_frac"] for item in labels], dtype=float)
        unique_cycles = np.unique(cycle_array)

        if TARGET_CYCLE_NO is not None and TARGET_CYCLE_NO in unique_cycles:
            cycle_no = int(TARGET_CYCLE_NO)
        else:
            if TARGET_CYCLE_NO is not None:
                print(
                    f"警告：{res['name']} 不包含第 {TARGET_CYCLE_NO} 周期，"
                    f"自动改用最后一个周期 {unique_cycles[-1]}。"
                )
            cycle_no = int(unique_cycles[-1])

        ids = np.where(cycle_array == cycle_no)[0]
        ids = ids[np.argsort(phase_array[ids])]
        phase = phase_array[ids]
        coeffs_cycle = coeffs[:, ids]
    else:
        n_available = coeffs.shape[1]
        n_phase = min(DEFAULT_PHASES_PER_CYCLE, n_available)
        ids = np.arange(n_available - n_phase, n_available)
        phase = np.linspace(0.0, 1.0, n_phase, endpoint=False)
        coeffs_cycle = coeffs[:, ids]
        cycle_no = None

    if phase.size < 2:
        raise ValueError(f"{res['name']} 选定周期内快照数不足。")

    # snapshot_labels.csv 中的 phase_frac 通常已经位于 [0, 1)。
    # 这里仅做轻量修正，不用“除以最后一个相位”的方式拉伸坐标，
    # 以免把 31/32 误改为 1 并丢失周期闭合点。
    phase = np.asarray(phase, dtype=float)
    if not np.all(np.isfinite(phase)):
        phase = np.linspace(0.0, 1.0, phase.size, endpoint=False)
    else:
        phase = phase - phase[0]
        if phase[-1] > 1.0 + 1e-8:
            # 仅在标签不是归一化相位时，按一个采样间隔外推周期长度。
            dphase = np.diff(phase)
            positive_steps = dphase[dphase > 0.0]
            step = float(np.median(positive_steps)) if positive_steps.size else 0.0
            period_span = phase[-1] + step
            if period_span > 0.0:
                phase = phase / period_span
        phase = np.clip(phase, 0.0, 1.0)

    # 系数随相位曲线在 t/T=1 处回到周期起点。
    if phase[-1] < 1.0 - 1e-10:
        phase_plot = np.r_[phase, 1.0]
        coeffs_plot = np.concatenate([coeffs_cycle, coeffs_cycle[:, :1]], axis=1)
    else:
        phase_plot = phase
        coeffs_plot = coeffs_cycle

    return phase, coeffs_cycle, phase_plot, coeffs_plot, cycle_no


def prepare_coefficients(coeffs: np.ndarray, normalise: bool):
    """按每阶最大绝对值选择性归一化系数。"""
    coeffs = np.asarray(coeffs, dtype=float).copy()
    if not normalise:
        return coeffs

    for i in range(coeffs.shape[0]):
        scale = np.nanmax(np.abs(coeffs[i]))
        if np.isfinite(scale) and scale > 0.0:
            coeffs[i] /= scale
    return coeffs


# ============================================================
# 5. 手动布局
# ============================================================

def compute_manual_layout(ncases: int, ncols: int = 4):
    """
    计算四个 2×4 主面板的 axes 位置。

    第一行仍按四个等宽长条流场图布置。

    第二行按“实际绘图区宽度”顺序布置：
    - 第1张时间系数图宽度为 axis_width；
    - 后3张相图的绘图区均为正方形，宽度等于 bottom_height；
    - 相邻四张图之间统一使用 BOTTOM_ROW_EQUAL_GAP_INCH。

    这样相图启用 box_aspect=1 后，不会因为 axes 容器宽度不同而产生
    视觉上不等的水平间距。
    """
    total_col_gap = COLUMN_GAP_INCH * (ncols - 1)
    usable_width = FIG_WIDTH - MARGIN_LEFT_INCH - MARGIN_RIGHT_INCH - total_col_gap
    if usable_width <= 0.0:
        raise ValueError("图宽、边距或列距设置不合理。")

    axis_width = usable_width / ncols
    flow_aspect = (Y_LIM[1] - Y_LIM[0]) / (X_LIM[1] - X_LIM[0])
    top_height = axis_width * flow_aspect
    bottom_height = axis_width * BOTTOM_ROW_HEIGHT_SCALE
    block_height = top_height + TOP_BOTTOM_ROW_GAP_INCH + bottom_height

    coeff_width = axis_width
    phase_width = bottom_height
    bottom_group_width = (
        coeff_width
        + 3.0 * phase_width
        + 3.0 * BOTTOM_ROW_EQUAL_GAP_INCH
    )

    available_bottom_width = FIG_WIDTH - MARGIN_LEFT_INCH - MARGIN_RIGHT_INCH
    if bottom_group_width > available_bottom_width:
        raise ValueError(
            "第二行总宽度超过可用图幅。请减小 "
            "BOTTOM_ROW_EQUAL_GAP_INCH 或 BOTTOM_ROW_HEIGHT_SCALE。"
        )

    bottom_group_left = (
        MARGIN_LEFT_INCH
        + 0.5 * (available_bottom_width - bottom_group_width)
        + BOTTOM_ROW_CENTER_SHIFT_INCH
    )
    bottom_group_right = bottom_group_left + bottom_group_width

    if bottom_group_left < 0.0 or bottom_group_right > FIG_WIDTH:
        raise ValueError(
            "BOTTOM_ROW_CENTER_SHIFT_INCH 使第二行超出图幅，请减小其绝对值。"
        )

    fig_height = (
        MARGIN_TOP_INCH + MARGIN_BOTTOM_INCH
        + ncases * block_height
        + (ncases - 1) * CASE_BLOCK_GAP_INCH
    )

    top_positions = []
    bottom_positions = []
    block_tops = []

    for i in range(ncases):
        block_top = (
            fig_height
            - MARGIN_TOP_INCH
            - i * (block_height + CASE_BLOCK_GAP_INCH)
        )
        top_bottom = block_top - top_height
        bottom_bottom = (
            top_bottom
            - TOP_BOTTOM_ROW_GAP_INCH
            - bottom_height
        )
        block_tops.append(block_top)

        row_top = []
        for j in range(ncols):
            left = MARGIN_LEFT_INCH + j * (axis_width + COLUMN_GAP_INCH)
            row_top.append([
                left / FIG_WIDTH,
                top_bottom / fig_height,
                axis_width / FIG_WIDTH,
                top_height / fig_height,
            ])

        row_bottom = []
        current_left = bottom_group_left

        row_bottom.append([
            current_left / FIG_WIDTH,
            bottom_bottom / fig_height,
            coeff_width / FIG_WIDTH,
            bottom_height / fig_height,
        ])
        current_left += coeff_width + BOTTOM_ROW_EQUAL_GAP_INCH

        for _ in range(3):
            row_bottom.append([
                current_left / FIG_WIDTH,
                bottom_bottom / fig_height,
                phase_width / FIG_WIDTH,
                bottom_height / fig_height,
            ])
            current_left += phase_width + BOTTOM_ROW_EQUAL_GAP_INCH

        top_positions.append(row_top)
        bottom_positions.append(row_bottom)

    return (
        fig_height,
        top_positions,
        bottom_positions,
        block_tops,
        axis_width,
        top_height,
        bottom_height,
    )


def create_axes(fig, top_positions, bottom_positions):
    ncases = len(top_positions)
    top_axes = np.empty((ncases, 4), dtype=object)
    bottom_axes = np.empty((ncases, 4), dtype=object)

    for i in range(ncases):
        for j in range(4):
            top_axes[i, j] = fig.add_axes(
                top_positions[i][j],
                facecolor="none",
            )
            bottom_axes[i, j] = fig.add_axes(
                bottom_positions[i][j],
                facecolor="none",
            )
            top_axes[i, j].patch.set_alpha(0.0)
            bottom_axes[i, j].patch.set_alpha(0.0)

    return top_axes, bottom_axes


# ============================================================
# 6. 单个小图绘制函数
# ============================================================

def style_axes(ax):
    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        length=2.5,
        width=0.6,
        pad=1.5,
    )


def plot_mode(ax, res: dict, mode_index: int, pcm_norm):
    X = res["Xc"]
    Y = res["Yc"]
    field = res["fields_plot"][mode_index]

    pcm = ax.pcolormesh(
        X,
        Y,
        field,
        cmap=CMAP,
        norm=pcm_norm,
        shading="auto",
        rasterized=True,
    )

    if SHOW_MASK_CONTOUR:
        ax.contour(
            X,
            Y,
            res["mask"].astype(float),
            levels=[0.5],
            colors="0.2",
            linewidths=0.4,
        )

    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)
    ax.set_aspect("equal", adjustable="box")

    # 保留模态图的空间坐标、刻度和坐标标签，但不显示 Mode 1–4 标题。
    ax.set_xticks(X_TICKS)
    ax.set_yticks(Y_TICKS)
    ax.set_xlabel(r"$x/c$", labelpad=AXIS_LABEL_PAD)
    style_axes(ax)

    # 仅第一列显示 y 轴刻度值和 y/c 标签，避免四列重复。
    if mode_index == 0:
        ax.set_ylabel(r"$y/c$", labelpad=AXIS_LABEL_PAD)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    ax.text(
        0.03,
        0.90,
        rf"$E_{mode_index + 1}={res['energy'][mode_index] * 100:.1f}\%$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ENERGY_LABEL_FONT_SIZE,
        zorder=10,
        bbox=dict(facecolor="none", edgecolor="none", alpha=0.0, pad=0.8),
    )

    return pcm


def plot_one_cycle_coefficients(ax, res: dict, global_ylim, show_legend: bool = False):
    phase_plot = res["phase_plot"]
    coeffs_plot = prepare_coefficients(res["coeffs_cycle_plot"], NORMALISE_TIME_COEFFS)

    for m in range(N_MODES):
        kwargs = {}
        if COEFF_SHOW_MARKERS:
            kwargs.update(marker="o", ms=COEFF_MARKER_SIZE, markevery=max(1, len(phase_plot) // 8))
        ax.plot(
            phase_plot,
            coeffs_plot[m],
            lw=COEFF_LINE_WIDTH,
            label=rf"$a_{m + 1}$",
            **kwargs,
        )

    ax.axhline(0.0, color="0.65", lw=0.45, ls="--", zorder=0)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(COEFF_X_TICKS, labels=COEFF_X_TICK_LABELS)
    ax.set_yticks(COEFF_Y_TICKS)
    if global_ylim is not None:
        ax.set_ylim(global_ylim)

    ax.set_xlabel(r"$t/T$", labelpad=AXIS_LABEL_PAD)
    if NORMALISE_TIME_COEFFS:
        ax.set_ylabel(r"$a_i/\max|a_i|$", labelpad=COEFF_Y_LABEL_PAD)
    else:
        ax.set_ylabel(r"$a_i$", rotation=0, ha="right", va="center", labelpad=COEFF_Y_LABEL_PAD)

    # 标题放入轴内，避免与上排模态图的横坐标刻度重叠。
    # ax.text(
    #     0.50, 0.985, r"$a_1$--$a_4$",
    #     transform=ax.transAxes,
    #     ha="center", va="top",
    #     fontsize=FONT_SIZE,
    #     bbox=dict(facecolor="none", edgecolor="none", alpha=0.0, pad=0.7),
    #     zorder=10,
    # )
    style_axes(ax)
    if show_legend:
        ax.legend(
            frameon=False,
            ncol=COEFF_LEGEND_NCOL,
            loc=COEFF_LEGEND_LOC,
            bbox_to_anchor=COEFF_LEGEND_BBOX,
            handlelength=1.4,
            columnspacing=0.35,
            handletextpad=0.25,
            borderaxespad=0.15,
        )


def plot_phase_portrait(ax, res: dict, x_mode: int, y_mode: int, show_ylabel=True):
    coeffs = prepare_coefficients(res["coeffs_cycle"], NORMALISE_PHASE_COEFFS)
    x = coeffs[x_mode]
    y = coeffs[y_mode]

    if CLOSE_PHASE_PORTRAIT and x.size > 1:
        x_line = np.r_[x, x[0]]
        y_line = np.r_[y, y[0]]
    else:
        x_line = x
        y_line = y

    # 相图仅绘制连续轨迹线，不显示圆形数据点和方形起始点。
    ax.plot(
        x_line,
        y_line,
        "-",
        lw=PHASE_LINE_WIDTH,
        alpha=0.9,
    )

    if PHASE_SHOW_ZERO_LINES:
        ax.axhline(0.0, color="0.65", lw=0.45, ls="--", zorder=0)
        ax.axvline(0.0, color="0.65", lw=0.45, ls="--", zorder=0)

    ax.set_xlim(PHASE_AXIS_MIN, PHASE_AXIS_MAX)
    ax.set_ylim(PHASE_AXIS_MIN, PHASE_AXIS_MAX)
    ax.set_xticks(PHASE_TICKS)
    ax.set_yticks(PHASE_TICKS)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1.0)
    style_axes(ax)

    if NORMALISE_PHASE_COEFFS:
        xlabel = rf"$a_{x_mode + 1}/\max|a_{x_mode + 1}|$"
        ylabel = rf"$a_{y_mode + 1}/\max|a_{y_mode + 1}|$"
    else:
        xlabel = rf"$a_{x_mode + 1}$"
        ylabel = rf"$a_{y_mode + 1}$"

    ax.set_xlabel(xlabel, labelpad=PHASE_AXIS_LABEL_PAD)
    if show_ylabel:
        ax.set_ylabel(ylabel, rotation=0, ha="right", va="center", labelpad=PHASE_Y_LABEL_PAD)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)



# ============================================================
# 7. 主绘图函数
# ============================================================

def plot_representative_pod_modes():
    case_results = []

    # ---------- 读取、加权、裁剪 ----------
    for case in REPRESENTATIVE_CASES:
        res = load_case_pod_result(case)

        weighted_modes_w, modal_weights, modal_field_label = apply_modal_vorticity_weighting(
            res["modes_w"],
            res["energy"],
            res.get("eigvals"),
        )

        Xc, Yc, Fc, Mc = crop_by_xlim_ylim(
            res["X"],
            res["Y"],
            weighted_modes_w,
            res["fluid_mask"],
            X_LIM,
            Y_LIM,
        )

        for m in range(N_MODES):
            Fc[m][~Mc] = np.nan

        phase, coeffs_cycle, phase_plot, coeffs_cycle_plot, cycle_no = select_one_cycle(res)

        res["Xc"] = Xc
        res["Yc"] = Yc
        res["fields"] = Fc
        res["mask"] = Mc
        res["modal_weights"] = modal_weights
        res["modal_field_label"] = modal_field_label
        res["phase"] = phase
        res["coeffs_cycle"] = coeffs_cycle
        res["phase_plot"] = phase_plot
        res["coeffs_cycle_plot"] = coeffs_cycle_plot
        res["selected_cycle_no"] = cycle_no
        case_results.append(res)

    if MODAL_VORTICITY_WEIGHT.lower() != "none" and NORMALISE_BY == "mode":
        print(
            "警告：NORMALISE_BY='mode' 会抵消不同模态之间的加权幅值差异，"
            "正文图建议使用 'global' 或 'row'。"
        )

    case_results = normalise_fields(case_results, NORMALISE_BY)

    # ---------- 单周期系数曲线统一纵轴 ----------
    if COEFF_Y_LIM is not None:
        coeff_global_ylim = tuple(COEFF_Y_LIM)
    else:
        values = []
        for res in case_results:
            coeffs_plot = prepare_coefficients(res["coeffs_cycle_plot"], NORMALISE_TIME_COEFFS)
            finite = np.abs(coeffs_plot[np.isfinite(coeffs_plot)])
            if finite.size:
                values.append(finite)
        if values:
            vmax = float(np.nanpercentile(np.concatenate(values), 99.5))
            vmax = max(vmax * 1.08, 1e-12)
            coeff_global_ylim = (-vmax, vmax)
        else:
            coeff_global_ylim = None

    # ---------- 创建图 ----------
    ncases = len(case_results)
    (
        fig_height,
        top_positions,
        bottom_positions,
        block_tops,
        axis_width,
        top_height,
        bottom_height,
    ) = compute_manual_layout(ncases)

    print(
        f"Figure size: {FIG_WIDTH:.3f} in × {fig_height:.3f} in; "
        f"column width: {axis_width:.3f} in; "
        f"mode height: {top_height:.3f} in; "
        f"bottom-row height: {bottom_height:.3f} in"
    )

    fig = plt.figure(
        figsize=(FIG_WIDTH, fig_height),
        facecolor="none",
    )
    fig.patch.set_alpha(0.0)
    top_axes, bottom_axes = create_axes(fig, top_positions, bottom_positions)
    pcm_norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    pcm_for_colorbar = None

    for i, res in enumerate(case_results):
        # 第一行：Mode 1–4。
        for j in range(N_MODES):
            pcm = plot_mode(top_axes[i, j], res, j, pcm_norm)
            if i == 0 and j == 3:
                pcm_for_colorbar = pcm
            add_inner_panel_label(top_axes[i, j], j)

        # 第二行：系数曲线 + 三组相图。
        plot_one_cycle_coefficients(
            bottom_axes[i, 0],
            res,
            coeff_global_ylim,
            show_legend=(i == 0),
        )
        add_inner_panel_label(bottom_axes[i, 0], 4)

        plot_phase_portrait(
            bottom_axes[i, 1], res,
            x_mode=0, y_mode=1,
            show_ylabel=True,
        )
        add_inner_panel_label(bottom_axes[i, 1], 5)

        plot_phase_portrait(
            bottom_axes[i, 2], res,
            x_mode=2, y_mode=3,
            show_ylabel=True,
        )
        add_inner_panel_label(bottom_axes[i, 2], 6)

        plot_phase_portrait(
            bottom_axes[i, 3], res,
            x_mode=0, y_mode=2,
            show_ylabel=True,
        )
        add_inner_panel_label(bottom_axes[i, 3], 7)

    # 共享色条。
    if SHOW_INSET_COLORBAR and pcm_for_colorbar is not None:
        target_ax = top_axes[0, 3]
        cax = target_ax.inset_axes(
            INSET_COLORBAR_BOUNDS,
            facecolor="none",
        )
        cax.patch.set_alpha(0.0)
        cbar = fig.colorbar(
            pcm_for_colorbar,
            cax=cax,
            orientation="horizontal",
            ticks=COLORBAR_TICKS,
        )
        cbar.ax.tick_params(
            direction="in",
            length=1.8,
            width=0.5,
            labelsize=COLORBAR_TICK_LABEL_SIZE,
            pad=0.8,
        )

    # --------------------------------------------------------
    # 主面板标签 (a)–(d)：按整个 2×4 子图组的外接边界定位。
    # get_tightbbox 会把刻度值、x/y 坐标标签等一并计入，
    # 因此标签中心位于“包含坐标标签后的整体左边界与上边界交点”。
    # --------------------------------------------------------
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for i in range(ncases):
        panel_axes = [top_axes[i, j] for j in range(N_MODES)] + [
            bottom_axes[i, j] for j in range(4)
        ]
        tight_bboxes = [
            ax.get_tightbbox(renderer)
            for ax in panel_axes
            if ax.get_visible()
        ]
        tight_bboxes = [bbox for bbox in tight_bboxes if bbox is not None]
        if not tight_bboxes:
            continue

        panel_bbox_display = Bbox.union(tight_bboxes)
        panel_bbox_figure = panel_bbox_display.transformed(fig.transFigure.inverted())
        panel_letter = chr(ord("a") + i)

        fig.text(
            panel_bbox_figure.x0,
            panel_bbox_figure.y1,
            rf"$({panel_letter})$",
            transform=fig.transFigure,
            ha="center",
            va="center",
            fontsize=MAIN_PANEL_LABEL_FONT_SIZE,
        )

    png_path = os.path.join(FIG_SAVE_DIR, OUT_BASENAME + ".png")
    svg_path = os.path.join(FIG_SAVE_DIR, OUT_BASENAME + ".svg")
    pdf_path = os.path.join(FIG_SAVE_DIR, OUT_BASENAME + ".pdf")

    fig.savefig(
        png_path,
        dpi=DPI,
        transparent=TRANSPARENT_SVG_BACKGROUND,
        facecolor="none" if TRANSPARENT_SVG_BACKGROUND else "white",
        edgecolor="none",
    )
    fig.savefig(
        svg_path,
        dpi=DPI,
        transparent=TRANSPARENT_SVG_BACKGROUND,
        facecolor="none" if TRANSPARENT_SVG_BACKGROUND else "white",
        edgecolor="none",
    )
    # 如需 PDF，取消下一行注释。
    # fig.savefig(pdf_path, dpi=DPI)
    plt.close(fig)

    print("\nPOD 前四阶模态、单周期系数与相图组合图已保存：")
    print(png_path)
    print(svg_path)
    print(pdf_path)

    print("\n读取的 POD 文件与周期：")
    for res in case_results:
        cycle_text = (
            str(res["selected_cycle_no"])
            if res["selected_cycle_no"] is not None
            else f"last {res['coeffs_cycle'].shape[1]} snapshots"
        )
        print(f"  {res['name']}: {res['npz_path']}")
        print(f"    selected cycle: {cycle_text}")
        print(
            f"    E1-E{N_MODES}: " +
            ", ".join(f"{e:.4f}" for e in res["energy"])
        )
        print(
            "    modal weights: " +
            ", ".join(f"{w:.4e}" for w in res["modal_weights"])
        )


# ============================================================
# 8. 执行
# ============================================================

if __name__ == "__main__":
    plot_representative_pod_modes()