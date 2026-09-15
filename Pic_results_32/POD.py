# -*- coding: utf-8 -*-
"""
pod_311_periodic_batch.py

用于论文 3.1.1 节的周期性尾迹 POD 分析。

核心功能
1. 对指定周期性尾迹工况，或指定工况及其邻域参数工况，批量执行速度脉动 POD。
2. 每个工况输出 POD 能谱、累计能量、时间系数、a1-a2 相图、平均涡量、模态涡量、瞬时涡量相位图。
3. 汇总输出 all_cases_pod_metrics_summary.csv。
4. 根据汇总指标绘制 k-St 图谱，包括 C2、C4、Delta12、H、E12、分类结果。

适用数据格式
Tecplot Ordered ASCII .dat
VARIABLES = "X" "Y" "U" "V" "Presure" "Vorticity" "body" "dvordt"
ZONE ... I=..., J=..., SOLUTIONTIME=...

本次处理策略
- 处理指定的 12 个非零后翼俯仰角工况。
- k=0.8、1.1 的工况仅使用第 15 周期的等相位快照进行 POD。
- k=1.4、1.7 的工况仅使用第 20 周期的等相位快照进行 POD。
- POD 区域优先取后翼下游尾迹区，减少翼扫掠区域对模态的污染。
"""

import os
import re
import csv
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps


# ============================================================
# 0. 用户配置区
# ============================================================

@dataclass
class CaseSpec:
    k: float
    st: float
    a: float = 0.0
    dc: float = 1.0
    tag: str = ""
    # 可选：对单个工况单独指定 POD 所取周期。
    # 若保持 None，则按 CYCLE_BY_CASE、CYCLE_BY_K、DEFAULT_CYCLE_* 的优先级自动选择。
    cycle_start: Optional[int] = None
    cycle_end: Optional[int] = None


# ---------- 运行模式 ----------
# 本代码仅处理 CASE_LIST 中显式列出的工况。
RUN_MODE = "selected_cases"

# ---------- 指定处理工况 ----------
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

# 转换为程序内部使用的 CaseSpec。
SELECTED_CASES = [
    CaseSpec(k=k, st=st, a=a, dc=1.0, tag="selected")
    for k, st, a in CASE_LIST
]

# ---------- 如果需要加入周围参数，使用完整参数网格 ----------
K_GRID = [0.8, 1.1, 1.4, 1.7]
ST_GRID = [0.3]
A_GRID = sorted({float(a) for _, _, a in CASE_LIST})
DC_GRID = [1.0]

# 邻域大小。1 表示目标工况上下左右相邻一个网格步长。
NEIGHBOR_RADIUS_K_INDEX = 1
NEIGHBOR_RADIUS_ST_INDEX = 1

# ---------- 路径设置 ----------
# 注意：模板中的 {k}, {st}, {a}, {dc_int} 会自动替换。
BASE_ROOT = r"E:/Rensq/DC_explore_F/data_1Tflow/"
CASE_DIR_TEMPLATE = (
    BASE_ROOT
    + r"k{k}/st{st}k{k}/st{st}k{k}a{a}/"
      r"st{st}k{k}a{a}dc{dc_int}/k{k}st{st}a{a}/"
)

SAVE_ROOT = r"E:/Rensq/DC_explore_F/POD_results/POD_results_3_1_2_aneq0"

# ---------- 周期选择 ----------
# 每个元组均为 (cycle_start, cycle_end)。
# 当起止周期相同时，只读取该单个周期内的全部快照。
#
# 本次设置：
#   k=0.8、1.1：仅使用第 15 周期；
#   k=1.4、1.7：仅使用第 20 周期。
#
# 周期选择优先级：
# 1) CaseSpec(..., cycle_start=..., cycle_end=...)，精确到单个工况；
# 2) CYCLE_BY_CASE，精确到 (k, St) 或 (k, St, a, dc)；
# 3) CYCLE_BY_K，只按 k 选择；
# 4) DEFAULT_CYCLE_START / DEFAULT_CYCLE_END。
#
# 未在 CYCLE_BY_K 或 CYCLE_BY_CASE 中指定的工况，默认取第 15 周期。
DEFAULT_CYCLE_START = 15
DEFAULT_CYCLE_END = 15

CYCLE_BY_K = {
    0.8: (15, 15),   # 第 15 周期
    1.1: (15, 15),   # 第 15 周期
    1.4: (20, 20),   # 第 20 周期
    1.7: (20, 20),   # 第 20 周期
}

# 若同一个 k 下不同 St 也需要不同周期，在这里覆盖 CYCLE_BY_K。
# 支持两种 key：
#   (k, St): (cycle_start, cycle_end)
#   (k, St, a, dc): (cycle_start, cycle_end)
# 例如：
# CYCLE_BY_CASE = {
#     (1.4, 0.3): (11, 15),
#     (1.4, 0.4, 0.0, 1.0): (11, 15),
# }
CYCLE_BY_CASE = {}

# ---------- POD 空间区域 ----------
# 3.1.1 用于尾迹分类时，建议聚焦后翼下游。
# 后翼 x2=1.5c 时，建议 x_min_pod=1.5 或 1.8。
# 如果要包含近场相互作用，可设 x_min_pod=0.5。
X_MIN_POD, X_MAX_POD = -1, 20.0
Y_MIN_POD, Y_MAX_POD = -5.0, 5.0

# ---------- 空间降采样 ----------
STRIDE_X = 2
STRIDE_Y = 2

# ---------- 固体区域掩膜 ----------
USE_BODY_MASK = True
BODY_THRESHOLD = 0.5

# ---------- POD 输出设置 ----------
N_MODES_SAVE = 12
N_MODES_PLOT = 6
N_RECON_PLOT = 4
N_COEFF_PLOT = 4

# ---------- 涡量图范围 ----------
VMIN_VORT, VMAX_VORT = -30.0, 30.0
DPI = 300

# ---------- 周期性/组织性分类阈值 ----------
# 这些阈值不是物理定律，只用于批量初筛；正式论文中建议结合瞬时涡量图和模态图解释。
CLASS_THRESHOLDS = {
    "periodic_pair": {"C4_min": 0.70, "H_max": 0.55, "Delta12_max": 0.35},
    "strongly_organized": {"C4_min": 0.55, "H_max": 0.68},
    "weak_or_complex": {"C4_max": 0.35, "H_min": 0.75},
}


# ============================================================
# 1. Matplotlib 设置
# ============================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "text.usetex": False,
})


def build_white_center_cmap(vmin: float, vmax: float, n: int = 512, sigma: float = 1.0):
    base = colormaps["RdBu_r"].resampled(n)
    new_colors = base(np.linspace(0, 1, n))
    vals = np.linspace(vmin, vmax, n)
    weights = np.exp(-0.5 * (vals / sigma) ** 2)
    for i in range(n):
        w = weights[i]
        new_colors[i] = (1 - w) * new_colors[i] + w * np.array([1, 1, 1, 1])
    return mcolors.ListedColormap(new_colors)


CMAP_WHITE = build_white_center_cmap(VMIN_VORT, VMAX_VORT)


# ============================================================
# 2. 工具函数
# ============================================================

def fmt_num(x: float) -> str:
    """保持路径中的 1.1、2.0 等格式。"""
    if float(x).is_integer():
        return f"{float(x):.1f}"
    return str(float(x)).rstrip("0").rstrip(".")


def fmt_num_short(x: float) -> str:
    """用于文件名，避免小数点造成不便。"""
    s = fmt_num(x)
    return s.replace(".", "p").replace("-", "m")


def case_name(case: CaseSpec) -> str:
    a_val = int(case.a) if float(case.a).is_integer() else case.a
    dc_val = int(case.dc) if float(case.dc).is_integer() else case.dc
    return f"k{fmt_num_short(case.k)}_St{fmt_num_short(case.st)}_a{a_val}_dc{dc_val}"


def get_case_dir(case: CaseSpec) -> str:
    return CASE_DIR_TEMPLATE.format(
        k=fmt_num(case.k),
        st=fmt_num(case.st),
        a=int(case.a) if float(case.a).is_integer() else fmt_num(case.a),
        dc_int=int(case.dc),
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def norm_float_key(x: float, ndigits: int = 10) -> float:
    """用于字典匹配，避免 1.4000000000000001 这类浮点误差。"""
    return round(float(x), ndigits)


def normalise_cycle_dict_keys() -> Tuple[Dict, Dict[float, Tuple[int, int]]]:
    """将用户配置的周期字典转换为标准浮点 key。"""
    by_case = {}
    for key, val in CYCLE_BY_CASE.items():
        if len(key) not in (2, 4):
            raise ValueError("CYCLE_BY_CASE 的 key 只能是 (k, St) 或 (k, St, a, dc)。")
        by_case[tuple(norm_float_key(v) for v in key)] = tuple(int(v) for v in val)

    by_k = {norm_float_key(k): tuple(int(v) for v in val) for k, val in CYCLE_BY_K.items()}
    return by_case, by_k


def get_cycle_range(case: CaseSpec) -> Tuple[int, int]:
    """按单工况、(k, St)、k、默认值的优先级返回该工况 POD 使用的周期范围。"""
    if case.cycle_start is not None or case.cycle_end is not None:
        start = DEFAULT_CYCLE_START if case.cycle_start is None else int(case.cycle_start)
        end = start if case.cycle_end is None else int(case.cycle_end)
        if end < start:
            raise ValueError(f"{case_name(case)} 的 cycle_end 小于 cycle_start。")
        return start, end

    by_case, by_k = normalise_cycle_dict_keys()
    key4 = (norm_float_key(case.k), norm_float_key(case.st), norm_float_key(case.a), norm_float_key(case.dc))
    key2 = (norm_float_key(case.k), norm_float_key(case.st))
    key1 = norm_float_key(case.k)

    if key4 in by_case:
        start, end = by_case[key4]
    elif key2 in by_case:
        start, end = by_case[key2]
    elif key1 in by_k:
        start, end = by_k[key1]
    else:
        start, end = DEFAULT_CYCLE_START, DEFAULT_CYCLE_END

    if end < start:
        raise ValueError(f"{case_name(case)} 的 cycle_end 小于 cycle_start。")
    return int(start), int(end)


# ============================================================
# 3. Tecplot dat 文件读取
# ============================================================

def read_tecplot_ordered_dat(
    filename: str,
    target_vars: Sequence[str] = ("X", "Y", "U", "V", "Vorticity", "body"),
) -> Dict[str, np.ndarray]:
    """
    读取 Tecplot Ordered ASCII .dat 文件。
    返回 result[var] shape=(J, I)。
    """
    var_map = {
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

    cols = [var_map[v] for v in target_vars]
    need_cols = max(cols) + 1

    I = None
    J = None
    sol_time = None
    data_start_idx = None
    zone_found = False

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        if (not zone_found) and stripped.upper().startswith("ZONE"):
            zone_found = True
            i_m = re.search(r"I\s*=\s*(\d+)", stripped, re.I)
            j_m = re.search(r"J\s*=\s*(\d+)", stripped, re.I)
            if i_m:
                I = int(i_m.group(1))
            if j_m:
                J = int(j_m.group(1))
            continue

        if zone_found:
            if sol_time is None:
                t_m = re.match(r"^\s*([\d.Ee+\-]+)\s*$", stripped)
                if t_m:
                    try:
                        sol_time = float(t_m.group(1))
                        continue
                    except ValueError:
                        pass

            parts = stripped.split()
            if len(parts) >= need_cols:
                try:
                    [float(parts[c]) for c in cols]
                    data_start_idx = idx
                    break
                except ValueError:
                    continue

    if I is None or J is None or data_start_idx is None:
        raise ValueError(f"无法解析 dat 文件头信息: {filename}")

    data = np.loadtxt(
        filename,
        skiprows=data_start_idx - 1,
        max_rows=I * J,
        usecols=cols,
        delimiter=None,
        dtype=np.float32,
    )

    if data.shape[0] != I * J:
        raise ValueError(f"{filename} 数据点数量不足，期望 {I * J}，实际 {data.shape[0]}")

    result = {"solution_time": sol_time, "I": I, "J": J}
    for i, var in enumerate(target_vars):
        result[var] = data[:, i].reshape(J, I)
    return result


# ============================================================
# 4. 文件扫描与周期识别
# ============================================================

def parse_time_from_filename(filename: str) -> Optional[float]:
    """
    支持 F165000.dat、F 165000.dat、F    165000.dat。
    每 10000 步 = 1 无量纲时间。
    """
    base = os.path.basename(filename)
    m = re.match(r"^F\s*(\d+)\.dat$", base, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)) / 10000.0


def is_valid_flow_filename(filename: str) -> bool:
    base = os.path.basename(filename)
    return re.match(r"^F\s*\d+\.dat$", base, re.IGNORECASE) is not None


def compute_cycle_info(t: float, T: float, eps: float = 1e-10) -> Tuple[int, float]:
    cycle_no = int(np.floor(t / T + eps)) + 1
    phase_frac = (t / T) - np.floor(t / T + eps)

    if abs(phase_frac) < 1e-8:
        phase_frac = 0.0
    elif abs(phase_frac - 1.0) < 1e-8:
        phase_frac = 0.0
        cycle_no += 1

    return cycle_no, phase_frac


def scan_cycles_data(
    case_dir: str,
    T: float,
    cycle_start: Optional[int] = None,
    cycle_end: Optional[int] = None,
) -> List[Dict]:
    all_files = []

    if not os.path.isdir(case_dir):
        return []

    for fname in os.listdir(case_dir):
        if is_valid_flow_filename(fname):
            t = parse_time_from_filename(fname)
            if t is None:
                continue
            cycle_no, phase_frac = compute_cycle_info(t, T)
            if cycle_start is not None and cycle_no < cycle_start:
                continue
            if cycle_end is not None and cycle_no > cycle_end:
                continue
            all_files.append({
                "filename": fname,
                "path": os.path.join(case_dir, fname),
                "t": t,
                "cycle_no": cycle_no,
                "phase_frac": phase_frac,
            })

    all_files.sort(key=lambda x: x["t"])
    grouped = {}
    for item in all_files:
        grouped.setdefault(item["cycle_no"], []).append(item)

    cycles_data = []
    for cyc in sorted(grouped.keys()):
        records = sorted(grouped[cyc], key=lambda x: x["phase_frac"])
        cycles_data.append({"i": cyc, "records": records})
    return cycles_data


# ============================================================
# 5. 构建 POD 数据集
# ============================================================

def find_crop_indices(
    X: np.ndarray,
    Y: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    stride_x: int = 1,
    stride_y: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
    if not np.any(mask):
        raise ValueError("POD 裁剪区域与流场坐标没有交集，请检查 x/y 范围。")

    rows = np.where(np.any(mask, axis=1))[0][::stride_y]
    cols = np.where(np.any(mask, axis=0))[0][::stride_x]
    return rows, cols


def flatten_cycles(cycles_data: List[Dict]) -> List[Dict]:
    records = []
    for cyc in cycles_data:
        records.extend(cyc["records"])
    return records


def load_snapshots_for_pod(cycles_data: List[Dict]) -> Tuple[np.ndarray, ...]:
    records = flatten_cycles(cycles_data)
    if not records:
        raise ValueError("没有可用于 POD 的流场文件。")

    U_list, V_list, W_list, B_list = [], [], [], []
    labels = []
    rows, cols = None, None
    Xc, Yc = None, None

    for idx, rec in enumerate(records):
        print(f"    [{idx+1:03d}/{len(records):03d}] 读取 {rec['filename']}")
        dat = read_tecplot_ordered_dat(rec["path"])

        X, Y = dat["X"], dat["Y"]
        U0, V0 = dat["U"], dat["V"]
        W0, B0 = dat["Vorticity"], dat["body"]

        if rows is None:
            rows, cols = find_crop_indices(
                X, Y, X_MIN_POD, X_MAX_POD, Y_MIN_POD, Y_MAX_POD, STRIDE_X, STRIDE_Y
            )
            Xc = X[np.ix_(rows, cols)].astype(np.float32)
            Yc = Y[np.ix_(rows, cols)].astype(np.float32)
            print(f"    裁剪网格: ny={Xc.shape[0]}, nx={Xc.shape[1]}, velocity dof={2 * Xc.size}")

        U_list.append(U0[np.ix_(rows, cols)].astype(np.float32))
        V_list.append(V0[np.ix_(rows, cols)].astype(np.float32))
        W_list.append(W0[np.ix_(rows, cols)].astype(np.float32))
        B_list.append(B0[np.ix_(rows, cols)].astype(np.float32))

        labels.append({
            "snapshot_index": idx,
            "filename": rec["filename"],
            "t_filename": rec["t"],
            "solution_time": dat["solution_time"],
            "cycle_no": rec["cycle_no"],
            "phase_frac": rec["phase_frac"],
        })

    U = np.stack(U_list, axis=0)
    V = np.stack(V_list, axis=0)
    W = np.stack(W_list, axis=0)
    BODY = np.stack(B_list, axis=0)
    return Xc, Yc, U, V, W, BODY, labels


# ============================================================
# 6. POD 计算与指标
# ============================================================

def build_fluid_mask(BODY: np.ndarray) -> np.ndarray:
    if not USE_BODY_MASK:
        return np.ones(BODY.shape[1:], dtype=bool)
    body_union = np.any(BODY > BODY_THRESHOLD, axis=0)
    fluid_mask = ~body_union
    if np.sum(fluid_mask) == 0:
        raise ValueError("fluid_mask 中没有可用流体点，请检查 body 字段或 POD 区域。")
    return fluid_mask


def compute_velocity_pod(
    U: np.ndarray,
    V: np.ndarray,
    fluid_mask: np.ndarray,
    n_modes_save: int = 12,
) -> Dict[str, np.ndarray]:
    ns, ny, nx = U.shape
    n_fluid = int(np.sum(fluid_mask))

    U_mean = np.mean(U, axis=0)
    V_mean = np.mean(V, axis=0)
    Up = U - U_mean[None, :, :]
    Vp = V - V_mean[None, :, :]

    Xmat = np.empty((2 * n_fluid, ns), dtype=np.float32)
    for s in range(ns):
        Xmat[:n_fluid, s] = Up[s][fluid_mask]
        Xmat[n_fluid:, s] = Vp[s][fluid_mask]

    # 去除数值残差均值，使 POD 严格作用于脉动场。
    Xmat -= np.mean(Xmat, axis=1, keepdims=True)

    if ns < 2:
        raise ValueError("POD 至少需要 2 个快照。")

    U_svd, S, Vt = np.linalg.svd(Xmat, full_matrices=False)
    eigvals = S ** 2 / (ns - 1)
    total_energy = np.sum(eigvals)
    if total_energy <= 0:
        raise ValueError("POD 总能量为 0，请检查速度场。")

    energy = eigvals / total_energy
    cumulative = np.cumsum(energy)
    nm = min(n_modes_save, ns)
    coeffs = np.diag(S[:nm]) @ Vt[:nm, :]

    modes_u = np.full((nm, ny, nx), np.nan, dtype=np.float32)
    modes_v = np.full((nm, ny, nx), np.nan, dtype=np.float32)
    for m in range(nm):
        mu = np.full((ny, nx), np.nan, dtype=np.float32)
        mv = np.full((ny, nx), np.nan, dtype=np.float32)
        mu[fluid_mask] = U_svd[:n_fluid, m]
        mv[fluid_mask] = U_svd[n_fluid:, m]
        modes_u[m] = mu
        modes_v[m] = mv

    return {
        "eigvals": eigvals,
        "energy": energy,
        "cumulative": cumulative,
        "singular_values": S,
        "coeffs": coeffs,
        "modes_u": modes_u,
        "modes_v": modes_v,
        "mean_u": U_mean,
        "mean_v": V_mean,
        "fluid_mask": fluid_mask,
    }


def compute_pair_balance(eigvals: np.ndarray, i: int, j: int) -> Optional[float]:
    if len(eigvals) <= max(i, j):
        return None
    denom = eigvals[i] + eigvals[j]
    if denom <= 0:
        return None
    return float(abs(eigvals[i] - eigvals[j]) / denom)


def compute_periodic_repetition_metrics(pod: Dict, labels: List[Dict]) -> Dict[str, Optional[float]]:
    """
    如果输入了多个周期，计算同相位快照的 POD 系数重复性。
    只有一个周期时返回空值。
    """
    coeffs = pod["coeffs"]
    cycles = np.array([lab["cycle_no"] for lab in labels])
    phases = np.array([lab["phase_frac"] for lab in labels])
    unique_cycles = np.unique(cycles)

    out = {
        "a12_cycle_repeat_corr_mean": None,
        "a12_cycle_repeat_corr_min": None,
    }
    if len(unique_cycles) < 2 or coeffs.shape[0] < 2:
        return out

    # 以第一个周期的相位为参考，查找其他周期最接近相位。
    ref_cycle = unique_cycles[0]
    ref_ids = np.where(cycles == ref_cycle)[0]
    corrs = []
    ref_signal = coeffs[:2, ref_ids].reshape(-1)
    for cyc in unique_cycles[1:]:
        ids = []
        for rid in ref_ids:
            ph = phases[rid]
            cand = np.where(cycles == cyc)[0]
            sid = cand[np.argmin(np.abs(phases[cand] - ph))]
            ids.append(sid)
        sig = coeffs[:2, ids].reshape(-1)
        if np.std(ref_signal) > 0 and np.std(sig) > 0:
            corrs.append(float(np.corrcoef(ref_signal, sig)[0, 1]))

    if corrs:
        out["a12_cycle_repeat_corr_mean"] = float(np.mean(corrs))
        out["a12_cycle_repeat_corr_min"] = float(np.min(corrs))
    return out


def classify_wake(metrics: Dict[str, object]) -> Tuple[int, str]:
    """
    返回 class_id 和 class_name。
    class_id 用于图谱绘制。
    0 weak_or_complex
    1 organized
    2 periodic_pair
    """
    C4 = metrics.get("C4")
    H = metrics.get("H")
    D12 = metrics.get("Delta12")

    if C4 is None or H is None:
        return -1, "not_available"

    th_pair = CLASS_THRESHOLDS["periodic_pair"]
    if (
        C4 >= th_pair["C4_min"]
        and H <= th_pair["H_max"]
        and (D12 is None or D12 <= th_pair["Delta12_max"])
    ):
        return 2, "periodic_pair_dominated"

    th_org = CLASS_THRESHOLDS["strongly_organized"]
    if C4 >= th_org["C4_min"] and H <= th_org["H_max"]:
        return 1, "organized_multimode"

    return 0, "weak_or_complex"


def compute_pod_metrics(
    case: CaseSpec,
    pod: Dict[str, np.ndarray],
    labels: List[Dict],
    cycle_start: int,
    cycle_end: int,
) -> Dict[str, object]:
    eigvals = pod["eigvals"]
    energy = pod["energy"]
    cumulative = pod["cumulative"]
    singular_values = pod["singular_values"]
    fluid_mask = pod["fluid_mask"]
    coeffs = pod["coeffs"]

    metrics: Dict[str, object] = {
        "case_name": case_name(case),
        "case_tag": case.tag,
        "k": float(case.k),
        "St": float(case.st),
        "a": float(case.a),
        "dc": float(case.dc),
        "cycle_start": int(cycle_start),
        "cycle_end": int(cycle_end),
        "x_min_pod": X_MIN_POD,
        "x_max_pod": X_MAX_POD,
        "y_min_pod": Y_MIN_POD,
        "y_max_pod": Y_MAX_POD,
        "stride_x": STRIDE_X,
        "stride_y": STRIDE_Y,
        "use_body_mask": int(USE_BODY_MASK),
        "body_threshold": BODY_THRESHOLD,
        "n_snapshots": int(coeffs.shape[1]),
        "n_modes_total": int(len(energy)),
        "n_modes_saved": int(coeffs.shape[0]),
        "n_grid_total": int(fluid_mask.size),
        "n_fluid_points": int(np.sum(fluid_mask)),
        "n_masked_points": int(np.sum(~fluid_mask)),
        "masked_fraction": float(np.sum(~fluid_mask) / fluid_mask.size),
    }

    if labels:
        metrics.update({
            "t_start": float(labels[0]["t_filename"]),
            "t_end": float(labels[-1]["t_filename"]),
            "phase_start": float(labels[0]["phase_frac"]),
            "phase_end": float(labels[-1]["phase_frac"]),
        })

    n_report = min(12, len(energy))
    for i in range(n_report):
        m = i + 1
        metrics[f"lambda{m}"] = float(eigvals[i])
        metrics[f"sigma{m}"] = float(singular_values[i])
        metrics[f"E{m}"] = float(energy[i])
        metrics[f"C{m}"] = float(cumulative[i])

    for m in [2, 4, 6, 8, 10, 12]:
        metrics[f"C{m}"] = float(cumulative[m - 1]) if len(cumulative) >= m else None

    metrics["Delta12"] = compute_pair_balance(eigvals, 0, 1)
    metrics["Delta34"] = compute_pair_balance(eigvals, 2, 3)
    metrics["E12_sum"] = float(energy[0] + energy[1]) if len(energy) >= 2 else None
    metrics["E34_sum"] = float(energy[2] + energy[3]) if len(energy) >= 4 else None
    metrics["E1_over_E2"] = float(energy[0] / energy[1]) if len(energy) >= 2 and energy[1] > 0 else None

    p = eigvals / np.sum(eigvals)
    p = p[p > 0]
    metrics["H"] = float(-np.sum(p * np.log(p)) / np.log(len(eigvals))) if len(p) > 1 else 0.0
    metrics["N_eff"] = float(np.exp(-np.sum(p * np.log(p)))) if len(p) > 1 else 1.0

    metrics.update(compute_periodic_repetition_metrics(pod, labels))
    class_id, class_name = classify_wake(metrics)
    metrics["wake_class_id"] = class_id
    metrics["wake_class_name"] = class_name
    return metrics


# ============================================================
# 7. 涡量、重构和相位槽
# ============================================================

def estimate_dx_dy(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    dx = float(np.nanmean(np.diff(X, axis=1)))
    dy = float(np.nanmean(np.diff(Y, axis=0)))
    return abs(dx), abs(dy)


def fill_nan_for_gradient(A: np.ndarray) -> np.ndarray:
    B = np.array(A, copy=True)
    B[~np.isfinite(B)] = 0.0
    return B


def compute_vorticity_from_uv(X: np.ndarray, Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    dx, dy = estimate_dx_dy(X, Y)
    du_dy = np.gradient(U, dy, axis=0)
    dv_dx = np.gradient(V, dx, axis=1)
    return dv_dx - du_dy


def compute_modal_vorticity(X: np.ndarray, Y: np.ndarray, modes_u: np.ndarray, modes_v: np.ndarray) -> np.ndarray:
    dx, dy = estimate_dx_dy(X, Y)
    nm, ny, nx = modes_u.shape
    modes_w = np.full_like(modes_u, np.nan, dtype=np.float32)
    for m in range(nm):
        u_m = fill_nan_for_gradient(modes_u[m])
        v_m = fill_nan_for_gradient(modes_v[m])
        du_dy = np.gradient(u_m, dy, axis=0)
        dv_dx = np.gradient(v_m, dx, axis=1)
        w_m = dv_dx - du_dy
        valid = np.isfinite(modes_u[m]) & np.isfinite(modes_v[m])
        w_m[~valid] = np.nan
        modes_w[m] = w_m
    return modes_w


def reconstruct_velocity(pod: Dict, snapshot_id: int, mode_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    ur = np.array(pod["mean_u"], copy=True)
    vr = np.array(pod["mean_v"], copy=True)
    for m in mode_ids:
        if m >= pod["coeffs"].shape[0]:
            continue
        ur += pod["coeffs"][m, snapshot_id] * np.nan_to_num(pod["modes_u"][m], nan=0.0)
        vr += pod["coeffs"][m, snapshot_id] * np.nan_to_num(pod["modes_v"][m], nan=0.0)
    return ur, vr


def select_phase_slots(labels: List[Dict], n_slots: int = 8, cycle_preference: Optional[int] = None) -> List[int]:
    cycle = np.array([lab["cycle_no"] for lab in labels])
    phase = np.array([lab["phase_frac"] for lab in labels])
    if cycle_preference is None:
        cycle_preference = int(np.min(cycle))
    ids_in_cycle = np.where(cycle == cycle_preference)[0]
    if len(ids_in_cycle) == 0:
        ids_in_cycle = np.arange(len(labels))

    targets = np.linspace(0.0, 1.0, n_slots, endpoint=False)
    selected = []
    for ph in targets:
        sid = ids_in_cycle[np.argmin(np.abs(phase[ids_in_cycle] - ph))]
        selected.append(int(sid))
    return sorted(set(selected))


# ============================================================
# 8. 绘图函数
# ============================================================

def plot_contour(
    X: np.ndarray,
    Y: np.ndarray,
    F: np.ndarray,
    fname: str,
    title: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap=CMAP_WHITE,
    levels: int = 128,
):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    if vmin is None or vmax is None:
        vmax0 = np.nanpercentile(np.abs(F), 99.0)
        vmax0 = 1.0 if vmax0 == 0 or not np.isfinite(vmax0) else vmax0
        vmin0, vmax0 = -vmax0, vmax0
    else:
        vmin0, vmax0 = vmin, vmax

    cf = ax.contourf(X, Y, F, levels=np.linspace(vmin0, vmax0, levels), cmap=cmap, extend="both")
    fig.colorbar(cf, ax=ax, shrink=0.86)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$y/c$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(fname, dpi=DPI)
    plt.close(fig)


def plot_phase_panel(
    X: np.ndarray,
    Y: np.ndarray,
    fields: Sequence[np.ndarray],
    labels_title: Sequence[str],
    fname: str,
    main_title: str,
    vmin: float,
    vmax: float,
    ncols: int = 4,
):
    n = len(fields)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.5 * nrows), squeeze=False)

    last_cf = None
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        last_cf = ax.contourf(
            X, Y, fields[i], levels=np.linspace(vmin, vmax, 128), cmap=CMAP_WHITE, extend="both"
        )
        ax.set_title(labels_title[i], fontsize=10)
        ax.set_xlabel(r"$x/c$")
        ax.set_ylabel(r"$y/c$")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(direction="in", top=True, right=True)

    if last_cf is not None:
        fig.colorbar(last_cf, ax=axes.ravel().tolist(), shrink=0.82, pad=0.01)
    fig.suptitle(main_title, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.96, 0.97])
    fig.savefig(fname, dpi=DPI)
    plt.close(fig)


def plot_pod_energy(pod: Dict, out_dir: str, cname: str):
    energy = pod["energy"]
    cumulative = pod["cumulative"]
    mode_id = np.arange(1, len(energy) + 1)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.semilogy(mode_id, energy, "o-", ms=3, lw=1.1)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Energy fraction")
    ax.set_title("POD energy spectrum")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{cname}_energy_spectrum.png"), dpi=DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(mode_id, cumulative, "o-", ms=3, lw=1.1)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Cumulative energy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Cumulative POD energy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{cname}_cumulative_energy.png"), dpi=DPI)
    plt.close(fig)


def plot_fluid_mask(X: np.ndarray, Y: np.ndarray, fluid_mask: np.ndarray, out_dir: str, cname: str):
    F = np.where(fluid_mask, 1.0, 0.0)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    cf = ax.contourf(X, Y, F, levels=np.linspace(0, 1, 3), cmap="Greys")
    fig.colorbar(cf, ax=ax, shrink=0.86, ticks=[0, 1])
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$y/c$")
    ax.set_title("POD fluid mask")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{cname}_fluid_mask.png"), dpi=DPI)
    plt.close(fig)


def plot_phase_portrait(pod: Dict, labels: List[Dict], out_dir: str, cname: str):
    coeffs = pod["coeffs"]
    if coeffs.shape[0] < 2:
        return
    a1, a2 = coeffs[0], coeffs[1]
    cycles = np.array([lab["cycle_no"] for lab in labels])

    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot(a1, a2, "-", lw=1.0, color="black", alpha=0.75)
    for cyc in sorted(np.unique(cycles)):
        ids = np.where(cycles == cyc)[0]
        ax.scatter(a1[ids], a2[ids], s=16, label=f"cycle {cyc}")
    ax.set_xlabel(r"$a_1(t)$")
    ax.set_ylabel(r"$a_2(t)$")
    ax.set_title(r"Phase portrait: $a_1-a_2$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{cname}_a1_a2_phase_portrait.png"), dpi=DPI)
    plt.close(fig)


def plot_temporal_coefficients(pod: Dict, labels: List[Dict], out_dir: str, cname: str, nplot: int = 4):
    coeffs = pod["coeffs"]
    nplot = min(nplot, coeffs.shape[0])
    phase = np.array([lab["phase_frac"] for lab in labels])
    t = np.array([lab["t_filename"] for lab in labels])

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    unique_cycles = np.unique([lab["cycle_no"] for lab in labels])
    one_cycle = len(unique_cycles) == 1
    x = phase if one_cycle else t
    xlabel = r"Phase $t/T$" if one_cycle else "Time"

    for m in range(nplot):
        a_m = coeffs[m]
        denom = np.max(np.abs(a_m))
        denom = 1.0 if denom == 0 else denom
        ax.plot(x, a_m / denom, "o-", ms=3, lw=1.0, label=f"mode {m+1}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalised coefficient")
    ax.set_title("POD temporal coefficients")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{cname}_temporal_coefficients.png"), dpi=DPI)
    plt.close(fig)


def plot_coefficient_spectrum(pod: Dict, labels: List[Dict], out_dir: str, cname: str, reference_frequency: float, nplot: int = 2):
    coeffs = pod["coeffs"]
    nplot = min(nplot, coeffs.shape[0])
    t = np.array([lab["t_filename"] for lab in labels], dtype=float)
    if len(t) < 4:
        return

    dt = float(np.mean(np.diff(t)))
    if dt <= 0:
        return
    freqs = np.fft.rfftfreq(coeffs.shape[1], d=dt)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for m in range(nplot):
        a_m = coeffs[m] - np.mean(coeffs[m])
        spec = np.abs(np.fft.rfft(a_m)) ** 2
        if np.max(spec) > 0:
            spec = spec / np.max(spec)
        ax.plot(freqs / reference_frequency, spec, lw=1.2, label=f"mode {m+1}")

    ax.set_xlabel(r"$f/f_1$")
    ax.set_ylabel("Normalised spectral energy")
    ax.set_title("Spectrum of POD coefficients")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{cname}_coefficient_spectrum_f_over_f1.png"), dpi=DPI)
    plt.close(fig)


def plot_mean_modes_and_examples(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    pod: Dict,
    modes_w: np.ndarray,
    labels: List[Dict],
    out_dir: str,
    cname: str,
):
    # 平均涡量
    W_mean = np.mean(W, axis=0)
    plot_contour(
        X, Y, W_mean,
        os.path.join(out_dir, f"{cname}_mean_vorticity.png"),
        title="Mean vorticity",
        vmin=VMIN_VORT,
        vmax=VMAX_VORT,
    )

    # 模态涡量单图
    for m in range(min(N_MODES_PLOT, modes_w.shape[0])):
        vmax_m = np.nanpercentile(np.abs(modes_w[m]), 99.0)
        vmax_m = 1.0 if vmax_m == 0 or not np.isfinite(vmax_m) else vmax_m
        plot_contour(
            X, Y, modes_w[m],
            os.path.join(out_dir, f"{cname}_mode{m+1:02d}_modal_vorticity.png"),
            title=f"Modal vorticity, mode {m+1}",
            vmin=-vmax_m,
            vmax=vmax_m,
        )

    # 模态涡量总览图：论文中可直接作为 3.1.1 的模态结构图。
    mode_fields = []
    mode_titles = []
    for m in range(min(N_MODES_PLOT, modes_w.shape[0])):
        mode_fields.append(modes_w[m])
        mode_titles.append(f"Mode {m+1}, E={pod['energy'][m]:.3f}")
    if mode_fields:
        vmax_all = np.nanpercentile(np.abs(np.stack(mode_fields)), 99.0)
        vmax_all = 1.0 if vmax_all == 0 or not np.isfinite(vmax_all) else vmax_all
        plot_phase_panel(
            X, Y, mode_fields, mode_titles,
            os.path.join(out_dir, f"{cname}_modal_vorticity_panel.png"),
            main_title="POD modal vorticity",
            vmin=-vmax_all,
            vmax=vmax_all,
            ncols=3,
        )

    # 一个周期的瞬时涡量相位图。
    selected = select_phase_slots(labels, n_slots=8)
    fields = [W[sid] for sid in selected]
    titles = [f"phase={labels[sid]['phase_frac']:.3f}T" for sid in selected]
    plot_phase_panel(
        X, Y, fields, titles,
        os.path.join(out_dir, f"{cname}_instant_vorticity_phase_panel.png"),
        main_title="Instantaneous vorticity over one cycle",
        vmin=VMIN_VORT,
        vmax=VMAX_VORT,
        ncols=4,
    )

    # 前两阶模态重构涡量相位图。
    recon_fields = []
    recon_titles = []
    for sid in selected[:N_RECON_PLOT]:
        ur, vr = reconstruct_velocity(pod, sid, mode_ids=[0, 1])
        wr = compute_vorticity_from_uv(X, Y, ur, vr)
        wr[~pod["fluid_mask"]] = np.nan
        recon_fields.append(wr)
        recon_titles.append(f"phase={labels[sid]['phase_frac']:.3f}T")
    if recon_fields:
        plot_phase_panel(
            X, Y, recon_fields, recon_titles,
            os.path.join(out_dir, f"{cname}_recon12_vorticity_phase_panel.png"),
            main_title="Vorticity reconstructed by modes 1-2",
            vmin=VMIN_VORT,
            vmax=VMAX_VORT,
            ncols=2,
        )


# ============================================================
# 9. CSV/NPZ 保存
# ============================================================

def none_to_blank(x):
    return "" if x is None else x


def write_dict_csv(path: str, rows: List[Dict[str, object]]):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: none_to_blank(row.get(k)) for k in fieldnames})


def save_labels(labels: List[Dict], out_dir: str, cname: str):
    path = os.path.join(out_dir, f"{cname}_snapshot_labels.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["snapshot_index", "filename", "t_filename", "solution_time", "cycle_no", "phase_frac"])
        for lab in labels:
            writer.writerow([
                lab["snapshot_index"], lab["filename"], lab["t_filename"],
                lab["solution_time"], lab["cycle_no"], lab["phase_frac"],
            ])


def save_energy_csv(pod: Dict, out_dir: str, cname: str):
    path = os.path.join(out_dir, f"{cname}_pod_energy.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "lambda", "singular_value", "energy_fraction_Ei", "cumulative_energy_Ci"])
        for i in range(len(pod["energy"])):
            writer.writerow([
                i + 1,
                float(pod["eigvals"][i]),
                float(pod["singular_values"][i]),
                float(pod["energy"][i]),
                float(pod["cumulative"][i]),
            ])


def save_coefficients_csv(pod: Dict, labels: List[Dict], out_dir: str, cname: str):
    path = os.path.join(out_dir, f"{cname}_pod_coefficients.csv")
    coeffs = pod["coeffs"]
    header = ["snapshot_index", "filename", "t_filename", "solution_time", "cycle_no", "phase_frac"]
    header += [f"a{i+1}" for i in range(coeffs.shape[0])]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for s, lab in enumerate(labels):
            row = [
                lab["snapshot_index"], lab["filename"], lab["t_filename"],
                lab["solution_time"], lab["cycle_no"], lab["phase_frac"],
            ] + [float(coeffs[i, s]) for i in range(coeffs.shape[0])]
            writer.writerow(row)


def save_spectrum_csv(pod: Dict, labels: List[Dict], out_dir: str, cname: str, reference_frequency: float, n_modes: int = 6):
    t = np.array([lab["t_filename"] for lab in labels], dtype=float)
    if len(t) < 4:
        return
    dt = float(np.mean(np.diff(t)))
    if dt <= 0:
        return
    coeffs = pod["coeffs"]
    freqs = np.fft.rfftfreq(coeffs.shape[1], d=dt)
    n_modes = min(n_modes, coeffs.shape[0])

    path = os.path.join(out_dir, f"{cname}_pod_coefficient_spectrum.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "frequency", "frequency_over_f1", "normalised_spectral_energy"])
        for m in range(n_modes):
            a_m = coeffs[m] - np.mean(coeffs[m])
            spec = np.abs(np.fft.rfft(a_m)) ** 2
            if np.max(spec) > 0:
                spec = spec / np.max(spec)
            for j, freq in enumerate(freqs):
                writer.writerow([m + 1, float(freq), float(freq / reference_frequency), float(spec[j])])


def save_mask_csv(X: np.ndarray, Y: np.ndarray, fluid_mask: np.ndarray, out_dir: str, cname: str):
    path = os.path.join(out_dir, f"{cname}_pod_fluid_mask.csv")
    ny, nx = fluid_mask.shape
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["j", "i", "x", "y", "fluid_mask"])
        for j in range(ny):
            for i in range(nx):
                writer.writerow([j, i, float(X[j, i]), float(Y[j, i]), int(fluid_mask[j, i])])


def save_all_case_outputs(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    pod: Dict,
    modes_w: np.ndarray,
    metrics: Dict,
    labels: List[Dict],
    out_dir: str,
    cname: str,
    reference_frequency: float,
):
    np.savez_compressed(
        os.path.join(out_dir, f"{cname}_pod_results.npz"),
        X=X,
        Y=Y,
        U_mean=pod["mean_u"],
        V_mean=pod["mean_v"],
        W_mean=np.mean(W, axis=0),
        eigvals=pod["eigvals"],
        energy=pod["energy"],
        cumulative=pod["cumulative"],
        coeffs=pod["coeffs"],
        modes_u=pod["modes_u"],
        modes_v=pod["modes_v"],
        modes_w=modes_w,
        fluid_mask=pod["fluid_mask"],
    )

    with open(os.path.join(out_dir, f"{cname}_pod_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    write_dict_csv(os.path.join(out_dir, f"{cname}_pod_metrics_summary.csv"), [metrics])
    save_labels(labels, out_dir, cname)
    save_energy_csv(pod, out_dir, cname)
    save_coefficients_csv(pod, labels, out_dir, cname)
    save_spectrum_csv(pod, labels, out_dir, cname, reference_frequency, n_modes=6)
    save_mask_csv(X, Y, pod["fluid_mask"], out_dir, cname)


# ============================================================
# 10. 批量工况选择
# ============================================================

def get_cases_to_run() -> List[CaseSpec]:
    """返回 CASE_LIST 中显式指定的 12 个非零俯仰角工况。"""
    return SELECTED_CASES


# ============================================================
# 11. 汇总图谱
# ============================================================

def read_metrics_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def to_float(x: object) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def plot_metric_vs_angle(
    rows: List[Dict[str, str]],
    value_key: str,
    out_dir: str,
    ylabel: str,
    title: str,
):
    """
    按 k 分组绘制 POD 指标随后翼俯仰角 a 的变化。

    本次所有工况的 St 均为 0.3，同一个 (k, St) 对应三个 a。
    因此不再使用原来的 k-St 网格，以免不同 a 的数据相互覆盖。
    """
    grouped: Dict[float, List[Tuple[float, float]]] = {}

    for row in rows:
        k = to_float(row.get("k"))
        a = to_float(row.get("a"))
        value = to_float(row.get(value_key))
        if k is None or a is None or value is None:
            continue
        grouped.setdefault(k, []).append((a, value))

    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.4))

    for k in sorted(grouped):
        data = sorted(grouped[k], key=lambda item: item[0])
        angles = np.asarray([item[0] for item in data], dtype=float)
        values = np.asarray([item[1] for item in data], dtype=float)

        ax.plot(
            angles,
            values,
            marker="o",
            linewidth=1.2,
            markersize=4.5,
            label=rf"$k={k:g}$",
        )

    all_angles = sorted({
        to_float(row.get("a"))
        for row in rows
        if to_float(row.get("a")) is not None
    })

    ax.set_xticks(all_angles)
    ax.set_xlabel(r"$a\ (^\circ)$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, f"{value_key}_versus_a.png"),
        dpi=DPI,
    )
    plt.close(fig)


def plot_organization_scatter(rows: List[Dict[str, str]], out_dir: str):
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    plotted = False

    unique_k = sorted({
        to_float(row.get("k"))
        for row in rows
        if to_float(row.get("k")) is not None
    })

    for k in unique_k:
        selected = []
        for row in rows:
            row_k = to_float(row.get("k"))
            C4 = to_float(row.get("C4"))
            H = to_float(row.get("H"))
            a = to_float(row.get("a"))
            if row_k == k and C4 is not None and H is not None and a is not None:
                selected.append((C4, H, a))

        if not selected:
            continue

        selected.sort(key=lambda item: item[2])
        x = [item[0] for item in selected]
        y = [item[1] for item in selected]
        ax.plot(x, y, "o-", linewidth=1.0, markersize=4.5, label=rf"$k={k:g}$")

        for C4, H, a in selected:
            ax.annotate(
                rf"$a={a:g}^\circ$",
                (C4, H),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel(r"$C_4$")
    ax.set_ylabel(r"$H$")
    ax.set_title("POD organization map")
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scatter_C4_H_by_angle.png"), dpi=DPI)
    plt.close(fig)


def plot_summary_maps(metrics_csv: str, out_dir: str):
    rows = read_metrics_csv(metrics_csv)
    if not rows:
        print("未找到汇总 CSV，跳过汇总图绘制。")
        return

    plot_metric_vs_angle(
        rows, "C2", out_dir,
        ylabel=r"$C_2$",
        title=r"Cumulative energy $C_2$",
    )
    plot_metric_vs_angle(
        rows, "C4", out_dir,
        ylabel=r"$C_4$",
        title=r"Cumulative energy $C_4$",
    )
    plot_metric_vs_angle(
        rows, "Delta12", out_dir,
        ylabel=r"$\Delta_{12}$",
        title=r"Pair imbalance $\Delta_{12}$",
    )
    plot_metric_vs_angle(
        rows, "H", out_dir,
        ylabel=r"$H$",
        title=r"Normalised modal entropy $H$",
    )
    plot_metric_vs_angle(
        rows, "N_eff", out_dir,
        ylabel=r"$N_{\mathrm{eff}}$",
        title=r"Effective number of modes $N_{\mathrm{eff}}$",
    )
    plot_organization_scatter(rows, out_dir)


# ============================================================
# 12. 单工况执行与主程序
# ============================================================

def run_one_case(case: CaseSpec) -> Optional[Dict[str, object]]:
    cname = case_name(case)
    case_dir = get_case_dir(case)
    out_dir = os.path.join(SAVE_ROOT, cname)
    ensure_dir(out_dir)

    f1 = case.k / np.pi
    T = 1.0 / f1

    print("=" * 72)
    print(f"工况: {cname}")
    print(f"case_dir = {case_dir}")
    cycle_start, cycle_end = get_cycle_range(case)
    print(f"周期范围: {cycle_start} 至 {cycle_end}; f1={f1:.8f}; T={T:.8f}")

    cycles_data = scan_cycles_data(case_dir, T, cycle_start, cycle_end)
    if not cycles_data:
        print(f"  跳过：未找到符合周期范围 {cycle_start}-{cycle_end} 的 dat 文件。")
        return None

    n_files = sum(len(cyc["records"]) for cyc in cycles_data)
    print(f"  找到 {n_files} 个快照，周期 {', '.join(str(c['i']) for c in cycles_data)}")
    for cyc in cycles_data:
        print(f"    cycle {cyc['i']}: {len(cyc['records'])} snapshots")

    X, Y, U, V, W, BODY, labels = load_snapshots_for_pod(cycles_data)
    fluid_mask = build_fluid_mask(BODY)
    print(f"  参与 POD 流体点: {int(np.sum(fluid_mask))}/{fluid_mask.size}")

    pod = compute_velocity_pod(U, V, fluid_mask, n_modes_save=N_MODES_SAVE)
    metrics = compute_pod_metrics(case, pod, labels, cycle_start, cycle_end)
    modes_w = compute_modal_vorticity(X, Y, pod["modes_u"], pod["modes_v"])

    save_all_case_outputs(X, Y, U, V, W, pod, modes_w, metrics, labels, out_dir, cname, reference_frequency=f1)

    plot_fluid_mask(X, Y, fluid_mask, out_dir, cname)
    plot_pod_energy(pod, out_dir, cname)
    plot_phase_portrait(pod, labels, out_dir, cname)
    plot_temporal_coefficients(pod, labels, out_dir, cname, nplot=N_COEFF_PLOT)
    plot_coefficient_spectrum(pod, labels, out_dir, cname, reference_frequency=f1, nplot=2)
    plot_mean_modes_and_examples(X, Y, W, pod, modes_w, labels, out_dir, cname)

    print("  关键指标:")
    for key in ["E1", "E2", "C2", "C4", "Delta12", "H", "N_eff", "wake_class_name"]:
        print(f"    {key}: {metrics.get(key)}")
    return metrics


def main():
    ensure_dir(SAVE_ROOT)
    cases = get_cases_to_run()

    print("=" * 72)
    print("非零后翼俯仰角工况 POD 批量分析")
    print(f"RUN_MODE = {RUN_MODE}")
    print(f"待处理工况数 = {len(cases)}")
    print(f"默认周期范围 = {DEFAULT_CYCLE_START} 至 {DEFAULT_CYCLE_END}")
    print(f"按 k 周期配置 = {CYCLE_BY_K}")
    print(f"按工况周期覆盖 = {CYCLE_BY_CASE}")
    print(f"输出目录 = {SAVE_ROOT}")
    print("=" * 72)

    metrics_all: List[Dict[str, object]] = []
    failed_cases: List[str] = []

    for case in cases:
        try:
            metrics = run_one_case(case)
            if metrics is not None:
                metrics_all.append(metrics)
        except Exception as e:
            failed_cases.append(case_name(case))
            print(f"  工况失败: {case_name(case)}")
            print(f"  错误信息: {repr(e)}")

    metrics_csv = os.path.join(SAVE_ROOT, "all_cases_pod_metrics_summary.csv")
    write_dict_csv(metrics_csv, metrics_all)
    print("=" * 72)
    print(f"汇总指标已保存: {metrics_csv}")

    if failed_cases:
        failed_txt = os.path.join(SAVE_ROOT, "failed_cases.txt")
        with open(failed_txt, "w", encoding="utf-8") as f:
            for name in failed_cases:
                f.write(name + "\n")
        print(f"失败工况已记录: {failed_txt}")

    plot_summary_maps(metrics_csv, SAVE_ROOT)
    print("POD 指标随俯仰角变化的汇总图已生成。")
    print("完成。")


if __name__ == "__main__":
    main()