import os
import re
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps


# ============================================================
# 0. 基本设置
# ============================================================

# ---------- 工况参数 ----------
u_inf = 1.0
c = 1.0
dc = 1
k = 1.1
st = 0.3
a = 0         # 可改为 0, 15, 16, 17, 20 等
case_name = f'k{k}st{st}a{a}'

# ---------- 路径设置 ----------
base_root = r'E:/Rensq/DC_explore_F/data/'
case_dir = base_root + f'k{k}/st{st}k{k}/st{st}k{k}a{a}/st{st}k{k}a{a}dc{int(dc)}/k{k}st{st}/'

save_root = r'E:/Rensq/DC_explore_F/POD_results/POD_results_test/' + case_name
os.makedirs(save_root, exist_ok=True)

# ---------- 读取周期范围 ----------
cycle_start = 13
cycle_end = 16

# ---------- POD 空间区域 ----------
# 若主要分析尾迹组织，建议使用后翼下游区域，避免翼扫掠区域造成大面积空白。
# 后翼 x2=1.5c 时，可优先使用 x_min_pod=1.2 或 1.8。
# 若仍想沿用原来的全近场区域，可改回 x_min_pod=-1.0。
x_min_pod, x_max_pod = -1, 8.0
y_min_pod, y_max_pod = -3.0, 3.0

# ---------- 空间降采样 ----------
stride_x = 2
stride_y = 2

# ---------- 是否剔除 body 区域 ----------
# True 表示剔除所有快照中曾经被翼占据过的网格点。
# 对运动边界 POD，正式结果建议保持 True。
use_body_mask = True
body_threshold = 0.5

# ---------- POD 设置 ----------
n_modes_save = 12
n_modes_plot = 6

# ---------- CSV 设置 ----------
# single_case_csv：保存当前工况的指标；
# all_cases_csv：追加保存到总表，便于后续画 k-St 图谱。
save_case_csv = True
append_global_metrics_csv = True
global_metrics_csv = r'E:/Rensq/DC_explore_F/POD_results_continue/all_cases_pod_metrics_summary.csv'

# ---------- 绘图设置 ----------
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": False,
})

dpi = 300

# 涡量图配色，沿用白化中心色表
vmin_vort, vmax_vort = -30, 30
numb = 512
base_cmap = colormaps['RdBu_r'].resampled(numb)
new_colors = base_cmap(np.linspace(0, 1, numb))
sigma = 1.0
vals = np.linspace(vmin_vort, vmax_vort, numb)
weights = np.exp(-0.5 * (vals / sigma) ** 2)

for i in range(numb):
    w = weights[i]
    new_colors[i] = (1 - w) * new_colors[i] + w * np.array([1, 1, 1, 1])

cmap_white = mcolors.ListedColormap(new_colors)


# ============================================================
# 1. Tecplot dat 文件读取函数
# ============================================================

def read_tecplot_ordered_dat(filename, target_vars=('X', 'Y', 'U', 'V', 'Vorticity', 'body')):
    """
    读取 Tecplot Ordered ASCII .dat 文件。

    文件变量顺序为：
    VARIABLES = "X" "Y" "U" "V" "Presure" "Vorticity" "body" "dvordt"

    返回：
        result[var] shape = (J, I)
    """
    var_map = {
        'X': 0,
        'Y': 1,
        'U': 2,
        'V': 3,
        'Presure': 4,
        'Pressure': 4,
        'Vorticity': 5,
        'body': 6,
        'dvordt': 7
    }

    cols = [var_map[v] for v in target_vars]
    need_cols = max(cols) + 1

    I = None
    J = None
    sol_time = None
    data_start_idx = None
    zone_found = False

    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        if (not zone_found) and stripped.upper().startswith('ZONE'):
            zone_found = True

            i_m = re.search(r'I\s*=\s*(\d+)', stripped, re.I)
            j_m = re.search(r'J\s*=\s*(\d+)', stripped, re.I)

            if i_m:
                I = int(i_m.group(1))
            if j_m:
                J = int(j_m.group(1))
            continue

        if zone_found:
            if sol_time is None:
                t_m = re.match(r'^\s*([\d.Ee+\-]+)\s*$', stripped)
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
        raise ValueError(f'无法解析 dat 文件头信息: {filename}')

    data = np.loadtxt(
        filename,
        skiprows=data_start_idx - 1,
        max_rows=I * J,
        usecols=cols,
        delimiter=None,
        dtype=np.float32
    )

    if data.shape[0] != I * J:
        raise ValueError(f'{filename} 数据点数量不足，期望 {I * J}，实际 {data.shape[0]}')

    result = {'solution_time': sol_time, 'I': I, 'J': J}

    for i, var in enumerate(target_vars):
        result[var] = data[:, i].reshape(J, I)

    return result


# ============================================================
# 2. 文件名解析、周期识别
# ============================================================

def parse_time_from_filename(filename):
    """
    支持：
        F165000.dat
        F 165000.dat
        F    165000.dat

    每 10000 步 = 1 无量纲时间。
    """
    base = os.path.basename(filename)
    m = re.match(r'^F\s*(\d+)\.dat$', base, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)) / 10000.0


def is_valid_flow_filename(filename):
    base = os.path.basename(filename)
    return re.match(r'^F\s*\d+\.dat$', base, re.IGNORECASE) is not None


def compute_cycle_info(t, T, eps=1e-10):
    """
    返回：
        cycle_no: 第几个周期，从 1 开始；
        phase_frac: 周期内相位，范围 [0, 1)。
    """
    cycle_no = int(np.floor(t / T + eps)) + 1
    phase_frac = (t / T) - np.floor(t / T + eps)

    if abs(phase_frac) < 1e-8:
        phase_frac = 0.0
    elif abs(phase_frac - 1.0) < 1e-8:
        phase_frac = 0.0
        cycle_no += 1

    return cycle_no, phase_frac


def scan_cycles_data(case_dir, T, cycle_start=None, cycle_end=None):
    """
    扫描 case_dir 下所有 F*.dat 文件，按周期和相位排序。
    """
    all_files = []

    for fname in os.listdir(case_dir):
        if is_valid_flow_filename(fname):
            t = parse_time_from_filename(fname)
            if t is not None:
                cycle_no, phase_frac = compute_cycle_info(t, T)

                if cycle_start is not None and cycle_no < cycle_start:
                    continue
                if cycle_end is not None and cycle_no > cycle_end:
                    continue

                all_files.append({
                    'filename': fname,
                    'path': os.path.join(case_dir, fname),
                    't': t,
                    'cycle_no': cycle_no,
                    'phase_frac': phase_frac
                })

    if not all_files:
        return []

    all_files.sort(key=lambda x: x['t'])

    grouped = {}
    for item in all_files:
        grouped.setdefault(item['cycle_no'], []).append(item)

    cycles_data = []
    for cyc in sorted(grouped.keys()):
        records = sorted(grouped[cyc], key=lambda x: x['phase_frac'])
        cycles_data.append({
            'i': cyc,
            'records': records
        })

    return cycles_data


# ============================================================
# 3. 构建 POD 数据集
# ============================================================

def find_crop_indices(X, Y, x_min, x_max, y_min, y_max, stride_x=1, stride_y=1):
    """
    根据二维 X,Y 找到裁剪区域的行列索引。
    """
    mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)

    if not np.any(mask):
        raise ValueError('POD 裁剪区域与流场坐标没有交集，请检查 x/y 范围。')

    row_any = np.any(mask, axis=1)
    col_any = np.any(mask, axis=0)

    rows = np.where(row_any)[0][::stride_y]
    cols = np.where(col_any)[0][::stride_x]

    return rows, cols


def load_snapshots_for_pod(cycles_data):
    """
    读取所有快照，裁剪并降采样。

    返回：
        Xc, Yc: 裁剪后的坐标，shape=(ny, nx)
        U, V, W, BODY: shape=(ns, ny, nx)
        labels: 每个快照的周期、相位、时间、文件名
    """
    records = []
    for cyc in cycles_data:
        for rec in cyc['records']:
            records.append(rec)

    if len(records) == 0:
        raise ValueError('没有可用于 POD 的流场文件。')

    U_list = []
    V_list = []
    W_list = []
    B_list = []

    labels = []

    rows = None
    cols = None
    Xc = None
    Yc = None

    for idx, rec in enumerate(records):
        print(f'[{idx+1:04d}/{len(records):04d}] 读取: {rec["filename"]}')

        dat = read_tecplot_ordered_dat(
            rec['path'],
            target_vars=('X', 'Y', 'U', 'V', 'Vorticity', 'body')
        )

        X = dat['X']
        Y = dat['Y']
        U0 = dat['U']
        V0 = dat['V']
        W0 = dat['Vorticity']
        B0 = dat['body']

        if rows is None:
            rows, cols = find_crop_indices(
                X, Y,
                x_min_pod, x_max_pod,
                y_min_pod, y_max_pod,
                stride_x=stride_x,
                stride_y=stride_y
            )

            Xc = X[np.ix_(rows, cols)].astype(np.float32)
            Yc = Y[np.ix_(rows, cols)].astype(np.float32)

            print(f'POD 裁剪后网格尺寸: ny={Xc.shape[0]}, nx={Xc.shape[1]}')
            print(f'POD 快照变量维度约为: 2 * ny * nx = {2 * Xc.size}')

        U_list.append(U0[np.ix_(rows, cols)].astype(np.float32))
        V_list.append(V0[np.ix_(rows, cols)].astype(np.float32))
        W_list.append(W0[np.ix_(rows, cols)].astype(np.float32))
        B_list.append(B0[np.ix_(rows, cols)].astype(np.float32))

        labels.append({
            'snapshot_index': idx,
            'filename': rec['filename'],
            't_filename': rec['t'],
            'solution_time': dat['solution_time'],
            'cycle_no': rec['cycle_no'],
            'phase_frac': rec['phase_frac']
        })

    U = np.stack(U_list, axis=0)
    V = np.stack(V_list, axis=0)
    W = np.stack(W_list, axis=0)
    BODY = np.stack(B_list, axis=0)

    return Xc, Yc, U, V, W, BODY, labels


# ============================================================
# 4. 速度场 POD
# ============================================================

def build_fluid_mask(BODY):
    """
    构建固定空间掩膜。

    对运动翼 POD，每个快照必须使用相同空间点。
    如果 use_body_mask=True，则剔除所有快照中曾经进入 body 的网格点。
    """
    if not use_body_mask:
        fluid_mask = np.ones(BODY.shape[1:], dtype=bool)
        print('未剔除 body 区域。')
        return fluid_mask

    body_union = np.any(BODY > body_threshold, axis=0)
    fluid_mask = ~body_union

    print(f'总网格点数: {fluid_mask.size}')
    print(f'参与 POD 的流体点数: {np.sum(fluid_mask)}')
    print(f'剔除 body/运动覆盖点数: {np.sum(~fluid_mask)}')
    print(f'剔除比例 masked_fraction = {np.sum(~fluid_mask) / fluid_mask.size:.6f}')

    if np.sum(fluid_mask) == 0:
        raise ValueError('fluid_mask 中没有可用流体点，请检查 body 字段或 POD 区域。')

    return fluid_mask


def compute_velocity_pod(U, V, fluid_mask, n_modes_save=12):
    """
    对速度脉动场 (u', v') 做 POD。

    U, V:
        shape = (ns, ny, nx)

    fluid_mask:
        shape = (ny, nx)
    """
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

    # 数值上再去除每个空间自由度的均值残差
    Xmat -= np.mean(Xmat, axis=1, keepdims=True)

    print(f'POD 矩阵维度: {Xmat.shape[0]} × {Xmat.shape[1]}')

    if ns < 2:
        raise ValueError('POD 至少需要 2 个快照。')

    U_svd, S, Vt = np.linalg.svd(Xmat, full_matrices=False)

    eigvals = S ** 2 / (ns - 1)
    total_energy = np.sum(eigvals)

    if total_energy <= 0:
        raise ValueError('POD 总能量为 0，请检查速度场是否全为常数或输入数据是否异常。')

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

    pod = {
        'eigvals': eigvals,
        'energy': energy,
        'cumulative': cumulative,
        'coeffs': coeffs,
        'modes_u': modes_u,
        'modes_v': modes_v,
        'mean_u': U_mean,
        'mean_v': V_mean,
        'fluid_mask': fluid_mask,
        'singular_values': S
    }

    return pod


def compute_pod_metrics(pod, labels=None):
    """
    计算 POD 标量指标。

    命名约定：
        E1, E2, ... 为单阶模态能量占比；
        C1, C2, ... 为前 m 阶累计能量；
        Delta12 为前两阶特征值差异；
        H 为归一化谱熵。
    """
    eigvals = pod['eigvals']
    energy = pod['energy']
    cumulative = pod['cumulative']
    singular_values = pod['singular_values']
    fluid_mask = pod['fluid_mask']

    n_modes = len(energy)
    nm_saved = pod['coeffs'].shape[0]
    ns = pod['coeffs'].shape[1]

    metrics = {
        'case_name': case_name,
        'k': float(k),
        'St': float(st),
        'a': float(a),
        'dc': float(dc),
        'cycle_start': int(cycle_start),
        'cycle_end': int(cycle_end),
        'x_min_pod': float(x_min_pod),
        'x_max_pod': float(x_max_pod),
        'y_min_pod': float(y_min_pod),
        'y_max_pod': float(y_max_pod),
        'stride_x': int(stride_x),
        'stride_y': int(stride_y),
        'use_body_mask': int(use_body_mask),
        'body_threshold': float(body_threshold),
        'n_modes_total': int(n_modes),
        'n_modes_saved': int(nm_saved),
        'n_snapshots': int(ns),
        'n_grid_total': int(fluid_mask.size),
        'n_fluid_points': int(np.sum(fluid_mask)),
        'n_masked_points': int(np.sum(~fluid_mask)),
        'masked_fraction': float(np.sum(~fluid_mask) / fluid_mask.size)
    }

    if labels is not None and len(labels) > 0:
        metrics['t_start'] = float(labels[0]['t_filename'])
        metrics['t_end'] = float(labels[-1]['t_filename'])
        metrics['phase_start'] = float(labels[0]['phase_frac'])
        metrics['phase_end'] = float(labels[-1]['phase_frac'])

    n_report = min(12, n_modes)

    for i in range(n_report):
        mode_id = i + 1
        metrics[f'lambda{mode_id}'] = float(eigvals[i])
        metrics[f'sigma{mode_id}'] = float(singular_values[i])
        metrics[f'E{mode_id}'] = float(energy[i])
        metrics[f'C{mode_id}'] = float(cumulative[i])

    for m in [2, 4, 6, 8, 10, 12]:
        if n_modes >= m:
            metrics[f'C{m}'] = float(cumulative[m - 1])
        else:
            metrics[f'C{m}'] = None

    if n_modes >= 2:
        metrics['Delta12'] = float(abs(eigvals[0] - eigvals[1]) / (eigvals[0] + eigvals[1]))
        metrics['E12_sum'] = float(energy[0] + energy[1])
        metrics['E1_over_E2'] = float(energy[0] / energy[1]) if energy[1] > 0 else None
    else:
        metrics['Delta12'] = None
        metrics['E12_sum'] = None
        metrics['E1_over_E2'] = None

    p = eigvals / np.sum(eigvals)
    p = p[p > 0]

    if len(p) > 1 and len(eigvals) > 1:
        metrics['H'] = float(-np.sum(p * np.log(p)) / np.log(len(eigvals)))
    else:
        metrics['H'] = 0.0

    return metrics


# ============================================================
# 5. 模态涡量、重构涡量
# ============================================================

def estimate_dx_dy(X, Y):
    dx = float(np.nanmean(np.diff(X, axis=1)))
    dy = float(np.nanmean(np.diff(Y, axis=0)))
    return abs(dx), abs(dy)


def fill_nan_for_gradient(A):
    """
    为了计算梯度，将 NaN 临时填 0。
    只用于计算模态涡量。
    """
    B = np.array(A, copy=True)
    B[~np.isfinite(B)] = 0.0
    return B


def compute_modal_vorticity(X, Y, modes_u, modes_v):
    """
    由速度 POD 模态计算对应的模态涡量。
    注意：这不是直接对涡量场做 POD。
    """
    dx, dy = estimate_dx_dy(X, Y)

    nm, ny, nx = modes_u.shape
    modes_w = np.full_like(modes_u, np.nan, dtype=np.float32)

    for m in range(nm):
        u_m = fill_nan_for_gradient(modes_u[m])
        v_m = fill_nan_for_gradient(modes_v[m])

        du_dy = np.gradient(u_m, dy, axis=0)
        dv_dx = np.gradient(v_m, dx, axis=1)

        modes_w[m] = dv_dx - du_dy

        valid_mask = np.isfinite(modes_u[m]) & np.isfinite(modes_v[m])
        modes_w[m][~valid_mask] = np.nan

    return modes_w


def reconstruct_velocity(pod, snapshot_id, mode_ids):
    """
    使用指定模态重构某个快照的速度场。
    mode_ids 为 0-based，例如 [0, 1] 表示第 1、2 模态。
    """
    mean_u = pod['mean_u']
    mean_v = pod['mean_v']
    modes_u = pod['modes_u']
    modes_v = pod['modes_v']
    coeffs = pod['coeffs']

    ur = np.array(mean_u, copy=True)
    vr = np.array(mean_v, copy=True)

    for m in mode_ids:
        if m >= coeffs.shape[0]:
            continue
        ur += coeffs[m, snapshot_id] * np.nan_to_num(modes_u[m], nan=0.0)
        vr += coeffs[m, snapshot_id] * np.nan_to_num(modes_v[m], nan=0.0)

    return ur, vr


def compute_vorticity_from_uv(X, Y, U, V):
    dx, dy = estimate_dx_dy(X, Y)
    du_dy = np.gradient(U, dy, axis=0)
    dv_dx = np.gradient(V, dx, axis=1)
    return dv_dx - du_dy


# ============================================================
# 6. 绘图函数
# ============================================================

def plot_contour(X, Y, F, fname, title='', vmin=None, vmax=None,
                 cmap=cmap_white, levels=128, equal=True):
    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    if vmin is None or vmax is None:
        vmax0 = np.nanpercentile(np.abs(F), 99.0)
        if vmax0 == 0 or not np.isfinite(vmax0):
            vmax0 = 1.0
        vmin0 = -vmax0
    else:
        vmin0 = vmin
        vmax0 = vmax

    cf = ax.contourf(
        X, Y, F,
        levels=np.linspace(vmin0, vmax0, levels),
        cmap=cmap,
        extend='both'
    )

    fig.colorbar(cf, ax=ax, shrink=0.85)
    ax.set_xlabel(r'$x/c$')
    ax.set_ylabel(r'$y/c$')
    ax.set_title(title)

    if equal:
        ax.set_aspect('equal', adjustable='box')

    ax.tick_params(direction='in', top=True, right=True)
    fig.tight_layout()
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)


def plot_pod_energy(pod, save_root):
    energy = pod['energy']
    cumulative = pod['cumulative']
    mode_id = np.arange(1, len(energy) + 1)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.semilogy(mode_id, energy, 'o-', ms=3, lw=1.1)
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Energy fraction')
    ax.set_title('POD energy spectrum')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_root, f'{case_name}_energy_spectrum.png'), dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(mode_id, cumulative, 'o-', ms=3, lw=1.1)
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Cumulative energy')
    ax.set_ylim(0, 1.02)
    ax.set_title('Cumulative POD energy')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_root, f'{case_name}_cumulative_energy.png'), dpi=dpi)
    plt.close(fig)


def plot_fluid_mask(X, Y, fluid_mask, save_root):
    """
    绘制参与 POD 的固定流体区域。
    白色/空白区域通常对应翼扫掠区域或固体区域。
    """
    F = np.where(fluid_mask, 1.0, 0.0)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    cf = ax.contourf(X, Y, F, levels=np.linspace(0, 1, 3), cmap='Greys')
    fig.colorbar(cf, ax=ax, shrink=0.85, ticks=[0, 1])
    ax.set_xlabel(r'$x/c$')
    ax.set_ylabel(r'$y/c$')
    ax.set_title('POD fluid mask')
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(direction='in', top=True, right=True)
    fig.tight_layout()
    fig.savefig(os.path.join(save_root, f'{case_name}_fluid_mask.png'), dpi=dpi)
    plt.close(fig)


def plot_phase_portrait(pod, labels, save_root):
    coeffs = pod['coeffs']
    if coeffs.shape[0] < 2:
        return

    a1 = coeffs[0]
    a2 = coeffs[1]

    fig, ax = plt.subplots(figsize=(4.8, 4.5))

    ax.plot(a1, a2, '-', lw=1.0, color='black', alpha=0.8)

    cycle_nos = np.array([lab['cycle_no'] for lab in labels])
    for cyc in sorted(np.unique(cycle_nos)):
        ids = np.where(cycle_nos == cyc)[0]
        ax.scatter(a1[ids], a2[ids], s=12, label=f'cycle {cyc}')

    ax.set_xlabel(r'$a_1(t)$')
    ax.set_ylabel(r'$a_2(t)$')
    ax.set_title(r'Phase portrait: $a_1-a_2$')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(save_root, f'{case_name}_a1_a2_phase_portrait.png'), dpi=dpi)
    plt.close(fig)


def plot_temporal_coefficients(pod, labels, save_root, nplot=4):
    coeffs = pod['coeffs']
    nplot = min(nplot, coeffs.shape[0])

    t = np.array([lab['t_filename'] for lab in labels])

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    for m in range(nplot):
        a_m = coeffs[m]
        denom = np.max(np.abs(a_m))
        if denom == 0:
            denom = 1.0
        a_norm = a_m / denom
        ax.plot(t, a_norm, lw=1.0, label=f'mode {m+1}')

    ax.set_xlabel('Time')
    ax.set_ylabel('Normalised coefficient')
    ax.set_title('POD temporal coefficients')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(save_root, f'{case_name}_temporal_coefficients.png'), dpi=dpi)
    plt.close(fig)


def plot_coefficient_spectrum_from_labels(pod, labels, save_root, nplot=2, reference_frequency=None):
    coeffs = pod['coeffs']
    nplot = min(nplot, coeffs.shape[0])

    t = np.array([lab['t_filename'] for lab in labels], dtype=float)
    dt = np.mean(np.diff(t))

    freqs = np.fft.rfftfreq(coeffs.shape[1], d=dt)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    for m in range(nplot):
        a_m = coeffs[m] - np.mean(coeffs[m])
        spec = np.abs(np.fft.rfft(a_m)) ** 2
        if np.max(spec) > 0:
            spec = spec / np.max(spec)
        ax.plot(freqs, spec, lw=1.2, label=f'mode {m+1}')

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Normalised spectral energy')
    ax.set_title('Spectrum of POD coefficients')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(save_root, f'{case_name}_coefficient_spectrum.png'), dpi=dpi)
    plt.close(fig)

    if reference_frequency is not None and reference_frequency > 0:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))

        for m in range(nplot):
            a_m = coeffs[m] - np.mean(coeffs[m])
            spec = np.abs(np.fft.rfft(a_m)) ** 2
            if np.max(spec) > 0:
                spec = spec / np.max(spec)
            ax.plot(freqs / reference_frequency, spec, lw=1.2, label=f'mode {m+1}')

        ax.set_xlabel(r'$f/f_1$')
        ax.set_ylabel('Normalised spectral energy')
        ax.set_title('Spectrum of POD coefficients')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(save_root, f'{case_name}_coefficient_spectrum_f_over_f1.png'), dpi=dpi)
        plt.close(fig)


def plot_mean_and_modes(X, Y, W, modes_w, save_root):
    W_mean = np.mean(W, axis=0)

    plot_contour(
        X, Y, W_mean,
        os.path.join(save_root, f'{case_name}_mean_vorticity.png'),
        title='Mean vorticity',
        vmin=vmin_vort, vmax=vmax_vort
    )

    nm = min(n_modes_plot, modes_w.shape[0])
    for m in range(nm):
        vmax_m = np.nanpercentile(np.abs(modes_w[m]), 99.0)
        if vmax_m == 0 or not np.isfinite(vmax_m):
            vmax_m = 1.0

        plot_contour(
            X, Y, modes_w[m],
            os.path.join(save_root, f'{case_name}_mode{m+1:02d}_modal_vorticity.png'),
            title=f'Modal vorticity, mode {m+1}',
            vmin=-vmax_m,
            vmax=vmax_m
        )


def plot_reconstruction_examples(X, Y, pod, labels, save_root):
    """
    用第 1、2 模态重构几个典型相位。
    """
    phase = np.array([lab['phase_frac'] for lab in labels])
    cycle = np.array([lab['cycle_no'] for lab in labels])
    first_cycle = np.min(cycle)

    ids_in_first = np.where(cycle == first_cycle)[0]

    target_phases = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    selected = []

    for ph in target_phases:
        sid = ids_in_first[np.argmin(np.abs(phase[ids_in_first] - ph))]
        selected.append(sid)

    selected = sorted(set(selected))

    for sid in selected:
        ur, vr = reconstruct_velocity(pod, sid, mode_ids=[0, 1])
        wr = compute_vorticity_from_uv(X, Y, ur, vr)
        wr[~pod['fluid_mask']] = np.nan

        plot_contour(
            X, Y, wr,
            os.path.join(save_root, f'{case_name}_recon_mode12_snap{sid:04d}.png'),
            title=f'Reconstructed vorticity by modes 1-2, snap {sid}, phase={phase[sid]:.3f}T',
            vmin=vmin_vort,
            vmax=vmax_vort
        )


def plot_instantaneous_examples(X, Y, W, labels, save_root):
    """
    绘制第一个保存周期内的若干瞬时涡量场。
    """
    cycle = np.array([lab['cycle_no'] for lab in labels])
    phase = np.array([lab['phase_frac'] for lab in labels])
    first_cycle = np.min(cycle)

    ids_in_first = np.where(cycle == first_cycle)[0]

    target_phases = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    selected = []

    for ph in target_phases:
        sid = ids_in_first[np.argmin(np.abs(phase[ids_in_first] - ph))]
        selected.append(sid)

    selected = sorted(set(selected))

    for sid in selected:
        plot_contour(
            X, Y, W[sid],
            os.path.join(save_root, f'{case_name}_instant_vorticity_snap{sid:04d}.png'),
            title=f'Instantaneous vorticity, snap {sid}, phase={phase[sid]:.3f}T',
            vmin=vmin_vort,
            vmax=vmax_vort
        )


# ============================================================
# 7. CSV 与数据保存函数
# ============================================================

def none_to_blank(x):
    if x is None:
        return ''
    return x


def save_labels(labels, save_root):
    csv_path = os.path.join(save_root, f'{case_name}_snapshot_labels.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['snapshot_index', 'filename', 't_filename', 'solution_time', 'cycle_no', 'phase_frac'])

        for lab in labels:
            writer.writerow([
                lab['snapshot_index'],
                lab['filename'],
                lab['t_filename'],
                lab['solution_time'],
                lab['cycle_no'],
                lab['phase_frac']
            ])

    print(f'快照标签已保存: {csv_path}')


def save_one_row_metrics_csv(metrics, save_root):
    """
    保存当前工况的 POD 标量指标。
    """
    csv_path = os.path.join(save_root, f'{case_name}_pod_metrics_summary.csv')

    fieldnames = list(metrics.keys())

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: none_to_blank(v) for k, v in metrics.items()})

    print(f'POD 标量指标 CSV 已保存: {csv_path}')


def append_metrics_to_global_csv(metrics, csv_path):
    """
    将当前工况指标追加保存到全局汇总表。
    多个工况循环运行时，可直接得到 all_cases_pod_metrics_summary.csv。
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = list(metrics.keys())
    file_exists = os.path.exists(csv_path)

    need_write_header = True
    if file_exists and os.path.getsize(csv_path) > 0:
        need_write_header = False

    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if need_write_header:
            writer.writeheader()

        writer.writerow({k: none_to_blank(v) for k, v in metrics.items()})

    print(f'POD 指标已追加到全局汇总表: {csv_path}')


def save_energy_csv(pod, save_root):
    """
    保存每一阶模态的特征值、奇异值、单阶能量和累计能量。
    """
    csv_path = os.path.join(save_root, f'{case_name}_pod_energy.csv')

    eigvals = pod['eigvals']
    energy = pod['energy']
    cumulative = pod['cumulative']
    singular_values = pod['singular_values']

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'lambda', 'singular_value', 'energy_fraction_Ei', 'cumulative_energy_Ci'])

        for i in range(len(energy)):
            writer.writerow([
                i + 1,
                float(eigvals[i]),
                float(singular_values[i]),
                float(energy[i]),
                float(cumulative[i])
            ])

    print(f'POD 能量谱 CSV 已保存: {csv_path}')


def save_coefficients_csv(pod, labels, save_root):
    """
    保存 POD 时间系数。
    每行对应一个快照，每列对应一个模态系数 a_i(t)。
    """
    csv_path = os.path.join(save_root, f'{case_name}_pod_coefficients.csv')

    coeffs = pod['coeffs']
    nm, ns = coeffs.shape

    header = [
        'snapshot_index',
        'filename',
        't_filename',
        'solution_time',
        'cycle_no',
        'phase_frac'
    ] + [f'a{i+1}' for i in range(nm)]

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for s, lab in enumerate(labels):
            row = [
                lab['snapshot_index'],
                lab['filename'],
                lab['t_filename'],
                lab['solution_time'],
                lab['cycle_no'],
                lab['phase_frac']
            ] + [float(coeffs[i, s]) for i in range(nm)]

            writer.writerow(row)

    print(f'POD 时间系数 CSV 已保存: {csv_path}')


def save_coefficient_spectrum_csv(pod, labels, save_root, n_modes=6, reference_frequency=None):
    """
    保存 POD 时间系数频谱。
    若 reference_frequency=f1，则额外保存 frequency_over_f1。
    """
    csv_path = os.path.join(save_root, f'{case_name}_pod_coefficient_spectrum.csv')

    coeffs = pod['coeffs']
    nm, ns = coeffs.shape
    n_modes = min(n_modes, nm)

    t = np.array([lab['t_filename'] for lab in labels], dtype=float)

    if len(t) < 2:
        print('快照数量不足，跳过频谱 CSV 保存。')
        return

    dt = float(np.mean(np.diff(t)))
    freqs = np.fft.rfftfreq(ns, d=dt)

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'frequency', 'frequency_over_f1', 'normalised_spectral_energy'])

        for m in range(n_modes):
            a_m = coeffs[m] - np.mean(coeffs[m])
            spec = np.abs(np.fft.rfft(a_m)) ** 2

            if np.max(spec) > 0:
                spec = spec / np.max(spec)

            for j in range(len(freqs)):
                if reference_frequency is not None and reference_frequency > 0:
                    freq_over_f1 = freqs[j] / reference_frequency
                else:
                    freq_over_f1 = None

                writer.writerow([
                    m + 1,
                    float(freqs[j]),
                    none_to_blank(None if freq_over_f1 is None else float(freq_over_f1)),
                    float(spec[j])
                ])

    print(f'POD 时间系数频谱 CSV 已保存: {csv_path}')


def save_mask_csv(X, Y, fluid_mask, save_root):
    """
    保存 POD 掩膜，便于检查翼运动扫掠区域。
    每行一个网格点。
    """
    csv_path = os.path.join(save_root, f'{case_name}_pod_fluid_mask.csv')

    ny, nx = fluid_mask.shape

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['j', 'i', 'x', 'y', 'fluid_mask'])

        for j in range(ny):
            for i in range(nx):
                writer.writerow([
                    j,
                    i,
                    float(X[j, i]),
                    float(Y[j, i]),
                    int(fluid_mask[j, i])
                ])

    print(f'POD 掩膜 CSV 已保存: {csv_path}')


def save_pod_csv_tables(pod, metrics, labels, X, Y, save_root, reference_frequency=None):
    """
    统一保存所有适合后处理和论文绘图的 CSV 表。
    """
    if save_case_csv:
        save_one_row_metrics_csv(metrics, save_root)
        save_energy_csv(pod, save_root)
        save_coefficients_csv(pod, labels, save_root)
        save_coefficient_spectrum_csv(
            pod,
            labels,
            save_root,
            n_modes=min(6, pod['coeffs'].shape[0]),
            reference_frequency=reference_frequency
        )
        save_mask_csv(X, Y, pod['fluid_mask'], save_root)

    if append_global_metrics_csv:
        append_metrics_to_global_csv(metrics, global_metrics_csv)


def save_pod_results(X, Y, U, V, W, BODY, pod, modes_w, metrics, labels, save_root, reference_frequency=None):
    """
    保存 npz、json、labels 和 CSV。
    """
    npz_path = os.path.join(save_root, f'{case_name}_pod_results.npz')

    np.savez_compressed(
        npz_path,
        X=X,
        Y=Y,
        U_mean=pod['mean_u'],
        V_mean=pod['mean_v'],
        W_mean=np.mean(W, axis=0),
        eigvals=pod['eigvals'],
        energy=pod['energy'],
        cumulative=pod['cumulative'],
        coeffs=pod['coeffs'],
        modes_u=pod['modes_u'],
        modes_v=pod['modes_v'],
        modes_w=modes_w,
        fluid_mask=pod['fluid_mask']
    )

    print(f'POD 数组结果已保存: {npz_path}')

    json_path = os.path.join(save_root, f'{case_name}_pod_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f'POD 指标 JSON 已保存: {json_path}')

    save_labels(labels, save_root)

    save_pod_csv_tables(
        pod,
        metrics,
        labels,
        X,
        Y,
        save_root,
        reference_frequency=reference_frequency
    )


# ============================================================
# 8. 主程序
# ============================================================

def main():
    f1 = k / np.pi
    T = 1.0 / f1

    print('============================================================')
    print(f'POD analysis for k={k}, St={st}, a={a}, dc={dc}')
    print(f'f1 = {f1:.10f}')
    print(f'T  = {T:.10f}')
    print(f'case_dir  = {case_dir}')
    print(f'save_root = {save_root}')
    print('============================================================')

    cycles_data = scan_cycles_data(
        case_dir,
        T,
        cycle_start=cycle_start,
        cycle_end=cycle_end
    )

    if not cycles_data:
        raise RuntimeError('未找到符合要求的 dat 文件，请检查 case_dir、文件名和周期范围。')

    n_files = sum(len(cyc['records']) for cyc in cycles_data)
    print(f'扫描到 {n_files} 个 dat 文件')
    print(f'识别到周期: {[cyc["i"] for cyc in cycles_data]}')

    for cyc in cycles_data:
        print(f'  cycle {cyc["i"]}: {len(cyc["records"])} snapshots')

    # 读取并裁剪快照
    X, Y, U, V, W, BODY, labels = load_snapshots_for_pod(cycles_data)

    # 建立固定 fluid mask
    fluid_mask = build_fluid_mask(BODY)

    # 绘制参与 POD 的固定流体区域
    plot_fluid_mask(X, Y, fluid_mask, save_root)

    # 速度场 POD
    pod = compute_velocity_pod(U, V, fluid_mask, n_modes_save=n_modes_save)

    # POD 指标
    metrics = compute_pod_metrics(pod, labels=labels)

    print('================ POD metrics ================')
    important_keys = [
        'E1', 'E2', 'C2', 'C4', 'C6',
        'Delta12', 'H',
        'masked_fraction',
        'n_snapshots',
        'n_fluid_points'
    ]

    for key in important_keys:
        if key in metrics:
            print(f'{key}: {metrics[key]}')

    # 模态涡量
    modes_w = compute_modal_vorticity(X, Y, pod['modes_u'], pod['modes_v'])

    # 保存数据
    save_pod_results(
        X,
        Y,
        U,
        V,
        W,
        BODY,
        pod,
        modes_w,
        metrics,
        labels,
        save_root,
        reference_frequency=f1
    )

    # 绘图
    plot_pod_energy(pod, save_root)
    plot_phase_portrait(pod, labels, save_root)
    plot_temporal_coefficients(pod, labels, save_root, nplot=4)
    plot_coefficient_spectrum_from_labels(
        pod,
        labels,
        save_root,
        nplot=2,
        reference_frequency=f1
    )
    plot_mean_and_modes(X, Y, W, modes_w, save_root)
    plot_instantaneous_examples(X, Y, W, labels, save_root)
    plot_reconstruction_examples(X, Y, pod, labels, save_root)

    print('============================================================')
    print('POD 分析完成。')
    print('主要输出包括：')
    print('1. POD 能谱和累计能量图')
    print('2. E1, E2, C2, C4, C6, Delta12, H 等 CSV 指标')
    print('3. a1-a2 相图')
    print('4. POD 时间系数和频谱')
    print('5. 平均涡量场、模态涡量场')
    print('6. 第 1-2 模态重构涡量场')
    print('7. fluid_mask 图和 CSV，用于检查翼运动扫掠区域')
    print('============================================================')


if __name__ == '__main__':
    main()