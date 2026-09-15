import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
import os
import re
import math

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{newtxtext,newtxmath}"
})

# ==================== 参数设置 ====================
u = 1.0
rou = 1.0
c = 1.0

a = 20

# 25, 50, 75 分别对应 DC=0.25, 0.50, 0.75
dc_list = [25, 50, 75]

# 平滑系数，需要与 motion_params.dat 生成代码保持一致
ks = 9

k_list = [0.8]
st_list = [0.3]

base_root = r'E:/Rensq/DC_explore_F/data/'
final_save_root = r'E:/Rensq/DC_explore_F/flow/vortex_results_intermittent_new'
os.makedirs(final_save_root, exist_ok=True)

# ==================== 固定相位槽设置 ====================
# 每个周期固定 8 个流场时刻：
# 1/8T, 2/8T, 3/8T, 4/8T, 5/8T, 6/8T, 7/8T, 8/8T
n_phase_per_cycle = 8

# 周期起算时间。
# 若流场文件时间从 t=0 开始，保持 0.0。
# 若前面有静止时间 static_time，并且希望从开始扑动时刻算周期，则设为 static_time。
t_ref = 0.0

# 只绘制前 4 个识别到的周期。
# 如果希望绘制全部周期，可改为 None。
max_cycles_to_plot = 4

# 文件名时间通常只有 4 位小数，T 可能是无理数，因此允许一定相位匹配误差。
slot_tol = 5e-3

# ==================== 翼型轮廓 ====================
profile = np.loadtxt(r'E:/Rensq/DC_explore_F/flow/boundary.dat')
x_prof = profile[:, 0] * c
y_prof = profile[:, 1] * c

# ==================== 新流场中心与绘图区域 ====================
flow_center_x = 0.0
flow_center_y = 0.0

x_min, x_max = -1.0, 20.0
y_min, y_max = -10, 10

wing_shift_x = flow_center_x
wing_shift_y = flow_center_y

# ==================== 坐标轴刻度设置 ====================
x_tick_step = 3
x_ticks = np.arange(int(np.ceil(x_min)), int(np.floor(x_max)) + 1, x_tick_step)

# ==================== 颜色表设置 ====================
numb = 512
base_cmap = colormaps['RdBu_r'].resampled(numb)
new_colors = base_cmap(np.linspace(0, 1, numb))

vmin, vmax = -30, 30
sigma = 1.0

vals = np.linspace(vmin, vmax, numb)
weights = np.exp(-0.5 * (vals / sigma) ** 2)

for i in range(numb):
    w = weights[i]
    new_colors[i] = (1 - w) * new_colors[i] + w * np.array([1, 1, 1, 1])

cmap_white = mcolors.ListedColormap(new_colors)
levels = np.linspace(vmin, vmax, numb + 1)


# ==================== Tecplot dat 文件读取函数 ====================
def read_tecplot_ordered_dat(filename, target_vars=('X', 'Y', 'Vorticity')):
    """
    读取 Tecplot Ordered ASCII .dat 文件。
    """
    var_map = {
        'X': 0,
        'Y': 1,
        'U': 2,
        'V': 3,
        'Presure': 4,
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
        delimiter=None
    )

    if data.shape[0] != I * J:
        raise ValueError(f'{filename} 数据点数量不足，期望 {I * J}，实际 {data.shape[0]}')

    result = {'solution_time': sol_time, 'I': I, 'J': J}

    for i, var in enumerate(target_vars):
        result[var] = data[:, i].reshape(J, I)

    return result


# ==================== 从文件名解析时间 ====================
def parse_time_from_filename(filename):
    """
    同时支持：
    F165000.dat
    F 165000.dat
    F    165000.dat

    当前规则：
    F165000.dat -> 16.5000
    """
    base = os.path.basename(filename)
    m = re.match(r'^F\s*(\d+)\.dat$', base, re.IGNORECASE)

    if not m:
        return None

    return int(m.group(1)) / 10000.0


# ==================== 判断是否为合法流场文件名 ====================
def is_valid_flow_filename(filename):
    """
    同时支持：
    F165000.dat
    F 165000.dat
    F    165000.dat
    """
    base = os.path.basename(filename)
    return re.match(r'^F\s*\d+\.dat$', base, re.IGNORECASE) is not None


# ==================== 固定 8 相位槽周期分组 ====================
def compute_cycle_info_fixed_slots(t, T, n_phase=8, t_ref=0.0, tol=5e-3):
    """
    用固定相位槽对流场文件进行周期分组。

    每周期固定 n_phase 个相位槽：
        phase_idx = 1, 2, ..., n_phase

    对 n_phase=8：
        1 -> 1/8T
        2 -> 1/4T
        3 -> 3/8T
        4 -> 1/2T
        5 -> 5/8T
        6 -> 3/4T
        7 -> 7/8T
        8 -> 1T

    关键处理：
        t = 1T 归入第 1 个周期的第 8 列；
        t = 2T 归入第 2 个周期的第 8 列；
        t = 3T 归入第 3 个周期的第 8 列。
    """

    r = (t - t_ref) / T
    slot_float = r * n_phase
    slot_global = int(np.round(slot_float))

    if slot_global < 0:
        return None, None, None

    # 如果 t 恰好等于 t_ref，记为第 1 周期的 0T。
    # 但本脚本主要绘制 1/8T 到 1T，因此后续会跳过 phase_idx=0。
    if slot_global == 0:
        cycle_no = 1
        phase_idx = 0
        phase_frac = 0.0
        return cycle_no, phase_frac, phase_idx

    err = abs(slot_float - slot_global)
    if err > tol:
        print(
            f"[警告] t={t:.10f}, (t-t_ref)/T={r:.10f}, "
            f"slot_float={slot_float:.6f}, "
            f"未严格落在 {n_phase} 等分相位上，"
            f"已近似归入 slot_global={slot_global}"
        )

    cycle_no = (slot_global - 1) // n_phase + 1
    phase_idx = (slot_global - 1) % n_phase + 1
    phase_frac = phase_idx / n_phase

    return cycle_no, phase_frac, phase_idx


def format_phase_label_by_idx(phase_idx, n_phase=8):
    """
    根据固定相位槽生成标签。

    n_phase=8 时：
        1 -> \\frac{1}{8}T
        2 -> \\frac{1}{4}T
        3 -> \\frac{3}{8}T
        4 -> \\frac{1}{2}T
        5 -> \\frac{5}{8}T
        6 -> \\frac{3}{4}T
        7 -> \\frac{7}{8}T
        8 -> 1T
    """

    if phase_idx == n_phase:
        return r'1T'

    g = math.gcd(phase_idx, n_phase)
    num = phase_idx // g
    den = n_phase // g

    if den == 1:
        return rf'{num}T'

    return rf'\frac{{{num}}}{{{den}}}T'


def scan_cycles_data_fixed_slots(case_dir, T, n_phase=8, t_ref=0.0, tol=5e-3):
    """
    自动扫描 dat 文件，并按固定相位槽分组。

    返回结构：
    cycles_data = [
        {
            'i': cycle_no,
            'records': [rec_or_None, rec_or_None, ..., rec_or_None]
        },
        ...
    ]

    records 长度固定为 n_phase。
    某个相位缺文件时，该位置为 None，绘图时对应空白，不会导致后续列错位。
    """

    all_files = []

    for fname in os.listdir(case_dir):
        if is_valid_flow_filename(fname):
            t = parse_time_from_filename(fname)

            if t is None:
                continue

            cycle_no, phase_frac, phase_idx = compute_cycle_info_fixed_slots(
                t,
                T,
                n_phase=n_phase,
                t_ref=t_ref,
                tol=tol
            )

            if cycle_no is None:
                continue

            # 只保留 1~n_phase 相位槽。
            # phase_idx=0 对应 t=t_ref 的 0T，默认不参与 8 列图。
            if phase_idx < 1 or phase_idx > n_phase:
                continue

            all_files.append({
                'filename': fname,
                't': t,
                'cycle_no': cycle_no,
                'phase_frac': phase_frac,
                'phase_idx': phase_idx
            })

    if not all_files:
        return []

    all_files.sort(key=lambda x: x['t'])

    grouped = {}

    for item in all_files:
        cycle_no = item['cycle_no']
        phase_idx = item['phase_idx']

        grouped.setdefault(cycle_no, {})

        # 如果同一周期、同一相位槽出现多个文件，保留最接近理论时刻的那个。
        target_t = t_ref + ((cycle_no - 1) + phase_idx / n_phase) * T

        if phase_idx not in grouped[cycle_no]:
            grouped[cycle_no][phase_idx] = item
        else:
            old_item = grouped[cycle_no][phase_idx]
            new_err = abs(item['t'] - target_t)
            old_err = abs(old_item['t'] - target_t)

            if new_err < old_err:
                grouped[cycle_no][phase_idx] = item

    cycles_data = []

    for cycle_no in sorted(grouped.keys()):
        records = []

        for phase_idx in range(1, n_phase + 1):
            records.append(grouped[cycle_no].get(phase_idx, None))

        cycles_data.append({
            'i': cycle_no,
            'records': records
        })

    return cycles_data


# ==================== 后翼间歇运动包络函数 ====================
def smooth_window(tau, T, ks):
    """
    Double-tanh smooth window for one isolated active cycle.
    """
    return -np.tanh(ks * tau / T) * np.tanh(ks * (tau / T - 1.0))


def smooth_start(tau, T, ks):
    """
    Smooth transition from inactive state to active motion.
    """
    return np.tanh(ks * tau / T)


def smooth_stop(tau, T, ks):
    """
    Smooth transition from active motion to inactive state.
    """
    return -np.tanh(ks * (tau / T - 1.0))


def get_hindwing_envelope(t, T1, dc_ratio, ks):
    """
    返回某一时刻后翼间歇运动的包络值。

    四周期窗口：

    DC=0.25:
        active, inactive, inactive, inactive

    DC=0.50:
        active, inactive, active, inactive

    DC=0.75:
        active, active, active, inactive
    """
    window_id = np.floor(t / (4.0 * T1))
    tau4 = t - window_id * 4.0 * T1

    envelope = 0.0

    if np.isclose(dc_ratio, 0.25):
        if 0.0 <= tau4 < T1:
            envelope = smooth_window(tau4, T1, ks)

    elif np.isclose(dc_ratio, 0.50):
        if 0.0 <= tau4 < T1:
            envelope = smooth_window(tau4, T1, ks)

        elif 2.0 * T1 <= tau4 < 3.0 * T1:
            tau_active = tau4 - 2.0 * T1
            envelope = smooth_window(tau_active, T1, ks)

    elif np.isclose(dc_ratio, 0.75):
        if 0.0 <= tau4 < T1:
            envelope = smooth_start(tau4, T1, ks)

        elif T1 <= tau4 < 2.0 * T1:
            envelope = 1.0

        elif 2.0 * T1 <= tau4 < 3.0 * T1:
            tau_active = tau4 - 2.0 * T1
            envelope = smooth_stop(tau_active, T1, ks)

        else:
            envelope = 0.0

    else:
        raise ValueError("Only DC=0.25, 0.5, and 0.75 are supported.")

    return envelope


# ==================== 绘制机翼函数 ====================
def plot_wing(ax, t, theta1, theta2, omega1, omega2, h0, alpha, phi,
              wing='fore', x_shift=0.0, y_shift=0.0,
              dc_ratio=1.0, T1=None, ks=9,
              zorder=2, facecolor='white', edgecolor='black', lw=0.5):
    """
    绘制前翼或后翼轮廓。
    """

    if wing == 'fore':
        theta = theta1 * np.sin(omega1 * t)
        h = h0 * np.sin(omega1 * t + phi)
        x_le = 0.0

    elif wing == 'hind':
        if T1 is None:
            raise ValueError("T1 must be provided when plotting the hindwing.")

        envelope = get_hindwing_envelope(t, T1, dc_ratio, ks)

        theta_base = -theta2 * np.sin(omega2 * t)
        h_base = h0 * np.sin(omega2 * t + phi) - h0

        theta = -alpha - theta_base * envelope
        h = h0 + h_base * envelope

        x_le = 1.5

    else:
        raise ValueError("wing must be 'fore' or 'hind'.")

    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    x_g = x_prof * cos_th - y_prof * sin_th + x_le + x_shift
    y_g = x_prof * sin_th + y_prof * cos_th + h + y_shift

    ax.fill(
        x_g,
        y_g,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=zorder
    )


# ==================== 主循环：遍历全部参数组合 ====================
for dc in dc_list:

    dc_ratio = dc / 100.0

    for k in k_list:
        for st in st_list:

            dir_name = f'st{st}k{k}/'
            dir2_name = f'st{st}k{k}a{a}/'

            dir3_name = f'st{st}k{k}a{a}dc{int(dc)}/k{k}st{st}a{a}dc{int(dc)}/'

            case_dir = os.path.join(
                base_root,
                f'k{k}',
                dir_name,
                dir2_name,
                dir3_name
            )

            print(f'正在处理: st={st}, k={k}, a={a}, dc{int(dc)}')
            print(f'case_dir = {case_dir}')

            theta1 = np.radians(30)
            theta2 = np.radians(30)

            f1 = k / np.pi
            f2 = f1

            T = 1.0 / f1

            h0 = st * u / (2.0 * f1)

            omega1 = 2.0 * np.pi * f1
            omega2 = 2.0 * np.pi * f2

            alpha = np.radians(a)
            phi = np.radians(90)

            if not os.path.exists(case_dir):
                print(f'目录不存在，跳过: {case_dir}')
                continue

            # ==================== 自动扫描 dat 文件并按固定 8 相位槽分组 ====================
            cycles_data = scan_cycles_data_fixed_slots(
                case_dir,
                T,
                n_phase=n_phase_per_cycle,
                t_ref=t_ref,
                tol=slot_tol
            )

            if not cycles_data:
                print(f'未找到符合要求的 dat 文件，跳过: st={st}, k={k}, a={a}, dc{int(dc)}')
                continue

            if max_cycles_to_plot is not None:
                cycles_data = cycles_data[:max_cycles_to_plot]

            total_file_count = 0
            for cyc in cycles_data:
                total_file_count += sum(rec is not None for rec in cyc['records'])

            print(f'共扫描并选取 {total_file_count} 个 dat 文件')
            print(f'共绘制 {len(cycles_data)} 个周期，每周期固定 {n_phase_per_cycle} 列')

            for cyc in cycles_data:
                phase_status = []
                for idx, rec in enumerate(cyc['records'], start=1):
                    if rec is None:
                        phase_status.append(f'{idx}: None')
                    else:
                        phase_status.append(f'{idx}: {rec["filename"]}')
                print(f'  Cycle {cyc["i"]}: ' + ' | '.join(phase_status))

            # ==================== 加载所有数据 ====================
            all_data = []

            for cycle in cycles_data:
                cycle_data = []

                for rec in cycle['records']:

                    # 某个相位槽缺文件时，保留空位，避免后续列错位。
                    if rec is None:
                        cycle_data.append(None)
                        continue

                    fname = rec['filename']
                    file_path = os.path.join(case_dir, fname)

                    if not os.path.exists(file_path):
                        print(f'缺少文件: {file_path}')
                        cycle_data.append(None)
                        continue

                    print(f'  读取文件: {file_path}')

                    try:
                        dat = read_tecplot_ordered_dat(
                            file_path,
                            target_vars=('X', 'Y', 'Vorticity')
                        )

                        X = dat['X']
                        Y = dat['Y']
                        W = dat['Vorticity']

                        W_plot = np.ma.masked_where(
                            (X < x_min) | (X > x_max) |
                            (Y < y_min) | (Y > y_max),
                            W
                        )

                        cycle_data.append({
                            'X': X,
                            'Y': Y,
                            'W_plot': W_plot,
                            't': rec['t'],
                            'phase_frac': rec['phase_frac'],
                            'phase_idx': rec['phase_idx'],
                            'filename': fname
                        })

                    except Exception as e:
                        print(f'读取失败: {file_path}')
                        print(f'错误信息: {e}')
                        cycle_data.append(None)

                all_data.append(cycle_data)

            # ==================== 创建组合图 ====================
            n_cycles = len(cycles_data)
            n_times_max = n_phase_per_cycle

            fig, axes = plt.subplots(
                n_cycles,
                n_times_max,
                figsize=(3 * n_times_max, 2.8 * n_cycles),
                dpi=300,
                sharex=True,
                sharey=True
            )

            plt.subplots_adjust(wspace=0.02, hspace=0.05)

            if n_cycles == 1 and n_times_max == 1:
                axes = np.array([[axes]])
            elif n_cycles == 1:
                axes = axes.reshape(1, -1)
            elif n_times_max == 1:
                axes = axes.reshape(-1, 1)

            tri = None

            for row_idx, cycle in enumerate(cycles_data):
                row_data = all_data[row_idx]

                for col_idx in range(n_times_max):
                    ax = axes[row_idx, col_idx]
                    data = row_data[col_idx]

                    # 缺失相位槽：保持空白，不让后面的时刻错列。
                    if data is None:
                        ax.axis('off')
                        continue

                    tri = ax.contourf(
                        data['X'],
                        data['Y'],
                        data['W_plot'],
                        levels=levels,
                        cmap=cmap_white,
                        extend='both'
                    )

                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    ax.set_aspect('equal', adjustable='box')
                    ax.tick_params(direction='in', top=True, right=True, length=3)

                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels([str(int(tick)) for tick in x_ticks])

                    if row_idx < n_cycles - 1:
                        ax.tick_params(labelbottom=False)

                    if col_idx > 0:
                        ax.tick_params(labelleft=False)

                    t_current = data['t']

                    # 前翼
                    plot_wing(
                        ax,
                        t_current,
                        theta1,
                        theta2,
                        omega1,
                        omega2,
                        h0,
                        alpha,
                        phi,
                        wing='fore',
                        x_shift=wing_shift_x,
                        y_shift=wing_shift_y,
                        edgecolor='black',
                        lw=0.5,
                        zorder=5
                    )

                    # 后翼
                    plot_wing(
                        ax,
                        t_current,
                        theta1,
                        theta2,
                        omega1,
                        omega2,
                        h0,
                        alpha,
                        phi,
                        wing='hind',
                        x_shift=wing_shift_x,
                        y_shift=wing_shift_y,
                        dc_ratio=dc_ratio,
                        T1=T,
                        ks=ks,
                        edgecolor='black',
                        lw=0.5,
                        zorder=5
                    )

                    phase_label = format_phase_label_by_idx(
                        data['phase_idx'],
                        n_phase=n_phase_per_cycle
                    )

                    t_label = rf'$t={phase_label}$'

                    ax.text(
                        0.05,
                        0.95,
                        t_label,
                        transform=ax.transAxes,
                        ha='left',
                        va='top',
                        fontsize=9,
                        color='black',
                        zorder=6
                    )

            # ==================== 行标签：第几个周期 ====================
            for row_idx, cycle in enumerate(cycles_data):
                y_pos = 1.0 - (row_idx + 0.5) / n_cycles

                fig.text(
                    0.02,
                    y_pos,
                    f'${cycle["i"]}T$',
                    rotation=90,
                    va='center',
                    ha='center',
                    fontsize=12,
                    fontweight='bold'
                )

            # ==================== 可选颜色条 ====================
            # 如果需要颜色条，可取消下面代码注释。
            # if tri is not None:
            #     cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            #     cbar = fig.colorbar(tri, cax=cbar_ax)
            #     cbar.set_label(r'$\omega$')

            # ==================== 保存 ====================
            final_png_name = f'k{k}st{st}a{a}dc{int(dc)}.png'
            final_png_path = os.path.join(final_save_root, final_png_name)

            plt.savefig(
                final_png_path,
                format='png',
                bbox_inches='tight',
                dpi=600
            )

            plt.close()

            print(f'已保存: {final_png_path}')

print('全部参数组合绘图完成。')