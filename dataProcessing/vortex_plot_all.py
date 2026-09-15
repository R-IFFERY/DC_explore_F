import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
import os
import shutil
import re

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
a = 0
dc = 1

# 可自行修改
k_list = [2.0]
# st_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
st_list = [0.2]

base_root = r'E:/Rensq/DC_explore_F/data_1Tflow/'
final_save_root = r'E:/Rensq/DC_explore_F/flow/test'
os.makedirs(final_save_root, exist_ok=True)

# ==================== 翼型轮廓 ====================
profile = np.loadtxt(r'E:/Rensq/DC_explore_F/flow/boundary.dat')
x_prof = profile[:, 0] * c
y_prof = profile[:, 1] * c

# ==================== 新流场中心与绘图区域 ====================
flow_center_x = 0.0
flow_center_y = 0.0

x_min, x_max = -1.0, 20.0
y_min, y_max = -10, 10

# 机翼整体平移到新流场中心附近
wing_shift_x = flow_center_x
wing_shift_y = flow_center_y

# ==================== 坐标轴刻度设置 ====================
# 横坐标只显示部分整数值，不显示小数
x_tick_step = 3
x_ticks = np.arange(int(np.ceil(x_min)), int(np.floor(x_max)) + 1, x_tick_step)

# 如需纵坐标也只显示整数，可启用下面两行
# y_tick_step = 2
# y_ticks = np.arange(int(np.ceil(y_min)), int(np.floor(y_max)) + 1, y_tick_step)

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
    读取 Tecplot Ordered ASCII .dat 文件
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


# ==================== 计算属于第几个周期及周期内位置 ====================
def compute_cycle_info(t, T, eps=1e-10):
    cycle_no = int(np.floor(t / T + eps)) + 1
    phase_frac = (t / T) - np.floor(t / T + eps)

    if abs(phase_frac) < 1e-8:
        phase_frac = 0.0
    elif abs(phase_frac - 1.0) < 1e-8:
        phase_frac = 0.0
        cycle_no += 1

    return cycle_no, phase_frac


def format_phase_label(phase_frac, decimals=2):
    s = f"{phase_frac:.{decimals}f}".rstrip('0').rstrip('.')
    if s == '':
        s = '0'
    return f'{s}T'


# ==================== 自动扫描 dat 文件并按周期分组 ====================
def scan_cycles_data(case_dir, T):
    all_files = []
    for fname in os.listdir(case_dir):
        if is_valid_flow_filename(fname):
            t = parse_time_from_filename(fname)
            if t is not None:
                cycle_no, phase_frac = compute_cycle_info(t, T)
                all_files.append({
                    'filename': fname,
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
    for cycle_no in sorted(grouped.keys()):
        records = sorted(grouped[cycle_no], key=lambda x: x['phase_frac'])
        cycles_data.append({
            'i': cycle_no,
            'records': records
        })

    return cycles_data


# ==================== 绘制机翼函数（增加整体平移） ====================
def plot_wing(ax, t, theta1, theta2, omega1, omega2, h0, alpha, phi,
              wing='fore', x_shift=0.0, y_shift=0.0,
              zorder=2, facecolor='white', edgecolor='black', lw=0.5):
    if wing == 'fore':
        theta = theta1 * np.sin(omega1 * t)
        h = h0 * np.sin(omega1 * t + phi)
        x_le = 0.0
    else:
        theta = theta2 * np.sin(omega2 * t) - alpha
        h = h0 * np.sin(omega2 * t + phi)
        x_le = 1.5

    cos_th, sin_th = np.cos(theta), np.sin(theta)
    x_g = x_prof * cos_th - y_prof * sin_th + x_le + x_shift
    y_g = x_prof * sin_th + y_prof * cos_th + h + y_shift

    ax.fill(
        x_g, y_g,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=zorder
    )


# ==================== 主循环：遍历全部参数组合 ====================
for k in k_list:
    for st in st_list:
        dir_name = f'st{st}k{k}/'
        dir2_name = f'st{st}k{k}a{a}/'
        dir3_name = f'st{st}k{k}a{a}dc{int(dc)}/k{k}st{st}/'
        case_dir = os.path.join(base_root, f'k{k}', dir_name, dir2_name, dir3_name)

        print(f'正在处理: st={st}, k={k}， a={a}')

        theta1 = np.radians(30)
        theta2 = np.radians(30)
        f1 = k / np.pi
        f2 = f1
        T = 1.0 / f1
        h0 = st * u / (2 * f1)
        omega1 = 2 * np.pi * f1
        omega2 = 2 * np.pi * f2
        alpha = np.radians(a)
        phi = np.radians(90)

        # ==================== 自动扫描 dat 文件并分组 ====================
        cycles_data = scan_cycles_data(case_dir, T)

        if not cycles_data:
            print(f'未找到符合要求的 dat 文件，跳过: st={st}, k={k}, a={a}')
            continue

        print(f'共扫描到 {sum(len(c["records"]) for c in cycles_data)} 个 dat 文件')
        print(f'共识别到 {len(cycles_data)} 个周期')

        # ==================== 加载所有数据 ====================
        all_data = []
        file_missing = False

        for cycle in cycles_data:
            cycle_data = []
            for rec in cycle['records']:
                fname = rec['filename']
                file_path = os.path.join(case_dir, fname)

                if not os.path.exists(file_path):
                    print(f'缺少文件，跳过该组: {file_path}')
                    file_missing = True
                    break

                print(f'  读取文件: {file_path}')

                try:
                    dat = read_tecplot_ordered_dat(file_path, target_vars=('X', 'Y', 'Vorticity'))
                    X = dat['X']
                    Y = dat['Y']
                    W = dat['Vorticity']

                    # 按 dat 的二维网格方式处理
                    W_plot = np.ma.masked_where(
                        (X < x_min) | (X > x_max) | (Y < y_min) | (Y > y_max),
                        W
                    )

                    cycle_data.append({
                        'X': X,
                        'Y': Y,
                        'W_plot': W_plot,
                        't': rec['t'],
                        'phase_frac': rec['phase_frac'],
                        'filename': fname
                    })

                except Exception as e:
                    print(f'读取失败，跳过该组: {file_path}')
                    print(f'错误信息: {e}')
                    file_missing = True
                    break

            if file_missing:
                break

            all_data.append(cycle_data)

        if file_missing:
            print(f'文件不完整，跳过: st={st}, k={k}, a={a}')
            continue

        # ==================== 创建组合图 ====================
        n_cycles = len(cycles_data)
        n_times_max = max(len(cycle['records']) for cycle in cycles_data)

        fig, axes = plt.subplots(
            n_cycles, n_times_max,
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
            n_this = len(row_data)

            for col_idx in range(n_times_max):
                ax = axes[row_idx, col_idx]

                if col_idx >= n_this:
                    ax.axis('off')
                    continue

                data = row_data[col_idx]

                tri = ax.contourf(
                    data['X'], data['Y'], data['W_plot'],
                    levels=levels, cmap=cmap_white, extend='both'
                )

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_aspect('equal', adjustable='box')
                ax.tick_params(direction='in', top=True, right=True, length=3)

                # 只显示部分整数横坐标，不显示小数
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([str(int(tick)) for tick in x_ticks])

                # 如果你也希望纵坐标只显示整数，可取消下面两行注释
                # ax.set_yticks(y_ticks)
                # ax.set_yticklabels([str(int(tick)) for tick in y_ticks])

                if row_idx < n_cycles - 1:
                    ax.tick_params(labelbottom=False)
                if col_idx > 0:
                    ax.tick_params(labelleft=False)

                t = data['t']
                plot_wing(ax, t, theta1, theta2, omega1, omega2, h0, alpha, phi,
                          wing='fore',
                          x_shift=wing_shift_x, y_shift=wing_shift_y,
                          edgecolor='black', lw=0.5, zorder=5)
                plot_wing(ax, t, theta1, theta2, omega1, omega2, h0, alpha, phi,
                          wing='hind',
                          x_shift=wing_shift_x, y_shift=wing_shift_y,
                          edgecolor='black', lw=0.5, zorder=5)

                t_label = f'$t={format_phase_label(data["phase_frac"])}$'
                ax.text(
                    0.05, 0.95, t_label,
                    transform=ax.transAxes,
                    ha='left', va='top',
                    fontsize=9, color='black', zorder=6
                )

        # 行标签：第几个周期
        for row_idx, cycle in enumerate(cycles_data):
            y_pos = 1 - (row_idx + 0.5) / n_cycles
            fig.text(
                0.02,
                y_pos,
                f'${cycle["i"]}T$',
                rotation=90, va='center', ha='center',
                fontsize=12, fontweight='bold'
            )

        # ==================== 保存 ====================
        final_png_name = f'k{k}st{st}a{a}.png'
        final_png_path = os.path.join(final_save_root, final_png_name)
        plt.savefig(final_png_path, format='png', bbox_inches='tight', dpi=600)
        plt.close()

print('全部参数组合绘图完成。')