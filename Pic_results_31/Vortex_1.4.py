import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import re

# ============================================================
# Figure style
# ============================================================
USE_TEX = True
FONT_SIZE = 10

plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": USE_TEX,
})

if USE_TEX:
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{newtxtext,newtxmath}"
    })

# ============================================================
# User parameters
# ============================================================
u = 1.0
rou = 1.0
c = 1.0

a = 0
dc = 1

# 仅绘制 k = 1.4
k = 1.4

# 四个分组子图分别对应 St = 0.3, 0.5, 0.7, 0.8
st_list = [0.3, 0.5, 0.7, 0.8]

# 每个 St 下绘制第 12 和第 13 个周期
cycle_list = [12, 13]

# 每个周期绘制四个相位
phase_list = [0.0, 0.25, 0.5, 0.75]

PHASE_TOL = 0.04

base_root = r'E:/Rensq/DC_explore_F/data_1Tflow/'
final_save_root = r'E:/Rensq/DC_explore_F/flow/vortex_k14_selected_cycles'
os.makedirs(final_save_root, exist_ok=True)

boundary_path = r'E:/Rensq/DC_explore_F/flow/boundary.dat'

# ============================================================
# Plot domain
# ============================================================
flow_center_x = 0.0
flow_center_y = 0.0

x_min, x_max = -1.0, 20.0

# St = 0.3 和 0.5 使用 y/c = [-5, 5]
# St = 0.7 和 0.8 使用 y/c = [-10, 10]
def get_y_range_by_st(st):
    if st <= 0.7 + 1e-12:
        return -5.0, 5.0
    else:
        return -10.0, 10.0


wing_shift_x = flow_center_x
wing_shift_y = flow_center_y

x_tick_step = 5
y_tick_step = 5

x_ticks = np.array([0, 5, 10, 15, 20])


def get_y_ticks_by_st(st):
    y_min_row, y_max_row = get_y_range_by_st(st)
    return np.arange(
        int(np.ceil(y_min_row)),
        int(np.floor(y_max_row)) + 1,
        y_tick_step
    )


Y_LABELPAD = -4

# 若要使用原始网格，设为 1
PLOT_STRIDE = 2

# 文件名 F165000.dat 表示 t = 16.5000
FILENAME_TIME_SCALE = 10000.0

# 是否在每个分组子图的第二行显示 x 坐标
# 若只想在整张图最下方显示 x 坐标，改为 False
SHOW_X_IN_EACH_PANEL = True

# ============================================================
# Colormap
# ============================================================
numb = 512
base_cmap = colormaps['RdBu_r'].resampled(numb)
new_colors = base_cmap(np.linspace(0, 1, numb))

vmin, vmax = -20, 20
sigma = 1.0

vals = np.linspace(vmin, vmax, numb)
weights = np.exp(-0.5 * (vals / sigma) ** 2)

for i in range(numb):
    w = weights[i]
    new_colors[i] = (1 - w) * new_colors[i] + w * np.array([1, 1, 1, 1])

cmap_white = mcolors.ListedColormap(new_colors)
levels = np.linspace(vmin, vmax, numb + 1)

norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# ============================================================
# Airfoil profile
# ============================================================
profile = np.loadtxt(boundary_path)
x_prof = profile[:, 0] * c
y_prof = profile[:, 1] * c

# ============================================================
# Utilities
# ============================================================
def fmt_num(x):
    return f'{x:g}'


def unique_keep_order(seq):
    out = []
    for item in seq:
        if item not in out:
            out.append(item)
    return out


def format_phase_for_title(phase):
    if abs(phase) < 1e-12:
        return r'0T'
    s = f'{phase:g}'
    return rf'{s}T'


def format_cycle_label(cycle_no):
    return rf'${cycle_no}^{{\mathrm{{th}}}}T$'


def get_case_dir_candidates(base_root, k, st, a, dc):
    """
    同时尝试 str(k) 和 g 格式，避免 2.0 / 2 的文件夹命名差异。

    当前主路径：
    base_root/k{k}/st{st}k{k}/st{st}k{k}a{a}/
    st{st}k{k}a{a}dc{dc}/k{k}st{st}

    同时附加一个带 a 的末级路径候选：
    .../k{k}st{st}a{a}
    """
    k_names = unique_keep_order([str(k), fmt_num(k)])
    st_names = unique_keep_order([str(st), fmt_num(st)])

    candidates = []

    for ks in k_names:
        for ss in st_names:
            case_dir_1 = os.path.join(
                base_root,
                f'k{ks}',
                f'st{ss}k{ks}',
                f'st{ss}k{ks}a{a}',
                f'st{ss}k{ks}a{a}dc{int(dc)}',
                f'k{ks}st{ss}'
            )

            case_dir_2 = os.path.join(
                base_root,
                f'k{ks}',
                f'st{ss}k{ks}',
                f'st{ss}k{ks}a{a}',
                f'st{ss}k{ks}a{a}dc{int(dc)}',
                f'k{ks}st{ss}a{a}'
            )

            candidates.append(case_dir_1)
            candidates.append(case_dir_2)

    return unique_keep_order(candidates)


def find_existing_case_dir(base_root, k, st, a, dc):
    candidates = get_case_dir_candidates(base_root, k, st, a, dc)

    for d in candidates:
        if os.path.isdir(d):
            return d

    return None

# ============================================================
# Tecplot dat reader
# ============================================================
def read_tecplot_ordered_dat(filename, target_vars=('X', 'Y', 'Vorticity')):
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
        raise ValueError(
            f'{filename} 数据点数量不足，期望 {I * J}，实际 {data.shape[0]}'
        )

    result = {
        'solution_time': sol_time,
        'I': I,
        'J': J
    }

    for i, var in enumerate(target_vars):
        result[var] = data[:, i].reshape(J, I)

    return result

# ============================================================
# Filename time parser
# ============================================================
def parse_time_from_filename(filename):
    """
    支持：
    F165000.dat
    F 165000.dat
    F    165000.dat
    """
    base = os.path.basename(filename)
    m = re.match(r'^F\s*(\d+)\.dat$', base, re.IGNORECASE)

    if not m:
        return None

    return int(m.group(1)) / FILENAME_TIME_SCALE


def is_valid_flow_filename(filename):
    base = os.path.basename(filename)
    return re.match(r'^F\s*\d+\.dat$', base, re.IGNORECASE) is not None


def compute_cycle_phase_from_time(t, T):
    cycle_no = int(np.floor(t / T + 1e-10)) + 1
    phase = t / T - np.floor(t / T + 1e-10)

    if abs(phase) < 1e-8:
        phase = 0.0

    return cycle_no, phase


def find_target_record_by_cycle_phase(case_dir, k, selected_cycle,
                                      target_phase, phase_tol=0.04):
    """
    对固定 k，寻找 selected_cycle 周期内 target_phase 对应的流场文件。

    T_k = pi / k
    t_target = (selected_cycle - 1 + target_phase) * T_k

    周期编号从 1 开始：
    第 12 周期 0T      -> t = 11T
    第 12 周期 0.25T   -> t = 11.25T
    第 13 周期 0T      -> t = 12T
    """
    if case_dir is None or not os.path.isdir(case_dir):
        return None

    T = np.pi / k
    target_time = (selected_cycle - 1 + target_phase) * T
    target_step = int(round(target_time * FILENAME_TIME_SCALE))

    records = []

    for fname in os.listdir(case_dir):
        if not is_valid_flow_filename(fname):
            continue

        t = parse_time_from_filename(fname)

        if t is None:
            continue

        cycle_no, phase = compute_cycle_phase_from_time(t, T)

        records.append({
            'filename': fname,
            'path': os.path.join(case_dir, fname),
            't': t,
            'cycle_no': cycle_no,
            'phase_frac': phase,
            'time_error': abs(t - target_time),
            'phase_error': abs((t - target_time) / T)
        })

    if not records:
        return None

    best = min(records, key=lambda r: r['time_error'])

    if best['phase_error'] <= phase_tol:
        print(
            f'  target cycle={selected_cycle}, phase={target_phase:.3f}T, '
            f'T={T:.6f}, target t={target_time:.6f}, ideal F{target_step:d}.dat'
        )
        print(
            f'  selected {best["filename"]}, '
            f't={best["t"]:.6f}, cycle={best["cycle_no"]}, '
            f'phase={best["phase_frac"]:.4f}T, '
            f'phase error={best["phase_error"]:.4f}T'
        )
        return best

    print(
        f'  nearest file is {best["filename"]}, but phase error '
        f'{best["phase_error"]:.4f}T > PHASE_TOL={phase_tol:.4f}T'
    )
    print(
        f'  target cycle={selected_cycle}, phase={target_phase:.3f}T, '
        f'T={T:.6f}, target t={target_time:.6f}, ideal F{target_step:d}.dat'
    )

    return None


def crop_and_downsample(X, Y, W, x_min, x_max, y_min, y_max, stride=1):
    """
    对绘图窗口裁剪并降采样。
    """
    x_line = X[0, :]
    y_line = Y[:, 0]

    ix = np.where((x_line >= x_min) & (x_line <= x_max))[0]
    iy = np.where((y_line >= y_min) & (y_line <= y_max))[0]

    if len(ix) == 0 or len(iy) == 0:
        return X[::stride, ::stride], Y[::stride, ::stride], W[::stride, ::stride]

    i0, i1 = ix[0], ix[-1] + 1
    j0, j1 = iy[0], iy[-1] + 1

    Xc = X[j0:j1:stride, i0:i1:stride]
    Yc = Y[j0:j1:stride, i0:i1:stride]
    Wc = W[j0:j1:stride, i0:i1:stride]

    return Xc, Yc, Wc

# ============================================================
# Wing plotting
# ============================================================
def plot_wing(ax, t, theta1, theta2, omega1, omega2, h0, alpha, phi,
              wing='fore', x_shift=0.0, y_shift=0.0,
              zorder=5, facecolor='white', edgecolor='black', lw=0.4):

    if wing == 'fore':
        theta = theta1 * np.sin(omega1 * t)
        h = h0 * np.sin(omega1 * t + phi)
        x_le = 0.0
    else:
        theta = theta2 * np.sin(omega2 * t) - alpha
        h = h0 * np.sin(omega2 * t + phi)
        x_le = 1.5

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


def setup_axis(ax, phase_col, panel_idx, cycle_row, st):
    """
    每个分组子图内部为 2 行 × 4 列。
    phase_col: 0--3
    cycle_row: 0 对应第 12 周期，1 对应第 13 周期
    """
    y_min_row, y_max_row = get_y_range_by_st(st)
    y_ticks_row = get_y_ticks_by_st(st)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min_row, y_max_row)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks_row)

    ax.tick_params(
        direction='in',
        top=True,
        right=True,
        length=2.5,
        width=0.6
    )

    # x 轴显示策略
    if SHOW_X_IN_EACH_PANEL:
        show_x = (cycle_row == 1)
    else:
        show_x = (panel_idx == len(st_list) - 1 and cycle_row == 1)

    if show_x:
        ax.tick_params(labelbottom=True)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([r'$0$', r'$5$', r'$10$', r'$15$', r'$20$'])
        ax.set_xlabel(r'$x/c$', labelpad=1)
    else:
        ax.tick_params(labelbottom=False)

    # 每个周期行的最左列显示 y 轴
    if phase_col == 0:
        ax.tick_params(labelleft=True)
        ax.set_ylabel(r'$y/c$', labelpad=Y_LABELPAD)
    else:
        ax.tick_params(labelleft=False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

# ============================================================
# Build grouped figure
# ============================================================
n_panels = len(st_list)
n_cycle_rows = len(cycle_list)
n_phase_cols = len(phase_list)

# 每个 St 分组包含两个周期行，因此高度比例为 2 * y_range
outer_height_ratios = []
for st in st_list:
    y_min_row, y_max_row = get_y_range_by_st(st)
    outer_height_ratios.append(2.0 * (y_max_row - y_min_row))

fig = plt.figure(figsize=(6.5, 9.0), dpi=300)

outer_gs = fig.add_gridspec(
    nrows=n_panels,
    ncols=1,
    left=0.115,
    right=0.985,
    bottom=0.070,
    top=0.955,
    hspace=0.28,
    height_ratios=outer_height_ratios
)

panel_axes = []

for panel_idx, st in enumerate(st_list):
    y_min_row, y_max_row = get_y_range_by_st(st)

    inner_gs = outer_gs[panel_idx].subgridspec(
        nrows=n_cycle_rows,
        ncols=n_phase_cols,
        wspace=0.000,
        hspace=0.090,
        height_ratios=[y_max_row - y_min_row] * n_cycle_rows
    )

    axes_this_panel = []

    for cycle_row in range(n_cycle_rows):
        row_axes = []
        for phase_col in range(n_phase_cols):
            ax = fig.add_subplot(inner_gs[cycle_row, phase_col])
            row_axes.append(ax)
        axes_this_panel.append(row_axes)

    panel_axes.append(axes_this_panel)

# ============================================================
# Main plotting loop
# ============================================================
for panel_idx, st in enumerate(st_list):
    case_dir = find_existing_case_dir(base_root, k, st, a, dc)

    for cycle_row, cycle_no in enumerate(cycle_list):
        for phase_col, phase in enumerate(phase_list):
            ax = panel_axes[panel_idx][cycle_row][phase_col]
            setup_axis(ax, phase_col, panel_idx, cycle_row, st)

            # 每个分组子图的第一行顶部标注相位
            if cycle_row == 0:
                phase_label = format_phase_for_title(phase)
                ax.set_title(rf'${phase_label}$', pad=4)

            print(
                f'Processing k={k:g}, St={st:g}, '
                f'cycle={cycle_no}, phase={phase:g}T'
            )

            if case_dir is None:
                print('  Case directory not found. Leave blank.')
                continue

            # ------------------------------------------------------------
            # Kinematics
            # ------------------------------------------------------------
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

            # ------------------------------------------------------------
            # Find target file
            # ------------------------------------------------------------
            rec = find_target_record_by_cycle_phase(
                case_dir=case_dir,
                k=k,
                selected_cycle=cycle_no,
                target_phase=phase,
                phase_tol=PHASE_TOL
            )

            if rec is None:
                print('  No suitable file. Leave blank.')
                continue

            file_path = rec['path']

            if not os.path.exists(file_path):
                print(f'  Missing file: {file_path}. Leave blank.')
                continue

            # ------------------------------------------------------------
            # Read and plot vorticity
            # ------------------------------------------------------------
            try:
                dat = read_tecplot_ordered_dat(
                    file_path,
                    target_vars=('X', 'Y', 'Vorticity')
                )

                X = dat['X']
                Y = dat['Y']
                W = dat['Vorticity']

                y_min_row, y_max_row = get_y_range_by_st(st)

                Xp, Yp, Wp = crop_and_downsample(
                    X, Y, W,
                    x_min, x_max,
                    y_min_row, y_max_row,
                    stride=PLOT_STRIDE
                )

                Wp = np.ma.masked_where(
                    (Xp < x_min) | (Xp > x_max) |
                    (Yp < y_min_row) | (Yp > y_max_row),
                    Wp
                )

                ax.contourf(
                    Xp,
                    Yp,
                    Wp,
                    levels=levels,
                    cmap=cmap_white,
                    extend='both'
                )

                t = rec['t']

                plot_wing(
                    ax,
                    t,
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
                    lw=0.4,
                    zorder=5
                )

                plot_wing(
                    ax,
                    t,
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
                    edgecolor='black',
                    lw=0.4,
                    zorder=5
                )

            except Exception as e:
                print(f'  Read or plot failed: {file_path}')
                print(f'  Error: {e}')
                continue

# ============================================================
# Panel labels, cycle labels, and parameter labels
# ============================================================
panel_letters = [f'({chr(ord("a") + i)})' for i in range(len(st_list))]

for panel_idx, st in enumerate(st_list):
    axes_this_panel = panel_axes[panel_idx]

    all_axes = [ax for row in axes_this_panel for ax in row]
    left = min(ax.get_position().x0 for ax in all_axes)
    right = max(ax.get_position().x1 for ax in all_axes)
    bottom = min(ax.get_position().y0 for ax in all_axes)
    top = max(ax.get_position().y1 for ax in all_axes)
    mid_y = 0.5 * (bottom + top)

    # 图外左上角标注 (a), (b), (c), (d)
    fig.text(
        left - 0.08,
        top + 0.010,
        panel_letters[panel_idx],
        ha='left',
        va='bottom',
        fontsize=FONT_SIZE
    )

    # 原来的 k=1.4, St=... 标签保留，并放在两个周期行的中间
    fig.text(
        left - 0.060,
        mid_y,
        rf'$k={k:g}$' + '\n' + rf'$St={st:g}$',
        rotation=90,
        ha='center',
        va='center',
        fontsize=FONT_SIZE
    )

    # # 第 12 和第 13 周期行标签
    # for cycle_row, cycle_no in enumerate(cycle_list):
    #     row_axes = axes_this_panel[cycle_row]
    #     row_bottom = min(ax.get_position().y0 for ax in row_axes)
    #     row_top = max(ax.get_position().y1 for ax in row_axes)
    #     row_mid_y = 0.5 * (row_bottom + row_top)
    #
    #     fig.text(
    #         left - 0.050,
    #         row_mid_y,
    #         format_cycle_label(cycle_no),
    #         rotation=0,
    #         ha='center',
    #         va='center',
    #         fontsize=FONT_SIZE
    #     )

# ============================================================
# Colorbar only in the top-right subplot
# ============================================================
cbar_host_ax = panel_axes[0][0][-1]

cax = inset_axes(
    cbar_host_ax,
    width="55%",
    height="7%",
    loc="upper right",
    borderpad=0.45
)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_white)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    cax=cax,
    orientation='horizontal',
    extend='both'
)

# 只保留色条刻度数字，不显示刻度线和标签
cbar.set_ticks([vmin, 0, vmax])
cbar.set_ticklabels([rf'${vmin:g}$', r'$0$', rf'${vmax:g}$'])

cbar.ax.tick_params(
    labelsize=FONT_SIZE,
    length=0,
    width=0,
    pad=1
)

cbar.outline.set_linewidth(0.5)

# ============================================================
# Force all text objects to 10 pt
# ============================================================
for text in fig.findobj(match=plt.Text):
    text.set_fontsize(FONT_SIZE)

# ============================================================
# Save
# ============================================================
final_png_name = 'k1p4_st03050708_cycles12_13_grouped.png'
final_png_path = os.path.join(final_save_root, final_png_name)

plt.savefig(
    final_png_path,
    format='png',
    bbox_inches='tight',
    dpi=600
)

final_svg_name = 'k1p4_st03050708_cycles12_13_grouped.svg'
final_svg_path = os.path.join(final_save_root, final_svg_name)

plt.savefig(
    final_svg_path,
    format='svg',
    bbox_inches='tight',
    dpi=600
)

plt.close(fig)

print(f'Finished. Saved to: {final_png_path}')
print(f'Finished. Saved to: {final_svg_path}')