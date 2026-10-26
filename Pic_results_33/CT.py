# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,

    "font.family": "serif",
    "font.serif": ["Times New Roman"],

    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble":
        r"\usepackage{amsmath} "
        r"\usepackage{newtxtext,newtxmath}",

    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",

    # 透明背景
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
})

# ============================================================
# 数据路径
# ============================================================

data_root = (
    r'E:/Rensq/DC_explore_F/results_intermittent/'
    r'averaged_results/'
)

save_root = os.path.join(
    data_root,
    'large_cycle_stacked_results'
)

os.makedirs(save_root, exist_ok=True)

# ============================================================
# 工况设置
# ============================================================

ST = 0.3

# k : alpha_0
CASE_DICT = {
    0.8: 20,
    1.1: 20,
    1.4: 17,
    1.7: 11,
}

K_LIST = [
    0.8,
    1.1,
    1.4,
    1.7,
]

# ============================================================
# DC 设置
# ============================================================

DC_CODES = [
    1,
    75,
    50,
    25,
]

DC_VALUE_DICT = {
    1: 1.00,
    75: 0.75,
    50: 0.50,
    25: 0.25,
}

DC_PLOT_ORDER = [
    1,
    75,
    50,
    25,
]

# ============================================================
# 间歇拍动的大周期设置（仅用于 DC != 1）
# ============================================================

BIG_CYCLES = [
    (13, 16),
    (17, 20),
    (21, 24),
    (25, 28),
    (29, 32)
]

REQUIRE_COMPLETE_BIG_CYCLE = True

# ============================================================
# DC = 1 的统计设置
# ============================================================

DC1_LAST_N_PERIODS = 5

# ============================================================
# 数字格式
# ============================================================

def format_num(x):
    """
    文件名中的数字格式。
    例如：
        0.8 -> '0.8'
        1.0 -> '1'
    """
    if float(x).is_integer():
        return str(int(x))
    return f'{x:g}'

# ============================================================
# 生成 CSV 路径
# ============================================================

def get_csv_path(k, a, dc_code):
    k_str = format_num(k)
    st_str = format_num(ST)
    a_str = format_num(a)

    filename = (
        f'k={k_str}'
        f'st={st_str}'
        f'a={a_str}'
        f'dc={dc_code}.csv'
    )

    return os.path.join(
        data_root,
        filename
    )

# ============================================================
# 计算单个统计单元的系统参数
# ============================================================

def calculate_system_values(temp):
    """
    temp 可以是：
    1. DC=1 时的单个普通周期；
    2. DC<1 时的一个 4T 大周期。

    返回：
        CL
        CT
        CP
        eta_L
        eta_T
    """
    cl = temp['cl_total_mean'].mean()
    ct = temp['ct_total_mean'].mean()
    cp = temp['cp_total_mean'].mean()

    if np.isclose(cp, 0.0):
        eta_l = np.nan
        eta_t = np.nan
    else:
        eta_l = cl / cp
        eta_t = ct / cp

    return cl, ct, cp, eta_l, eta_t

# ============================================================
# DC = 1：使用最后 5 个普通周期
# ============================================================

def process_dc1(df, k, a, dc_code):
    available_periods = sorted(
        df['period']
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if len(available_periods) < DC1_LAST_N_PERIODS:
        print(
            f'[警告] k={k}, DC=1 '
            f'只有 {len(available_periods)} 个有效周期，'
            f'不足 {DC1_LAST_N_PERIODS} 个'
        )
        return None

    selected_periods = available_periods[-DC1_LAST_N_PERIODS:]

    print(
        f'    DC=1 使用最后 {DC1_LAST_N_PERIODS} 个周期: '
        f'{selected_periods}'
    )

    results = []

    for sample_index, period_idx in enumerate(selected_periods, start=1):
        temp = df[df['period'] == period_idx].copy()

        if len(temp) == 0:
            continue

        cl, ct, cp, eta_l, eta_t = calculate_system_values(temp)

        results.append({
            'k': k,
            'St': ST,
            'a': a,

            'dc_code': dc_code,
            'dc': DC_VALUE_DICT[dc_code],

            'sample_index': sample_index,
            'sample_type': 'single_cycle',

            'period_start': period_idx,
            'period_end': period_idx,
            'sample_label': f'{period_idx}',

            'n_base_cycles': 1,

            'cl': cl,
            'ct': ct,
            'cp': cp,

            'eta_l': eta_l,
            'eta_t': eta_t,
        })

    if len(results) == 0:
        return None

    return pd.DataFrame(results)

# ============================================================
# DC < 1：使用五个 4T 大周期
# ============================================================

def process_intermittent_dc(df, k, a, dc_code):
    results = []

    for sample_index, (p_start, p_end) in enumerate(BIG_CYCLES, start=1):
        expected_periods = list(range(p_start, p_end + 1))

        temp = df[df['period'].isin(expected_periods)].copy()

        actual_periods = sorted(
            temp['period']
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if REQUIRE_COMPLETE_BIG_CYCLE:
            if actual_periods != expected_periods:
                print(
                    f'[警告] '
                    f'k={k}, '
                    f'DC={DC_VALUE_DICT[dc_code]}: '
                    f'{p_start}-{p_end} 周期不完整'
                )
                print(f'        需要: {expected_periods}')
                print(f'        实际: {actual_periods}')
                continue

        if len(temp) == 0:
            continue

        cl, ct, cp, eta_l, eta_t = calculate_system_values(temp)

        results.append({
            'k': k,
            'St': ST,
            'a': a,

            'dc_code': dc_code,
            'dc': DC_VALUE_DICT[dc_code],

            'sample_index': sample_index,
            'sample_type': 'four_cycle_block',

            'period_start': p_start,
            'period_end': p_end,
            'sample_label': f'{p_start}-{p_end}',

            'n_base_cycles': 4,

            'cl': cl,
            'ct': ct,
            'cp': cp,

            'eta_l': eta_l,
            'eta_t': eta_t,
        })

    if len(results) == 0:
        return None

    return pd.DataFrame(results)

# ============================================================
# 处理一个工况
# ============================================================

def process_one_case(k, a, dc_code):
    csv_path = get_csv_path(k=k, a=a, dc_code=dc_code)

    if not os.path.exists(csv_path):
        print(
            f'[缺少文件] '
            f'k={k}, '
            f'a={a}, '
            f'DC={DC_VALUE_DICT[dc_code]}'
        )
        print(f'    {csv_path}')
        return None

    print(
        f'\n正在处理: '
        f'k={k}, '
        f'St={ST}, '
        f'a={a}, '
        f'DC={DC_VALUE_DICT[dc_code]}'
    )

    df = pd.read_csv(csv_path)

    required_columns = [
        'period',
        'cl_total_mean',
        'ct_total_mean',
        'cp_total_mean',
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if len(missing_columns) > 0:
        print(f'[跳过] 缺少列: {missing_columns}')
        return None

    df['period'] = pd.to_numeric(df['period'], errors='coerce')
    df = df.dropna(subset=['period'])
    df['period'] = df['period'].astype(int)

    if dc_code == 1:
        return process_dc1(df=df, k=k, a=a, dc_code=dc_code)
    else:
        return process_intermittent_dc(df=df, k=k, a=a, dc_code=dc_code)

# ============================================================
# 批量处理所有工况
# ============================================================

all_sample_results = []

for k in K_LIST:
    a = CASE_DICT[k]

    for dc_code in DC_CODES:
        temp_df = process_one_case(k=k, a=a, dc_code=dc_code)

        if temp_df is not None:
            all_sample_results.append(temp_df)

# ============================================================
# 检查
# ============================================================

if len(all_sample_results) == 0:
    raise RuntimeError('没有读取到任何有效数据。')

# ============================================================
# 合并所有统计样本
# ============================================================

sample_df = pd.concat(
    all_sample_results,
    ignore_index=True
)

# ============================================================
# 保存所有统计单元
# ============================================================

sample_csv_path = os.path.join(
    save_root,
    'system_statistical_samples.csv'
)

sample_df.to_csv(
    sample_csv_path,
    index=False,
    encoding='utf-8-sig',
    float_format='%.8e'
)

# ============================================================
# 打印具体采用的数据
# ============================================================

print('\n' + '=' * 60)
print('各工况实际使用的统计周期')
print('=' * 60)

for (k, dc), group in sample_df.groupby(['k', 'dc']):
    labels = group['sample_label'].astype(str).tolist()
    print(f'k={k}, DC={dc}: {labels}')

# ============================================================
# 最终统计
# ============================================================

METRICS = [
    'cl',
    'ct',
    'cp',
    'eta_l',
    'eta_t',
]

summary_results = []

for (k, a, dc_code, dc), group in sample_df.groupby(['k', 'a', 'dc_code', 'dc']):
    result = {
        'k': k,
        'St': ST,
        'a': a,

        'dc_code': dc_code,
        'dc': dc,

        'n_samples': len(group),
    }

    if dc_code == 1:
        result['statistical_method'] = 'last_5_single_cycles'
    else:
        result['statistical_method'] = 'five_4T_blocks'

    for metric in METRICS:
        values = (
            group[metric]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .values
        )

        if len(values) == 0:
            mean_value = np.nan
            std_value = np.nan
        else:
            mean_value = np.mean(values)
            if len(values) > 1:
                std_value = np.std(values, ddof=1)
            else:
                std_value = np.nan

        result[f'mean_{metric}'] = mean_value
        result[f'std_{metric}'] = std_value

    summary_results.append(result)

summary_df = pd.DataFrame(summary_results)

# ============================================================
# 排序
# ============================================================

summary_df = summary_df.sort_values(
    by=['k', 'dc'],
    ascending=[True, False]
)

# ============================================================
# 保存最终统计
# ============================================================

summary_csv_path = os.path.join(
    save_root,
    'system_final_statistics.csv'
)

summary_df.to_csv(
    summary_csv_path,
    index=False,
    encoding='utf-8-sig',
    float_format='%.8e'
)

# ============================================================
# 构造推力云图数据
# ============================================================

DC_HEATMAP_ORDER = [
    1.00,
    0.75,
    0.50,
    0.25,
]

K_HEATMAP_ORDER = [
    0.8,
    1.1,
    1.4,
    1.7,
]

# 行：k（从下到上）
# 列：DC（从左到右）
ct_matrix = np.full(
    (len(K_HEATMAP_ORDER), len(DC_HEATMAP_ORDER)),
    np.nan
)

for i, k in enumerate(K_HEATMAP_ORDER):
    for j, dc in enumerate(DC_HEATMAP_ORDER):
        temp = summary_df[
            np.isclose(summary_df['k'], k)
            &
            np.isclose(summary_df['dc'], dc)
        ]

        if len(temp) == 0:
            print(f'[警告] 缺少云图数据: k={k}, DC={dc}')
            continue

        ct_matrix[i, j] = temp['mean_ct'].iloc[0]

# ============================================================
# 打印矩阵，便于检查
# ============================================================

print('\n' + '=' * 60)
print('用于绘制云图的平均推力系数矩阵')
print(f'行对应 k  = {K_HEATMAP_ORDER}')
print(f'列对应 DC = {DC_HEATMAP_ORDER}')
print(ct_matrix)
print('=' * 60)

# ============================================================
# 保存云图数据
# ============================================================

heatmap_df = pd.DataFrame(
    ct_matrix,
    index=K_HEATMAP_ORDER,
    columns=DC_HEATMAP_ORDER
)
heatmap_df.index.name = 'k'
heatmap_df.columns.name = 'DC'

heatmap_csv_path = os.path.join(
    save_root,
    'thrust_heatmap_data.csv'
)

heatmap_df.to_csv(
    heatmap_csv_path,
    encoding='utf-8-sig',
    float_format='%.8e'
)

# ============================================================
# 平滑插值绘图
# ============================================================

from scipy.interpolate import RectBivariateSpline

# ------------------------------------------------------------
# 注意：
# RectBivariateSpline 要求 x 和 y 单调递增
# 所以这里把 DC 改成递增顺序：0.25 -> 1.0
# 并同步调整矩阵列顺序
# ------------------------------------------------------------

dc_asc = np.array([0.25, 0.50, 0.75, 1.00])
k_asc  = np.array([0.8, 1.1, 1.4, 1.7])

# 原矩阵列顺序是 [1.00, 0.75, 0.50, 0.25]
# 这里倒过来，变成 [0.25, 0.50, 0.75, 1.00]
ct_matrix_asc = ct_matrix[:, ::-1]

# ------------------------------------------------------------
# 检查是否存在缺失值
# ------------------------------------------------------------

if np.any(~np.isfinite(ct_matrix_asc)):
    raise RuntimeError('ct_matrix 中存在 NaN，无法进行二维样条插值。')

# ------------------------------------------------------------
# 二维三次样条插值
# 行对应 k，列对应 DC
# spline(k, dc)
# ------------------------------------------------------------

spline = RectBivariateSpline(
    k_asc,
    dc_asc,
    ct_matrix_asc,
    kx=3,
    ky=3
)

# ------------------------------------------------------------
# 构造高密度网格
# ------------------------------------------------------------

dc_dense = np.linspace(0.25, 1.00, 300)
k_dense  = np.linspace(0.8, 1.7, 300)

ct_dense = spline(k_dense, dc_dense)

# ============================================================
# 色标范围：以 0 为中心的对称范围
# ============================================================

# valid_ct = ct_matrix[np.isfinite(ct_matrix)]
#
# if len(valid_ct) == 0:
#     raise RuntimeError('没有有效的推力数据可用于绘图。')
#
# ct_abs_max = np.max(np.abs(valid_ct))
#
# if np.isclose(ct_abs_max, 0.0):
#     ct_abs_max = 1.0
#
# vmin = -ct_abs_max
# vmax = ct_abs_max
# ============================================================
# 固定色条范围
# ============================================================
vmin = -0.2
vmax = 0.2

norm = mcolors.TwoSlopeNorm(
    vmin=vmin,
    vcenter=0.0,
    vmax=vmax
)

# ============================================================
# 绘图：平滑连续云图
# ============================================================

fig, ax = plt.subplots(figsize=(4.4, 3.6))

# 透明背景
fig.patch.set_facecolor('none')
fig.patch.set_alpha(0.0)
ax.set_facecolor('none')
ax.patch.set_facecolor('none')
ax.patch.set_alpha(0.0)

# ------------------------------------------------------------
# 连续填色云图
# levels 取得足够多，使过渡平滑
# ------------------------------------------------------------

levels = np.linspace(
    -0.2,
    0.2,
    201
)

cf = ax.contourf(
    dc_dense,
    k_dense,
    ct_dense,
    levels=levels,
    cmap='RdBu_r',
    norm=norm,
    # extend='both'
)

# # ------------------------------------------------------------
# # 可选：叠加很淡的等值线，让层次更清晰
# # 如果不想要等值线，可直接删掉这一段
# # ------------------------------------------------------------
#
# # ============================================================
# # 普通等值线
# # ============================================================
#
# ax.contour(
#     dc_dense,
#     k_dense,
#     ct_dense,
#     levels=np.linspace(vmin, vmax, 5),
#     colors='black',
#     linewidths=0.30,
#     alpha=0.25
# )


# ============================================================
# CT = 0 界限线
#
# 明确区分：
# CT > 0 ：推力
# CT < 0 ：阻力
# ============================================================

zero_contour = ax.contour(
    dc_dense,
    k_dense,
    ct_dense,

    levels=[0.0],

    colors='black',
    linewidths=1.4,
    linestyles='-',

    zorder=5
)
# ------------------------------------------------------------
# 可选：标出原始 16 个计算点位置
# 如果不想显示采样点，把这一段删掉即可
# ------------------------------------------------------------

DC_pts, K_pts = np.meshgrid(DC_HEATMAP_ORDER, K_HEATMAP_ORDER)

ax.scatter(
    DC_pts.flatten(),
    K_pts.flatten(),
    s=12,
    c='black',
    marker='o',
    linewidths=0.0,
    zorder=3,
    alpha=0.8
)

# ============================================================
# 坐标轴设置
# ============================================================

ax.set_xlabel(r'$DC$')

ax.set_ylabel(r'$k$', rotation=0)
ax.yaxis.set_label_coords(-0.12, 0.5)

# x 轴仍然按你的要求从左到右为 1 -> 0.25
ax.set_xlim(1.00, 0.25)
ax.set_ylim(0.8, 1.7)

ax.set_xticks([1.00, 0.75, 0.50, 0.25])
ax.set_xticklabels(['1', '0.75', '0.5', '0.25'])

ax.set_yticks([0.8, 1.1, 1.4, 1.7])
ax.set_yticklabels(['0.8', '1.1', '1.4', '1.7'])

# ============================================================
# 色条
# ============================================================

cbar = fig.colorbar(
    cf,
    ax=ax,
    orientation='vertical',
    fraction=0.05,
    pad=0.04,
    aspect=22,
    ticks=[-0.2, -0.1, 0.0, 0.1, 0.2]
)

cbar.set_label(
    r'$\overline{C}_{TS}$',
    rotation=0,
    labelpad=14
)

cbar.ax.tick_params(direction='in')
cbar.ax.tick_params(direction='in')

# ============================================================
# 坐标样式
# ============================================================

ax.tick_params(axis='both', direction='in')
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.tight_layout(pad=0.6)

# ============================================================
# 保存
# ============================================================

png_path = os.path.join(
    save_root,
    'thrust_heatmap_smooth.png'
)

svg_path = os.path.join(
    save_root,
    'thrust_heatmap_smooth.svg'
)

pdf_path = os.path.join(
    save_root,
    'thrust_heatmap_smooth.pdf'
)

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches='tight',
    transparent=True,
    facecolor='none',
    edgecolor='none'
)

fig.savefig(
    svg_path,
    format='svg',
    bbox_inches='tight',
    transparent=True,
    facecolor='none',
    edgecolor='none'
)

# fig.savefig(
#     pdf_path,
#     format='pdf',
#     bbox_inches='tight',
#     transparent=True,
#     facecolor='none',
#     edgecolor='none'
# )

plt.show()

# ============================================================
# 完成
# ============================================================

print('\n' + '=' * 60)
print('平滑推力云图绘制完成')
print('=' * 60)

print(f'\n统计样本数据：\n{sample_csv_path}')
print(f'\n最终统计数据：\n{summary_csv_path}')
print(f'\n云图数据：\n{heatmap_csv_path}')
print(f'\nPNG：\n{png_path}')
print(f'\nSVG：\n{svg_path}')
print(f'\nPDF：\n{pdf_path}')