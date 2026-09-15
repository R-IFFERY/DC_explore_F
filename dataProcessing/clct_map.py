import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

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


class Config:
    """统一配置管理"""

    # 参数范围
    ST_LIST = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    K_LIST = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
    A_LIST = [0]
    DC_LIST = [1]

    # 数据基础路径
    BASE_DIR = 'E:/Rensq/DC_explore_F/data/'

    # 统一输出目录：只在这个 heatmap 文件夹输出
    OUTPUT_DIR = 'E:/Rensq/DC_explore_F/results_a0/heatmap/'

    METRICS = {
        # 升力系数
        'cl1': {
            'cols': ['cl1_mean'],
            'plot': {'title': r'$\overline{C}_{LF}$', 'cmap': 'RdYlGn_r', 'vmin': -1, 'vmax': 1, 'fmt': '{:.2f}'}
        },
        'cl2': {
            'cols': ['cl2_mean'],
            'plot': {'title': r'$\overline{C}_{LH}$', 'cmap': 'RdYlGn_r', 'vmin': -1, 'vmax': 1, 'fmt': '{:.2f}'}
        },
        'cl_total': {
            'cols': ['cl_total_mean'],
            'plot': {'title': r'$\overline{C}_{LS}$', 'cmap': 'RdYlGn_r', 'vmin': -1, 'vmax': 1, 'fmt': '{:.2f}'}
        },

        # 推力系数
        'ct1': {
            'cols': ['ct1_mean'],
            'plot': {'title': r'$\overline{C}_{TF}$', 'cmap': 'RdBu_r', 'vmin': -1, 'vmax': 7, 'fmt': '{:.2f}'}
        },
        'ct2': {
            'cols': ['ct2_mean'],
            'plot': {'title': r'$\overline{C}_{TH}$', 'cmap': 'RdBu_r', 'vmin': -1, 'vmax': 7, 'fmt': '{:.2f}'}
        },
        'ct_total': {
            'cols': ['ct_total_mean'],
            'plot': {'title': r'$\overline{C}_{TS}$', 'cmap': 'RdBu_r', 'vmin': -1, 'vmax': 7, 'fmt': '{:.2f}'}
        },

        # 功率系数
        'cp1': {
            'cols': ['cp1_mean'],
            'plot': {'title': r'$\overline{C}_{PF}$', 'cmap': 'YlOrRd', 'vmin': 0, 'vmax': 8, 'fmt': '{:.2f}'}
        },
        'cp2': {
            'cols': ['cp2_mean'],
            'plot': {'title': r'$\overline{C}_{PH}$', 'cmap': 'YlOrRd', 'vmin': 0, 'vmax': 8, 'fmt': '{:.2f}'}
        },
        'cp_total': {
            'cols': ['cp_total_mean'],
            'plot': {'title': r'$\overline{C}_{PS}$', 'cmap': 'YlOrRd', 'vmin': 0, 'vmax': 8, 'fmt': '{:.2f}'}
        },
    }

    FIG_SIZE = (6.5, 5)
    DPI = 300
    FONT_SIZE = 9
    OUTPUT_FORMATS = ['png']


def build_path(st, k, a, dc):
    """构建CSV文件路径"""
    dir_name = f'st{st}k{k}/st{st}k{k}a{a}/st{st}k{k}a{a}dc{dc}/'
    return os.path.join(
        Config.BASE_DIR,
        f'k{k}',
        dir_name,
        'One_period',
        'selected_cycle_mean.csv'
    )


def get_all_target_cols():
    """获取所有需要读取的列名（去重）"""
    cols = set()
    for metric in Config.METRICS.values():
        cols.update(metric['cols'])
    return list(cols)


def reshape_to_matrix(df, value_col, st_list, k_list, a_val=0, dc_val=1):
    """将长格式DataFrame重塑为 st×k 矩阵"""
    matrix = np.full((len(st_list), len(k_list)), np.nan)

    mask = (
        (df['st'].isin(st_list)) &
        (df['k'].isin(k_list)) &
        (df['a'] == a_val) &
        (df['dc'] == dc_val)
    )
    subset = df[mask]

    for i, st in enumerate(st_list):
        for j, k in enumerate(k_list):
            data = subset[(subset['st'] == st) & (subset['k'] == k)]
            if len(data) > 0 and value_col in data.columns:
                matrix[i, j] = data[value_col].iloc[0]

    return matrix


def calculate_missing_rate(matrix):
    """计算缺失率"""
    total = matrix.size
    missing = np.isnan(matrix).sum()
    return missing / total if total > 0 else 0


def load_data():
    """批量读取所有参数组合的数据"""
    all_cols = get_all_target_cols()
    results = []
    stats = {'success': 0, 'missing': 0, 'error': 0}

    for st, k, a, dc in product(Config.ST_LIST, Config.K_LIST, Config.A_LIST, Config.DC_LIST):
        csv_path = build_path(st, k, a, dc)

        if not os.path.exists(csv_path):
            stats['missing'] += 1
            continue

        try:
            df = pd.read_csv(csv_path)

            if len(df) == 0:
                stats['error'] += 1
                continue

            row = {'st': st, 'k': k, 'a': a, 'dc': dc}
            for col in all_cols:
                row[col] = df[col].iloc[0] if col in df.columns else np.nan

            results.append(row)
            stats['success'] += 1

        except Exception as e:
            print(f"读取错误 {csv_path}: {e}")
            stats['error'] += 1

    if not results:
        raise ValueError("未读取到任何有效数据！")

    result_df = pd.DataFrame(results)

    print("数据加载完成")
    print(f"成功: {stats['success']} | 文件缺失: {stats['missing']} | 读取错误: {stats['error']}")
    print(f"数据框形状: {result_df.shape}")

    return result_df, stats


def plot_heatmap(matrix, st_list, k_list, config, output_name):
    """绘制 st-k 参数矩阵热力图"""
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE, dpi=Config.DPI)

    cmap = config['cmap']
    vmin, vmax = config['vmin'], config['vmax']

    if vmin < 0 < vmax and cmap in ['RdYlGn_r', 'RdBu_r', 'coolwarm']:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im = ax.pcolor(
            np.arange(len(k_list) + 1),
            np.arange(len(st_list) + 1),
            matrix,
            cmap=cmap,
            norm=norm,
            edgecolors='white',
            linewidths=0.5
        )
    else:
        im = ax.pcolor(
            np.arange(len(k_list) + 1),
            np.arange(len(st_list) + 1),
            matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors='white',
            linewidths=0.5
        )

    for i in range(len(st_list)):
        for j in range(len(k_list)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j + 0.5, i + 0.5,
                    config['fmt'].format(val),
                    ha='center', va='center',
                    fontsize=8, color='black', fontweight='bold'
                )
            else:
                ax.text(
                    j + 0.5, i + 0.5,
                    'N/A',
                    ha='center', va='center',
                    fontsize=6, color='gray', alpha=0.6
                )

    ax.set_xticks(np.arange(len(k_list)) + 0.5)
    ax.set_xticklabels([f'{k:g}' for k in k_list], fontsize=Config.FONT_SIZE)
    ax.set_yticks(np.arange(len(st_list)) + 0.5)
    ax.set_yticklabels([f'{st:g}' for st in st_list], fontsize=Config.FONT_SIZE)

    ax.set_xlabel(r'Reduced frequency $k$', fontsize=10, labelpad=5)
    ax.set_ylabel(r'Strouhal number $St$', fontsize=10, labelpad=5)
    ax.set_title(config['title'], fontsize=12, pad=10)

    ax.set_xlim(0, len(k_list))
    ax.set_ylim(0, len(st_list))
    ax.invert_yaxis()

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Value', rotation=270, labelpad=15, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    plt.tight_layout()

    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for fmt in Config.OUTPUT_FORMATS:
        save_path = os.path.join(Config.OUTPUT_DIR, f'{output_name}.{fmt}')
        plt.savefig(save_path, dpi=Config.DPI if fmt == 'png' else None,
                    bbox_inches='tight', facecolor='white')

    plt.close()


def plot_all_heatmaps(result_df):
    """绘制所有指标热图，只输出到 heatmap 文件夹"""
    stats = {'success': 0, 'skip': 0}

    for short_name, metric in Config.METRICS.items():
        col_name = metric['cols'][0]
        plot_config = metric['plot']

        if col_name not in result_df.columns:
            print(f"列 '{col_name}' 不存在，跳过 {short_name}")
            stats['skip'] += 1
            continue

        matrix = reshape_to_matrix(
            result_df,
            col_name,
            Config.ST_LIST,
            Config.K_LIST,
            a_val=Config.A_LIST[0],
            dc_val=Config.DC_LIST[0]
        )

        missing_rate = calculate_missing_rate(matrix)
        if missing_rate > 0.95:
            print(f"{short_name}: 缺失率 {missing_rate:.1%}，跳过")
            stats['skip'] += 1
            continue

        config = plot_config.copy()

        if config['vmin'] is None or config['vmax'] is None:
            valid_vals = matrix[~np.isnan(matrix)]
            if len(valid_vals) > 0:
                margin = (valid_vals.max() - valid_vals.min()) * 0.1
                config['vmin'] = valid_vals.min() - margin
                config['vmax'] = valid_vals.max() + margin

        output_name = f'heatmap_{short_name}_a{Config.A_LIST[0]}dc{Config.DC_LIST[0]}'
        plot_heatmap(matrix, Config.ST_LIST, Config.K_LIST, config, output_name)

        stats['success'] += 1
        print(f"✓ {short_name}: 缺失率 {missing_rate:.1%}")

    print(f"\n热力图绘制完成！成功: {stats['success']}, 跳过: {stats['skip']}")
    return stats


def export_summary(result_df):
    """只导出一个CSV汇总文件到 heatmap 文件夹"""
    export_cols = ['st', 'k', 'a', 'dc'] + get_all_target_cols()
    summary_path = os.path.join(Config.OUTPUT_DIR, 'processed_results_summary.csv')
    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    result_df[export_cols].to_csv(summary_path, index=False, float_format='%.6e')
    print(f"数据已导出: {summary_path}")


if __name__ == '__main__':
    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    result_df, load_stats = load_data()
    plot_stats = plot_all_heatmaps(result_df)
    export_summary(result_df)