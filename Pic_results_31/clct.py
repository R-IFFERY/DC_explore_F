# -*- coding: utf-8 -*-
"""
Plot smoothed phase-averaged instantaneous lift and thrust coefficients.

Cases:
1. k=0.8, St=0.6: cycles 16-20
2. k=2.0, St=0.3: cycles 16-20
3. k=0.5, St=0.6: cycles 11-15

Processing:
1. Extract five cycles.
2. Interpolate each cycle onto a unified phase grid.
3. Smooth each individual cycle using a periodic moving-average window.
4. Calculate the phase-resolved mean and sample standard deviation.
5. Plot the mean curve and mean +/- one standard-deviation band.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Basic parameters
# ============================================================
a = 0
dc = 1

base_root = r'E:/Rensq/DC_explore_F/data_1Tflow'
save_root = r'E:/Rensq/DC_explore_F/results_a0/fig8'
os.makedirs(save_root, exist_ok=True)

# Time corresponding to the beginning of cycle 1
TIME_ORIGIN = 0.0

# Number of phase points, including both t/T=0 and t/T=1
N_PHASE_POINTS = 1001

# Minimum number of raw samples required in one cycle
MIN_SAMPLES_PER_CYCLE = 10

# Sample standard deviation
STD_DDOF = 1


# ============================================================
# Moving-average smoothing
# ============================================================
APPLY_SMOOTHING = True

# Must be an odd integer.
# 21 points correspond to approximately 2% of one cycle when
# N_PHASE_POINTS=1001.
SMOOTH_WINDOW = 21

# Error-band transparency
STD_ALPHA = 0.30

# Mean-line width
MEAN_LINEWIDTH = 1.00


# ============================================================
# Cases
# ============================================================
cases = [
    {
        'k': 0.8,
        'st': 0.6,
        'cycle_start': 16,
        'cycle_end': 20,
        'color': 'tab:blue',
    },
    {
        'k': 2.0,
        'st': 0.3,
        'cycle_start': 16,
        'cycle_end': 20,
        'color': 'tab:orange',
    },
    {
        'k': 0.5,
        'st': 0.6,
        'cycle_start': 11,
        'cycle_end': 15,
        'color': 'tab:green',
    },
]


# ============================================================
# Figure style
# ============================================================
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,

    'font.family': 'serif',
    'font.serif': ['Times New Roman'],

    'text.usetex': True,
    'pgf.texsystem': 'pdflatex',
    'text.latex.preamble': (
        r'\usepackage{amsmath}'
        r'\usepackage{newtxtext,newtxmath}'
    ),

    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})


def format_num(value):
    """Format values according to the directory naming convention."""
    return f'{value:.1f}'


def build_force_path(k, st):
    """Construct the path to force.dat."""
    k_str = format_num(k)
    st_str = format_num(st)

    force_dir = os.path.join(
        base_root,
        f'k{k_str}',
        f'st{st_str}k{k_str}',
        f'st{st_str}k{k_str}a{a}',
        f'st{st_str}k{k_str}a{a}dc{int(dc)}',
        f'k{k_str}st{st_str}'
    )

    return os.path.join(force_dir, 'force.dat')


def read_force_data(force_path):
    """
    Read total instantaneous lift and thrust coefficients.

    The normalization remains consistent with the original code:

        C_L,S = F_y,total / 2
        C_T,S = -F_x,total / 2
    """
    if not os.path.exists(force_path):
        raise FileNotFoundError(
            f'找不到文件：{force_path}'
        )

    df = pd.read_csv(
        force_path,
        header=None,
        sep=r'\s+',
        engine='python',
        on_bad_lines='skip'
    )

    if df.shape[1] < 9:
        raise ValueError(
            f'数据列数不足：当前为 {df.shape[1]} 列，至少需要9列。'
        )

    required_columns = [0, 7, 8]

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors='coerce'
        )

    df = df.dropna(subset=required_columns)

    time = df.iloc[:, 0].to_numpy(dtype=float)
    force_x_total = df.iloc[:, 7].to_numpy(dtype=float)
    force_y_total = df.iloc[:, 8].to_numpy(dtype=float)

    cl_total = force_y_total / 2.0
    ct_total = -force_x_total / 2.0

    # Sort by time
    order = np.argsort(time)

    time = time[order]
    cl_total = cl_total[order]
    ct_total = ct_total[order]

    # Remove duplicate time points
    time, unique_indices = np.unique(
        time,
        return_index=True
    )

    cl_total = cl_total[unique_indices]
    ct_total = ct_total[unique_indices]

    return time, cl_total, ct_total


def periodic_moving_average(signal, window_size):
    """
    Apply a circular moving average to one periodic signal.

    The signal contains both phase 0 and phase 1. Because these two
    points represent the same phase, smoothing is performed on
    signal[:-1]. The first smoothed point is then appended to the end.

    Circular padding avoids artificial edge deformation near t/T=0
    and t/T=1.
    """
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError('滑动平均输入必须为一维数组。')

    if window_size <= 1:
        return signal.copy()

    if window_size % 2 == 0:
        raise ValueError(
            f'SMOOTH_WINDOW必须为奇数，当前值为 {window_size}。'
        )

    if len(signal) < 3:
        raise ValueError('输入数据点数不足，无法进行平滑。')

    # Remove the repeated phase endpoint t/T=1
    periodic_core = signal[:-1]

    if window_size >= len(periodic_core):
        raise ValueError(
            f'滑动窗口 {window_size} 不能大于或等于'
            f'单周期有效点数 {len(periodic_core)}。'
        )

    half_window = window_size // 2

    # Circular padding
    padded_signal = np.pad(
        periodic_core,
        pad_width=(half_window, half_window),
        mode='wrap'
    )

    kernel = np.ones(window_size, dtype=float) / window_size

    smoothed_core = np.convolve(
        padded_signal,
        kernel,
        mode='valid'
    )

    # Enforce periodic closure: value at phase 1 equals phase 0
    smoothed_signal = np.concatenate([
        smoothed_core,
        [smoothed_core[0]]
    ])

    return smoothed_signal


def interpolate_cycle(
    time,
    signal,
    cycle_number,
    period,
    phase_grid
):
    """
    Extract one cycle and interpolate it onto the unified phase grid.

    Cycle definition:
        cycle 1: 0T <= t <= 1T
        cycle 2: 1T <= t <= 2T
        ...
    """
    cycle_t_start = (
        TIME_ORIGIN
        + (cycle_number - 1) * period
    )

    cycle_t_end = (
        TIME_ORIGIN
        + cycle_number * period
    )

    tolerance = max(
        1.0e-10,
        period * 1.0e-8
    )

    mask = (
        (time >= cycle_t_start - tolerance)
        & (time <= cycle_t_end + tolerance)
    )

    cycle_time = time[mask]
    cycle_signal = signal[mask]

    valid = (
        np.isfinite(cycle_time)
        & np.isfinite(cycle_signal)
    )

    cycle_time = cycle_time[valid]
    cycle_signal = cycle_signal[valid]

    if len(cycle_time) < MIN_SAMPLES_PER_CYCLE:
        raise ValueError(
            f'第 {cycle_number} 周期仅找到'
            f'{len(cycle_time)}个有效数据点。'
        )

    cycle_phase = (
        cycle_time - cycle_t_start
    ) / period

    valid_phase = (
        (cycle_phase >= -1.0e-8)
        & (cycle_phase <= 1.0 + 1.0e-8)
    )

    cycle_phase = cycle_phase[valid_phase]
    cycle_signal = cycle_signal[valid_phase]

    cycle_phase = np.clip(
        cycle_phase,
        0.0,
        1.0
    )

    # Sort according to phase
    order = np.argsort(cycle_phase)

    cycle_phase = cycle_phase[order]
    cycle_signal = cycle_signal[order]

    # Remove duplicate phase points
    cycle_phase, unique_indices = np.unique(
        cycle_phase,
        return_index=True
    )

    cycle_signal = cycle_signal[unique_indices]

    if len(cycle_phase) < 2:
        raise ValueError(
            f'第 {cycle_number} 周期无法完成相位插值。'
        )

    if cycle_phase[0] > 0.05 or cycle_phase[-1] < 0.95:
        raise ValueError(
            f'第 {cycle_number} 周期覆盖不完整：'
            f'{cycle_phase[0]:.4f}--{cycle_phase[-1]:.4f}'
        )

    interpolated_signal = np.interp(
        phase_grid,
        cycle_phase,
        cycle_signal
    )

    return interpolated_signal


def calculate_cycle_statistics(
    time,
    signal,
    k,
    cycle_start,
    cycle_end,
    phase_grid
):
    """
    Calculate the smoothed phase-resolved mean and sample standard deviation.

    Each individual cycle is smoothed before calculating the mean and
    standard deviation. Consequently, the mean curve and error band
    are based on the same smoothed data.
    """
    period = np.pi / k

    raw_cycle_signals = []
    smoothed_cycle_signals = []

    for cycle_number in range(
        cycle_start,
        cycle_end + 1
    ):
        raw_cycle_signal = interpolate_cycle(
            time=time,
            signal=signal,
            cycle_number=cycle_number,
            period=period,
            phase_grid=phase_grid
        )

        raw_cycle_signals.append(raw_cycle_signal)

        if APPLY_SMOOTHING:
            smoothed_cycle_signal = periodic_moving_average(
                raw_cycle_signal,
                SMOOTH_WINDOW
            )
        else:
            smoothed_cycle_signal = raw_cycle_signal.copy()

        smoothed_cycle_signals.append(
            smoothed_cycle_signal
        )

    raw_cycle_signals = np.asarray(
        raw_cycle_signals,
        dtype=float
    )

    smoothed_cycle_signals = np.asarray(
        smoothed_cycle_signals,
        dtype=float
    )

    # Five-cycle smoothed phase mean
    mean_signal = np.mean(
        smoothed_cycle_signals,
        axis=0
    )

    # Phase-resolved sample standard deviation of smoothed cycles
    std_signal = np.std(
        smoothed_cycle_signals,
        axis=0,
        ddof=STD_DDOF
    )

    return (
        mean_signal,
        std_signal,
        raw_cycle_signals,
        smoothed_cycle_signals
    )


def load_and_process_case(case, phase_grid):
    """Read and process one parameter combination."""
    k = case['k']
    st = case['st']
    cycle_start = case['cycle_start']
    cycle_end = case['cycle_end']

    force_path = build_force_path(k, st)

    print(
        f'\n[读取] k={k:.1f}, St={st:.1f}, '
        f'周期={cycle_start}-{cycle_end}'
    )
    print(f'       {force_path}')

    time, cl_total, ct_total = read_force_data(
        force_path
    )

    (
        cl_mean,
        cl_std,
        cl_cycles_raw,
        cl_cycles_smoothed
    ) = calculate_cycle_statistics(
        time=time,
        signal=cl_total,
        k=k,
        cycle_start=cycle_start,
        cycle_end=cycle_end,
        phase_grid=phase_grid
    )

    (
        ct_mean,
        ct_std,
        ct_cycles_raw,
        ct_cycles_smoothed
    ) = calculate_cycle_statistics(
        time=time,
        signal=ct_total,
        k=k,
        cycle_start=cycle_start,
        cycle_end=cycle_end,
        phase_grid=phase_grid
    )

    print(
        f'       平滑窗口：{SMOOTH_WINDOW}个相位点'
    )

    print(
        f'       CL标准差：'
        f'mean={np.mean(cl_std):.6e}, '
        f'max={np.max(cl_std):.6e}'
    )

    print(
        f'       CT标准差：'
        f'mean={np.mean(ct_std):.6e}, '
        f'max={np.max(ct_std):.6e}'
    )

    result = case.copy()

    result.update({
        'force_path': force_path,

        'cl_mean': cl_mean,
        'cl_std': cl_std,
        'cl_cycles_raw': cl_cycles_raw,
        'cl_cycles_smoothed': cl_cycles_smoothed,

        'ct_mean': ct_mean,
        'ct_std': ct_std,
        'ct_cycles_raw': ct_cycles_raw,
        'ct_cycles_smoothed': ct_cycles_smoothed,
    })

    return result


def draw_error_band(
    ax,
    x,
    mean_signal,
    std_signal,
    color,
    band_zorder=1
):
    """
    Draw one independent mean +/- one-standard-deviation band.

    No boundary lines are drawn.
    """
    lower_bound = mean_signal - std_signal
    upper_bound = mean_signal + std_signal

    ax.fill_between(
        x,
        lower_bound,
        upper_bound,
        facecolor=color,
        edgecolor='none',
        linewidth=0.0,
        alpha=STD_ALPHA,
        interpolate=True,
        zorder=band_zorder
    )

    return lower_bound, upper_bound


def draw_mean_curve(
    ax,
    x,
    mean_signal,
    color,
    label=None
):
    """Draw the smoothed mean curve as a solid line."""
    line, = ax.plot(
        x,
        mean_signal,
        color=color,
        linestyle='-',
        linewidth=MEAN_LINEWIDTH,
        label=label,
        zorder=10
    )

    return line


def plot_results(results, phase_grid):
    """Plot lift and thrust coefficients in a 1 x 2 layout."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.5, 2),
        sharex=True
    )

    ax_cl, ax_ct = axes

    legend_handles = []
    legend_labels = []

    # ========================================================
    # Draw the three independent error bands
    # ========================================================
    for index, result in enumerate(results):
        color = result['color']
        band_zorder = 1 + index * 0.1

        draw_error_band(
            ax=ax_cl,
            x=phase_grid,
            mean_signal=result['cl_mean'],
            std_signal=result['cl_std'],
            color=color,
            band_zorder=band_zorder
        )

        draw_error_band(
            ax=ax_ct,
            x=phase_grid,
            mean_signal=result['ct_mean'],
            std_signal=result['ct_std'],
            color=color,
            band_zorder=band_zorder
        )

    # ========================================================
    # Draw the three smoothed mean curves
    # ========================================================
    for result in results:
        k = result['k']
        st = result['st']
        color = result['color']

        label = rf'$k={k:.1f},\ St={st:.1f}$'

        line_cl = draw_mean_curve(
            ax=ax_cl,
            x=phase_grid,
            mean_signal=result['cl_mean'],
            color=color,
            label=label
        )

        draw_mean_curve(
            ax=ax_ct,
            x=phase_grid,
            mean_signal=result['ct_mean'],
            color=color
        )

        legend_handles.append(line_cl)
        legend_labels.append(label)

    # ========================================================
    # Axis labels and fixed ranges
    # ========================================================
    ax_cl.set_xlabel(r'$t/T$')
    ax_ct.set_xlabel(r'$t/T$')

    ax_cl.set_ylabel(r'$C_{LS}$', rotation=0)
    ax_ct.set_ylabel(r'$C_{TS}$', rotation=0)

    # Lift coefficient
    ax_cl.set_xlim(0.0, 1.0)
    ax_cl.set_xticks([0.0, 0.5, 1.0])

    ax_cl.set_ylim(-10.0, 10.0)
    ax_cl.set_yticks([
        -10.0,
        -5.0,
        0.0,
        5.0,
        10.0
    ])

    # Thrust coefficient
    ax_ct.set_xlim(0.0, 1.0)
    ax_ct.set_xticks([0.0, 0.5, 1.0])

    ax_ct.set_ylim(-2.0, 6.0)
    ax_ct.set_yticks([
        -2.0,
        0.0,
        2.0,
        4.0,
        6.0
    ])

    for ax in axes:
        # ========================================================
        # Downstroke stage: 0 <= t/T <= 0.5
        # ========================================================
        ax.axvspan(
            0.0,
            0.5,
            facecolor='lightgray',  # 浅灰色背景
            edgecolor='none',
            alpha=1.0,
            zorder=-10
        )

        # Zero reference line
        ax.axhline(
            0.0,
            color='0.35',
            linewidth=0.65,
            linestyle=':',
            zorder=0
        )

        ax.grid(
            True,
            which='major',
            linestyle=':',
            linewidth=0.45,
            alpha=0.45,
            zorder=0
        )

        ax.tick_params(
            axis='both',
            which='both',
            direction='in',
            top=True,
            right=True
        )

    # ========================================================
    # Panel labels
    # ========================================================
    ax_cl.text(
        -0.145,
        1.045,
        r'$(d)$',
        transform=ax_cl.transAxes,
        ha='left',
        va='bottom',
        fontsize=10,
        clip_on=False
    )

    ax_ct.text(
        -0.145,
        1.045,
        r'$(e)$',
        transform=ax_ct.transAxes,
        ha='left',
        va='bottom',
        fontsize=10,
        clip_on=False
    )

    # ========================================================
    # Legend: lines only, inside upper-right of the left panel
    # ========================================================
    ax_cl.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='upper right',
        frameon=False,
        handlelength=2.5,
        borderaxespad=0.6
    )

    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.175,
        top=0.93,
        wspace=0.28
    )

    output_base = os.path.join(
        save_root,
        'phase_averaged_CL_CT_smoothed'
    )

    png_path = output_base + '.png'
    svg_path = output_base + '.svg'
    pdf_path = output_base + '.pdf'

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches='tight'
    )

    fig.savefig(
        svg_path,
        bbox_inches='tight'
    )

    fig.savefig(
        pdf_path,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(f'\n[完成] PNG：{png_path}')
    print(f'[完成] SVG：{svg_path}')
    print(f'[完成] PDF：{pdf_path}')


def main():
    phase_grid = np.linspace(
        0.0,
        1.0,
        N_PHASE_POINTS,
        endpoint=True
    )

    results = []

    for case in cases:
        try:
            result = load_and_process_case(
                case=case,
                phase_grid=phase_grid
            )

            results.append(result)

        except Exception as error:
            print(
                f'\n[报错] k={case["k"]:.1f}, '
                f'St={case["st"]:.1f}：{error}'
            )

    if len(results) != len(cases):
        print(
            f'\n[终止绘图] 仅成功读取'
            f'{len(results)}/{len(cases)}组数据。'
        )
        return

    plot_results(
        results=results,
        phase_grid=phase_grid
    )

    print(
        f'\n全部处理完成，结果保存于：'
        f'{save_root}'
    )


if __name__ == '__main__':
    main()