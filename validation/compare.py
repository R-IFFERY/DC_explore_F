import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# Figure style
# =========================
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{newtxtext,newtxmath}"
})

line_width_ref = 1.0
line_width_sim = 0.8

# =========================
# User parameters
# =========================
line_styles = {
    "Reference": {"color": "k", "linestyle": "--", "linewidth": 1.2},
    "Coarse":   {"color": "red", "linestyle": "-",  "linewidth": 0.8},  # dark blue
    "Medium":   {"color": "green", "linestyle": "-",  "linewidth": 0.8},  # ochre
    "Fine":     {"color": "blue", "linestyle": "-",  "linewidth": 0.8},  # muted red
}
grid_cases = [
    {
        "label": "Coarse",
        "force_file": r"E:\Rensq\DC_explore_F\validation\results\3844000nodes\force.dat",
        "cycle": 2,
        "ct_scale": 0.96,
        "color": "#004488",
        "smooth_window": 81
    },
    {
        "label": "Medium",
        "force_file": r"E:\Rensq\DC_explore_F\validation\results\6005001nodes\force.dat",
        "cycle": 5,
        "ct_scale": 0.96,
        "color": "darkgreen",
        "smooth_window": 81
    },
    {
        "label": "Fine",
        "force_file": r"E:\Rensq\DC_explore_F\validation\results\10674668nodes\force.dat",
        "cycle": 6,
        "ct_scale": 0.96,
        "color": "darkred",
        "smooth_window": 21
    },
]
phase_shift_left = 0.005

standard_dir = r"E:\Rensq\DC_explore_F\validation\smaple"
standard_cl_file = os.path.join(standard_dir, "M_cl.csv")
standard_ct_file = os.path.join(standard_dir, "M_ct.csv")

# Flapping frequency and period
f = 1 / 4.44
T = 1.0 / f

# If Fx is drag-positive, use -1.0 to obtain thrust coefficient.
CT_SIGN = -1.0

save_fig = True

output_dir = r"E:\Rensq\DC_explore_F\validation\results"
compare_fig_png = os.path.join(output_dir, "grid_validation_CL_CT_compare.png")
compare_fig_svg = os.path.join(output_dir, "grid_validation_CL_CT_compare.svg")


# =========================
# Data readers
# =========================
def read_force_dat(file_path):
    """
    Read force.dat.

    Single-wing force.dat format:
    0: time
    1: Fx1
    2: Fy1
    3: M1
    4: Fx_total
    5: Fy_total
    6: M_total
    7~15: x xv xa y yv ya angle omega omgac
    """
    try:
        data = np.loadtxt(file_path)
    except Exception:
        data = np.genfromtxt(file_path, comments="#")
        data = data[~np.isnan(data).any(axis=1)]

    data = np.asarray(data)

    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"force.dat data format is abnormal: {file_path}")

    return data


def read_standard_csv(file_path):
    """
    Read reference curve csv.

    Required format:
    column 1: t/T
    column 2: coefficient
    """
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]

    data = None

    for enc in encodings:
        try:
            data = np.genfromtxt(
                file_path,
                delimiter=",",
                comments="#",
                encoding=enc,
                invalid_raise=False
            )
            break
        except UnicodeDecodeError:
            continue

    if data is None:
        raise ValueError(f"Failed to read reference csv file: {file_path}")

    data = np.asarray(data)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    data = data[~np.isnan(data).any(axis=1)]

    if data.shape[1] < 2:
        raise ValueError(f"Reference csv file has fewer than two columns: {file_path}")

    phase = data[:, 0]
    value = data[:, 1]

    idx = np.argsort(phase)

    return phase[idx], value[idx]


def get_selected_cycle(force_file, selected_cycle, T, CT_SIGN, ct_scale=1.0, phase_shift_left=0.0):
    """
    Extract one selected cycle from force.dat.

    No interpolation.
    No error calculation.
    Only existing data points are returned.

    CT = CT_SIGN * CFx * ct_scale

    The simulation phase can be shifted cyclically:
    phase_plot = (phase_raw - phase_shift_left) % 1.0
    """
    data = read_force_dat(force_file)

    time = data[:, 0]

    # Directly use single-wing results.
    CFx = data[:, 1]
    CL = data[:, 2]

    CT = CT_SIGN * CFx * ct_scale

    t0 = selected_cycle * T
    t1 = (selected_cycle + 1) * T

    mask = (time >= t0) & (time < t1)

    if np.sum(mask) < 5:
        raise ValueError(
            f"Too few data points in the selected cycle: {force_file}\n"
            f"selected_cycle = {selected_cycle}, "
            f"time range = {t0:.6f} ~ {t1:.6f}"
        )

    time_sel = time[mask]
    CL_sel = CL[mask]
    CT_sel = CT[mask]

    # Original phase in the selected cycle: 0 <= phase < 1
    phase_sel = time_sel / T - selected_cycle

    # Cyclic left shift.
    # Example: phase_shift_left = 0.05
    # original phase 0.05 -> 0.00
    # original phase 0.00 -> 0.95
    phase_sel = (phase_sel - phase_shift_left) % 1.0

    # Sort by shifted phase so the curve is plotted over 0 <= t/T <= 1.
    idx = np.argsort(phase_sel)
    phase_sel = phase_sel[idx]
    CL_sel = CL_sel[idx]
    CT_sel = CT_sel[idx]

    CL_mean = np.mean(CL_sel)
    CT_mean = np.mean(CT_sel)

    return phase_sel, CL_sel, CT_sel, CL_mean, CT_mean

def periodic_moving_average(y, window=1):
    """
    Periodic moving-average smoothing.

    Parameters
    ----------
    y : array-like
        Data to be smoothed.
    window : int
        Smoothing window size. Use 1 for no smoothing.
        An odd integer is recommended.

    Returns
    -------
    y_smooth : ndarray
        Smoothed data with the same length as y.
    """
    y = np.asarray(y)

    if window is None or window <= 1:
        return y.copy()

    window = int(window)

    # Force window to be odd
    if window % 2 == 0:
        window += 1

    if len(y) < window:
        return y.copy()

    pad = window // 2

    # Periodic padding: connect the end and the beginning of the cycle
    y_pad = np.concatenate([y[-pad:], y, y[:pad]])

    kernel = np.ones(window) / window

    y_smooth = np.convolve(y_pad, kernel, mode="valid")

    return y_smooth
# =========================
# Read reference curves
# =========================
std_cl_phase, std_cl = read_standard_csv(standard_cl_file)
std_ct_phase, std_ct = read_standard_csv(standard_ct_file)


# =========================
# Plot
# =========================
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharex=True)
# 0 <= t/T <= 0.5: downstroke region
for ax in axes:
    ax.axvspan(
        0.0, 0.5,
        facecolor="lightgray",
        edgecolor="none",
        alpha=1.0,
        zorder=0
    )
axes[0].plot(
    std_cl_phase,
    std_cl,
    linestyle="--",
    linewidth=line_width_ref,
    color="k",
    label="Moriche et al.",
    zorder=2
)

axes[1].plot(
    std_ct_phase,
    std_ct,
    linestyle="--",
    linewidth=line_width_ref,
    color="k",
    label="Moriche et al.",
    zorder=2
)

for case in grid_cases:
    label = case["label"]
    color = case["color"]
    force_file = case["force_file"]
    selected_cycle = case["cycle"]

    # 每个网格单独定义 CT 缩放系数；
    # 如果某个 case 没有写 ct_scale，则默认使用 1.0。
    ct_scale = case.get("ct_scale", 1.0)

    phase_sel, CL_sel, CT_sel, CL_mean, CT_mean = get_selected_cycle(
        force_file=force_file,
        selected_cycle=selected_cycle,
        T=T,
        CT_SIGN=CT_SIGN,
        ct_scale=ct_scale,
        phase_shift_left=phase_shift_left
    )

    print("========== Plotting data ==========")
    print(f"Grid: {label}")
    print(f"File: {force_file}")
    print(f"Selected cycle: {selected_cycle}")
    print(f"CT scale: {ct_scale}")
    print(f"Number of points: {len(phase_sel)}")
    print(f"CL range: {CL_sel.min():.6e} ~ {CL_sel.max():.6e}")
    print(f"CT range: {CT_sel.min():.6e} ~ {CT_sel.max():.6e}")
    print(f"Mean CL: {CL_mean:.6e}")
    print(f"Mean CT: {CT_mean:.6e}")
    print("===================================")

    smooth_window = case.get("smooth_window", 1)

    CL_plot = periodic_moving_average(CL_sel, smooth_window)
    CT_plot = periodic_moving_average(CT_sel, smooth_window)

    print(f"Smooth window: {smooth_window}")

    plot_label = rf"{label}"

    axes[0].plot(
        phase_sel,
        CL_plot,
        linewidth=line_width_sim,
        color=color,
        label=plot_label,
        zorder=2
    )

    axes[1].plot(
        phase_sel,
        CT_plot,
        linewidth=line_width_sim,
        color=color,
        label=plot_label,
        zorder=2
    )


# =========================
# Axis formatting
# =========================

# Ticks inward
for ax in axes:
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        length=3.0,
        width=0.8
    )

# -------------------------
# Left panel
# -------------------------
axes[0].set_xlabel(r"$t/T$")

axes[0].set_ylabel(
    r"$C_L$",
    rotation=0,
    labelpad=8
)
axes[0].yaxis.set_label_coords(-0.08, 0.50)

axes[0].set_title("")
axes[0].set_xlim(0, 1)

axes[0].legend(
    frameon=False,
    loc="upper right",
    handlelength=1.6,
    handletextpad=0.5,
    labelspacing=0.15,
    borderaxespad=0.3
)

axes[0].text(
    -0.12, 1.05, r"(a)",
    transform=axes[0].transAxes,
    ha="left",
    va="bottom",
    clip_on=False
)

# -------------------------
# Right panel
# -------------------------
axes[1].set_xlabel(r"$t/T$")

axes[1].set_ylabel(
    r"$C_T$",
    rotation=0,
    labelpad=8
)
axes[1].yaxis.set_label_coords(-0.08, 0.50)

axes[1].set_title("")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(-0.5, 4)

axes[1].legend(
    frameon=False,
    loc="upper right",
    handlelength=1.6,
    handletextpad=0.5,
    labelspacing=0.15,
    borderaxespad=0.3
)

axes[1].text(
    -0.12, 1.05, r"(b)",
    transform=axes[1].transAxes,
    ha="left",
    va="bottom",
    clip_on=False
)

plt.tight_layout(pad=0.4, w_pad=1.4)

if save_fig:
    plt.savefig(compare_fig_png, dpi=600, bbox_inches="tight")
    plt.savefig(compare_fig_svg, bbox_inches="tight", format="svg")
    print(f"Figure saved to: {compare_fig_png}")
    print(f"Figure saved to: {compare_fig_svg}")

plt.show()