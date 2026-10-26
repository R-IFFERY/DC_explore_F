# -*- coding: utf-8 -*-
"""
Plot continuous and intermittent tandem-wing kinematics.

Layout:
    Columns:
        1. Continuous
        2. DC = 0.75
        3. DC = 0.50
        4. DC = 0.25

    Rows:
        1. Heaving motion
        2. Pitching motion

Line colors:
    Forewing: black
    Hindwing: red
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# User settings
# ============================================================
USE_TEX = True
SAVE_FIG = True

OUTPUT_DIR = r"/Pic_results_31"
OUTPUT_BASENAME = "intermittent_kinematics_2x4"

# Column order
CASE_LIST = ["continuous", 0.75, 0.50, 0.25]

# Motion parameters
a = 11.0                       # Hindwing mean pitching angle, deg
alpha = np.radians(a)
theta0 = np.radians(30.0)      # Pitching amplitude, rad

k_reduced = 1.4                # Reduced frequency
st = 0.3                       # Strouhal number
c = 1.0                        # Chord length
u = 1.0                        # Free-stream velocity
phi = np.radians(90.0)         # Heave-pitch phase difference
ks = 9.0                       # Smoothing coefficient
dt = 1.0e-4                    # Time step

# Line settings
LINEWIDTH = 0.8
FOREWING_COLOR = "black"
HINDWING_COLOR = "red"


# ============================================================
# Figure style
# ============================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": USE_TEX,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
})

if USE_TEX:
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.latex.preamble":
            r"\usepackage{amsmath}"
            r"\usepackage{newtxtext,newtxmath}"
    })


# ============================================================
# Smooth transition functions
# ============================================================
def smooth_window(tau, T, smoothness):
    """
    Smooth envelope for one isolated active cycle:
    inactive -> active -> inactive.
    """
    return (
        -np.tanh(smoothness * tau / T)
        * np.tanh(smoothness * (tau / T - 1.0))
    )


def smooth_start(tau, T, smoothness):
    """
    Smooth transition from inactive motion to continuous active motion.
    """
    return np.tanh(smoothness * tau / T)


def smooth_stop(tau, T, smoothness):
    """
    Smooth transition from continuous active motion to inactive motion.
    """
    return -np.tanh(smoothness * (tau / T - 1.0))


def build_intermittent_envelope(t, T, dc, smoothness):
    """
    Construct the hindwing intermittent envelope over repeated 4T windows.

    DC = 0.25:
        Burst-Coast-Coast-Coast

    DC = 0.50:
        Burst-Coast-Burst-Coast

    DC = 0.75:
        Burst-Burst-Burst-Coast
    """
    envelope = np.zeros_like(t, dtype=float)

    # Local time in each four-cycle intermittent window
    window_id = np.floor(t / (4.0 * T)).astype(int)
    tau4 = t - window_id * 4.0 * T

    if np.isclose(dc, 0.25):
        # Cycle 1: isolated active cycle
        mask1 = (tau4 >= 0.0) & (tau4 < T)
        envelope[mask1] = smooth_window(
            tau4[mask1], T, smoothness
        )

    elif np.isclose(dc, 0.50):
        # Cycle 1: isolated active cycle
        mask1 = (tau4 >= 0.0) & (tau4 < T)
        envelope[mask1] = smooth_window(
            tau4[mask1], T, smoothness
        )

        # Cycle 3: isolated active cycle
        mask3 = (tau4 >= 2.0 * T) & (tau4 < 3.0 * T)
        envelope[mask3] = smooth_window(
            tau4[mask3] - 2.0 * T,
            T,
            smoothness
        )

    elif np.isclose(dc, 0.75):
        # Cycle 1: smooth start
        mask1 = (tau4 >= 0.0) & (tau4 < T)
        envelope[mask1] = smooth_start(
            tau4[mask1], T, smoothness
        )

        # Cycle 2: fully active
        mask2 = (tau4 >= T) & (tau4 < 2.0 * T)
        envelope[mask2] = 1.0

        # Cycle 3: smooth stop
        mask3 = (tau4 >= 2.0 * T) & (tau4 < 3.0 * T)
        envelope[mask3] = smooth_stop(
            tau4[mask3] - 2.0 * T,
            T,
            smoothness
        )

        # Cycle 4 remains inactive

    else:
        raise ValueError(
            "Only DC = 0.25, 0.50 and 0.75 are supported."
        )

    return envelope


# ============================================================
# Motion calculation over one four-cycle window
# ============================================================
frequency = k_reduced * u / (np.pi * c)
T = 1.0 / frequency
omega = 2.0 * np.pi * frequency
h0 = st * u / (2.0 * frequency)

# Complete four-cycle intermittent window
t = np.arange(0.0, 4.0 * T + 0.5 * dt, dt)
t_normalized = t / (4.0 * T)

# ------------------------------------------------------------
# Forewing: continuous motion
# ------------------------------------------------------------
theta_forewing = -theta0 * np.sin(omega * t)
h_forewing = h0 * np.sin(omega * t + phi)

# ------------------------------------------------------------
# Hindwing motions
# ------------------------------------------------------------
motions = {}

# Continuous hindwing
motions["continuous"] = {
    "heave": h0 * np.sin(omega * t + phi) / c,
    "pitch": np.degrees(
        alpha - theta0 * np.sin(omega * t)
    ),
    "title": "Continuous",
}

# Intermittent hindwing cases
for dc in [0.75, 0.50, 0.25]:
    envelope = build_intermittent_envelope(
        t,
        T,
        dc,
        ks,
    )

    # Hindwing heaving motion
    # Inactive state: h = h0
    # Active state:   h = h0*sin(omega*t + phi)
    h_base = h0 * np.sin(omega * t + phi) - h0
    h_hindwing = h0 + h_base * envelope

    # Hindwing pitching motion
    # Inactive state: theta = alpha
    # Active state:   theta = alpha - theta0*sin(omega*t)
    theta_base = -theta0 * np.sin(omega * t)
    theta_hindwing = alpha + theta_base * envelope

    motions[dc] = {
        "heave": h_hindwing / c,
        "pitch": np.degrees(theta_hindwing),
        "title": rf"$DC={dc:.2f}$",
    }


# ============================================================
# Axis settings
# ============================================================
heave_amp = h0 / c
heave_margin = max(0.03 * heave_amp, 0.005)

heave_ylim = (
    -heave_amp - heave_margin,
    heave_amp + heave_margin,
)

heave_ticks = [
    -heave_amp,
    0.0,
    heave_amp,
]

heave_ticklabels = [
    rf"${-heave_amp:.2f}$",
    r"$0$",
    rf"${heave_amp:.2f}$",
]

# Required pitching ticks
pitch_ylim = (-32.0, 47.0)
pitch_ticks = [-30.0, 0.0, 15.0, 45.0]
pitch_ticklabels = [
    r"$-30$",
    r"$0$",
    r"$15$",
    r"$45$",
]

x_ticks = [0.0, 0.25, 0.50, 0.75, 1.0]
x_ticklabels = [
    r"$0$",
    r"$0.25$",
    r"$0.5$",
    r"$0.75$",
    r"$1$",
]


# ============================================================
# Plot: 2 rows x 4 columns
# ============================================================
fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(6.5, 3),
    sharex=True,
)

fig.subplots_adjust(
    left=0.09,
    right=0.99,
    bottom=0.20,
    top=0.88,
    wspace=0.12,
    hspace=0.18,
)

for col, case in enumerate(CASE_LIST):
    ax_h = axes[0, col]
    ax_theta = axes[1, col]

    # --------------------------------------------------------
    # First row: heaving motion
    # --------------------------------------------------------
    ax_h.plot(
        t_normalized,
        h_forewing / c,
        color=FOREWING_COLOR,
        linewidth=LINEWIDTH,
        label="Forewing",
        clip_on=True,
    )

    ax_h.plot(
        t_normalized,
        motions[case]["heave"],
        color=HINDWING_COLOR,
        linewidth=LINEWIDTH,
        label="Hindwing",
        clip_on=True,
    )

    ax_h.set_xlim(0.0, 1.0)
    ax_h.set_ylim(*heave_ylim)
    ax_h.set_yticks(heave_ticks)

    # Only the first column shows y tick labels and ylabel
    if col == 0:
        ax_h.set_yticklabels(heave_ticklabels)
        ax_h.set_ylabel(
            r"$h/c$",
            rotation=0,
            labelpad=12,
            va="center",
        )
    else:
        ax_h.set_yticklabels([])

    # Column title
    ax_h.set_title(motions[case]["title"])

    # Upper row: hide bottom axis
    ax_h.spines["top"].set_visible(False)
    ax_h.spines["bottom"].set_visible(False)
    ax_h.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False,
    )

    # --------------------------------------------------------
    # Second row: pitching motion
    # --------------------------------------------------------
    ax_theta.plot(
        t_normalized,
        np.degrees(theta_forewing),
        color=FOREWING_COLOR,
        linewidth=LINEWIDTH,
        label="Forewing",
        clip_on=True,
    )

    ax_theta.plot(
        t_normalized,
        motions[case]["pitch"],
        color=HINDWING_COLOR,
        linewidth=LINEWIDTH,
        label="Hindwing",
        clip_on=True,
    )

    ax_theta.set_xlim(0.0, 1.0)
    ax_theta.set_ylim(*pitch_ylim)
    ax_theta.set_yticks(pitch_ticks)

    # Only the first column shows y tick labels and ylabel
    if col == 0:
        ax_theta.set_yticklabels(pitch_ticklabels)
        ax_theta.set_ylabel(
            r"$\alpha/^\circ$",
            rotation=0,
            labelpad=14,
            va="center",
        )
    else:
        ax_theta.set_yticklabels([])

    ax_theta.set_xticks(x_ticks)
    ax_theta.set_xticklabels(x_ticklabels)

    # Pitch zero reference line
    ax_theta.axhline(
        y=0.0,
        color="black",
        linewidth=0.65,
        linestyle="-.",
        dashes=(8, 4, 2, 4),
        zorder=0,
    )

    ax_theta.spines["top"].set_visible(False)

    # --------------------------------------------------------
    # Common styling
    # --------------------------------------------------------
    for ax in (ax_h, ax_theta):
        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            left=True,
            right=True,
            top=False,
        )

        # 右边三列保留左侧纵轴，只隐藏标签，不隐藏 spine
        ax.spines["left"].set_visible(True)

    # Bound vertical spines for ALL columns
    ax_h.spines["left"].set_bounds(
        -heave_amp,
        heave_amp,
    )
    ax_h.spines["right"].set_bounds(
        -heave_amp,
        heave_amp,
    )

    ax_theta.spines["left"].set_bounds(
        -30.0,
        45.0,
    )
    ax_theta.spines["right"].set_bounds(
        -30.0,
        45.0,
    )


# ============================================================
# Legend and shared x label
# ============================================================
axes[0, 0].legend(
    loc="upper right",
    frameon=False,
    handlelength=2.0,
    handletextpad=0.5,
    borderaxespad=0.2,
)

fig.supxlabel(
    r"$t/T_{cyc}$",
    x=0.54,
    y=0.045,
)


# ============================================================
# Save and display
# ============================================================
if SAVE_FIG:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    svg_path = os.path.join(
        OUTPUT_DIR,
        OUTPUT_BASENAME + ".svg",
    )
    pdf_path = os.path.join(
        OUTPUT_DIR,
        OUTPUT_BASENAME + ".pdf",
    )
    png_path = os.path.join(
        OUTPUT_DIR,
        OUTPUT_BASENAME + ".png",
    )

    fig.savefig(
        svg_path,
        format="svg",
        dpi=600,
        bbox_inches="tight",
    )
    # fig.savefig(
    #     pdf_path,
    #     format="pdf",
    #     dpi=600,
    #     bbox_inches="tight",
    # )
    fig.savefig(
        png_path,
        format="png",
        dpi=600,
        bbox_inches="tight",
    )

    print("Figures saved:")
    print(svg_path)
    print(pdf_path)
    print(png_path)

plt.show()