import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib import rcParams

# ============================================================
# Publication-style settings
# ============================================================
rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


# ============================================================
# Geometry utilities
# ============================================================
def naca0012(chord=1.0, n=220):
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    t = 0.12

    yt = 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    xu = x * chord
    yu = yt * chord
    xl = x[::-1] * chord
    yl = -yt[::-1] * chord

    return np.r_[xu, xl], np.r_[yu, yl]


def transform(x, y, x0=0.0, y0=0.0, angle=0.0, pivot=(0.0, 0.0)):
    th = np.deg2rad(angle)

    xp = x - pivot[0]
    yp = y - pivot[1]

    xr = xp * np.cos(th) - yp * np.sin(th) + pivot[0] + x0
    yr = xp * np.sin(th) + yp * np.cos(th) + pivot[1] + y0

    return xr, yr


def draw_airfoil(ax, x_le, y_mid, chord=2.55, angle=0.0, lw=1.55):
    x, y = naca0012(chord=chord)

    xpiv_local = 0.25 * chord

    xx, yy = transform(
        x, y,
        x0=x_le,
        y0=y_mid,
        angle=angle,
        pivot=(xpiv_local, 0.0)
    )

    ax.plot(
        xx, yy,
        color="black",
        lw=lw,
        solid_joinstyle="round",
        solid_capstyle="round"
    )

    xpiv = x_le + xpiv_local
    ypiv = y_mid

    ax.plot(xpiv, ypiv, "o", ms=6.0, color="black", zorder=5)

    return xpiv, ypiv


def draw_small_airfoil(ax, x_le, y_mid, chord=0.78, lw=1.05):
    x, y = naca0012(chord=chord, n=120)

    ax.plot(
        x + x_le,
        y + y_mid,
        color="black",
        lw=lw,
        solid_joinstyle="round"
    )


# ============================================================
# Arrow and dimension utilities
# ============================================================
def arrow(ax, p0, p1, lw=1.1, ms=12, color="black", ls="-", zorder=3):
    patch = FancyArrowPatch(
        p0, p1,
        arrowstyle="-|>",
        mutation_scale=ms,
        lw=lw,
        color=color,
        linestyle=ls,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder
    )
    ax.add_patch(patch)
    return patch


def double_arrow(ax, p0, p1, lw=1.1, ms=12, color="black"):
    patch = FancyArrowPatch(
        p0, p1,
        arrowstyle="<|-|>",
        mutation_scale=ms,
        lw=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(patch)
    return patch


def pitch_arrow(ax, cx, cy, label, dx_text=0.56):
    p0 = (cx + 0.05, cy + 0.40)
    p1 = (cx + 0.78, cy + 0.42)

    patch = FancyArrowPatch(
        p0, p1,
        connectionstyle="arc3,rad=-0.65",
        arrowstyle="-|>",
        mutation_scale=14,
        lw=1.25,
        color="black"
    )
    ax.add_patch(patch)

    ax.text(
        cx + dx_text,
        cy + 0.68,
        label,
        ha="center",
        va="center",
        fontsize=18
    )


def dim_line(ax, x1, x2, y, label, label_dy=-0.27, tick=0.36, fs=19):
    ax.plot(
        [x1, x1],
        [y - tick, y + tick],
        color="black",
        lw=0.95,
        ls=(0, (4, 3))
    )

    ax.plot(
        [x2, x2],
        [y - tick, y + tick],
        color="black",
        lw=0.95,
        ls=(0, (4, 3))
    )

    double_arrow(
        ax,
        (x1 + 0.04, y),
        (x2 - 0.04, y),
        lw=1.05,
        ms=13
    )

    ax.text(
        (x1 + x2) / 2.0,
        y + label_dy,
        label,
        ha="center",
        va="center",
        fontsize=fs
    )


# ============================================================
# Vortex and wake utilities
# ============================================================
def draw_spiral(
    ax,
    cx,
    cy,
    r0=0.18,
    turns=1.8,
    direction=1,
    lw=1.0,
    color="black",
    n=180
):
    th = np.linspace(0.0, turns * 2.0 * np.pi, n)
    r = np.linspace(r0, 0.018, n)

    x = cx + r * np.cos(direction * th)
    y = cy + r * np.sin(direction * th)

    ax.plot(
        x, y,
        color=color,
        lw=lw,
        solid_capstyle="round"
    )


def draw_near_wake(ax):
    x = np.linspace(4.25, 6.25, 260)

    curves = [
        5.20 + 0.18 * np.sin(3.8 * (x - 4.25)) + 0.035 * (x - 4.25),
        5.02 + 0.12 * np.sin(4.5 * (x - 4.25) + 1.0),
        4.78 + 0.14 * np.sin(4.2 * (x - 4.25) + 2.1) - 0.015 * (x - 4.25),
        4.92 + 0.10 * np.sin(5.0 * (x - 4.25) + 2.7),
    ]

    for y in curves:
        ax.plot(
            x, y,
            color="0.48",
            lw=0.95,
            solid_capstyle="round"
        )

    draw_spiral(ax, 5.00, 5.13, r0=0.14, direction=1, color="0.25", lw=1.0)
    draw_spiral(ax, 4.75, 4.85, r0=0.12, direction=-1, color="0.25", lw=1.0)
    draw_spiral(ax, 5.55, 4.70, r0=0.13, direction=1, color="0.25", lw=1.0)
    draw_spiral(ax, 5.95, 5.12, r0=0.11, direction=-1, color="0.25", lw=1.0)
    draw_spiral(ax, 6.10, 4.70, r0=0.11, direction=1, color="0.25", lw=1.0)


# ============================================================
# Wake boxes
# ============================================================
def wake_box(ax, x0, y0, w, h, title, kind):
    ax.add_patch(
        Rectangle(
            (x0, y0),
            w,
            h,
            fill=False,
            lw=1.20,
            ec="black"
        )
    )

    ax.text(
        x0 + 0.25,
        y0 + h - 0.32,
        title,
        ha="left",
        va="center",
        fontsize=15
    )

    if kind == "weak":
        draw_wake_weak(ax, x0, y0, w, h)
    elif kind == "pair":
        draw_wake_pair(ax, x0, y0, w, h)
    elif kind == "deflected":
        draw_wake_deflected(ax, x0, y0, w, h)
    elif kind == "complex":
        draw_wake_complex(ax, x0, y0, w, h)


def draw_wake_weak(ax, x0, y0, w, h):
    draw_small_airfoil(
        ax,
        x0 + 0.13,
        y0 + 0.52 * h,
        chord=0.86,
        lw=1.05
    )

    xs = np.linspace(x0 + 1.00, x0 + w - 0.12, 230)
    eta = (xs - xs[0]) / (xs[-1] - xs[0])

    y_top = y0 + 0.60 * h + 0.040 * np.sin(7.5 * np.pi * eta) + 0.10 * eta
    y_bot = y0 + 0.39 * h + 0.045 * np.sin(6.3 * np.pi * eta + 0.9) - 0.11 * eta

    ax.plot(xs, y_top, color="black", lw=1.05, ls=(0, (4.2, 3.0)))
    ax.plot(xs, y_bot, color="black", lw=1.05, ls=(0, (4.2, 3.0)))


def draw_wake_pair(ax, x0, y0, w, h):
    draw_small_airfoil(
        ax,
        x0 + 0.13,
        y0 + 0.52 * h,
        chord=0.86,
        lw=1.05
    )

    xs = np.linspace(x0 + 0.90, x0 + w - 0.12, 260)
    eta = (xs - xs[0]) / (xs[-1] - xs[0])

    y_mid = y0 + 0.50 * h

    y1 = y_mid + 0.24 * np.sin(8.0 * np.pi * eta - 0.25)
    y2 = y_mid - 0.24 * np.sin(8.0 * np.pi * eta - 0.25)

    ax.plot(xs, y1, color="black", lw=1.10)
    ax.plot(xs, y2, color="black", lw=1.10)

    xcenters = np.linspace(x0 + 1.55, x0 + w - 0.65, 4)

    for xc in xcenters:
        draw_spiral(ax, xc, y0 + 0.61 * h, r0=0.16, direction=1, lw=1.05)
        draw_spiral(ax, xc + 0.18, y0 + 0.34 * h, r0=0.16, direction=-1, lw=1.05)


def draw_wake_deflected(ax, x0, y0, w, h):
    draw_small_airfoil(
        ax,
        x0 + 0.13,
        y0 + 0.52 * h,
        chord=0.86,
        lw=1.05
    )

    xa = x0 + 1.12
    xb = x0 + w - 0.20
    ya = y0 + 0.40 * h
    yb = y0 + 0.61 * h

    arrow(
        ax,
        (xa, ya),
        (xb, yb),
        lw=1.00,
        ms=10,
        ls="-.",
        zorder=1
    )

    xs = np.linspace(xa + 0.04, xb - 0.22, 220)
    eta = (xs - xs[0]) / (xs[-1] - xs[0])

    center = ya + (yb - ya) * eta
    amp = 0.18

    ax.plot(
        xs,
        center + amp * np.sin(5.0 * np.pi * eta),
        color="black",
        lw=1.00
    )

    ax.plot(
        xs,
        center - amp * np.sin(5.0 * np.pi * eta + 0.12),
        color="black",
        lw=1.00
    )

    xcenters = np.linspace(x0 + 1.65, x0 + w - 0.70, 4)

    for xc in xcenters:
        eta_c = (xc - xa) / (xb - xa)
        yc = ya + (yb - ya) * eta_c

        draw_spiral(ax, xc, yc + 0.17, r0=0.16, direction=1, lw=1.05)
        draw_spiral(ax, xc + 0.28, yc - 0.18, r0=0.16, direction=-1, lw=1.05)


def draw_wake_complex(ax, x0, y0, w, h):
    draw_small_airfoil(
        ax,
        x0 + 0.13,
        y0 + 0.52 * h,
        chord=0.86,
        lw=1.05
    )

    vortices = [
        (1.28, 0.67, 0.14, 1),
        (1.70, 0.37, 0.20, -1),
        (2.18, 0.59, 0.11, 1),
        (2.55, 0.34, 0.13, -1),
        (3.05, 0.52, 0.07, 1),
        (3.48, 0.71, 0.20, 1),
        (3.95, 0.40, 0.30, -1),
        (4.42, 0.62, 0.18, 1),
        (4.78, 0.33, 0.16, -1),
    ]

    for xx, yy, rr, sgn in vortices:
        draw_spiral(
            ax,
            x0 + xx,
            y0 + yy * h,
            r0=rr,
            direction=sgn,
            lw=1.0
        )

    xs = np.linspace(x0 + 1.10, x0 + w - 0.18, 260)
    eta = (xs - xs[0]) / (xs[-1] - xs[0])

    for amp, phase, off in [
        (0.18, 0.2, 0.49),
        (0.13, 1.8, 0.33),
        (0.11, 3.3, 0.66)
    ]:
        ys = (
            y0
            + off * h
            + amp * np.sin(5.7 * np.pi * eta + phase)
            + 0.06 * np.sin(13.0 * np.pi * eta + 0.3 * phase)
        )

        ax.plot(
            xs,
            ys,
            color="0.22",
            lw=0.85
        )


# ============================================================
# Main drawing
# ============================================================
def main():
    fig, ax = plt.subplots(figsize=(14.91, 10.55))

    ax.set_xlim(0.0, 14.91)
    ax.set_ylim(0.0, 10.55)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------
    # Left schematic
    # ------------------------------
    chord = 2.55
    fore_le = 1.55
    hind_le = 6.20
    y_wing = 5.00

    pivot_f = draw_airfoil(
        ax,
        fore_le,
        y_wing,
        chord=chord,
        lw=1.65
    )

    pivot_h = draw_airfoil(
        ax,
        hind_le,
        y_wing,
        chord=chord,
        lw=1.65
    )

    # Incoming flow
    arrow(
        ax,
        (0.10, y_wing),
        (0.98, y_wing),
        lw=1.70,
        ms=19
    )

    ax.text(
        0.25,
        y_wing + 0.34,
        r"$U_\infty$",
        fontsize=20,
        ha="center",
        va="center"
    )

    # Plunging arrows
    double_arrow(
        ax,
        (fore_le + 0.10, 5.25),
        (fore_le + 0.10, 6.88),
        lw=1.30,
        ms=16
    )

    ax.text(
        fore_le - 0.18,
        6.03,
        r"$h_F$",
        fontsize=19,
        ha="center",
        va="center"
    )

    double_arrow(
        ax,
        (hind_le + 0.10, 5.25),
        (hind_le + 0.10, 6.88),
        lw=1.30,
        ms=16
    )

    ax.text(
        hind_le - 0.18,
        6.03,
        r"$h_H$",
        fontsize=19,
        ha="center",
        va="center"
    )

    # Pitching arrows
    pitch_arrow(ax, pivot_f[0], pivot_f[1], r"$\alpha_F$")
    pitch_arrow(ax, pivot_h[0], pivot_h[1], r"$\alpha_H$")

    # Pivot reference lines
    for px, py in [pivot_f, pivot_h]:
        ax.plot(
            [px, px],
            [py - 0.10, 3.95],
            color="black",
            lw=0.95,
            ls=(0, (4, 3))
        )

    # Pivot annotations
    ax.plot(
        [pivot_f[0] - 0.10, pivot_f[0] - 0.34],
        [pivot_f[1] - 0.05, 4.35],
        color="black",
        lw=1.0
    )

    ax.text(
        pivot_f[0] - 0.58,
        4.23,
        r"$0.25c$" + "\n" + r"$(\mathit{pivot\ point})$",
        fontsize=14,
        ha="center",
        va="center",
        linespacing=1.1
    )

    ax.plot(
        [pivot_h[0] - 0.10, pivot_h[0] - 0.28],
        [pivot_h[1] - 0.05, 4.35],
        color="black",
        lw=1.0
    )

    ax.text(
        pivot_h[0] - 0.45,
        4.23,
        r"$0.25c$" + "\n" + r"$(\mathit{pivot\ point})$",
        fontsize=14,
        ha="center",
        va="center",
        linespacing=1.1
    )

    # Dimension lines
    dim_line(
        ax,
        fore_le,
        fore_le + chord,
        3.62,
        r"$c$",
        fs=20
    )

    dim_line(
        ax,
        pivot_f[0],
        pivot_h[0],
        3.02,
        r"$L_x$",
        fs=22
    )

    # Near wake and annotation
    draw_near_wake(ax)

    arrow(
        ax,
        (5.55, 6.70),
        (5.35, 5.65),
        lw=1.00,
        ms=12
    )

    ax.text(
        5.43,
        7.10,
        "strong near-wake\ninteraction",
        fontsize=16,
        ha="center",
        va="center",
        linespacing=1.05
    )

    # Wing labels
    ax.text(
        pivot_f[0] + 0.15,
        2.02,
        "Forewing\n(NACA 0012)",
        fontsize=18,
        ha="center",
        va="center",
        linespacing=1.4
    )

    ax.text(
        pivot_h[0] + 0.15,
        2.02,
        "Hindwing\n(NACA 0012)",
        fontsize=18,
        ha="center",
        va="center",
        linespacing=1.4
    )

    # ------------------------------
    # Right wake-pattern column
    # ------------------------------
    x0 = 9.65
    w = 4.95
    h = 1.82
    y_top = 7.80
    gap = 0.13

    ax.text(
        x0 + w / 2.0,
        10.12,
        "Representative wake patterns",
        fontsize=20,
        ha="center",
        va="center"
    )

    wake_box(
        ax,
        x0,
        y_top,
        w,
        h,
        "1. Weak shear-layer wake",
        "weak"
    )

    wake_box(
        ax,
        x0,
        y_top - (h + gap),
        w,
        h,
        "2. Organized vortex-pair wake",
        "pair"
    )

    wake_box(
        ax,
        x0,
        y_top - 2 * (h + gap),
        w,
        h,
        "3. Deflected reverse Kármán wake",
        "deflected"
    )

    wake_box(
        ax,
        x0,
        y_top - 3 * (h + gap),
        w,
        h,
        "4. Complex multi-vortex wake",
        "complex"
    )

    # Export
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fig.savefig(
        "fig1a_tandem_wake_reconstructed.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04
    )

    # fig.savefig(
    #     "fig1a_tandem_wake_reconstructed.pdf",
    #     bbox_inches="tight",
    #     pad_inches=0.04
    # )

    fig.savefig(
        "fig1a_tandem_wake_reconstructed.svg",
        bbox_inches="tight",
        pad_inches=0.04
    )

    plt.show()


if __name__ == "__main__":
    main()