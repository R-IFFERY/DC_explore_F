import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble":
        r"\usepackage{amsmath} \usepackage{newtxtext,newtxmath}",
})

# =========================
# Basic parameters
# =========================
a = 20
alpha = np.radians(a)

dc_list = [0.25, 0.5, 0.75]   # Hindwing duty cycle
p = 0                         # Reserved parameter, currently unused
ks = 9                        # Smoothing coefficient
c = 1.0                       # Chord length
u = 1.0                       # Free-stream velocity
dt = 0.0001                   # Time step
theta0 = np.radians(30)       # Pitching amplitude
phi = np.radians(90)          # Heaving phase

# Parameter groups
# k_list = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
# st_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# periods_list = [9, 12, 12, 16, 16, 16, 18]

k_list = [0.8]
st_list = [0.3]
periods_list = [32]

# Mapping between k and the number of periods
k_periods_map = dict(zip(k_list, periods_list))

# Root output directory
base_output_dir = r'E:/Rensq/DC_explore_F/data_1Tflow'


# =========================
# Smooth transition functions
# =========================
def smooth_window(tau, T, ks):
    """
    Double-tanh smooth window for one isolated active cycle.

    This function is used for DC=0.25 and DC=0.50,
    where each active cycle is isolated by inactive cycles.

    tau=0   -> approximately 0
    tau=T/2 -> approximately 1
    tau=T   -> approximately 0
    """
    return -np.tanh(ks * tau / T) * np.tanh(ks * (tau / T - 1.0))


def smooth_start(tau, T, ks):
    """
    Smooth transition from inactive state to active motion.

    tau=0 -> 0
    tau=T -> approximately 1
    """
    return np.tanh(ks * tau / T)


def smooth_stop(tau, T, ks):
    """
    Smooth transition from active motion to inactive state.

    tau=0 -> approximately 1
    tau=T -> 0
    """
    return -np.tanh(ks * (tau / T - 1.0))


# =========================
# Hindwing intermittent-motion envelope
# =========================
def build_intermittent_envelope(t, T1, dc, ks):
    """
    Build the intermittent-motion envelope for the hindwing.

    A four-cycle window is used:

    DC=0.25:
        active, inactive, inactive, inactive

    DC=0.50:
        active, inactive, active, inactive

    DC=0.75:
        active, active, active, inactive

    For DC=0.75:
        Cycle 1 uses single-cycle start smoothing.
        Cycle 2 is fully active with envelope=1.
        Cycle 3 uses single-cycle stop smoothing.
        Cycle 4 is inactive with envelope=0.

    Therefore, the smoothing scale for DC=0.75 is T1, not 3T1.
    """

    envelope = np.zeros_like(t)

    # Index of the current four-cycle intermittent window
    window_id = np.floor(t / (4.0 * T1)).astype(int)

    # Local time within the current four-cycle window
    tau4 = t - window_id * 4.0 * T1

    if np.isclose(dc, 0.25):
        # active, inactive, inactive, inactive
        # One isolated active cycle with a full single-cycle window.
        mask = (tau4 >= 0.0) & (tau4 < T1)
        tau_active = tau4[mask]
        envelope[mask] = smooth_window(tau_active, T1, ks)

    elif np.isclose(dc, 0.50):
        # active, inactive, active, inactive
        # Two isolated active cycles, each using the same full single-cycle window.

        # First active cycle
        mask1 = (tau4 >= 0.0) & (tau4 < T1)
        tau_active1 = tau4[mask1]
        envelope[mask1] = smooth_window(tau_active1, T1, ks)

        # Third active cycle
        mask2 = (tau4 >= 2.0 * T1) & (tau4 < 3.0 * T1)
        tau_active2 = tau4[mask2] - 2.0 * T1
        envelope[mask2] = smooth_window(tau_active2, T1, ks)

    elif np.isclose(dc, 0.75):
        # active, active, active, inactive
        #
        # Cycle 1:
        # inactive -> active, using single-cycle start smoothing.
        mask1 = (tau4 >= 0.0) & (tau4 < T1)
        tau1 = tau4[mask1]
        envelope[mask1] = smooth_start(tau1, T1, ks)

        # Cycle 2:
        # fully active, no smoothing.
        mask2 = (tau4 >= T1) & (tau4 < 2.0 * T1)
        envelope[mask2] = 1.0

        # Cycle 3:
        # active -> inactive, using single-cycle stop smoothing.
        mask3 = (tau4 >= 2.0 * T1) & (tau4 < 3.0 * T1)
        tau3 = tau4[mask3] - 2.0 * T1
        envelope[mask3] = smooth_stop(tau3, T1, ks)

        # Cycle 4:
        # inactive, envelope remains 0.

    else:
        raise ValueError("Only dc=0.25, 0.5, and 0.75 are supported.")

    return envelope


# =========================
# Main loop
# =========================
for k in k_list:

    periods = k_periods_map[k]

    for st in st_list:

        # =========================
        # Parameter calculation
        # =========================
        f1 = k * u / np.pi / c
        T1 = 1.0 / f1
        total_time = periods * T1
        omega1 = 2.0 * np.pi * f1

        h0 = st * u / (2.0 * f1)

        # The hindwing base frequency is the same as that of the forewing.
        # The intermittency is controlled by the envelope.
        f2 = f1
        T2 = T1
        omega2 = omega1

        t = np.arange(0.0, total_time + dt, dt)

        for dc in dc_list:

            # =========================
            # Output directory
            # =========================
            dir_name = f'st{st}k{k}/'
            dir2_name = f'st{st}k{k}a{a}/'

            # Use dc25, dc50, and dc75 to avoid directory-name conflicts.
            filename = f'st{st}k{k}a{a}dc{int(dc * 100)}/'

            output_dir = os.path.join(
                base_output_dir,
                f'k{k}',
                dir_name,
                dir2_name,
                filename
            )

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # =========================
            # Forewing pitching motion
            # =========================
            theta_position1 = -theta0 * np.sin(omega1 * t)
            theta_velocity1 = -theta0 * omega1 * np.cos(omega1 * t)
            theta_acceleration1 = theta0 * omega1**2 * np.sin(omega1 * t)

            # =========================
            # Forewing heaving motion
            # =========================
            h_position1 = h0 * np.sin(omega1 * t + phi)
            h_velocity1 = h0 * omega1 * np.cos(omega1 * t + phi)
            h_acceleration1 = -h0 * omega1**2 * np.sin(omega1 * t + phi)

            # =========================
            # Forewing x-direction motion
            # =========================
            x_position1 = np.full_like(t, -0.0)
            x_velocity1 = np.zeros_like(t)
            x_acceleration1 = np.zeros_like(t)

            # =========================
            # Hindwing intermittent envelope
            # =========================
            envelope = build_intermittent_envelope(t, T1, dc, ks)

            # =========================
            # Hindwing pitching motion
            # =========================
            # In inactive segments:
            # theta_position2 = alpha
            #
            # In active segments:
            # theta_position2 = alpha - theta0 * sin(omega2*t)
            theta_base2 = -theta0 * np.sin(omega2 * t)

            theta_position2 = alpha + theta_base2 * envelope

            # Numerical differentiation for hindwing pitching velocity and acceleration
            theta_velocity2 = np.gradient(theta_position2, dt)
            theta_acceleration2 = np.gradient(theta_velocity2, dt)

            # =========================
            # Hindwing heaving motion
            # =========================
            # In inactive segments:
            # h_position2 = h0
            #
            # In active segments:
            # h_position2 = h0 * sin(omega2*t + phi)
            h_base2 = h0 * np.sin(omega2 * t + phi) - h0

            h_position2 = h0 + h_base2 * envelope

            # Numerical differentiation for hindwing heaving velocity and acceleration
            h_velocity2 = np.gradient(h_position2, dt)
            h_acceleration2 = np.gradient(h_velocity2, dt)

            # =========================
            # Hindwing x-direction motion
            # =========================
            x_position2 = np.full_like(t, 1.50)
            x_velocity2 = np.zeros_like(t)
            x_acceleration2 = np.zeros_like(t)

            # =========================
            # Save data
            # =========================
            data = np.column_stack([
                t,
                theta_position1, theta_velocity1, theta_acceleration1,
                x_position1, x_velocity1, x_acceleration1,
                h_position1, h_velocity1, h_acceleration1,
                theta_position2, theta_velocity2, theta_acceleration2,
                x_position2, x_velocity2, x_acceleration2,
                h_position2, h_velocity2, h_acceleration2
            ])

            out_filename = os.path.join(output_dir, f"k{k}st{st}a{a}dc{dc}.dat")
            np.savetxt(out_filename, data, delimiter=' ', fmt='%.8f')

            print(f"Completed: k={k}, St={st}, alpha={a}, DC={dc}, periods={periods}")
            print(f"Forewing frequency f1 = {f1:.4f} Hz, T1 = {T1:.4f} s")
            print(f"Hindwing base frequency f2 = {f2:.4f} Hz, T2 = {T2:.4f} s")
            print(f"Heaving amplitude h0 = {h0:.4f}")

            if np.isclose(dc, 0.25):
                print("Hindwing intermittent mode: active inactive inactive inactive")
            elif np.isclose(dc, 0.50):
                print("Hindwing intermittent mode: active inactive active inactive")
            elif np.isclose(dc, 0.75):
                print("Hindwing intermittent mode: active active active inactive")

            print(f"Motion parameter file saved to: {out_filename}")

            # =========================
            # Plot check: full time range
            # =========================
            plt.figure(figsize=(14, 10))

            # 1. Pitch angle
            plt.subplot(3, 2, 1)
            plt.plot(t, np.degrees(theta_position1), label='Forewing pitch')
            plt.plot(t, np.degrees(theta_position2), label='Hindwing pitch')
            plt.xlabel('Time (s)')
            plt.ylabel('Pitch angle (deg)')
            plt.title(f'Pitch angle: k={k}, St={st}, alpha={a}, DC={dc}')
            plt.legend()
            plt.grid()

            # 2. Heaving displacement
            plt.subplot(3, 2, 2)
            plt.plot(t, h_position1, label='Forewing heave')
            plt.plot(t, h_position2, label='Hindwing heave')
            plt.xlabel('Time (s)')
            plt.ylabel('Heaving displacement')
            plt.title(f'Heaving displacement: k={k}, St={st}, alpha={a}, DC={dc}')
            plt.legend()
            plt.grid()

            # 3. Pitch angular velocity
            plt.subplot(3, 2, 3)
            plt.plot(t, theta_velocity1, label='Forewing pitch velocity')
            plt.plot(t, theta_velocity2, label='Hindwing pitch velocity')
            plt.xlabel('Time (s)')
            plt.ylabel('Pitch angular velocity (rad/s)')
            plt.title('Pitch angular velocity')
            plt.legend()
            plt.grid()

            # 4. Heaving velocity
            plt.subplot(3, 2, 4)
            plt.plot(t, h_velocity1, label='Forewing heave velocity')
            plt.plot(t, h_velocity2, label='Hindwing heave velocity')
            plt.xlabel('Time (s)')
            plt.ylabel('Heaving velocity')
            plt.title('Heaving velocity')
            plt.legend()
            plt.grid()

            # 5. Pitch angular acceleration
            plt.subplot(3, 2, 5)
            plt.plot(t, theta_acceleration1, label='Forewing pitch acceleration')
            plt.plot(t, theta_acceleration2, label='Hindwing pitch acceleration')
            plt.xlabel('Time (s)')
            plt.ylabel('Pitch angular acceleration (rad/s²)')
            plt.title('Pitch angular acceleration')
            plt.legend()
            plt.grid()

            # 6. Heaving acceleration
            plt.subplot(3, 2, 6)
            plt.plot(t, h_acceleration1, label='Forewing heave acceleration')
            plt.plot(t, h_acceleration2, label='Hindwing heave acceleration')
            plt.xlabel('Time (s)')
            plt.ylabel('Heaving acceleration')
            plt.title('Heaving acceleration')
            plt.legend()
            plt.grid()

            plt.tight_layout()

            fig_filename = os.path.join(output_dir, "kinematics_check.png")
            plt.savefig(fig_filename, dpi=300)
            plt.show()

            # =========================
            # Plot check: first four cycles
            # =========================
            window_mask = t <= 4.0 * T1

            plt.figure(figsize=(14, 10))

            # 1. Pitch angle
            plt.subplot(3, 2, 1)
            plt.plot(t[window_mask], np.degrees(theta_position1[window_mask]), label='Forewing pitch')
            plt.plot(t[window_mask], np.degrees(theta_position2[window_mask]), label='Hindwing pitch')
            plt.xlabel('Time (s)')
            plt.ylabel('Pitch angle (deg)')
            plt.title(f'Pitch angle within one intermittent window: DC={dc}')
            plt.legend()
            plt.grid()

            # 2. Heaving displacement
            plt.subplot(3, 2, 2)
            plt.plot(t[window_mask], h_position1[window_mask], label='Forewing heave')
            plt.plot(t[window_mask], h_position2[window_mask], label='Hindwing heave')
            plt.xlabel('Time (s)')
            plt.ylabel('Heaving displacement')
            plt.title(f'Heaving displacement within one intermittent window: DC={dc}')
            plt.legend()
            plt.grid()

            # 3. Pitch angular velocity
            plt.subplot(3, 2, 3)
            plt.plot(t[window_mask], theta_velocity1[window_mask], label='Forewing pitch velocity')
            plt.plot(t[window_mask], theta_velocity2[window_mask], label='Hindwing pitch velocity')
            plt.xlabel('Time (s)')
            plt.ylabel('Pitch angular velocity (rad/s)')
            plt.title('Pitch angular velocity')
            plt.legend()
            plt.grid()

            # 4. Heaving velocity
            plt.subplot(3, 2, 4)
            plt.plot(t[window_mask], h_velocity1[window_mask], label='Forewing heave velocity')
            plt.plot(t[window_mask], h_velocity2[window_mask], label='Hindwing heave velocity')
            plt.xlabel('Time (s)')
            plt.ylabel('Heaving velocity')
            plt.title('Heaving velocity')
            plt.legend()
            plt.grid()

            # 5. Pitch angular acceleration
            plt.subplot(3, 2, 5)
            plt.plot(t[window_mask], theta_acceleration1[window_mask], label='Forewing pitch acceleration')
            plt.plot(t[window_mask], theta_acceleration2[window_mask], label='Hindwing pitch acceleration')
            plt.xlabel('Time (s)')
            plt.ylabel('Pitch angular acceleration (rad/s²)')
            plt.title('Pitch angular acceleration')
            plt.legend()
            plt.grid()

            # 6. Heaving acceleration
            plt.subplot(3, 2, 6)
            plt.plot(t[window_mask], h_acceleration1[window_mask], label='Forewing heave acceleration')
            plt.plot(t[window_mask], h_acceleration2[window_mask], label='Hindwing heave acceleration')
            plt.xlabel('Time (s)')
            plt.ylabel('Heaving acceleration')
            plt.title('Heaving acceleration')
            plt.legend()
            plt.grid()

            plt.tight_layout()

            fig_filename = os.path.join(output_dir, "kinematics_check_one_window.png")
            plt.savefig(fig_filename, dpi=300)
            plt.show()

print("All parameter combinations have been generated.")