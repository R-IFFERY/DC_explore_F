import numpy as np
import matplotlib.pyplot as plt

npz_path = (
    r"E:/Rensq/DC_explore_F/flow/"
    r"figure_rho_rhostar_wake_map/"
    r"rho_rhostar_all_k_st_metrics.npz"
)

data = np.load(npz_path)

k_list = data["k_list"]
st_list = data["st_list"]
rho_samples = data["rho_samples"]
rho_star_samples = data["rho_star_samples"]
best_shift_samples = data["best_shift_samples"]

cycle_pairs = ["11-12", "12-13", "13-14", "14-15"]

k_target = 1.1
k_index = np.where(np.isclose(k_list, k_target))[0][0]

for st_target in [0.5, 0.6]:
    st_index = np.where(np.isclose(st_list, st_target))[0][0]

    rho = rho_samples[st_index, k_index]
    rho_star = rho_star_samples[st_index, k_index]
    shift = best_shift_samples[st_index, k_index]

    print("=" * 70)
    print(f"k={k_target}, St={st_target}")
    print(f"rho mean      = {np.nanmean(rho):.6f}")
    print(f"rho_star mean = {np.nanmean(rho_star):.6f}")
    print(f"mean shift    = {np.nanmean(shift):.6f} c")

    print("\n各周期对平均值：")
    for pair_index, pair_name in enumerate(cycle_pairs):
        print(
            f"{pair_name}: "
            f"rho={np.nanmean(rho[pair_index]):.6f}, "
            f"rho_star={np.nanmean(rho_star[pair_index]):.6f}, "
            f"shift={np.nanmean(shift[pair_index]):.6f}c"
        )

    print("\n各相位平均值：")
    for phase_index in range(32):
        print(
            f"phase {phase_index:02d}: "
            f"rho={np.nanmean(rho[:, phase_index]):.6f}, "
            f"rho_star={np.nanmean(rho_star[:, phase_index]):.6f}, "
            f"shift={np.nanmean(shift[:, phase_index]):.6f}c"
        )

    # 4个周期对 × 32个相位
    plt.figure(figsize=(6.2, 2.6))
    image = plt.imshow(
        rho,
        aspect="auto",
        origin="lower",
        vmin=-1,
        vmax=1
    )
    plt.colorbar(image, label=r"$\rho_\omega$")
    plt.yticks(range(4), cycle_pairs)
    plt.xticks(
        np.arange(0, 32, 4),
        [f"{i}/32" for i in range(0, 32, 4)]
    )
    plt.xlabel("Phase")
    plt.ylabel("Cycle pair")
    plt.title(f"k={k_target}, St={st_target}: raw correlation")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6.2, 2.6))
    image = plt.imshow(
        rho_star,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1
    )
    plt.colorbar(image, label=r"$\rho_\omega^*$")
    plt.yticks(range(4), cycle_pairs)
    plt.xticks(
        np.arange(0, 32, 4),
        [f"{i}/32" for i in range(0, 32, 4)]
    )
    plt.xlabel("Phase")
    plt.ylabel("Cycle pair")
    plt.title(f"k={k_target}, St={st_target}: translation-corrected correlation")
    plt.tight_layout()
    plt.show()