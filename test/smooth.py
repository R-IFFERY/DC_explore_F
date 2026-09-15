import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{newtxtext,newtxmath}"
})

# ========== 参数设置 ==========
T = 1.0          # 时间尺度T，按需修改
sample_num = 2000# 采样点数，越大曲线越平滑
# ==============================

# 生成自变量 t
t = np.linspace(-0.1 * T, 1.1 * T, sample_num)
x = t / T  # 无量纲化 x = t/T

# 计算目标函数
term1 = np.tanh(9 * x)
term2 = np.tanh(9 * (x - 1))
f = -term1 * term2

# 绘图
plt.figure(figsize=(6, 6), dpi=100)
plt.plot(t, f, linewidth=2.2, color="#2E86AB")

# 辅助线条
# plt.axhline(y=0, c="black", ls="--", alpha=0.5)
# plt.axvline(x=0, c="black", ls="--", alpha=0.5)
# plt.axvline(x=T, c="crimson", ls="--", alpha=0.6, label=r"$t=T$")

# 图表美化
plt.grid(True, alpha=0.3)
plt.xlabel("$t/T$", fontsize=18)
plt.xlim(0, 1)
plt.ylabel("$S(t)$", fontsize=18)
plt.ylim(0,1.1)
plt.title(f"Function Curve (T = {T})", fontsize=18)
# plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# 计算 [0, T] 区间最大值
valid_range = (t >= 0) & (t <= T)
max_amp = np.max(f[valid_range])
print(f"t ∈ [0, T] 区间内函数最大值：{max_amp:.4f}")