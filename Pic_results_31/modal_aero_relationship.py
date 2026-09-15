# -*- coding: utf-8 -*-
"""
plot_figure9_1x2_normalized_kprime_ct_c4.py

绘制新的 1×2 图：
(a) 归一化总脉动能量 K' 沿两条参数截面的变化；
(b) C_{T,S} 与 C_4 的关系，颜色表示 C_{P,S}。

数据来源：
1. POD 批处理结果：
   POD_ROOT/all_cases_pod_metrics_summary.csv
   POD_ROOT/k..._St..._a0_dc1/k..._St..._a0_dc1_pod_results.npz
2. 周期平均力结果：
   FORCE_AVG_ROOT/k=0.5st=0.5.csv 等

说明：
- panel (a) 在同一个子图内同时画两条参数截面：
  * 底部 x 轴：k（固定 St=0.3）
  * 顶部 x 轴：St（固定 k=1.4）
  * y 轴：归一化 K'，默认每条截面分别除以各自最大值
- panel (b) 画 C_{T,S} 与 C_4 的关系，颜色表示 C_{P,S}，点形区分两条截面。
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 1. 用户配置区
# ============================================================

POD_ROOT = Path(r"E:/Rensq/DC_explore_F/POD_results/POD_results_3_1_1/")
POD_SUMMARY_CSV = POD_ROOT / "all_cases_pod_metrics_summary.csv"

FORCE_AVG_ROOT = Path(r"E:/Rensq/DC_explore_F/results_a0/cycle_averaged_results/")

FIG_SAVE_DIR = Path(r"E:/Rensq/DC_explore_F/POD_results/paper_figures/")
FIG_SAVE_DIR.mkdir(parents=True, exist_ok=True)
OUT_BASENAME = "Figure9_1x2_normalized_Kprime_CT_C4"
DPI = 600

# 两条参数截面
CUT1_ST = 0.3
CUT1_CASES = [
    {"k": 0.8, "St": 0.3, "a": 0.0, "dc": 1.0},
    {"k": 1.1, "St": 0.3, "a": 0.0, "dc": 1.0},
    {"k": 1.4, "St": 0.3, "a": 0.0, "dc": 1.0},
    {"k": 1.7, "St": 0.3, "a": 0.0, "dc": 1.0},
    {"k": 2.0, "St": 0.3, "a": 0.0, "dc": 1.0},
]

CUT2_K = 1.4
CUT2_CASES = [
    {"k": 1.4, "St": 0.2, "a": 0.0, "dc": 1.0},
    {"k": 1.4, "St": 0.3, "a": 0.0, "dc": 1.0},
    {"k": 1.4, "St": 0.4, "a": 0.0, "dc": 1.0},
    {"k": 1.4, "St": 0.5, "a": 0.0, "dc": 1.0},
]

# 周期平均力取值策略
FORCE_PERIOD_SELECTION = "last_n"   # "last_n" or "period_list"
FORCE_LAST_N_PERIODS = 3
DEFAULT_FORCE_PERIODS: Optional[Sequence[int]] = None
FORCE_PERIODS_BY_CASE: Dict[Tuple[float, ...], Sequence[int]] = {
    # (1.4, 0.3): [10, 11, 12],
}

# 气动力统计方式
CL_ABS_MODE = "abs_of_mean"   # "abs_of_mean" or "mean_abs"
RF_COLUMNS = ["cl_total_mean", "ct_total_mean"]  # 此脚本不直接使用 RF，但保留接口以兼容前一版本逻辑

# K' 归一化方式
# "each_cut_max"：每条截面分别除以各自最大值（推荐）
# "global_max"：两条截面共同除以全局最大值
KPRIME_NORMALISE = "each_cut_max"

# panel (b) 是否标注工况文本
ANNOTATE_POINTS = False

# 图像尺寸与间距
FIG_SIZE = (6.5, 3)
SUBPLOT_LEFT = 0.10
SUBPLOT_RIGHT = 0.96
SUBPLOT_BOTTOM = 0.18
SUBPLOT_TOP = 0.90
SUBPLOT_WSPACE = 0.35

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
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{newtxtext,newtxmath}",
})


# ============================================================
# 2. 基础工具函数
# ============================================================


def fmt_num(x: float) -> str:
    if float(x).is_integer():
        return f"{float(x):.1f}"
    return str(float(x)).rstrip("0").rstrip(".")


def fmt_num_short(x: float) -> str:
    return fmt_num(x).replace(".", "p").replace("-", "m")


def case_name_from_values(k: float, St: float, a: float = 0.0, dc: float = 1.0) -> str:
    a_val = int(a) if float(a).is_integer() else fmt_num(a)
    dc_val = int(dc) if float(dc).is_integer() else fmt_num(dc)
    return f"k{fmt_num_short(k)}_St{fmt_num_short(St)}_a{a_val}_dc{dc_val}"


def case_key(case: Dict) -> Tuple[float, float, float, float]:
    return (
        float(case["k"]),
        float(case["St"]),
        float(case.get("a", 0.0)),
        float(case.get("dc", 1.0)),
    )


def unique_cases(cases1: Sequence[Dict], cases2: Sequence[Dict]) -> List[Dict]:
    out: List[Dict] = []
    seen = set()
    for c in list(cases1) + list(cases2):
        key = case_key(c)
        if key not in seen:
            seen.add(key)
            out.append({"k": key[0], "St": key[1], "a": key[2], "dc": key[3]})
    return out


def read_table_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path)
        if len(df.columns) > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=r"\s+", engine="python")


def parse_force_filename(path: Path) -> Optional[Tuple[float, float]]:
    name = path.stem
    pattern = r"k\s*=?\s*([0-9]+(?:\.[0-9]+)?)\s*[_\- ]*\s*st\s*=?\s*([0-9]+(?:\.[0-9]+)?)"
    m = re.search(pattern, name, flags=re.IGNORECASE)
    if m is None:
        return None
    return float(m.group(1)), float(m.group(2))


def find_force_file(k: float, St: float) -> Path:
    if not FORCE_AVG_ROOT.exists():
        raise FileNotFoundError(f"周期平均力文件夹不存在：{FORCE_AVG_ROOT}")

    candidates = []
    for path in FORCE_AVG_ROOT.glob("*.csv"):
        parsed = parse_force_filename(path)
        if parsed is None:
            continue
        kk, ss = parsed
        if np.isclose(kk, k) and np.isclose(ss, St):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"未找到周期平均力文件：k={k}, St={St}\n"
            f"期望文件名类似：k={fmt_num(k)}st={fmt_num(St)}.csv\n"
            f"搜索目录：{FORCE_AVG_ROOT}"
        )

    candidates = sorted(candidates, key=lambda p: (len(p.name), p.name))
    return candidates[0]


def get_selected_periods_for_case(case: Dict, df: pd.DataFrame) -> pd.DataFrame:
    if "period" not in df.columns:
        raise KeyError(f"周期平均力文件中缺少 period 列。实际列：{list(df.columns)}")

    d = df.copy()
    d["period"] = pd.to_numeric(d["period"], errors="coerce")
    d = d.dropna(subset=["period"]).sort_values("period")

    key4 = case_key(case)
    key2 = (key4[0], key4[1])

    periods = None
    if key4 in FORCE_PERIODS_BY_CASE:
        periods = FORCE_PERIODS_BY_CASE[key4]
    elif key2 in FORCE_PERIODS_BY_CASE:
        periods = FORCE_PERIODS_BY_CASE[key2]
    elif FORCE_PERIOD_SELECTION == "period_list" and DEFAULT_FORCE_PERIODS is not None:
        periods = DEFAULT_FORCE_PERIODS

    if periods is not None:
        selected = d[d["period"].isin(list(periods))]
    elif FORCE_PERIOD_SELECTION == "last_n":
        selected = d.tail(FORCE_LAST_N_PERIODS)
    else:
        raise ValueError("FORCE_PERIOD_SELECTION 必须为 'last_n' 或 'period_list'。")

    if selected.empty:
        raise ValueError(f"k={case['k']}, St={case['St']} 没有选中任何周期。")
    return selected


def compute_force_metrics_from_file(case: Dict) -> Dict[str, float]:
    path = find_force_file(case["k"], case["St"])
    df = read_table_flexible(path)

    required = ["cl_total_mean", "ct_total_mean", "cp_total_mean"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"{path} 中缺少 {col} 列。实际列：{list(df.columns)}")

    selected = get_selected_periods_for_case(case, df)

    cl = pd.to_numeric(selected["cl_total_mean"], errors="coerce")
    ct = pd.to_numeric(selected["ct_total_mean"], errors="coerce")
    cp = pd.to_numeric(selected["cp_total_mean"], errors="coerce")

    if CL_ABS_MODE == "abs_of_mean":
        cls_abs = abs(float(cl.mean()))
    elif CL_ABS_MODE == "mean_abs":
        cls_abs = float(cl.abs().mean())
    else:
        raise ValueError("CL_ABS_MODE 必须为 'abs_of_mean' 或 'mean_abs'。")

    return {
        "force_file": str(path),
        "force_period_start": int(selected["period"].min()),
        "force_period_end": int(selected["period"].max()),
        "force_n_periods": int(len(selected)),
        "CLS": cls_abs,
        "CTS": float(ct.mean()),
        "CPS": float(cp.mean()),
    }


def load_kprime_from_npz(case: Dict) -> float:
    cname = case_name_from_values(case["k"], case["St"], case.get("a", 0.0), case.get("dc", 1.0))
    npz_path = POD_ROOT / cname / f"{cname}_pod_results.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"未找到 POD 结果文件：{npz_path}")
    data = np.load(npz_path)
    if "eigvals" not in data:
        raise KeyError(f"{npz_path} 中缺少 eigvals，无法计算 K'.")
    eigvals = np.asarray(data["eigvals"], dtype=float)
    return float(np.nansum(eigvals))


def get_case_membership(k: float, St: float) -> str:
    in_cut1 = any(np.isclose(k, c["k"]) and np.isclose(St, c["St"]) for c in CUT1_CASES)
    in_cut2 = any(np.isclose(k, c["k"]) and np.isclose(St, c["St"]) for c in CUT2_CASES)
    if in_cut1 and in_cut2:
        return "both"
    if in_cut1:
        return "cut1"
    if in_cut2:
        return "cut2"
    return "cut1"


def maybe_annotate(ax, x, y, text, dx=4, dy=4):
    if ANNOTATE_POINTS:
        ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8)


# ============================================================
# 3. 数据合并
# ============================================================


def build_dataframe() -> pd.DataFrame:
    if not POD_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"未找到 POD 汇总表：{POD_SUMMARY_CSV}")

    pod_df = pd.read_csv(POD_SUMMARY_CSV)
    required_pod_cols = ["k", "St", "C4"]
    for col in required_pod_cols:
        if col not in pod_df.columns:
            raise KeyError(f"POD 汇总表缺少 {col} 列。实际列：{list(pod_df.columns)}")

    for col in ["k", "St", "a", "dc", "C4"]:
        if col in pod_df.columns:
            pod_df[col] = pd.to_numeric(pod_df[col], errors="coerce")

    rows = []
    for case in unique_cases(CUT1_CASES, CUT2_CASES):
        k = float(case["k"])
        St = float(case["St"])
        a = float(case.get("a", 0.0))
        dc = float(case.get("dc", 1.0))

        mask = np.isclose(pod_df["k"], k) & np.isclose(pod_df["St"], St)
        if "a" in pod_df.columns:
            mask &= np.isclose(pod_df["a"].fillna(a), a)
        if "dc" in pod_df.columns:
            mask &= np.isclose(pod_df["dc"].fillna(dc), dc)

        sub = pod_df.loc[mask]
        if sub.empty:
            raise ValueError(f"POD 汇总表中找不到工况 k={k}, St={St}, a={a}, dc={dc}。")

        pod_row = sub.iloc[0].to_dict()
        force_metrics = compute_force_metrics_from_file(case)

        row = {
            "k": k,
            "St": St,
            "a": a,
            "dc": dc,
            "case_name": case_name_from_values(k, St, a, dc),
            "label": f"k={fmt_num(k)}, St={fmt_num(St)}",
            "membership": get_case_membership(k, St),
            "Kprime": load_kprime_from_npz(case),
            "C4": float(pod_row["C4"]),
        }
        row.update(force_metrics)
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 4. 绘图函数
# ============================================================


def add_panel_label(ax, label: str):
    ax.text(-0.18, 1.05, label, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")


def style_ax(ax):
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)


def normalise_kprime(d1: pd.DataFrame, d2: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    y1 = d1["Kprime"].to_numpy(dtype=float)
    y2 = d2["Kprime"].to_numpy(dtype=float)

    if KPRIME_NORMALISE == "each_cut_max":
        s1 = np.nanmax(np.abs(y1)) if np.nanmax(np.abs(y1)) > 0 else 1.0
        s2 = np.nanmax(np.abs(y2)) if np.nanmax(np.abs(y2)) > 0 else 1.0
        return y1 / s1, y2 / s2
    elif KPRIME_NORMALISE == "global_max":
        s = max(np.nanmax(np.abs(y1)), np.nanmax(np.abs(y2)))
        s = s if s > 0 else 1.0
        return y1 / s, y2 / s
    else:
        raise ValueError("KPRIME_NORMALISE 必须为 'each_cut_max' 或 'global_max'。")


def plot_panel_a(ax, df: pd.DataFrame):
    d1 = df[np.isclose(df["St"], CUT1_ST)].sort_values("k").copy()
    d2 = df[np.isclose(df["k"], CUT2_K)].sort_values("St").copy()

    y1, y2 = normalise_kprime(d1, d2)

    # 底部 x 轴：k 截面
    line1, = ax.plot(d1["k"], y1, marker="o", linewidth=1.0, label=rf"$St={fmt_num(CUT1_ST)}$")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"Normalised $K^{\prime}$")
    ax.set_xlim(min(d1["k"]) - 0.05, max(d1["k"]) + 0.05)
    style_ax(ax)

    # 顶部 x 轴：St 截面
    ax_top = ax.twiny()
    line2, = ax_top.plot(d2["St"], y2, marker="s", linewidth=1.0, label=rf"$k={fmt_num(CUT2_K)}$")
    ax_top.set_xlabel(r"$St$")
    ax_top.set_xlim(min(d2["St"]) - 0.01, max(d2["St"]) + 0.01)
    ax_top.tick_params(direction="in", top=True)

    ax.legend(handles=[line1, line2], loc="best", frameon=False)
    add_panel_label(ax, "(a)")


def scatter_by_membership(ax, df: pd.DataFrame, color_col: str = "CPS"):
    markers = {"cut1": "o", "cut2": "s", "both": "D"}
    labels = {
        "cut1": rf"$St={fmt_num(CUT1_ST)}$",
        "cut2": rf"$k={fmt_num(CUT2_K)}$",
        "both": "shared",
    }

    cp = df[color_col].to_numpy(dtype=float)
    vmin = float(np.nanmin(cp))
    vmax = float(np.nanmax(cp))
    norm = None if np.isclose(vmin, vmax) else plt.Normalize(vmin=vmin, vmax=vmax)

    handles = []
    first_scatter = None
    for key, marker in markers.items():
        sub = df[df["membership"] == key]
        if sub.empty:
            continue
        sc = ax.scatter(
            sub["C4"], sub["CTS"],
            s=34,
            marker=marker,
            c=sub[color_col],
            cmap="viridis",
            norm=norm,
            edgecolors="k",
            linewidths=0.7,
        )
        if first_scatter is None:
            first_scatter = sc
        handles.append(
            Line2D([0], [0], marker=marker, color="k", linestyle="None", markersize=5,
                   markerfacecolor="none", label=labels[key])
        )
        for _, r in sub.iterrows():
            maybe_annotate(ax, r["C4"], r["CTS"], r["label"])

    ax.set_xlabel(r"$C_4$")
    ax.set_ylabel(r"$C_{T,S}$")
    style_ax(ax)
    add_panel_label(ax, "(b)")
    return first_scatter, handles


# ============================================================
# 5. 主程序
# ============================================================


def main():
    df = build_dataframe()

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
    fig.subplots_adjust(left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT, bottom=SUBPLOT_BOTTOM, top=SUBPLOT_TOP, wspace=SUBPLOT_WSPACE)

    # (a)
    plot_panel_a(axes[0], df)

    # (b)
    scatter, handles = scatter_by_membership(axes[1], df, color_col="CPS")
    if handles:
        axes[1].legend(handles=handles, loc="best", frameon=False)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label(r"$C_{P,S}$")
        cbar.ax.tick_params(direction="in")

    png_path = FIG_SAVE_DIR / f"{OUT_BASENAME}.png"
    pdf_path = FIG_SAVE_DIR / f"{OUT_BASENAME}.pdf"
    svg_path = FIG_SAVE_DIR / f"{OUT_BASENAME}.svg"
    csv_path = FIG_SAVE_DIR / f"{OUT_BASENAME}_merged_data.csv"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    fig.savefig(png_path, dpi=DPI)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)

    print("新的 1x2 图已保存：")
    print(png_path)
    print(pdf_path)
    print(svg_path)
    print("用于绘图的合并数据表已保存：")
    print(csv_path)


if __name__ == "__main__":
    main()