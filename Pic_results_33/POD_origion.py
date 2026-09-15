# -*- coding: utf-8 -*-
"""
pod_dc_big_period.py

DC 工况的大周期 POD 分析。

分析原则
--------
1. St 固定为 0.3。
2. 四个 k 对应不同后翼平均俯仰角：
       k=0.8 -> a=20
       k=1.1 -> a=20
       k=1.4 -> a=17
       k=1.7 -> a=11

3. 每个 k 分析：
       DC = 1.00
       DC = 0.75
       DC = 0.50
       DC = 0.25

4. POD 统一使用 4T 时间窗口：

   DC < 1：
       第 25、26、27、28 周期
       即一个完整间歇运动大周期。

   DC = 1：
       自动选择保存数据中最后 4 个完整普通周期，
       组成一个长度同样为 4T 的分析窗口。

5. 每个普通周期抽取 32 个相位：
       4 × 32 = 128 snapshots

   因此所有 DC 的 POD 快照数保持一致。

6. POD 对速度脉动 (u', v') 进行。
7. 输出：
       - POD energy spectrum
       - cumulative energy
       - temporal coefficients
       - a1-a2 phase portrait
       - coefficient spectrum
       - mean vorticity
       - modal vorticity
       - instantaneous vorticity over 4T
       - NPZ
       - CSV
       - JSON
       - all_cases_pod_metrics_summary.csv
       - k-DC 指标图
"""

import os
import re
import csv
import json

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps


# ============================================================
# 0. 工况定义
# ============================================================

@dataclass
class CaseSpec:
    k: float
    st: float
    a: float
    dc: float
    tag: str = ""


ST = 0.3

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

DC_LIST = [
    1.00,
    0.75,
    0.50,
    0.25,
]


CASES = [

    CaseSpec(
        k=k,
        st=ST,
        a=CASE_DICT[k],
        dc=dc,
        tag="dc_big_period"
    )

    for k in K_LIST
    for dc in DC_LIST
]


# ============================================================
# 1. 路径
# ============================================================

BASE_ROOT = (
    r"E:/Rensq/DC_explore_F/data_1Tflow/"
)

SAVE_ROOT = (
    r"E:/Rensq/DC_explore_F/"
    r"POD_results/POD_333_intermittent"
)

os.makedirs(
    SAVE_ROOT,
    exist_ok=True
)


# ============================================================
# 2. 大周期设置
# ============================================================

# DC != 1 固定使用 25-28
INTERMITTENT_CYCLES = [
    25,
    26,
    27,
    28,
]

# DC=1 使用最后几个完整普通周期
DC1_N_CYCLES = 4

# 每个普通 T 周期统一抽取 32 个相位
N_PHASES_PER_CYCLE = 32

# 相位匹配容差
#
# 32相位理论间隔 = 1/32 = 0.03125T
# 这里设 0.02T，一般足够。
PHASE_TOL = 0.020

# 一个完整大周期总快照数
EXPECTED_BIG_PERIOD_SNAPSHOTS = (
    4 * N_PHASES_PER_CYCLE
)


# ============================================================
# 3. POD 空间范围
# ============================================================

X_MIN_POD = -1.0
X_MAX_POD = 20.0

Y_MIN_POD = -5.0
Y_MAX_POD = 5.0

# 空间降采样
STRIDE_X = 2
STRIDE_Y = 2


# ============================================================
# 4. 固体区域掩膜
# ============================================================

USE_BODY_MASK = True

BODY_THRESHOLD = 0.5


# ============================================================
# 5. POD 输出
# ============================================================

N_MODES_SAVE = 12

N_MODES_PLOT = 6

N_RECON_PLOT = 4

N_COEFF_PLOT = 4

VMIN_VORT = -30.0
VMAX_VORT = 30.0

DPI = 300


# ============================================================
# 6. Matplotlib
# ============================================================

plt.rcParams.update({

    "font.size": 10,

    "axes.titlesize": 10,
    "axes.labelsize": 10,

    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "legend.fontsize": 10,

    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "DejaVu Serif"
    ],

    "axes.linewidth": 0.8,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "text.usetex": False,
})


# ============================================================
# 7. Colormap
# ============================================================

def build_white_center_cmap(
    vmin: float,
    vmax: float,
    n: int = 512,
    sigma: float = 1.0
):

    base = (
        colormaps[
            "RdBu_r"
        ].resampled(n)
    )

    new_colors = base(
        np.linspace(
            0,
            1,
            n
        )
    )

    vals = np.linspace(
        vmin,
        vmax,
        n
    )

    weights = np.exp(
        -0.5
        * (vals / sigma) ** 2
    )

    for i in range(n):

        w = weights[i]

        new_colors[i] = (
            (1 - w)
            * new_colors[i]
            +
            w
            * np.array([
                1,
                1,
                1,
                1
            ])
        )

    return mcolors.ListedColormap(
        new_colors
    )


CMAP_WHITE = build_white_center_cmap(
    VMIN_VORT,
    VMAX_VORT
)


# ============================================================
# 8. 基本工具
# ============================================================

def fmt_num(x: float) -> str:
    """
    用于实际目录。

    0.8 -> 0.8
    1.0 -> 1
    0.50 -> 0.5
    """

    if float(x).is_integer():

        return str(
            int(x)
        )

    return f"{x:g}"


def fmt_num_short(x: float) -> str:

    return (
        fmt_num(x)
        .replace(".", "p")
        .replace("-", "m")
    )


def case_name(
    case: CaseSpec
) -> str:

    return (
        f"k{fmt_num_short(case.k)}"
        f"_St{fmt_num_short(case.st)}"
        f"_a{fmt_num_short(case.a)}"
        f"_dc{fmt_num_short(case.dc)}"
    )


def ensure_dir(
    path: str
):

    os.makedirs(
        path,
        exist_ok=True
    )


# ============================================================
# 9. DC 数据路径
# ============================================================

def get_case_dir(
    case: CaseSpec
) -> str:
    """
    实际目录规则。

    DC=1:

    .../
    st0.3k0.8a20dc1/
    k0.8st0.3a20/

    DC=0.5:

    .../
    st0.3k0.8a20dc50/
    k0.8st0.3a20dc0.5/
    """

    k = fmt_num(
        case.k
    )

    st = fmt_num(
        case.st
    )

    a = fmt_num(
        case.a
    )

    # --------------------------------------------------------
    # DC = 1
    # --------------------------------------------------------

    if np.isclose(
        case.dc,
        1.0
    ):

        parent = os.path.join(

            BASE_ROOT,

            f"k{k}",

            f"st{st}k{k}",

            f"st{st}k{k}a{a}",

            f"st{st}k{k}a{a}dc1"
        )

        final_dir = (
            f"k{k}"
            f"st{st}"
            f"a{a}"
        )

    # --------------------------------------------------------
    # DC != 1
    # --------------------------------------------------------

    else:

        dc_int = int(
            round(
                case.dc * 100
            )
        )

        dc_str = fmt_num(
            case.dc
        )

        parent = os.path.join(

            BASE_ROOT,

            f"k{k}",

            f"st{st}k{k}",

            f"st{st}k{k}a{a}",

            f"st{st}k{k}a{a}dc{dc_int}"
        )

        final_dir = (
            f"k{k}"
            f"st{st}"
            f"a{a}"
            f"dc{dc_str}"
        )

    return os.path.join(
        parent,
        final_dir
    )


# ============================================================
# 10. Tecplot Ordered DAT reader
# ============================================================

def read_tecplot_ordered_dat(

    filename: str,

    target_vars: Sequence[str] = (
        "X",
        "Y",
        "U",
        "V",
        "Vorticity",
        "body"
    )

) -> Dict[str, np.ndarray]:

    var_map = {

        "X": 0,
        "Y": 1,

        "U": 2,
        "V": 3,

        "Presure": 4,
        "Pressure": 4,

        "Vorticity": 5,

        "body": 6,

        "dvordt": 7,
    }

    cols = [
        var_map[v]
        for v in target_vars
    ]

    need_cols = (
        max(cols)
        + 1
    )

    I = None
    J = None

    sol_time = None

    data_start_idx = None

    zone_found = False

    with open(
        filename,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        lines = f.readlines()

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    for idx, line in enumerate(
        lines,
        start=1
    ):

        stripped = line.strip()

        if not stripped:
            continue

        if (
            not zone_found
            and
            stripped.upper().startswith(
                "ZONE"
            )
        ):

            zone_found = True

            i_m = re.search(
                r"I\s*=\s*(\d+)",
                stripped,
                re.I
            )

            j_m = re.search(
                r"J\s*=\s*(\d+)",
                stripped,
                re.I
            )

            if i_m:

                I = int(
                    i_m.group(1)
                )

            if j_m:

                J = int(
                    j_m.group(1)
                )

            continue

        if zone_found:

            if sol_time is None:

                t_m = re.match(
                    r"^\s*([\d.Ee+\-]+)\s*$",
                    stripped
                )

                if t_m:

                    try:

                        sol_time = float(
                            t_m.group(1)
                        )

                        continue

                    except ValueError:

                        pass

            parts = stripped.split()

            if len(parts) >= need_cols:

                try:

                    [
                        float(parts[c])
                        for c in cols
                    ]

                    data_start_idx = idx

                    break

                except ValueError:

                    continue

    if (
        I is None
        or
        J is None
        or
        data_start_idx is None
    ):

        raise ValueError(
            f"无法解析 dat 文件头信息: {filename}"
        )

    data = np.loadtxt(

        filename,

        skiprows=
            data_start_idx - 1,

        max_rows=
            I * J,

        usecols=
            cols,

        delimiter=None,

        dtype=np.float32
    )

    if data.shape[0] != I * J:

        raise ValueError(

            f"{filename} 数据点数量不足，"
            f"期望 {I * J}，"
            f"实际 {data.shape[0]}"
        )

    result = {

        "solution_time":
            sol_time,

        "I":
            I,

        "J":
            J,
    }

    for i, var in enumerate(
        target_vars
    ):

        result[var] = (
            data[:, i]
            .reshape(
                J,
                I
            )
        )

    return result


# ============================================================
# 11. 文件时间及普通周期识别
# ============================================================

def parse_time_from_filename(
    filename: str
) -> Optional[float]:

    base = os.path.basename(
        filename
    )

    m = re.match(
        r"^F\s*(\d+)\.dat$",
        base,
        re.IGNORECASE
    )

    if not m:

        return None

    return (
        int(
            m.group(1)
        )
        /
        10000.0
    )


def is_valid_flow_filename(
    filename: str
) -> bool:

    return (
        re.match(
            r"^F\s*\d+\.dat$",
            os.path.basename(filename),
            re.IGNORECASE
        )
        is not None
    )


def compute_cycle_info(

    t: float,

    T: float,

    eps: float = 1e-10

) -> Tuple[int, float]:

    cycle_no = (
        int(
            np.floor(
                t / T
                + eps
            )
        )
        + 1
    )

    phase_frac = (
        t / T
        -
        np.floor(
            t / T
            + eps
        )
    )

    if abs(
        phase_frac
    ) < 1e-8:

        phase_frac = 0.0

    elif abs(
        phase_frac - 1.0
    ) < 1e-8:

        phase_frac = 0.0

        cycle_no += 1

    return (
        cycle_no,
        phase_frac
    )


# ============================================================
# 12. 扫描全部流场文件
# ============================================================

def scan_all_records(

    case_dir: str,

    T: float

) -> List[Dict]:

    records = []

    if not os.path.isdir(
        case_dir
    ):

        return records

    for fname in os.listdir(
        case_dir
    ):

        if not is_valid_flow_filename(
            fname
        ):

            continue

        t = parse_time_from_filename(
            fname
        )

        if t is None:
            continue

        cycle_no, phase_frac = (
            compute_cycle_info(
                t,
                T
            )
        )

        records.append({

            "filename":
                fname,

            "path":
                os.path.join(
                    case_dir,
                    fname
                ),

            "t":
                t,

            "cycle_no":
                cycle_no,

            "phase_frac":
                phase_frac,
        })

    records.sort(
        key=lambda x: x["t"]
    )

    return records


# ============================================================
# 13. 单周期抽取固定32相位
# ============================================================

def select_cycle_32_phases(

    all_records: List[Dict],

    cycle_no: int,

    n_phases: int =
        N_PHASES_PER_CYCLE,

    phase_tol: float =
        PHASE_TOL

) -> Optional[List[Dict]]:
    """
    从指定普通周期中抽取：

        0/32
        1/32
        ...
        31/32

    共 32 个快照。

    使用最邻近相位匹配。
    """

    cycle_records = [

        rec

        for rec in all_records

        if rec[
            "cycle_no"
        ] == cycle_no
    ]

    if not cycle_records:

        return None

    targets = np.linspace(
        0.0,
        1.0,
        n_phases,
        endpoint=False
    )

    phases = np.array([
        rec["phase_frac"]
        for rec in cycle_records
    ])

    selected = []

    used_paths = set()

    for phase_id, target in enumerate(
        targets
    ):

        idx = int(
            np.argmin(
                np.abs(
                    phases - target
                )
            )
        )

        rec = cycle_records[
            idx
        ]

        error = abs(
            rec["phase_frac"]
            - target
        )

        if error > phase_tol:

            print(
                f"    cycle {cycle_no}: "
                f"target phase={target:.5f}, "
                f"nearest={rec['phase_frac']:.5f}, "
                f"error={error:.5f} > "
                f"{phase_tol:.5f}"
            )

            return None

        # 避免同一个文件被选两次
        if rec["path"] in used_paths:

            print(
                f"    cycle {cycle_no}: "
                f"相同文件被多个目标相位选中，"
                f"认为该周期不完整。"
            )

            return None

        rec_new = dict(
            rec
        )

        rec_new[
            "phase_target"
        ] = float(
            target
        )

        rec_new[
            "phase_id"
        ] = int(
            phase_id
        )

        selected.append(
            rec_new
        )

        used_paths.add(
            rec["path"]
        )

    return selected


# ============================================================
# 14. 判断周期是否完整
# ============================================================

def cycle_is_complete(

    all_records: List[Dict],

    cycle_no: int

) -> bool:

    selected = select_cycle_32_phases(
        all_records,
        cycle_no
    )

    return (
        selected is not None
        and
        len(selected)
        ==
        N_PHASES_PER_CYCLE
    )


# ============================================================
# 15. DC=1：寻找最后4个连续完整周期
# ============================================================

def find_last_four_complete_cycles(

    all_records: List[Dict]

) -> List[int]:

    available = sorted(
        {
            rec["cycle_no"]
            for rec in all_records
        }
    )

    if len(available) < 4:

        return []

    # 从最后向前寻找连续4周期
    for end_idx in range(
        len(available) - 1,
        2,
        -1
    ):

        end_cycle = available[
            end_idx
        ]

        candidate = [
            end_cycle - 3,
            end_cycle - 2,
            end_cycle - 1,
            end_cycle,
        ]

        # 必须全部存在
        if not all(
            cyc in available
            for cyc in candidate
        ):

            continue

        # 每个周期必须能匹配32相位
        ok = True

        for cyc in candidate:

            if not cycle_is_complete(
                all_records,
                cyc
            ):

                ok = False

                break

        if ok:

            return candidate

    return []


# ============================================================
# 16. 构建4T大周期快照
# ============================================================

def build_big_period_records(

    case: CaseSpec,

    all_records: List[Dict]

) -> Tuple[List[Dict], List[int]]:
    """
    返回：

        selected_records
        selected_cycles

    selected_records 固定 128 个。
    """

    # ========================================================
    # DC = 1
    # ========================================================

    if np.isclose(
        case.dc,
        1.0
    ):

        selected_cycles = (
            find_last_four_complete_cycles(
                all_records
            )
        )

        if len(
            selected_cycles
        ) != 4:

            raise ValueError(
                "DC=1 未找到最后4个连续完整周期。"
            )

    # ========================================================
    # DC < 1
    # ========================================================

    else:

        selected_cycles = (
            INTERMITTENT_CYCLES.copy()
        )

    # ========================================================
    # 每个周期抽取32相位
    # ========================================================

    selected_records = []

    for local_cycle_id, cycle_no in enumerate(
        selected_cycles
    ):

        cycle_records = (
            select_cycle_32_phases(
                all_records,
                cycle_no
            )
        )

        if cycle_records is None:

            raise ValueError(
                f"周期 {cycle_no} "
                f"无法获得完整的 "
                f"{N_PHASES_PER_CYCLE} "
                f"个相位。"
            )

        # ----------------------------------------------------
        # 定义大周期相位
        #
        # local_cycle_id = 0,1,2,3
        #
        # big_phase:
        #
        # 0T -> 0
        # 1T -> 0.25
        # 2T -> 0.50
        # 3T -> 0.75
        # 4T -> 1
        # ----------------------------------------------------

        for rec in cycle_records:

            rec_new = dict(
                rec
            )

            rec_new[
                "local_cycle_id"
            ] = int(
                local_cycle_id
            )

            rec_new[
                "big_phase_frac"
            ] = float(
                (
                    local_cycle_id
                    +
                    rec[
                        "phase_target"
                    ]
                )
                /
                4.0
            )

            selected_records.append(
                rec_new
            )

    selected_records.sort(
        key=lambda x:
        x["big_phase_frac"]
    )

    if len(
        selected_records
    ) != EXPECTED_BIG_PERIOD_SNAPSHOTS:

        raise ValueError(

            f"4T大周期快照数错误："
            f"期望 "
            f"{EXPECTED_BIG_PERIOD_SNAPSHOTS}，"
            f"实际 "
            f"{len(selected_records)}"
        )

    return (
        selected_records,
        selected_cycles
    )


# ============================================================
# 17. POD空间裁剪
# ============================================================

def find_crop_indices(

    X: np.ndarray,

    Y: np.ndarray,

    x_min: float,

    x_max: float,

    y_min: float,

    y_max: float,

    stride_x: int = 1,

    stride_y: int = 1

) -> Tuple[np.ndarray, np.ndarray]:

    mask = (
        (X >= x_min)
        &
        (X <= x_max)
        &
        (Y >= y_min)
        &
        (Y <= y_max)
    )

    if not np.any(
        mask
    ):

        raise ValueError(
            "POD裁剪区域与流场没有交集。"
        )

    rows = (
        np.where(
            np.any(
                mask,
                axis=1
            )
        )[0][::stride_y]
    )

    cols = (
        np.where(
            np.any(
                mask,
                axis=0
            )
        )[0][::stride_x]
    )

    return (
        rows,
        cols
    )


# ============================================================
# 18. 加载128个快照
# ============================================================

def load_snapshots_for_pod(

    records: List[Dict]

) -> Tuple[np.ndarray, ...]:

    if not records:

        raise ValueError(
            "没有可用于POD的快照。"
        )

    U_list = []
    V_list = []

    W_list = []
    B_list = []

    labels = []

    rows = None
    cols = None

    Xc = None
    Yc = None

    for idx, rec in enumerate(
        records
    ):

        print(

            f"    "
            f"[{idx + 1:03d}/"
            f"{len(records):03d}] "
            f"读取 "
            f"{rec['filename']}"
        )

        dat = read_tecplot_ordered_dat(
            rec["path"]
        )

        X = dat["X"]
        Y = dat["Y"]

        U0 = dat["U"]
        V0 = dat["V"]

        W0 = dat["Vorticity"]

        B0 = dat["body"]

        if rows is None:

            rows, cols = (
                find_crop_indices(

                    X,
                    Y,

                    X_MIN_POD,
                    X_MAX_POD,

                    Y_MIN_POD,
                    Y_MAX_POD,

                    STRIDE_X,
                    STRIDE_Y
                )
            )

            Xc = (
                X[
                    np.ix_(
                        rows,
                        cols
                    )
                ]
                .astype(
                    np.float32
                )
            )

            Yc = (
                Y[
                    np.ix_(
                        rows,
                        cols
                    )
                ]
                .astype(
                    np.float32
                )
            )

            print(
                f"    裁剪网格: "
                f"ny={Xc.shape[0]}, "
                f"nx={Xc.shape[1]}, "
                f"velocity dof="
                f"{2 * Xc.size}"
            )

        U_list.append(

            U0[
                np.ix_(
                    rows,
                    cols
                )
            ].astype(
                np.float32
            )
        )

        V_list.append(

            V0[
                np.ix_(
                    rows,
                    cols
                )
            ].astype(
                np.float32
            )
        )

        W_list.append(

            W0[
                np.ix_(
                    rows,
                    cols
                )
            ].astype(
                np.float32
            )
        )

        B_list.append(

            B0[
                np.ix_(
                    rows,
                    cols
                )
            ].astype(
                np.float32
            )
        )

        labels.append({

            "snapshot_index":
                idx,

            "filename":
                rec["filename"],

            "t_filename":
                rec["t"],

            "solution_time":
                dat[
                    "solution_time"
                ],

            "cycle_no":
                rec[
                    "cycle_no"
                ],

            "phase_frac":
                rec[
                    "phase_frac"
                ],

            "phase_target":
                rec[
                    "phase_target"
                ],

            "local_cycle_id":
                rec[
                    "local_cycle_id"
                ],

            "big_phase_frac":
                rec[
                    "big_phase_frac"
                ],
        })

    U = np.stack(
        U_list,
        axis=0
    )

    V = np.stack(
        V_list,
        axis=0
    )

    W = np.stack(
        W_list,
        axis=0
    )

    BODY = np.stack(
        B_list,
        axis=0
    )

    return (
        Xc,
        Yc,

        U,
        V,

        W,
        BODY,

        labels
    )


# ============================================================
# 19. 流体区域
# ============================================================

def build_fluid_mask(
    BODY: np.ndarray
) -> np.ndarray:

    if not USE_BODY_MASK:

        return np.ones(
            BODY.shape[1:],
            dtype=bool
        )

    # 整个4T过程中任意时刻属于实体的区域
    # 均排除在POD之外
    body_union = np.any(
        BODY > BODY_THRESHOLD,
        axis=0
    )

    fluid_mask = (
        ~body_union
    )

    if np.sum(
        fluid_mask
    ) == 0:

        raise ValueError(
            "fluid_mask 中无流体点。"
        )

    return fluid_mask


# ============================================================
# 20. 速度脉动 POD
# ============================================================

def compute_velocity_pod(

    U: np.ndarray,

    V: np.ndarray,

    fluid_mask: np.ndarray,

    n_modes_save: int =
        N_MODES_SAVE

) -> Dict[str, np.ndarray]:

    ns, ny, nx = (
        U.shape
    )

    n_fluid = int(
        np.sum(
            fluid_mask
        )
    )

    # --------------------------------------------------------
    # 4T平均场
    # --------------------------------------------------------

    U_mean = np.mean(
        U,
        axis=0
    )

    V_mean = np.mean(
        V,
        axis=0
    )

    # --------------------------------------------------------
    # 脉动
    # --------------------------------------------------------

    Up = (
        U
        -
        U_mean[
            None,
            :,
            :
        ]
    )

    Vp = (
        V
        -
        V_mean[
            None,
            :,
            :
        ]
    )

    Xmat = np.empty(

        (
            2 * n_fluid,
            ns
        ),

        dtype=np.float32
    )

    for s in range(ns):

        Xmat[
            :n_fluid,
            s
        ] = (
            Up[s][
                fluid_mask
            ]
        )

        Xmat[
            n_fluid:,
            s
        ] = (
            Vp[s][
                fluid_mask
            ]
        )

    # 数值上再次严格去均值
    Xmat -= np.mean(
        Xmat,
        axis=1,
        keepdims=True
    )

    if ns < 2:

        raise ValueError(
            "POD至少需要2个快照。"
        )

    # --------------------------------------------------------
    # SVD
    # --------------------------------------------------------

    U_svd, S, Vt = np.linalg.svd(

        Xmat,

        full_matrices=False
    )

    eigvals = (
        S ** 2
        /
        (ns - 1)
    )

    total_energy = np.sum(
        eigvals
    )

    if total_energy <= 0:

        raise ValueError(
            "POD总能量为0。"
        )

    energy = (
        eigvals
        /
        total_energy
    )

    cumulative = np.cumsum(
        energy
    )

    nm = min(
        n_modes_save,
        ns
    )

    coeffs = (
        np.diag(
            S[:nm]
        )
        @
        Vt[
            :nm,
            :
        ]
    )

    modes_u = np.full(
        (
            nm,
            ny,
            nx
        ),
        np.nan,
        dtype=np.float32
    )

    modes_v = np.full_like(
        modes_u,
        np.nan
    )

    for m in range(nm):

        mu = np.full(
            (ny, nx),
            np.nan,
            dtype=np.float32
        )

        mv = np.full(
            (ny, nx),
            np.nan,
            dtype=np.float32
        )

        mu[
            fluid_mask
        ] = U_svd[
            :n_fluid,
            m
        ]

        mv[
            fluid_mask
        ] = U_svd[
            n_fluid:,
            m
        ]

        modes_u[m] = mu
        modes_v[m] = mv

    return {

        "eigvals":
            eigvals,

        "energy":
            energy,

        "cumulative":
            cumulative,

        "singular_values":
            S,

        "coeffs":
            coeffs,

        "modes_u":
            modes_u,

        "modes_v":
            modes_v,

        "mean_u":
            U_mean,

        "mean_v":
            V_mean,

        "fluid_mask":
            fluid_mask,
    }


# ============================================================
# 21. POD 指标
# ============================================================

def compute_pair_balance(

    eigvals: np.ndarray,

    i: int,

    j: int

) -> Optional[float]:

    if len(
        eigvals
    ) <= max(
        i,
        j
    ):

        return None

    denominator = (
        eigvals[i]
        +
        eigvals[j]
    )

    if denominator <= 0:

        return None

    return float(

        abs(
            eigvals[i]
            -
            eigvals[j]
        )
        /
        denominator
    )


def compute_pod_metrics(

    case: CaseSpec,

    pod: Dict[str, np.ndarray],

    labels: List[Dict],

    selected_cycles: List[int]

) -> Dict[str, object]:

    eigvals = pod[
        "eigvals"
    ]

    energy = pod[
        "energy"
    ]

    cumulative = pod[
        "cumulative"
    ]

    singular_values = pod[
        "singular_values"
    ]

    fluid_mask = pod[
        "fluid_mask"
    ]

    coeffs = pod[
        "coeffs"
    ]

    # 基本拍动频率
    f1 = (
        case.k
        /
        np.pi
    )

    # 大周期频率
    f_big = (
        f1
        /
        4.0
    )

    metrics = {

        "case_name":
            case_name(case),

        "case_tag":
            case.tag,

        "k":
            float(case.k),

        "St":
            float(case.st),

        "a":
            float(case.a),

        "dc":
            float(case.dc),

        "analysis_unit":
            "4T_big_period",

        "cycle_start":
            int(
                selected_cycles[0]
            ),

        "cycle_end":
            int(
                selected_cycles[-1]
            ),

        "selected_cycles":
            "-".join(
                str(x)
                for x in selected_cycles
            ),

        "n_base_cycles":
            4,

        "n_phase_per_cycle":
            N_PHASES_PER_CYCLE,

        "n_snapshots":
            int(
                coeffs.shape[1]
            ),

        "f1":
            float(
                f1
            ),

        "f_big":
            float(
                f_big
            ),

        "T_big_over_T":
            4.0,

        "x_min_pod":
            X_MIN_POD,

        "x_max_pod":
            X_MAX_POD,

        "y_min_pod":
            Y_MIN_POD,

        "y_max_pod":
            Y_MAX_POD,

        "stride_x":
            STRIDE_X,

        "stride_y":
            STRIDE_Y,

        "n_grid_total":
            int(
                fluid_mask.size
            ),

        "n_fluid_points":
            int(
                np.sum(
                    fluid_mask
                )
            ),

        "masked_fraction":
            float(
                np.sum(
                    ~fluid_mask
                )
                /
                fluid_mask.size
            ),
    }

    # --------------------------------------------------------
    # 前12阶
    # --------------------------------------------------------

    n_report = min(
        12,
        len(
            energy
        )
    )

    for i in range(
        n_report
    ):

        m = (
            i + 1
        )

        metrics[
            f"lambda{m}"
        ] = float(
            eigvals[i]
        )

        metrics[
            f"sigma{m}"
        ] = float(
            singular_values[i]
        )

        metrics[
            f"E{m}"
        ] = float(
            energy[i]
        )

        metrics[
            f"C{m}"
        ] = float(
            cumulative[i]
        )

    # --------------------------------------------------------
    # 论文主要指标
    # --------------------------------------------------------

    for m in [
        2,
        4,
        6,
        8,
        10,
        12
    ]:

        if len(
            cumulative
        ) >= m:

            metrics[
                f"C{m}"
            ] = float(
                cumulative[
                    m - 1
                ]
            )

        else:

            metrics[
                f"C{m}"
            ] = None

    metrics[
        "Delta12"
    ] = compute_pair_balance(
        eigvals,
        0,
        1
    )

    metrics[
        "Delta34"
    ] = compute_pair_balance(
        eigvals,
        2,
        3
    )

    metrics[
        "E12_sum"
    ] = (

        float(
            energy[0]
            +
            energy[1]
        )

        if len(
            energy
        ) >= 2

        else None
    )

    metrics[
        "E34_sum"
    ] = (

        float(
            energy[2]
            +
            energy[3]
        )

        if len(
            energy
        ) >= 4

        else None
    )

    # --------------------------------------------------------
    # 模态熵
    #
    # 所有工况都是128 snapshots，
    # 因此H可以直接跨DC比较。
    # --------------------------------------------------------

    p = (
        eigvals
        /
        np.sum(
            eigvals
        )
    )

    p = p[
        p > 0
    ]

    if len(p) > 1:

        entropy = (
            -np.sum(
                p * np.log(p)
            )
        )

        H = (
            entropy
            /
            np.log(
                len(
                    eigvals
                )
            )
        )

        N_eff = np.exp(
            entropy
        )

    else:

        H = 0.0
        N_eff = 1.0

    metrics[
        "H"
    ] = float(
        H
    )

    metrics[
        "N_eff"
    ] = float(
        N_eff
    )

    return metrics


# ============================================================
# 22. 模态涡量
# ============================================================

def estimate_dx_dy(

    X: np.ndarray,

    Y: np.ndarray

) -> Tuple[float, float]:

    dx = float(
        np.nanmean(
            np.diff(
                X,
                axis=1
            )
        )
    )

    dy = float(
        np.nanmean(
            np.diff(
                Y,
                axis=0
            )
        )
    )

    return (
        abs(dx),
        abs(dy)
    )


def fill_nan_for_gradient(
    A: np.ndarray
) -> np.ndarray:

    B = np.array(
        A,
        copy=True
    )

    B[
        ~np.isfinite(B)
    ] = 0.0

    return B


def compute_vorticity_from_uv(

    X: np.ndarray,

    Y: np.ndarray,

    U: np.ndarray,

    V: np.ndarray

) -> np.ndarray:

    dx, dy = (
        estimate_dx_dy(
            X,
            Y
        )
    )

    du_dy = np.gradient(
        U,
        dy,
        axis=0
    )

    dv_dx = np.gradient(
        V,
        dx,
        axis=1
    )

    return (
        dv_dx
        -
        du_dy
    )


def compute_modal_vorticity(

    X: np.ndarray,

    Y: np.ndarray,

    modes_u: np.ndarray,

    modes_v: np.ndarray

) -> np.ndarray:

    dx, dy = (
        estimate_dx_dy(
            X,
            Y
        )
    )

    nm, ny, nx = (
        modes_u.shape
    )

    modes_w = np.full_like(
        modes_u,
        np.nan,
        dtype=np.float32
    )

    for m in range(nm):

        u_m = fill_nan_for_gradient(
            modes_u[m]
        )

        v_m = fill_nan_for_gradient(
            modes_v[m]
        )

        du_dy = np.gradient(
            u_m,
            dy,
            axis=0
        )

        dv_dx = np.gradient(
            v_m,
            dx,
            axis=1
        )

        w_m = (
            dv_dx
            -
            du_dy
        )

        valid = (
            np.isfinite(
                modes_u[m]
            )
            &
            np.isfinite(
                modes_v[m]
            )
        )

        w_m[
            ~valid
        ] = np.nan

        modes_w[m] = w_m

    return modes_w


# ============================================================
# 23. 重构
# ============================================================

def reconstruct_velocity(

    pod: Dict,

    snapshot_id: int,

    mode_ids: Sequence[int]

) -> Tuple[np.ndarray, np.ndarray]:

    ur = np.array(
        pod["mean_u"],
        copy=True
    )

    vr = np.array(
        pod["mean_v"],
        copy=True
    )

    for m in mode_ids:

        if m >= pod[
            "coeffs"
        ].shape[0]:

            continue

        ur += (
            pod[
                "coeffs"
            ][
                m,
                snapshot_id
            ]
            *
            np.nan_to_num(
                pod[
                    "modes_u"
                ][m],
                nan=0.0
            )
        )

        vr += (
            pod[
                "coeffs"
            ][
                m,
                snapshot_id
            ]
            *
            np.nan_to_num(
                pod[
                    "modes_v"
                ][m],
                nan=0.0
            )
        )

    return (
        ur,
        vr
    )


# ============================================================
# 24. 大周期相位槽
# ============================================================

def select_big_phase_slots(

    labels: List[Dict],

    n_slots: int = 8

) -> List[int]:

    big_phase = np.array([
        lab[
            "big_phase_frac"
        ]
        for lab in labels
    ])

    targets = np.linspace(
        0,
        1,
        n_slots,
        endpoint=False
    )

    selected = []

    for target in targets:

        sid = int(
            np.argmin(
                np.abs(
                    big_phase
                    -
                    target
                )
            )
        )

        selected.append(
            sid
        )

    return sorted(
        set(
            selected
        )
    )


# ============================================================
# 25. 通用 contour
# ============================================================

def plot_contour(

    X,

    Y,

    F,

    fname,

    title="",

    vmin=None,

    vmax=None,

    cmap=CMAP_WHITE,

    levels=128
):

    fig, ax = plt.subplots(
        figsize=(
            7.2,
            3.2
        )
    )

    if (
        vmin is None
        or
        vmax is None
    ):

        vmax0 = np.nanpercentile(
            np.abs(F),
            99.0
        )

        if (
            vmax0 == 0
            or
            not np.isfinite(
                vmax0
            )
        ):

            vmax0 = 1.0

        vmin0 = (
            -vmax0
        )

    else:

        vmin0 = vmin
        vmax0 = vmax

    cf = ax.contourf(

        X,
        Y,
        F,

        levels=np.linspace(
            vmin0,
            vmax0,
            levels
        ),

        cmap=cmap,

        extend="both"
    )

    fig.colorbar(
        cf,
        ax=ax,
        shrink=0.86
    )

    ax.set_xlabel(
        r"$x/c$"
    )

    ax.set_ylabel(
        r"$y/c$"
    )

    ax.set_title(
        title
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True
    )

    fig.tight_layout()

    fig.savefig(
        fname,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# 26. 多相位图
# ============================================================

def plot_phase_panel(

    X,

    Y,

    fields,

    labels_title,

    fname,

    main_title,

    vmin,

    vmax,

    ncols=4

):

    n = len(
        fields
    )

    ncols = min(
        ncols,
        n
    )

    nrows = int(
        np.ceil(
            n / ncols
        )
    )

    fig, axes = plt.subplots(

        nrows,
        ncols,

        figsize=(
            3.8 * ncols,
            2.5 * nrows
        ),

        squeeze=False
    )

    last_cf = None

    for i, ax in enumerate(
        axes.flat
    ):

        if i >= n:

            ax.axis(
                "off"
            )

            continue

        last_cf = ax.contourf(

            X,
            Y,
            fields[i],

            levels=np.linspace(
                vmin,
                vmax,
                128
            ),

            cmap=CMAP_WHITE,

            extend="both"
        )

        ax.set_title(
            labels_title[i]
        )

        ax.set_xlabel(
            r"$x/c$"
        )

        ax.set_ylabel(
            r"$y/c$"
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        ax.tick_params(
            direction="in",
            top=True,
            right=True
        )

    if last_cf is not None:

        fig.colorbar(

            last_cf,

            ax=
                axes.ravel().tolist(),

            shrink=0.82,

            pad=0.01
        )

    fig.suptitle(
        main_title,
        y=0.995
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            0.96,
            0.97
        ]
    )

    fig.savefig(
        fname,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# 27. 能谱
# ============================================================

def plot_pod_energy(

    pod,

    out_dir,

    cname

):

    energy = pod[
        "energy"
    ]

    cumulative = pod[
        "cumulative"
    ]

    mode_id = np.arange(
        1,
        len(energy) + 1
    )

    # Energy
    fig, ax = plt.subplots(
        figsize=(
            5.6,
            4.0
        )
    )

    ax.semilogy(
        mode_id,
        energy,
        "o-",
        ms=3,
        lw=1.1
    )

    ax.set_xlabel(
        "Mode index"
    )

    ax.set_ylabel(
        "Energy fraction"
    )

    ax.set_title(
        "POD energy spectrum"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(

        os.path.join(
            out_dir,
            f"{cname}_energy_spectrum.png"
        ),

        dpi=DPI
    )

    plt.close(
        fig
    )

    # Cumulative
    fig, ax = plt.subplots(
        figsize=(
            5.6,
            4.0
        )
    )

    ax.plot(
        mode_id,
        cumulative,
        "o-",
        ms=3,
        lw=1.1
    )

    ax.set_xlabel(
        "Mode index"
    )

    ax.set_ylabel(
        "Cumulative energy"
    )

    ax.set_ylim(
        0,
        1.02
    )

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(

        os.path.join(
            out_dir,
            f"{cname}_cumulative_energy.png"
        ),

        dpi=DPI
    )

    plt.close(
        fig
    )


# ============================================================
# 28. 时间系数
# ============================================================

def plot_temporal_coefficients(

    pod,

    labels,

    out_dir,

    cname,

    nplot=4

):

    coeffs = pod[
        "coeffs"
    ]

    nplot = min(
        nplot,
        coeffs.shape[0]
    )

    big_phase = np.array([
        lab[
            "big_phase_frac"
        ]
        for lab in labels
    ])

    fig, ax = plt.subplots(
        figsize=(
            7.0,
            4.0
        )
    )

    for m in range(
        nplot
    ):

        a_m = coeffs[m]

        denominator = np.max(
            np.abs(
                a_m
            )
        )

        if denominator == 0:

            denominator = 1.0

        ax.plot(

            big_phase,

            a_m
            /
            denominator,

            "-",

            lw=1.0,

            label=
                rf"$a_{m+1}$"
        )

    # 普通周期边界
    for x in [
        0.25,
        0.50,
        0.75
    ]:

        ax.axvline(
            x,
            color="0.7",
            lw=0.7,
            ls="--"
        )

    ax.set_xlim(
        0,
        1
    )

    ax.set_xlabel(
        r"Big-period phase $\tau/(4T)$"
    )

    ax.set_ylabel(
        "Normalised coefficient"
    )

    ax.legend(
        frameon=False,
        ncol=2
    )

    ax.grid(
        True,
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(

        os.path.join(
            out_dir,
            f"{cname}_temporal_coefficients.png"
        ),

        dpi=DPI
    )

    plt.close(
        fig
    )


# ============================================================
# 29. a1-a2 相图
# ============================================================

def plot_phase_portrait(

    pod,

    labels,

    out_dir,

    cname

):

    coeffs = pod[
        "coeffs"
    ]

    if coeffs.shape[0] < 2:

        return

    a1 = coeffs[0]
    a2 = coeffs[1]

    cycle_local = np.array([
        lab[
            "local_cycle_id"
        ]
        for lab in labels
    ])

    fig, ax = plt.subplots(
        figsize=(
            4.8,
            4.5
        )
    )

    ax.plot(
        a1,
        a2,
        "-",
        lw=0.8,
        color="black"
    )

    markers = [
        "o",
        "s",
        "^",
        "D"
    ]

    for local_id in range(
        4
    ):

        ids = np.where(
            cycle_local
            ==
            local_id
        )[0]

        if len(ids) == 0:

            continue

        ax.scatter(

            a1[ids],

            a2[ids],

            s=14,

            marker=
                markers[
                    local_id
                ],

            label=
                f"T block {local_id + 1}"
        )

    ax.set_xlabel(
        r"$a_1$"
    )

    ax.set_ylabel(
        r"$a_2$"
    )

    ax.set_title(
        r"$a_1-a_2$ phase portrait over $4T$"
    )

    ax.legend(
        frameon=False,
        fontsize=8
    )

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(

        os.path.join(
            out_dir,
            f"{cname}_a1_a2_phase_portrait.png"
        ),

        dpi=DPI
    )

    plt.close(
        fig
    )


# ============================================================
# 30. 时间系数频谱
# ============================================================

def plot_coefficient_spectrum(

    pod,

    labels,

    out_dir,

    cname,

    f_big,

    nplot=2

):

    coeffs = pod[
        "coeffs"
    ]

    nplot = min(
        nplot,
        coeffs.shape[0]
    )

    t = np.array([
        lab[
            "t_filename"
        ]
        for lab in labels
    ])

    if len(t) < 4:

        return

    dt = float(
        np.mean(
            np.diff(t)
        )
    )

    if dt <= 0:

        return

    freqs = np.fft.rfftfreq(
        coeffs.shape[1],
        d=dt
    )

    fig, ax = plt.subplots(
        figsize=(
            6.2,
            4.0
        )
    )

    for m in range(
        nplot
    ):

        a_m = (
            coeffs[m]
            -
            np.mean(
                coeffs[m]
            )
        )

        spec = np.abs(
            np.fft.rfft(
                a_m
            )
        ) ** 2

        if np.max(
            spec
        ) > 0:

            spec = (
                spec
                /
                np.max(
                    spec
                )
            )

        ax.plot(

            freqs
            /
            f_big,

            spec,

            lw=1.2,

            label=
                rf"$a_{m+1}$"
        )

    ax.set_xlabel(
        r"$f/f_B$"
    )

    ax.set_ylabel(
        "Normalised spectral energy"
    )

    ax.set_title(
        r"$f_B=1/(4T)$"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(

        os.path.join(
            out_dir,
            f"{cname}_coefficient_spectrum.png"
        ),

        dpi=DPI
    )

    plt.close(
        fig
    )


# ============================================================
# 31. 平均场 / 模态 / 4T瞬时图
# ============================================================

def plot_mean_modes_and_examples(

    X,

    Y,

    W,

    pod,

    modes_w,

    labels,

    out_dir,

    cname

):

    # --------------------------------------------------------
    # 4T平均涡量
    # --------------------------------------------------------

    W_mean = np.mean(
        W,
        axis=0
    )

    plot_contour(

        X,
        Y,
        W_mean,

        os.path.join(
            out_dir,
            f"{cname}_mean_vorticity.png"
        ),

        title=
            r"Mean vorticity over $4T$",

        vmin=
            VMIN_VORT,

        vmax=
            VMAX_VORT
    )

    # --------------------------------------------------------
    # 单个模态
    # --------------------------------------------------------

    for m in range(
        min(
            N_MODES_PLOT,
            modes_w.shape[0]
        )
    ):

        vmax_m = np.nanpercentile(
            np.abs(
                modes_w[m]
            ),
            99.0
        )

        if (
            vmax_m == 0
            or
            not np.isfinite(
                vmax_m
            )
        ):

            vmax_m = 1.0

        plot_contour(

            X,
            Y,

            modes_w[m],

            os.path.join(
                out_dir,
                f"{cname}_mode{m+1:02d}_modal_vorticity.png"
            ),

            title=
                f"Modal vorticity, mode {m+1}",

            vmin=
                -vmax_m,

            vmax=
                vmax_m
        )

    # --------------------------------------------------------
    # 前6模态总览
    # --------------------------------------------------------

    mode_fields = []

    mode_titles = []

    for m in range(
        min(
            N_MODES_PLOT,
            modes_w.shape[0]
        )
    ):

        mode_fields.append(
            modes_w[m]
        )

        mode_titles.append(

            f"Mode {m+1}, "
            f"E="
            f"{pod['energy'][m]:.3f}"
        )

    if mode_fields:

        vmax_all = np.nanpercentile(

            np.abs(
                np.stack(
                    mode_fields
                )
            ),

            99.0
        )

        if (
            vmax_all == 0
            or
            not np.isfinite(
                vmax_all
            )
        ):

            vmax_all = 1.0

        plot_phase_panel(

            X,
            Y,

            mode_fields,

            mode_titles,

            os.path.join(
                out_dir,
                f"{cname}_modal_vorticity_panel.png"
            ),

            main_title=
                "POD modal vorticity",

            vmin=
                -vmax_all,

            vmax=
                vmax_all,

            ncols=3
        )

    # --------------------------------------------------------
    # 整个4T取8个相位
    # --------------------------------------------------------

    selected = select_big_phase_slots(
        labels,
        n_slots=8
    )

    fields = [
        W[sid]
        for sid in selected
    ]

    titles = [

        (
            rf"$\tau/(4T)="
            f"{labels[sid]['big_phase_frac']:.3f}$"
        )

        for sid in selected
    ]

    plot_phase_panel(

        X,
        Y,

        fields,

        titles,

        os.path.join(
            out_dir,
            f"{cname}_instant_vorticity_big_period.png"
        ),

        main_title=
            r"Instantaneous vorticity over one $4T$ period",

        vmin=
            VMIN_VORT,

        vmax=
            VMAX_VORT,

        ncols=4
    )

    # --------------------------------------------------------
    # 前两阶重构
    # --------------------------------------------------------

    recon_fields = []

    recon_titles = []

    for sid in selected[
        :N_RECON_PLOT
    ]:

        ur, vr = reconstruct_velocity(

            pod,

            sid,

            mode_ids=[
                0,
                1
            ]
        )

        wr = compute_vorticity_from_uv(
            X,
            Y,
            ur,
            vr
        )

        wr[
            ~pod[
                "fluid_mask"
            ]
        ] = np.nan

        recon_fields.append(
            wr
        )

        recon_titles.append(

            rf"$\tau/(4T)="
            f"{labels[sid]['big_phase_frac']:.3f}$"
        )

    if recon_fields:

        plot_phase_panel(

            X,
            Y,

            recon_fields,

            recon_titles,

            os.path.join(
                out_dir,
                f"{cname}_recon12_vorticity_big_period.png"
            ),

            main_title=
                "Vorticity reconstructed by modes 1--2",

            vmin=
                VMIN_VORT,

            vmax=
                VMAX_VORT,

            ncols=2
        )


# ============================================================
# 32. CSV utilities
# ============================================================

def none_to_blank(
    x
):

    return (
        ""
        if x is None
        else x
    )


def write_dict_csv(

    path,

    rows

):

    if not rows:

        return

    fieldnames = []

    for row in rows:

        for key in row.keys():

            if key not in fieldnames:

                fieldnames.append(
                    key
                )

    with open(

        path,

        "w",

        newline="",

        encoding="utf-8-sig"

    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:

            writer.writerow({

                key:
                    none_to_blank(
                        row.get(
                            key
                        )
                    )

                for key in fieldnames
            })


# ============================================================
# 33. Snapshot labels
# ============================================================

def save_labels(

    labels,

    out_dir,

    cname

):

    path = os.path.join(
        out_dir,
        f"{cname}_snapshot_labels.csv"
    )

    header = [

        "snapshot_index",

        "filename",

        "t_filename",

        "solution_time",

        "cycle_no",

        "phase_frac",

        "phase_target",

        "local_cycle_id",

        "big_phase_frac",
    ]

    with open(

        path,

        "w",

        newline="",

        encoding="utf-8-sig"

    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            header
        )

        for lab in labels:

            writer.writerow([

                lab[
                    "snapshot_index"
                ],

                lab[
                    "filename"
                ],

                lab[
                    "t_filename"
                ],

                lab[
                    "solution_time"
                ],

                lab[
                    "cycle_no"
                ],

                lab[
                    "phase_frac"
                ],

                lab[
                    "phase_target"
                ],

                lab[
                    "local_cycle_id"
                ],

                lab[
                    "big_phase_frac"
                ],
            ])


# ============================================================
# 34. 能量 CSV
# ============================================================

def save_energy_csv(

    pod,

    out_dir,

    cname

):

    path = os.path.join(
        out_dir,
        f"{cname}_pod_energy.csv"
    )

    with open(

        path,

        "w",

        newline="",

        encoding="utf-8-sig"

    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([

            "mode",

            "lambda",

            "singular_value",

            "energy_fraction_Ei",

            "cumulative_energy_Ci",
        ])

        for i in range(
            len(
                pod[
                    "energy"
                ]
            )
        ):

            writer.writerow([

                i + 1,

                float(
                    pod[
                        "eigvals"
                    ][i]
                ),

                float(
                    pod[
                        "singular_values"
                    ][i]
                ),

                float(
                    pod[
                        "energy"
                    ][i]
                ),

                float(
                    pod[
                        "cumulative"
                    ][i]
                ),
            ])


# ============================================================
# 35. POD coefficients CSV
# ============================================================

def save_coefficients_csv(

    pod,

    labels,

    out_dir,

    cname

):

    path = os.path.join(
        out_dir,
        f"{cname}_pod_coefficients.csv"
    )

    coeffs = pod[
        "coeffs"
    ]

    header = [

        "snapshot_index",

        "filename",

        "t_filename",

        "cycle_no",

        "phase_frac",

        "big_phase_frac",
    ]

    header += [

        f"a{i+1}"

        for i in range(
            coeffs.shape[0]
        )
    ]

    with open(

        path,

        "w",

        newline="",

        encoding="utf-8-sig"

    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            header
        )

        for s, lab in enumerate(
            labels
        ):

            row = [

                lab[
                    "snapshot_index"
                ],

                lab[
                    "filename"
                ],

                lab[
                    "t_filename"
                ],

                lab[
                    "cycle_no"
                ],

                lab[
                    "phase_frac"
                ],

                lab[
                    "big_phase_frac"
                ],

            ] + [

                float(
                    coeffs[
                        i,
                        s
                    ]
                )

                for i in range(
                    coeffs.shape[0]
                )
            ]

            writer.writerow(
                row
            )


# ============================================================
# 36. 保存 NPZ
# ============================================================

def save_all_case_outputs(

    X,

    Y,

    U,

    V,

    W,

    pod,

    modes_w,

    metrics,

    labels,

    out_dir,

    cname

):

    np.savez_compressed(

        os.path.join(
            out_dir,
            f"{cname}_pod_results.npz"
        ),

        X=X,

        Y=Y,

        U_mean=
            pod[
                "mean_u"
            ],

        V_mean=
            pod[
                "mean_v"
            ],

        W_mean=
            np.mean(
                W,
                axis=0
            ),

        eigvals=
            pod[
                "eigvals"
            ],

        energy=
            pod[
                "energy"
            ],

        cumulative=
            pod[
                "cumulative"
            ],

        coeffs=
            pod[
                "coeffs"
            ],

        modes_u=
            pod[
                "modes_u"
            ],

        modes_v=
            pod[
                "modes_v"
            ],

        modes_w=
            modes_w,

        fluid_mask=
            pod[
                "fluid_mask"
            ],

        cycle_no=
            np.array([
                x[
                    "cycle_no"
                ]
                for x in labels
            ]),

        phase_frac=
            np.array([
                x[
                    "phase_frac"
                ]
                for x in labels
            ]),

        big_phase_frac=
            np.array([
                x[
                    "big_phase_frac"
                ]
                for x in labels
            ]),
    )

    # JSON
    with open(

        os.path.join(
            out_dir,
            f"{cname}_pod_metrics.json"
        ),

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            metrics,

            f,

            indent=4,

            ensure_ascii=False
        )

    # metrics CSV
    write_dict_csv(

        os.path.join(
            out_dir,
            f"{cname}_pod_metrics_summary.csv"
        ),

        [
            metrics
        ]
    )

    save_labels(
        labels,
        out_dir,
        cname
    )

    save_energy_csv(
        pod,
        out_dir,
        cname
    )

    save_coefficients_csv(
        pod,
        labels,
        out_dir,
        cname
    )


# ============================================================
# 37. 单个工况
# ============================================================

def run_one_case(

    case: CaseSpec

) -> Optional[Dict]:

    cname = case_name(
        case
    )

    case_dir = get_case_dir(
        case
    )

    out_dir = os.path.join(
        SAVE_ROOT,
        cname
    )

    ensure_dir(
        out_dir
    )

    # 基本周期
    f1 = (
        case.k
        /
        np.pi
    )

    T = (
        1.0
        /
        f1
    )

    # 大周期
    T_big = (
        4.0
        *
        T
    )

    f_big = (
        1.0
        /
        T_big
    )

    print(
        "=" * 76
    )

    print(
        f"工况: {cname}"
    )

    print(
        f"case_dir = {case_dir}"
    )

    print(
        f"f1 = {f1:.8f}"
    )

    print(
        f"T = {T:.8f}"
    )

    print(
        f"T_big = {T_big:.8f}"
    )

    print(
        f"f_big = {f_big:.8f}"
    )

    # --------------------------------------------------------
    # 目录检查
    # --------------------------------------------------------

    if not os.path.isdir(
        case_dir
    ):

        print(
            "  跳过：目录不存在。"
        )

        return None

    # --------------------------------------------------------
    # 扫描全部文件
    # --------------------------------------------------------

    all_records = scan_all_records(
        case_dir,
        T
    )

    if not all_records:

        print(
            "  跳过：没有找到 F*.dat。"
        )

        return None

    print(
        f"  共扫描到 "
        f"{len(all_records)} "
        f"个流场文件。"
    )

    # --------------------------------------------------------
    # 构造4T大周期
    # --------------------------------------------------------

    big_records, selected_cycles = (
        build_big_period_records(
            case,
            all_records
        )
    )

    print(
        f"  POD使用普通周期: "
        f"{selected_cycles}"
    )

    print(
        f"  POD快照数: "
        f"{len(big_records)}"
    )

    if len(
        big_records
    ) != 128:

        raise ValueError(
            "POD快照数不是128。"
        )

    # --------------------------------------------------------
    # 加载
    # --------------------------------------------------------

    (
        X,
        Y,

        U,
        V,

        W,
        BODY,

        labels

    ) = load_snapshots_for_pod(
        big_records
    )

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    fluid_mask = build_fluid_mask(
        BODY
    )

    print(

        f"  POD流体点: "
        f"{int(np.sum(fluid_mask))}"
        f"/"
        f"{fluid_mask.size}"
    )

    # --------------------------------------------------------
    # POD
    # --------------------------------------------------------

    pod = compute_velocity_pod(

        U,

        V,

        fluid_mask,

        n_modes_save=
            N_MODES_SAVE
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = compute_pod_metrics(

        case,

        pod,

        labels,

        selected_cycles
    )

    # --------------------------------------------------------
    # Modal vorticity
    # --------------------------------------------------------

    modes_w = compute_modal_vorticity(

        X,

        Y,

        pod[
            "modes_u"
        ],

        pod[
            "modes_v"
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_all_case_outputs(

        X,
        Y,

        U,
        V,

        W,

        pod,

        modes_w,

        metrics,

        labels,

        out_dir,

        cname
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    plot_pod_energy(
        pod,
        out_dir,
        cname
    )

    plot_phase_portrait(
        pod,
        labels,
        out_dir,
        cname
    )

    plot_temporal_coefficients(
        pod,
        labels,
        out_dir,
        cname,
        nplot=
            N_COEFF_PLOT
    )

    plot_coefficient_spectrum(

        pod,

        labels,

        out_dir,

        cname,

        f_big=
            f_big,

        nplot=2
    )

    plot_mean_modes_and_examples(

        X,
        Y,

        W,

        pod,

        modes_w,

        labels,

        out_dir,

        cname
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "  关键POD指标："
    )

    for key in [

        "E1",

        "E2",

        "C2",

        "C4",

        "Delta12",

        "Delta34",

        "H",

        "N_eff"

    ]:

        print(
            f"    "
            f"{key}: "
            f"{metrics.get(key)}"
        )

    return metrics


# ============================================================
# 38. k-DC 汇总图
# ============================================================

def plot_summary_vs_dc(

    summary_rows: List[Dict],

    save_root: str

):

    if not summary_rows:

        return

    metrics_to_plot = [

        (
            "C2",
            r"$C_2$"
        ),

        (
            "C4",
            r"$C_4$"
        ),

        (
            "Delta12",
            r"$\Delta_{12}$"
        ),

        (
            "H",
            r"$H$"
        ),
    ]

    fig, axes = plt.subplots(

        2,
        2,

        figsize=(
            6.5,
            4.8
        )
    )

    marker_map = {

        0.8:
            "o",

        1.1:
            "s",

        1.4:
            "^",

        1.7:
            "D",
    }

    for ax, (
        metric_key,
        ylabel
    ) in zip(
        axes.flat,
        metrics_to_plot
    ):

        for k in K_LIST:

            rows_k = [

                row

                for row in summary_rows

                if np.isclose(
                    row[
                        "k"
                    ],
                    k
                )
            ]

            rows_k = sorted(

                rows_k,

                key=lambda row:
                    -row[
                        "dc"
                    ]
            )

            if not rows_k:

                continue

            x = np.array([
                row[
                    "dc"
                ]
                for row in rows_k
            ])

            y = np.array([
                row.get(
                    metric_key,
                    np.nan
                )
                for row in rows_k
            ])

            ax.plot(

                x,

                y,

                marker=
                    marker_map[
                        k
                    ],

                lw=1.0,

                ms=4,

                label=
                    rf"$k={k:g}$"
            )

        ax.set_xlim(
            1.05,
            0.20
        )

        ax.set_xticks([
            1.0,
            0.75,
            0.5,
            0.25
        ])

        ax.set_xticklabels([
            "1",
            "0.75",
            "0.5",
            "0.25"
        ])

        ax.set_xlabel(
            r"$DC$"
        )

        ax.set_ylabel(
            ylabel,
            rotation=0,
            labelpad=10
        )

        ax.tick_params(
            direction="in",
            top=True,
            right=True
        )

    handles, labels = (
        axes[
            0,
            0
        ]
        .get_legend_handles_labels()
    )

    fig.legend(

        handles,

        labels,

        frameon=False,

        ncol=4,

        loc=
            "lower center",

        bbox_to_anchor=(
            0.5,
            -0.01
        )
    )

    fig.tight_layout(
        rect=[
            0,
            0.07,
            1,
            1
        ]
    )

    fig.savefig(

        os.path.join(
            save_root,
            "POD_metrics_vs_DC.png"
        ),

        dpi=600,

        bbox_inches="tight"
    )

    fig.savefig(

        os.path.join(
            save_root,
            "POD_metrics_vs_DC.svg"
        ),

        format="svg",

        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# 39. Main
# ============================================================

def main():

    ensure_dir(
        SAVE_ROOT
    )

    print(
        "=" * 76
    )

    print(
        "DC big-period POD analysis"
    )

    print(
        f"St = {ST}"
    )

    print(
        f"Cases = {len(CASES)}"
    )

    print(
        "DC < 1: cycles 25--28"
    )

    print(
        "DC = 1: last four complete cycles"
    )

    print(
        "Snapshots = 4 x 32 = 128"
    )

    print(
        f"SAVE_ROOT = {SAVE_ROOT}"
    )

    print(
        "=" * 76
    )

    metrics_all = []

    failed_cases = []

    for case in CASES:

        try:

            metrics = run_one_case(
                case
            )

            if metrics is not None:

                metrics_all.append(
                    metrics
                )

        except Exception as e:

            cname = case_name(
                case
            )

            failed_cases.append(
                cname
            )

            print(
                f"\n[失败] {cname}"
            )

            print(
                f"错误信息: {repr(e)}"
            )

    # ========================================================
    # 汇总 CSV
    # ========================================================

    metrics_csv = os.path.join(

        SAVE_ROOT,

        "all_cases_pod_metrics_summary.csv"
    )

    write_dict_csv(
        metrics_csv,
        metrics_all
    )

    print(
        "=" * 76
    )

    print(
        f"汇总指标保存至："
        f"{metrics_csv}"
    )

    # ========================================================
    # 失败列表
    # ========================================================

    if failed_cases:

        failed_txt = os.path.join(
            SAVE_ROOT,
            "failed_cases.txt"
        )

        with open(
            failed_txt,
            "w",
            encoding="utf-8"
        ) as f:

            for name in failed_cases:

                f.write(
                    name
                    +
                    "\n"
                )

        print(
            f"失败工况："
            f"{failed_txt}"
        )

    # ========================================================
    # k-DC POD指标图
    # ========================================================

    plot_summary_vs_dc(
        metrics_all,
        SAVE_ROOT
    )

    print(
        "=" * 76
    )

    print(
        "全部 POD 分析完成。"
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()