# -*- coding: utf-8 -*-
"""
big_period_local_correlation_dc.py

间歇扑翼大周期局部相关性分析

分析内容
--------
1. St = 0.3

2. 推进平衡工况：
       k=0.8 -> a=20
       k=1.1 -> a=20
       k=1.4 -> a=17
       k=1.7 -> a=11

3. 对每个 k 分析：
       DC = 0.75
       DC = 0.50
       DC = 0.25

4. 比较两个完整间歇大周期：

       大周期 A：普通周期 21,22,23,24
       大周期 B：普通周期 25,26,27,28

5. 每个普通周期固定抽取 32 个相位：

       4 × 32 = 128 snapshots / big period

6. 在每个空间网格点比较：

       omega_21-24(x,y,t)
       omega_25-28(x,y,t)

   两个长度为 128 的涡量时间序列，并计算 Pearson 相关系数：

       rho_big(x,y)

7. 输出：
       - 每个 DC 的 NPZ
       - 每个 k 的三 DC 对比图
       - 每个 k 的统计 CSV
       - 全部工况汇总 CSV
       - 失败工况记录
"""

import os
import re
import csv
import gc

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# ============================================================
# 0. 工况定义
# ============================================================

@dataclass
class CaseSpec:
    k: float
    st: float
    a: float
    dc: float


ST = 0.3

# 推进平衡角
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

# 这里只处理间歇模式
DC_LIST = [
    0.75,
    0.50,
    0.25,
]


# ============================================================
# 1. 路径
# ============================================================

BASE_ROOT = (
    r"E:/Rensq/DC_explore_F/data_1Tflow/"
)

SAVE_ROOT = (
    r"E:/Rensq/DC_explore_F/flow/"
    r"local_correlation_DC"
)

os.makedirs(
    SAVE_ROOT,
    exist_ok=True
)


# ============================================================
# 2. 大周期设置
# ============================================================

# 前一个完整大周期
BIG_PERIOD_1_CYCLES = [
    21,
    22,
    23,
    24,
]

# 后一个完整大周期
BIG_PERIOD_2_CYCLES = [
    25,
    26,
    27,
    28,
]

# 每个普通周期固定 32 相位
N_PHASES_PER_CYCLE = 32

# 一个完整 4T 大周期
N_SNAPSHOTS_BIG_PERIOD = (
    4
    * N_PHASES_PER_CYCLE
)

# = 128

# 相位匹配容差
PHASE_TOL = 0.020


# ============================================================
# 3. 相关性分析区域
# ============================================================

c = 1.0

X_MIN = -1.0
X_MAX = 20.0

Y_MIN = -10.0
Y_MAX = 10.0


# ============================================================
# 4. 相关图空间降采样
# ============================================================

# 希望最终相关图网格间距约为 0.1c
CORR_TARGET_DX = 0.10
CORR_TARGET_DY = 0.10


# ============================================================
# 5. 固体区域
# ============================================================

USE_BODY_MASK = True

BODY_THRESHOLD = 0.5


# ============================================================
# 6. Pearson 相关设置
# ============================================================

# 原来单周期：
#
# 12 / 32 = 37.5%
#
# 大周期为128个点，因此保持相同有效比例：
#
# 128 × 0.375 = 48
#
MIN_VALID_FRACTION = (
    12.0 / 32.0
)

MIN_VALID_BIG_PHASES = int(
    np.ceil(
        N_SNAPSHOTS_BIG_PERIOD
        * MIN_VALID_FRACTION
    )
)

# 弱流过滤
LOCAL_ACTIVITY_FRAC = 0.02

ACTIVITY_PERCENTILE = 99.0

CORR_EPS = 1.0e-14

# CSV中统计 rho < 0.9 的区域比例
LOW_CORRELATION_THRESHOLD = 0.90


# ============================================================
# 7. 输出与缓存
# ============================================================

# False：
# 如果 NPZ 已存在且参数一致，则直接读取。
#
# True：
# 强制重新读取全部原始流场。
#
FORCE_RECOMPUTE = False

STOP_ON_ERROR = False


# ============================================================
# 8. 绘图
# ============================================================

USE_TEX = True

FONT_SIZE = 10

plt.rcParams.update({

    "font.size":
        FONT_SIZE,

    "axes.titlesize":
        FONT_SIZE,

    "axes.labelsize":
        FONT_SIZE,

    "xtick.labelsize":
        FONT_SIZE,

    "ytick.labelsize":
        FONT_SIZE,

    "legend.fontsize":
        FONT_SIZE,

    "font.family":
        "serif",

    "font.serif": [
        "Times New Roman"
    ],

    "text.usetex":
        USE_TEX,

    "axes.linewidth":
        0.7,

    "xtick.direction":
        "in",

    "ytick.direction":
        "in",

    "xtick.major.width":
        0.7,

    "ytick.major.width":
        0.7,

    "xtick.major.size":
        3.0,

    "ytick.major.size":
        3.0,
})

if USE_TEX:

    plt.rcParams.update({

        "pgf.texsystem":
            "pdflatex",

        "text.latex.preamble":
            r"\usepackage{amsmath}"
            r"\usepackage{newtxtext,newtxmath}"
    })


# ============================================================
# 9. 基本工具
# ============================================================

def fmt_num(
    x: float
) -> str:
    """
    用于实际目录：

    0.8  -> 0.8
    1.0  -> 1
    0.50 -> 0.5
    """

    if float(x).is_integer():

        return str(
            int(x)
        )

    return f"{x:g}"


def fmt_num_short(
    x: float
) -> str:

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


def k_case_name(
    k: float,
    st: float,
    a: float
) -> str:

    return (
        f"k{fmt_num_short(k)}"
        f"_St{fmt_num_short(st)}"
        f"_a{fmt_num_short(a)}"
    )


def ensure_dir(
    path: str
):

    os.makedirs(
        path,
        exist_ok=True
    )


# ============================================================
# 10. DC 数据路径
# ============================================================

def get_case_dir(
    case: CaseSpec
) -> str:
    """
    完全按照 POD 大周期代码的目录规则。

    例如 DC=0.5：

    E:/.../
        k0.8/
        st0.3k0.8/
        st0.3k0.8a20/
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
    # DC < 1
    # --------------------------------------------------------

    dc_int = int(
        round(
            case.dc
            * 100
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
# 11. Tecplot Ordered DAT reader
# ============================================================

def read_tecplot_ordered_dat(

    filename: str,

    target_vars: Sequence[str] = (
        "X",
        "Y",
        "Vorticity",
        "body"
    )

) -> Dict[str, np.ndarray]:
    """
    流场读取方式与 POD 大周期代码保持一致。
    """

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

            # ------------------------------------------------
            # 某些 DAT 文件在 ZONE 后单独保存 solution time
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 判断数据开始行
            # ------------------------------------------------

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
            f"无法解析 dat 文件头信息: "
            f"{filename}"
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

    if data.ndim == 1:

        data = data[
            None,
            :
        ]

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
# 12. 文件名时间
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

            os.path.basename(
                filename
            ),

            re.IGNORECASE

        )
        is not None
    )


# ============================================================
# 13. 普通周期识别
# ============================================================

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
# 14. 扫描全部流场文件
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
        key=lambda x:
            x["t"]
    )

    return records


# ============================================================
# 15. 单周期抽取固定 32 相位
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
    从指定周期抽取：

        0/32
        1/32
        ...
        31/32

    共32个快照。

    与 POD 大周期代码保持一致。
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
                    phases
                    -
                    target
                )
            )
        )

        rec = cycle_records[
            idx
        ]

        error = abs(
            rec["phase_frac"]
            -
            target
        )

        if error > phase_tol:

            print(

                f"    cycle {cycle_no}: "
                f"target={target:.5f}, "
                f"nearest="
                f"{rec['phase_frac']:.5f}, "
                f"error={error:.5f} "
                f"> {phase_tol:.5f}"
            )

            return None

        # ----------------------------------------------------
        # 同一个文件不能被两个目标相位重复使用
        # ----------------------------------------------------

        if rec["path"] in used_paths:

            print(

                f"    cycle {cycle_no}: "
                f"同一流场文件被多个目标相位选中，"
                f"该周期判定为不完整。"
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
# 16. 构造指定 4T 大周期
# ============================================================

def build_big_period_records(

    all_records: List[Dict],

    selected_cycles: Sequence[int]

) -> List[Dict]:
    """
    例如：

        selected_cycles = [21,22,23,24]

    返回128个按大周期时间顺序排列的流场记录。
    """

    if len(
        selected_cycles
    ) != 4:

        raise ValueError(
            "一个大周期必须恰好包含4个普通周期。"
        )

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

        for rec in cycle_records:

            rec_new = dict(
                rec
            )

            rec_new[
                "local_cycle_id"
            ] = int(
                local_cycle_id
            )

            # ------------------------------------------------
            # 大周期归一化相位
            #
            # cycle 1 -> 0 ~ 0.25
            # cycle 2 -> 0.25 ~ 0.50
            # cycle 3 -> 0.50 ~ 0.75
            # cycle 4 -> 0.75 ~ 1.00
            # ------------------------------------------------

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

                / 4.0
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
    ) != N_SNAPSHOTS_BIG_PERIOD:

        raise ValueError(

            "大周期快照数错误："
            f"期望 "
            f"{N_SNAPSHOTS_BIG_PERIOD}，"
            f"实际 "
            f"{len(selected_records)}"
        )

    return selected_records


# ============================================================
# 17. 空间降采样
# ============================================================

def choose_stride(

    coordinate_line: np.ndarray,

    target_spacing: float

) -> Tuple[int, float]:

    differences = np.diff(
        np.asarray(
            coordinate_line,
            dtype=float
        )
    )

    differences = differences[
        np.isfinite(
            differences
        )
        &
        (
            np.abs(
                differences
            ) > 0.0
        )
    ]

    if differences.size == 0:

        raise ValueError(
            "无法确定网格间距。"
        )

    spacing = float(

        np.median(
            np.abs(
                differences
            )
        )
    )

    stride = max(

        1,

        int(
            round(
                target_spacing
                /
                spacing
            )
        )
    )

    return (
        stride,
        spacing
    )


def build_selected_indices(

    X: np.ndarray,

    Y: np.ndarray

) -> Tuple[
    np.ndarray,
    np.ndarray,
    Dict
]:

    x_line = np.asarray(
        X[0, :],
        dtype=float
    )

    y_line = np.asarray(
        Y[:, 0],
        dtype=float
    )

    stride_x, raw_dx = (
        choose_stride(
            x_line,
            CORR_TARGET_DX
        )
    )

    stride_y, raw_dy = (
        choose_stride(
            y_line,
            CORR_TARGET_DY
        )
    )

    x_candidates = np.arange(
        0,
        x_line.size,
        stride_x,
        dtype=int
    )

    y_candidates = np.arange(
        0,
        y_line.size,
        stride_y,
        dtype=int
    )

    x_keep = x_candidates[

        (
            x_line[x_candidates]
            >= X_MIN
        )

        &

        (
            x_line[x_candidates]
            <= X_MAX
        )
    ]

    y_keep = y_candidates[

        (
            y_line[y_candidates]
            >= Y_MIN
        )

        &

        (
            y_line[y_candidates]
            <= Y_MAX
        )
    ]

    if (
        len(x_keep) < 3
        or
        len(y_keep) < 3
    ):

        raise ValueError(
            "相关性计算区域内网格点过少。"
        )

    diagnostics = {

        "stride_x":
            int(stride_x),

        "stride_y":
            int(stride_y),

        "raw_dx":
            float(raw_dx),

        "raw_dy":
            float(raw_dy),

        "effective_dx":
            float(
                stride_x
                * raw_dx
            ),

        "effective_dy":
            float(
                stride_y
                * raw_dy
            ),

        "nx":
            int(
                len(x_keep)
            ),

        "ny":
            int(
                len(y_keep)
            ),
    }

    return (
        x_keep,
        y_keep,
        diagnostics
    )


# ============================================================
# 18. 加载两个128快照大周期
# ============================================================

def load_two_big_periods(

    big_records_1: List[Dict],

    big_records_2: List[Dict]

) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict
]:
    """
    返回：

        X
        Y
        W1 : (128, ny, nx)
        W2 : (128, ny, nx)
        metadata
    """

    if (
        len(big_records_1)
        != N_SNAPSHOTS_BIG_PERIOD
        or
        len(big_records_2)
        != N_SNAPSHOTS_BIG_PERIOD
    ):

        raise ValueError(
            "两个大周期都必须包含128个快照。"
        )

    # --------------------------------------------------------
    # 首个文件建立空间索引
    # --------------------------------------------------------

    first = read_tecplot_ordered_dat(

        big_records_1[0]["path"],

        target_vars=(
            "X",
            "Y",
            "Vorticity",
            "body"
        )
    )

    X0 = first["X"]
    Y0 = first["Y"]

    full_shape = (
        X0.shape
    )

    (
        cols,
        rows,
        sampling
    ) = (
        None,
        None,
        None
    )

    # 注意 build_selected_indices 返回 x_indices, y_indices
    x_indices, y_indices, sampling = (
        build_selected_indices(
            X0,
            Y0
        )
    )

    Xc = (
        X0[
            np.ix_(
                y_indices,
                x_indices
            )
        ]
        .astype(
            np.float32
        )
        / c
    )

    Yc = (
        Y0[
            np.ix_(
                y_indices,
                x_indices
            )
        ]
        .astype(
            np.float32
        )
        / c
    )

    ny, nx = (
        Xc.shape
    )

    W1 = np.full(

        (
            N_SNAPSHOTS_BIG_PERIOD,
            ny,
            nx
        ),

        np.nan,

        dtype=np.float32
    )

    W2 = np.full_like(
        W1,
        np.nan
    )

    # --------------------------------------------------------
    # 内部读取函数
    # --------------------------------------------------------

    def load_one_period(
        records: List[Dict],
        output: np.ndarray,
        label: str
    ):

        for sid, rec in enumerate(
            records
        ):

            print(

                f"    {label}: "
                f"[{sid + 1:03d}/"
                f"{len(records):03d}] "
                f"cycle="
                f"{rec['cycle_no']}, "
                f"phase="
                f"{rec['phase_target']:.5f}, "
                f"{rec['filename']}"
            )

            dat = (
                read_tecplot_ordered_dat(

                    rec["path"],

                    target_vars=(
                        "X",
                        "Y",
                        "Vorticity",
                        "body"
                    )
                )
            )

            if (
                dat["X"].shape
                != full_shape
            ):

                raise ValueError(

                    f"网格尺寸发生变化："
                    f"{dat['X'].shape} "
                    f"!= {full_shape}"
                )

            W0 = dat[
                "Vorticity"
            ][
                np.ix_(
                    y_indices,
                    x_indices
                )
            ].astype(
                np.float32
            )

            B0 = dat[
                "body"
            ][
                np.ix_(
                    y_indices,
                    x_indices
                )
            ].astype(
                np.float32
            )

            finite = np.isfinite(
                W0
            )

            if USE_BODY_MASK:

                valid = (
                    finite
                    &
                    (
                        B0
                        < BODY_THRESHOLD
                    )
                )

            else:

                valid = finite

            # ------------------------------------------------
            # 与原局部相关方法一致：
            # 实体占据的相位单独设为 NaN，
            # 而不是永久删除整个空间点。
            # ------------------------------------------------

            output[sid] = np.where(
                valid,
                W0,
                np.nan
            )

    # --------------------------------------------------------
    # 加载 21-24
    # --------------------------------------------------------

    load_one_period(
        big_records_1,
        W1,
        "21-24"
    )

    # --------------------------------------------------------
    # 加载 25-28
    # --------------------------------------------------------

    load_one_period(
        big_records_2,
        W2,
        "25-28"
    )

    metadata = {

        "sampling":
            sampling,

        "n_snapshots":
            N_SNAPSHOTS_BIG_PERIOD,
    }

    return (
        Xc,
        Yc,
        W1,
        W2,
        metadata
    )


# ============================================================
# 19. 大周期逐网格 Pearson 相关
# ============================================================

def pointwise_big_period_pearson(

    W1: np.ndarray,

    W2: np.ndarray

) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict
]:
    """
    W1, W2:

        shape = (128, ny, nx)

    对每个空间点计算：

        rho = corr(
            W1[:, y, x],
            W2[:, y, x]
        )
    """

    field_1 = np.asarray(
        W1,
        dtype=np.float64
    )

    field_2 = np.asarray(
        W2,
        dtype=np.float64
    )

    if (
        field_1.shape
        != field_2.shape
    ):

        raise ValueError(
            "两个大周期数组尺寸不一致。"
        )

    if (
        field_1.ndim != 3
    ):

        raise ValueError(
            "输入必须为 "
            "(128, ny, nx)。"
        )

    if (
        field_1.shape[0]
        != N_SNAPSHOTS_BIG_PERIOD
    ):

        raise ValueError(

            f"大周期快照数应为 "
            f"{N_SNAPSHOTS_BIG_PERIOD}，"
            f"实际 "
            f"{field_1.shape[0]}"
        )

    # --------------------------------------------------------
    # 两个大周期该相位都必须为有效流体
    # --------------------------------------------------------

    valid = (

        np.isfinite(
            field_1
        )

        &

        np.isfinite(
            field_2
        )
    )

    valid_count = np.sum(
        valid,
        axis=0
    )

    safe_count = np.maximum(
        valid_count,
        1
    )

    values_1 = np.where(
        valid,
        field_1,
        0.0
    )

    values_2 = np.where(
        valid,
        field_2,
        0.0
    )

    # --------------------------------------------------------
    # 各空间点的时间均值
    # --------------------------------------------------------

    mean_1 = (
        np.sum(
            values_1,
            axis=0
        )
        /
        safe_count
    )

    mean_2 = (
        np.sum(
            values_2,
            axis=0
        )
        /
        safe_count
    )

    centered_1 = np.where(

        valid,

        field_1
        -
        mean_1[
            None,
            :,
            :
        ],

        0.0
    )

    centered_2 = np.where(

        valid,

        field_2
        -
        mean_2[
            None,
            :,
            :
        ],

        0.0
    )

    numerator = np.sum(

        centered_1
        *
        centered_2,

        axis=0
    )

    energy_1 = np.sum(
        centered_1 ** 2,
        axis=0
    )

    energy_2 = np.sum(
        centered_2 ** 2,
        axis=0
    )

    denominator = np.sqrt(
        energy_1
        *
        energy_2
    )

    # ========================================================
    # 局部活动度
    # ========================================================

    activity_numerator = np.sum(

        np.where(

            valid,

            0.5
            *
            (
                field_1 ** 2
                +
                field_2 ** 2
            ),

            0.0
        ),

        axis=0
    )

    activity = np.sqrt(

        activity_numerator
        /
        safe_count
    )

    finite_activity = activity[

        np.isfinite(
            activity
        )

        &

        (
            valid_count
            >=
            MIN_VALID_BIG_PHASES
        )
    ]

    if (
        LOCAL_ACTIVITY_FRAC <= 0.0
        or
        finite_activity.size == 0
    ):

        activity_reference = np.nan

        activity_threshold = 0.0

    else:

        activity_reference = float(

            np.nanpercentile(

                finite_activity,

                ACTIVITY_PERCENTILE
            )
        )

        activity_threshold = (

            LOCAL_ACTIVITY_FRAC
            *
            activity_reference
        )

    # ========================================================
    # 有效 rho
    # ========================================================

    valid_output = (

        (
            valid_count
            >=
            MIN_VALID_BIG_PHASES
        )

        &

        np.isfinite(
            denominator
        )

        &

        (
            denominator
            >
            CORR_EPS
        )

        &

        np.isfinite(
            activity
        )

        &

        (
            activity
            >=
            activity_threshold
        )
    )

    rho = np.full(

        valid_count.shape,

        np.nan,

        dtype=np.float64
    )

    rho[
        valid_output
    ] = (

        numerator[
            valid_output
        ]

        /

        denominator[
            valid_output
        ]
    )

    rho[
        valid_output
    ] = np.clip(

        rho[
            valid_output
        ],

        -1.0,
        1.0
    )

    diagnostics = {

        "activity_reference":
            activity_reference,

        "activity_threshold":
            activity_threshold,

        "valid_point_count":
            int(
                np.sum(
                    np.isfinite(
                        rho
                    )
                )
            ),

        "total_point_count":
            int(
                rho.size
            ),
    }

    return (
        rho,
        valid_count,
        activity,
        diagnostics
    )


# ============================================================
# 20. 统计量
# ============================================================

def build_statistics(

    case: CaseSpec,

    rho: np.ndarray,

    valid_count: np.ndarray,

    diagnostics: Dict

) -> Dict:

    finite = np.isfinite(
        rho
    )

    values = rho[
        finite
    ]

    valid_samples = (
        valid_count[
            finite
        ]
    )

    row = {

        "k":
            float(case.k),

        "St":
            float(case.st),

        "a":
            float(case.a),

        "DC":
            float(case.dc),

        "big_period_1":
            "21-24",

        "big_period_2":
            "25-28",

        "n_snapshots_each":
            N_SNAPSHOTS_BIG_PERIOD,

        "min_valid_snapshots":
            MIN_VALID_BIG_PHASES,

        "valid_spatial_points":
            int(
                len(values)
            ),

        "activity_threshold":
            diagnostics[
                "activity_threshold"
            ],
    }

    if values.size == 0:

        row.update({

            "mean_local_rho":
                np.nan,

            "median_local_rho":
                np.nan,

            "std_local_rho":
                np.nan,

            "minimum_local_rho":
                np.nan,

            "maximum_local_rho":
                np.nan,

            "fraction_rho_below_0p9":
                np.nan,

            "mean_valid_snapshots":
                np.nan,
        })

    else:

        row.update({

            "mean_local_rho":
                float(
                    np.mean(
                        values
                    )
                ),

            "median_local_rho":
                float(
                    np.median(
                        values
                    )
                ),

            "std_local_rho":
                float(
                    np.std(
                        values
                    )
                ),

            "minimum_local_rho":
                float(
                    np.min(
                        values
                    )
                ),

            "maximum_local_rho":
                float(
                    np.max(
                        values
                    )
                ),

            "fraction_rho_below_0p9":
                float(
                    np.mean(
                        values
                        <
                        LOW_CORRELATION_THRESHOLD
                    )
                ),

            "mean_valid_snapshots":
                float(
                    np.mean(
                        valid_samples
                    )
                ),
        })

    return row


# ============================================================
# 21. NPZ
# ============================================================

def get_npz_path(
    case: CaseSpec
) -> str:

    out_dir = os.path.join(

        SAVE_ROOT,

        k_case_name(
            case.k,
            case.st,
            case.a
        )
    )

    ensure_dir(
        out_dir
    )

    return os.path.join(

        out_dir,

        (
            f"{case_name(case)}"
            f"_rho_big_21-24_vs_25-28.npz"
        )
    )


def save_npz(

    path: str,

    case: CaseSpec,

    X: np.ndarray,

    Y: np.ndarray,

    rho: np.ndarray,

    valid_count: np.ndarray,

    activity: np.ndarray,

    metadata: Dict,

    diagnostics: Dict

):

    sampling = metadata[
        "sampling"
    ]

    np.savez_compressed(

        path,

        k=
            float(case.k),

        St=
            float(case.st),

        a=
            float(case.a),

        dc=
            float(case.dc),

        cycles_1=
            np.asarray(
                BIG_PERIOD_1_CYCLES,
                dtype=int
            ),

        cycles_2=
            np.asarray(
                BIG_PERIOD_2_CYCLES,
                dtype=int
            ),

        n_phase_per_cycle=
            N_PHASES_PER_CYCLE,

        n_snapshots_big_period=
            N_SNAPSHOTS_BIG_PERIOD,

        phase_tolerance=
            PHASE_TOL,

        x_over_c=
            X,

        y_over_c=
            Y,

        x_range=
            np.asarray([
                X_MIN,
                X_MAX
            ]),

        y_range=
            np.asarray([
                Y_MIN,
                Y_MAX
            ]),

        target_dx=
            CORR_TARGET_DX,

        target_dy=
            CORR_TARGET_DY,

        effective_dx=
            sampling[
                "effective_dx"
            ],

        effective_dy=
            sampling[
                "effective_dy"
            ],

        minimum_valid_snapshots=
            MIN_VALID_BIG_PHASES,

        local_activity_fraction=
            LOCAL_ACTIVITY_FRAC,

        activity_percentile=
            ACTIVITY_PERCENTILE,

        activity_threshold=
            diagnostics[
                "activity_threshold"
            ],

        rho_map=
            rho,

        valid_count_map=
            valid_count,

        activity_map=
            activity
    )

    print(
        f"    保存 NPZ：{path}"
    )


def npz_matches_current_settings(

    path: str,

    case: CaseSpec

) -> bool:

    if not os.path.isfile(
        path
    ):

        return False

    try:

        with np.load(
            path,
            allow_pickle=False
        ) as data:

            checks = [

                np.isclose(
                    float(data["k"]),
                    case.k
                ),

                np.isclose(
                    float(data["St"]),
                    case.st
                ),

                np.isclose(
                    float(data["a"]),
                    case.a
                ),

                np.isclose(
                    float(data["dc"]),
                    case.dc
                ),

                np.array_equal(
                    data["cycles_1"],
                    np.asarray(
                        BIG_PERIOD_1_CYCLES
                    )
                ),

                np.array_equal(
                    data["cycles_2"],
                    np.asarray(
                        BIG_PERIOD_2_CYCLES
                    )
                ),

                int(
                    data[
                        "n_phase_per_cycle"
                    ]
                )
                ==
                N_PHASES_PER_CYCLE,

                int(
                    data[
                        "n_snapshots_big_period"
                    ]
                )
                ==
                N_SNAPSHOTS_BIG_PERIOD,

                np.isclose(
                    float(
                        data[
                            "phase_tolerance"
                        ]
                    ),
                    PHASE_TOL
                ),

                np.isclose(
                    float(
                        data[
                            "target_dx"
                        ]
                    ),
                    CORR_TARGET_DX
                ),

                np.isclose(
                    float(
                        data[
                            "target_dy"
                        ]
                    ),
                    CORR_TARGET_DY
                ),

                int(
                    data[
                        "minimum_valid_snapshots"
                    ]
                )
                ==
                MIN_VALID_BIG_PHASES,

                np.isclose(
                    float(
                        data[
                            "local_activity_fraction"
                        ]
                    ),
                    LOCAL_ACTIVITY_FRAC
                ),
            ]

            return bool(
                all(checks)
            )

    except Exception:

        return False


def load_npz(
    path: str
) -> Dict:

    with np.load(
        path,
        allow_pickle=False
    ) as data:

        return {

            "X":
                np.asarray(
                    data[
                        "x_over_c"
                    ]
                ),

            "Y":
                np.asarray(
                    data[
                        "y_over_c"
                    ]
                ),

            "rho":
                np.asarray(
                    data[
                        "rho_map"
                    ]
                ),

            "valid_count":
                np.asarray(
                    data[
                        "valid_count_map"
                    ]
                ),

            "activity":
                np.asarray(
                    data[
                        "activity_map"
                    ]
                ),

            "activity_threshold":
                float(
                    data[
                        "activity_threshold"
                    ]
                ),
        }


# ============================================================
# 22. CSV
# ============================================================

SUMMARY_FIELDS = [

    "k",
    "St",
    "a",
    "DC",

    "big_period_1",
    "big_period_2",

    "n_snapshots_each",

    "min_valid_snapshots",

    "valid_spatial_points",

    "mean_local_rho",

    "median_local_rho",

    "std_local_rho",

    "minimum_local_rho",

    "maximum_local_rho",

    "fraction_rho_below_0p9",

    "mean_valid_snapshots",

    "activity_threshold",
]


def write_summary_csv(

    path: str,

    rows: List[Dict]

):

    with open(

        path,

        "w",

        newline="",

        encoding="utf-8-sig"

    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=
                SUMMARY_FIELDS
        )

        writer.writeheader()

        for row in rows:

            output = {}

            for key in SUMMARY_FIELDS:

                value = row.get(
                    key,
                    ""
                )

                if isinstance(
                    value,
                    (
                        float,
                        np.floating
                    )
                ):

                    if np.isfinite(
                        value
                    ):

                        output[key] = (
                            f"{float(value):.8e}"
                        )

                    else:

                        output[key] = (
                            "nan"
                        )

                else:

                    output[key] = (
                        value
                    )

            writer.writerow(
                output
            )

    print(
        f"保存 CSV：{path}"
    )


# ============================================================
# 23. 单个 DC
# ============================================================

def run_one_dc(
    case: CaseSpec
) -> Dict:

    cname = case_name(
        case
    )

    case_dir = get_case_dir(
        case
    )

    npz_path = get_npz_path(
        case
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"工况：{cname}"
    )

    print(
        f"case_dir = {case_dir}"
    )

    print(
        "比较："
        "cycles 21-24 "
        "vs "
        "cycles 25-28"
    )

    # ========================================================
    # Cache
    # ========================================================

    if (
        not FORCE_RECOMPUTE
        and
        npz_matches_current_settings(
            npz_path,
            case
        )
    ):

        print(
            "    找到兼容 NPZ，"
            "直接读取缓存。"
        )

        cached = load_npz(
            npz_path
        )

        X = cached["X"]
        Y = cached["Y"]

        rho = cached["rho"]

        valid_count = (
            cached[
                "valid_count"
            ]
        )

        activity = (
            cached[
                "activity"
            ]
        )

        diagnostics = {

            "activity_threshold":
                cached[
                    "activity_threshold"
                ],

            "valid_point_count":
                int(
                    np.sum(
                        np.isfinite(
                            rho
                        )
                    )
                ),

            "total_point_count":
                int(
                    rho.size
                ),
        }

        statistics = build_statistics(

            case,

            rho,

            valid_count,

            diagnostics
        )

        return {

            "case":
                case,

            "X":
                X,

            "Y":
                Y,

            "rho":
                rho,

            "valid_count":
                valid_count,

            "activity":
                activity,

            "statistics":
                statistics,
        }

    # ========================================================
    # 目录
    # ========================================================

    if not os.path.isdir(
        case_dir
    ):

        raise FileNotFoundError(

            "工况目录不存在：\n"
            f"{case_dir}"
        )

    # ========================================================
    # 基本周期
    # ========================================================

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

    print(
        f"    f1 = {f1:.8f}"
    )

    print(
        f"    T  = {T:.8f}"
    )

    # ========================================================
    # 扫描流场
    # ========================================================

    all_records = (
        scan_all_records(
            case_dir,
            T
        )
    )

    if not all_records:

        raise FileNotFoundError(
            "没有找到 F*.dat 流场文件。"
        )

    print(
        f"    共找到 "
        f"{len(all_records)} "
        f"个流场文件。"
    )

    # ========================================================
    # 构造两个完整 4T 大周期
    # ========================================================

    records_1 = (
        build_big_period_records(

            all_records,

            BIG_PERIOD_1_CYCLES
        )
    )

    records_2 = (
        build_big_period_records(

            all_records,

            BIG_PERIOD_2_CYCLES
        )
    )

    print(
        f"    cycles 21-24："
        f"{len(records_1)} snapshots"
    )

    print(
        f"    cycles 25-28："
        f"{len(records_2)} snapshots"
    )

    # ========================================================
    # 加载涡量
    # ========================================================

    (
        X,
        Y,
        W1,
        W2,
        metadata

    ) = load_two_big_periods(

        records_1,
        records_2
    )

    print(
        f"    相关网格："
        f"{X.shape[1]} × "
        f"{X.shape[0]}"
    )

    print(
        f"    W1 shape = "
        f"{W1.shape}"
    )

    print(
        f"    W2 shape = "
        f"{W2.shape}"
    )

    # ========================================================
    # Pearson
    # ========================================================

    (
        rho,
        valid_count,
        activity,
        diagnostics

    ) = pointwise_big_period_pearson(

        W1,
        W2
    )

    finite_rho = rho[
        np.isfinite(
            rho
        )
    ]

    if finite_rho.size:

        print(
            f"    mean rho   = "
            f"{np.mean(finite_rho):.6f}"
        )

        print(
            f"    median rho = "
            f"{np.median(finite_rho):.6f}"
        )

        print(
            f"    std rho    = "
            f"{np.std(finite_rho):.6f}"
        )

        print(
            f"    min rho    = "
            f"{np.min(finite_rho):.6f}"
        )

    else:

        print(
            "    警告：没有有效相关系数。"
        )

    # ========================================================
    # Save NPZ
    # ========================================================

    save_npz(

        npz_path,

        case,

        X,
        Y,

        rho,
        valid_count,
        activity,

        metadata,

        diagnostics
    )

    # ========================================================
    # Statistics
    # ========================================================

    statistics = build_statistics(

        case,

        rho,

        valid_count,

        diagnostics
    )

    # 释放两个128快照流场
    del W1
    del W2

    gc.collect()

    return {

        "case":
            case,

        "X":
            X,

        "Y":
            Y,

        "rho":
            rho,

        "valid_count":
            valid_count,

        "activity":
            activity,

        "statistics":
            statistics,
    }


# ============================================================
# 24. 三个 DC 的相关图
# ============================================================

def plot_dc_comparison(

    k: float,

    st: float,

    a: float,

    results: List[Dict],

    out_dir: str

):

    # 按 DC = 0.75, 0.5, 0.25
    results = sorted(

        results,

        key=lambda x:
            -x[
                "case"
            ].dc
    )

    fig, axes = plt.subplots(

        1,
        3,

        figsize=(
            7.2,
            2.75
        ),

        dpi=300,

        sharex=True,
        sharey=True
    )

    cmap = plt.get_cmap(
        "RdBu_r"
    ).copy()

    cmap.set_bad(
        "#E6E6E6"
    )

    norm = Normalize(
        vmin=-1.0,
        vmax=1.0
    )

    last_mesh = None

    for panel_id, (
        ax,
        result
    ) in enumerate(
        zip(
            axes,
            results
        )
    ):

        case = result[
            "case"
        ]

        X = result["X"]
        Y = result["Y"]

        rho = result[
            "rho"
        ]

        last_mesh = ax.pcolormesh(

            X,
            Y,

            np.ma.masked_invalid(
                rho
            ),

            cmap=cmap,

            norm=norm,

            shading="auto",

            rasterized=True
        )

        ax.set_xlim(
            X_MIN,
            X_MAX
        )

        ax.set_ylim(
            Y_MIN,
            Y_MAX
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        ax.tick_params(
            top=True,
            right=True
        )

        ax.set_xlabel(
            r"$x/c$"
        )

        if panel_id == 0:

            ax.set_ylabel(
                r"$y/c$"
            )

        ax.set_title(

            rf"$DC={case.dc:g}$",

            pad=3.0
        )

        # ----------------------------------------------------
        # (a), (b), (c)
        # ----------------------------------------------------

        letter = chr(
            97 + panel_id
        )

        ax.text(

            -0.02,
            1.025,

            rf"$({letter})$",

            transform=
                ax.transAxes,

            ha="right",

            va="bottom",

            clip_on=False
        )

    # ========================================================
    # Layout
    # ========================================================

    fig.subplots_adjust(

        left=0.080,

        right=0.985,

        bottom=0.245,

        top=0.835,

        wspace=0.12
    )

    # ========================================================
    # Colorbar
    # ========================================================

    cbar_ax = fig.add_axes([

        0.275,
        0.090,
        0.45,
        0.032
    ])

    cbar = fig.colorbar(

        last_mesh,

        cax=cbar_ax,

        orientation=
            "horizontal",

        ticks=[
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0
        ]
    )

    cbar.set_label(
        r"$\rho_{\mathrm{big}}$"
    )

    cbar.outline.set_linewidth(
        0.6
    )

    # ========================================================
    # Save
    # ========================================================

    base = os.path.join(

        out_dir,

        (
            f"k{fmt_num_short(k)}"
            f"_St{fmt_num_short(st)}"
            f"_a{fmt_num_short(a)}"
            f"_rho_big_21-24_vs_25-28"
        )
    )

    fig.savefig(

        base + ".png",

        dpi=600,

        bbox_inches="tight",

        transparent=True
    )

    fig.savefig(

        base + ".svg",

        format="svg",

        bbox_inches="tight",

        transparent=True
    )

    fig.savefig(

        base + ".pdf",

        bbox_inches="tight",

        transparent=True
    )

    plt.close(
        fig
    )

    print(
        f"保存图像：{base}"
    )


# ============================================================
# 25. 单个 k
# ============================================================

def run_one_k(
    k: float
) -> Tuple[List[Dict], List[Dict]]:

    st = ST

    a = CASE_DICT[
        k
    ]

    out_dir = os.path.join(

        SAVE_ROOT,

        k_case_name(
            k,
            st,
            a
        )
    )

    ensure_dir(
        out_dir
    )

    results = []

    statistics_rows = []

    for dc in DC_LIST:

        case = CaseSpec(

            k=k,

            st=st,

            a=a,

            dc=dc
        )

        result = run_one_dc(
            case
        )

        results.append(
            result
        )

        statistics_rows.append(
            result[
                "statistics"
            ]
        )

        gc.collect()

    # ========================================================
    # 检查三个 DC 的网格是否一致
    # ========================================================

    X_ref = results[0][
        "X"
    ]

    Y_ref = results[0][
        "Y"
    ]

    for result in results[1:]:

        if (
            result["X"].shape
            !=
            X_ref.shape
        ):

            raise ValueError(
                "不同 DC 的 X 网格尺寸不一致。"
            )

        if (
            result["Y"].shape
            !=
            Y_ref.shape
        ):

            raise ValueError(
                "不同 DC 的 Y 网格尺寸不一致。"
            )

        if not np.allclose(
            result["X"],
            X_ref
        ):

            raise ValueError(
                "不同 DC 的 X 网格坐标不一致。"
            )

        if not np.allclose(
            result["Y"],
            Y_ref
        ):

            raise ValueError(
                "不同 DC 的 Y 网格坐标不一致。"
            )

    # ========================================================
    # 每个 k 的 CSV
    # ========================================================

    csv_path = os.path.join(

        out_dir,

        (
            f"k{fmt_num_short(k)}"
            f"_big_period_rho_summary.csv"
        )
    )

    write_summary_csv(

        csv_path,

        statistics_rows
    )

    # ========================================================
    # 三 DC 对比图
    # ========================================================

    plot_dc_comparison(

        k,

        st,

        a,

        results,

        out_dir
    )

    return (
        results,
        statistics_rows
    )


# ============================================================
# 26. Main
# ============================================================

def main():

    ensure_dir(
        SAVE_ROOT
    )

    print(
        "=" * 80
    )

    print(
        "Intermittent big-period "
        "local correlation analysis"
    )

    print(
        f"St = {ST}"
    )

    print(
        "Compare:"
    )

    print(
        f"    {BIG_PERIOD_1_CYCLES}"
    )

    print(
        "vs"
    )

    print(
        f"    {BIG_PERIOD_2_CYCLES}"
    )

    print(
        f"Snapshots per cycle = "
        f"{N_PHASES_PER_CYCLE}"
    )

    print(
        f"Snapshots per big period = "
        f"{N_SNAPSHOTS_BIG_PERIOD}"
    )

    print(
        f"Minimum valid snapshots = "
        f"{MIN_VALID_BIG_PHASES}"
    )

    print(
        f"DC = {DC_LIST}"
    )

    print(
        f"SAVE_ROOT = "
        f"{SAVE_ROOT}"
    )

    print(
        "=" * 80
    )

    all_statistics = []

    failed_cases = []

    # ========================================================
    # k loop
    # ========================================================

    for k in K_LIST:

        print(
            "\n"
            + "#" * 80
        )

        print(
            f"k = {k:g}, "
            f"St = {ST:g}, "
            f"a = {CASE_DICT[k]:g}"
        )

        print(
            "#" * 80
        )

        try:

            (
                results,
                statistics_rows
            ) = run_one_k(
                k
            )

            all_statistics.extend(
                statistics_rows
            )

        except Exception as e:

            failed_cases.append({

                "k":
                    k,

                "St":
                    ST,

                "a":
                    CASE_DICT[k],

                "error_type":
                    type(e).__name__,

                "error_message":
                    str(e),
            })

            print(
                "\n[失败]"
            )

            print(
                f"k = {k:g}"
            )

            print(
                f"错误：{repr(e)}"
            )

            if STOP_ON_ERROR:

                raise

        finally:

            gc.collect()

    # ========================================================
    # 全部工况汇总 CSV
    # ========================================================

    summary_path = os.path.join(

        SAVE_ROOT,

        "all_big_period_rho_summary.csv"
    )

    write_summary_csv(

        summary_path,

        all_statistics
    )

    # ========================================================
    # Failure log
    # ========================================================

    failed_path = os.path.join(

        SAVE_ROOT,

        "failed_cases.csv"
    )

    with open(

        failed_path,

        "w",

        newline="",

        encoding="utf-8-sig"

    ) as f:

        fieldnames = [

            "k",
            "St",
            "a",

            "error_type",
            "error_message"
        ]

        writer = csv.DictWriter(

            f,

            fieldnames=
                fieldnames
        )

        writer.writeheader()

        for row in failed_cases:

            writer.writerow(
                row
            )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "全部计算完成"
    )

    print(
        f"成功结果数量："
        f"{len(all_statistics)}"
    )

    print(
        f"失败 k 工况数量："
        f"{len(failed_cases)}"
    )

    print(
        f"汇总 CSV："
        f"{summary_path}"
    )

    print(
        f"失败记录："
        f"{failed_path}"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()