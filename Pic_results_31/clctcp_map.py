import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase


# ============================================================
# Configuration
# ============================================================
class Config:
    ST_LIST = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    K_LIST = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0]

    DATA_DIR = Path(
        r"E:\Rensq\DC_explore_F\results_a0\cycle_averaged_results"
    )
    OUTPUT_DIR = Path(
        r"E:\Rensq\DC_explore_F\results_a0\heatmap\new"
    )

    # System-total columns
    CL_COL = "cl_total_mean"
    CT_COL = "ct_total_mean"
    CP_COL = "cp_total_mean"

    # --------------------------------------------------------
    # Periods used for all four panels.
    #
    # One period:
    #     1.4: [12]
    #
    # Several periods:
    #     1.4: [12, 13, 14, 15, 16]
    #
    # Final period(s):
    #     1.4: "last"
    #     1.4: "last_two"
    #     1.4: "last_three"
    # --------------------------------------------------------
    MEAN_PERIODS_BY_K = {
        0.2: [11, 12, 13, 14, 15],
        0.5: [11, 12, 13, 14, 15],
        0.8: [11, 12, 13, 14, 15],
        1.1: [11, 12, 13, 14, 15],
        1.4: [16, 17, 18, 19, 20],
        1.7: [16, 17, 18, 19, 20],
        2.0: [16, 17, 18, 19, 20],
    }

    # Case-specific period settings have higher priority than
    # MEAN_PERIODS_BY_K. Thus, only k=0.8, St=0.6 is changed
    # from periods 11-15 to periods 16-20.
    MEAN_PERIODS_BY_CASE = {
        (0.8, 0.6): [16, 17, 18, 19, 20],
    }

    DEFAULT_MEAN_PERIODS = "last"

    # --------------------------------------------------------
    # Five-cycle repeatability criterion
    # --------------------------------------------------------
    # For each coefficient q = CL, CT and CP, calculate the maximum
    # difference among the five selected cycle-averaged values:
    #
    #   Delta_q = max(q_n) - min(q_n).
    #
    # Each coefficient is assessed independently:
    #   Delta_CL > 0.05 -> hatch only panel (a)
    #   Delta_CT > 0.05 -> hatch only panel (b)
    #   Delta_CP > 0.10 -> hatch only panel (c)
    #
    # The efficiency panel (d) is never hatched.
    PERIODICITY_REQUIRED_CYCLES = 5

    NONPERIODIC_MAX_DIFFERENCE_THRESHOLDS = {
        CL_COL: 0.1,
        CT_COL: 0.1,
        CP_COL: 0.10,
    }

    # These reference floors are retained only for reporting the
    # normalised sample standard deviation in the CSV. They do not
    # participate in the nonperiodic decision.
    PERIODICITY_REFERENCE_FLOORS = {
        CL_COL: 1.0,
        CT_COL: 0.10,
        CP_COL: 0.10,
    }

    # Explicit diagonal line settings. Ordinary SVG paths are used
    # instead of Matplotlib hatch patterns so that the markings remain
    # visible and editable after opening the SVG in Adobe Illustrator.
    NONPERIODIC_LINE_COLOUR = "0.15"
    NONPERIODIC_LINEWIDTH = 0.60
    NONPERIODIC_LINE_SPACING = 0.18
    SHOW_NONPERIODIC_LEGEND = True

    # True:
    # Any missing requested period makes the corresponding case NaN.
    #
    # False:
    # Use the requested periods that are actually available.
    STRICT_PERIOD_SELECTION = True

    # eta_T = mean(CT) / mean(CP)
    #
    # A denominator with abs(CP) <= ETA_CP_EPS is treated as invalid.
    ETA_CP_EPS = 1.0e-12

    # False:
    # Plot the direct ratio CT/CP, including negative values in drag regimes.
    #
    # True:
    # Only retain cases satisfying CT > 0 and CP > 0.
    ETA_MASK_NON_PROPULSIVE = False

    USE_TEX = True
    FONT_SIZE = 10
    VALUE_FONT_SIZE = 10

    # Overall figure size in inches: (width, height)
    FIG_SIZE = (6.5, 6.8)

    DPI = 300
    OUTPUT_FORMATS = ("png", "svg")
    INVERT_ST_AXIS = True

    PANELS = {
        "cl_mean": {
            "title": r"$\overline{C}_{LS}$",
            "cmap": "RdBu_r",
            "vmin": -1.0,
            "vmax": 1.0,
            "vcenter": 0.0,
            "fmt": "{:.2f}",
            "cbar_ticks": [-1.0, 0.0, 1.0],
        },
        "ct_mean": {
            "title": r"$\overline{C}_{TS}$",
            "cmap": "RdBu_r",
            "vmin": -1.0,
            "vmax": 7.0,
            "vcenter": 0.0,
            "fmt": "{:.2f}",
            "cbar_ticks": None,
        },
        "cp_mean": {
            "title": r"$\overline{C}_{PS}$",
            "cmap": "YlOrRd",
            "vmin": 0.0,
            "vmax": 30.0,
            "vcenter": None,
            "fmt": "{:.2f}",
            "cbar_ticks": None,
        },
        "eta_t": {
            "title": r"$\eta_{TS}$",
            "cmap": "RdYlGn",
            # None means that the limits are determined from valid data.
            # For a fixed range, for example, use:
            # "vmin": -1.3,
            # "vmax": 0.4,
            "vmin": -0.5,
            "vmax": 0.5,
            "vcenter": 0.0,
            "fmt": "{:.2f}",
            "cbar_ticks": [-0.5, 0.0, 0.5],
        },
    }


def configure_matplotlib():
    plt.rcParams.update({
        "font.size": Config.FONT_SIZE,
        "axes.titlesize": Config.FONT_SIZE,
        "axes.labelsize": Config.FONT_SIZE,
        "xtick.labelsize": Config.FONT_SIZE,
        "ytick.labelsize": Config.FONT_SIZE,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "text.usetex": Config.USE_TEX,
    })

    if Config.USE_TEX:
        plt.rcParams["pgf.texsystem"] = "pdflatex"
        plt.rcParams["text.latex.preamble"] = (
            r"\usepackage{amsmath}\usepackage{newtxtext,newtxmath}"
        )


# ============================================================
# File location and robust CSV reading
# ============================================================
def number_tokens(value):
    value = float(value)
    return list(dict.fromkeys([
        f"{value:g}",
        f"{value:.1f}",
        str(value),
    ]))


def find_case_file(k, st):
    """Recognise both k=2st=0.5.csv and k=2.0st=0.5.csv."""
    for k_text in number_tokens(k):
        for st_text in number_tokens(st):
            path = Config.DATA_DIR / f"k={k_text}st={st_text}a=0dc=100.csv"
            if path.exists():
                return path

    pattern = re.compile(
        r"^k=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"st=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\.csv$",
        re.IGNORECASE,
    )

    for path in Config.DATA_DIR.glob("*.csv"):
        match = pattern.match(path.name.replace(" ", ""))
        if (
            match
            and np.isclose(float(match.group(1)), k)
            and np.isclose(float(match.group(2)), st)
        ):
            return path

    return None


def normalize_column_name(column):
    """Remove BOM, quotes, spaces and inconsistent separators."""
    name = str(column)
    name = name.replace("\ufeff", "").replace("ï»¿", "")
    name = name.strip().strip('"').strip("'").strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)
    return name


def read_case(path):
    """
    Read comma-, tab-, or whitespace-delimited cycle-average files.

    The function also handles UTF-8 BOM characters in the first header.
    """
    required = {
        "period",
        Config.CL_COL,
        Config.CT_COL,
        Config.CP_COL,
    }

    encodings = ("utf-8-sig", "utf-8", "gb18030", "latin1")
    separators = (
        (",", "c"),
        ("\t", "c"),
        (r"\s+", "python"),
        (None, "python"),
    )

    candidates = []
    errors = []

    for encoding in encodings:
        for separator, engine in separators:
            try:
                df_try = pd.read_csv(
                    filepath_or_buffer=path,
                    sep=separator,
                    engine=engine,
                    encoding=encoding,
                    skipinitialspace=True,
                )

                df_try.columns = [
                    normalize_column_name(column)
                    for column in df_try.columns
                ]

                # Remove fully empty columns caused by trailing delimiters.
                df_try = df_try.dropna(axis=1, how="all")

                score = len(required.intersection(df_try.columns))
                candidates.append(
                    (
                        score,
                        len(df_try.columns),
                        encoding,
                        separator,
                        df_try,
                    )
                )

            except Exception as error:
                errors.append(
                    f"encoding={encoding}, separator={separator!r}: {error}"
                )

    if not candidates:
        raise ValueError(
            "unable to parse file; "
            + " | ".join(errors[:4])
        )

    # Prefer the candidate recognising the largest number of required columns.
    candidates.sort(
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )

    score, column_count, encoding, separator, df = candidates[0]

    missing = required.difference(df.columns)
    if missing:
        detected = ", ".join(repr(column) for column in df.columns)
        raise ValueError(
            f"missing columns: {sorted(missing)}; "
            f"detected columns ({column_count}) using "
            f"encoding={encoding}, separator={separator!r}: [{detected}]"
        )

    numeric_columns = [
        "period",
        Config.CL_COL,
        Config.CT_COL,
        Config.CP_COL,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric_columns).copy()

    if df.empty:
        raise ValueError(
            "required columns were found, but no valid numerical rows remain"
        )

    df["period"] = np.rint(df["period"]).astype(int)

    # Collapse duplicate period rows, if present.
    df = (
        df.groupby("period", as_index=False)[
            [Config.CL_COL, Config.CT_COL, Config.CP_COL]
        ]
        .mean()
        .sort_values("period")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# Period selection and metric calculation
# ============================================================
def resolve_periods(df, specification, k, st):
    available = sorted(df["period"].unique().tolist())

    if isinstance(specification, str):
        final_period_count = {
            "last": 1,
            "last_two": 2,
            "last_three": 3,
            "last_four": 4,
            "last_five": 5,
        }.get(specification.lower())

        if final_period_count is None:
            raise ValueError(
                f"unsupported period selector: {specification}"
            )

        if len(available) < final_period_count:
            warnings.warn(
                f"k={k:g}, St={st:g}: only {len(available)} periods "
                "are available"
            )
            return (
                []
                if Config.STRICT_PERIOD_SELECTION
                else available
            )

        return available[-final_period_count:]

    if isinstance(specification, (int, np.integer)):
        requested = [int(specification)]
    else:
        requested = sorted(
            set(int(value) for value in specification)
        )

    missing = [
        value for value in requested
        if value not in available
    ]
    selected = [
        value for value in requested
        if value in available
    ]

    if missing:
        warnings.warn(
            f"k={k:g}, St={st:g}: missing periods {missing}; "
            f"available periods are {available}"
        )

        if Config.STRICT_PERIOD_SELECTION:
            return []

    return selected


def rows_for_periods(df, periods):
    if not periods:
        return df.iloc[0:0].copy()

    return (
        df[df["period"].isin(periods)]
        .sort_values("period")
        .reset_index(drop=True)
    )


def mean_or_nan(df, column):
    if df.empty:
        return np.nan
    return float(df[column].mean())


def calculate_eta_t(ct_mean, cp_mean):
    """
    Calculate the thrust efficiency

        eta_T = mean(CT) / mean(CP).

    The ratio is formed after averaging CT and CP over the selected cycles.
    """
    if not np.isfinite(ct_mean) or not np.isfinite(cp_mean):
        return np.nan

    if abs(cp_mean) <= Config.ETA_CP_EPS:
        return np.nan

    if (
        Config.ETA_MASK_NON_PROPULSIVE
        and (ct_mean <= 0.0 or cp_mean <= 0.0)
    ):
        return np.nan

    return float(ct_mean / cp_mean)


def get_period_specification(k, st):
    """
    Return the period specification for one parameter combination.

    Priority:
        1. exact (k, St) case override;
        2. setting based on k;
        3. default setting.
    """
    for (case_k, case_st), specification in (
        Config.MEAN_PERIODS_BY_CASE.items()
    ):
        if (
            np.isclose(k, case_k)
            and np.isclose(st, case_st)
        ):
            return specification

    for configured_k, specification in (
        Config.MEAN_PERIODS_BY_K.items()
    ):
        if np.isclose(k, configured_k):
            return specification

    return Config.DEFAULT_MEAN_PERIODS


def cycle_repeatability_statistics(values, reference_floor):
    """
    Calculate the maximum cycle difference together with supplementary
    sample-standard-deviation statistics.

    For N selected cycle-averaged values q_n,

        sample_std =
            sqrt(sum((q_n - mean(q_n))**2) / (N - 1))

        max_difference = max(q_n) - min(q_n)

        normalized_std =
            sample_std / max(abs(mean(q_n)), reference_floor)

    The calculation uses ddof=1, so the standard deviation is the
    sample standard deviation rather than the population standard
    deviation.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 2:
        return {
            "cycle_mean": np.nan,
            "sample_std": np.nan,
            "max_difference": np.nan,
            "reference": np.nan,
            "normalized_std": np.nan,
        }

    cycle_mean = float(np.mean(values))
    sample_std = float(np.std(values, ddof=1))
    max_difference = float(np.max(values) - np.min(values))
    reference = max(
        abs(cycle_mean),
        float(reference_floor),
    )
    normalized_std = float(sample_std / reference)

    return {
        "cycle_mean": cycle_mean,
        "sample_std": sample_std,
        "max_difference": max_difference,
        "reference": reference,
        "normalized_std": normalized_std,
    }


def assess_five_cycle_periodicity(selected_rows):
    """
    Assess CL, CT and CP repeatability independently over five cycles.

    For each coefficient q, calculate the maximum difference

        Delta_q = max(q_n) - min(q_n).

    The three Boolean flags are independent and use coefficient-specific
    thresholds:
        Delta_CL > 0.05 -> hatch only panel (a),
        Delta_CT > 0.05 -> hatch only panel (b),
        Delta_CP > 0.10 -> hatch only panel (c).

    The sample standard deviation and normalised sample standard deviation
    are retained as supplementary CSV statistics, but they do not determine
    the hatching. eta_T is not assessed and panel (d) is never hatched.
    """
    result = {
        "periodicity_valid": False,
        "cl_cycle_sample_std": np.nan,
        "ct_cycle_sample_std": np.nan,
        "cp_cycle_sample_std": np.nan,
        "cl_cycle_max_difference": np.nan,
        "ct_cycle_max_difference": np.nan,
        "cp_cycle_max_difference": np.nan,
        "cl_cycle_normalized_std": np.nan,
        "ct_cycle_normalized_std": np.nan,
        "cp_cycle_normalized_std": np.nan,
        "cl_cycle_reference": np.nan,
        "ct_cycle_reference": np.nan,
        "cp_cycle_reference": np.nan,
        "cl_nonperiodic": False,
        "ct_nonperiodic": False,
        "cp_nonperiodic": False,
    }

    if len(selected_rows) != Config.PERIODICITY_REQUIRED_CYCLES:
        return result

    metric_map = {
        "cl": Config.CL_COL,
        "ct": Config.CT_COL,
        "cp": Config.CP_COL,
    }

    max_differences = {}

    for short_name, column in metric_map.items():
        statistics = cycle_repeatability_statistics(
            selected_rows[column].to_numpy(),
            Config.PERIODICITY_REFERENCE_FLOORS[column],
        )

        result[f"{short_name}_cycle_sample_std"] = (
            statistics["sample_std"]
        )
        result[f"{short_name}_cycle_max_difference"] = (
            statistics["max_difference"]
        )
        result[f"{short_name}_cycle_normalized_std"] = (
            statistics["normalized_std"]
        )
        result[f"{short_name}_cycle_reference"] = (
            statistics["reference"]
        )
        max_differences[short_name] = statistics["max_difference"]

    if not all(
        np.isfinite(value)
        for value in max_differences.values()
    ):
        return result

    result["periodicity_valid"] = True

    for short_name, column in metric_map.items():
        result[f"{short_name}_nonperiodic"] = bool(
            max_differences[short_name]
            > Config.NONPERIODIC_MAX_DIFFERENCE_THRESHOLDS[column]
        )

    return result


def period_text(periods):
    return ",".join(str(value) for value in periods)


def empty_record(k, st, source_file=""):
    return {
        "k": k,
        "st": st,
        "source_file": source_file,
        "mean_periods": "",
        "selected_period_count": 0,
        "cl_mean": np.nan,
        "ct_mean": np.nan,
        "cp_mean": np.nan,
        "eta_t": np.nan,
        "periodicity_valid": False,
        "cl_cycle_sample_std": np.nan,
        "ct_cycle_sample_std": np.nan,
        "cp_cycle_sample_std": np.nan,
        "cl_cycle_max_difference": np.nan,
        "ct_cycle_max_difference": np.nan,
        "cp_cycle_max_difference": np.nan,
        "cl_cycle_normalized_std": np.nan,
        "ct_cycle_normalized_std": np.nan,
        "cp_cycle_normalized_std": np.nan,
        "cl_cycle_reference": np.nan,
        "ct_cycle_reference": np.nan,
        "cp_cycle_reference": np.nan,
        "cl_nonperiodic": False,
        "ct_nonperiodic": False,
        "cp_nonperiodic": False,
    }


def process_all_cases():
    records = []
    stats = {
        "success": 0,
        "missing": 0,
        "error": 0,
    }

    for st in Config.ST_LIST:
        for k in Config.K_LIST:
            path = find_case_file(k, st)

            if path is None:
                print(f"[Missing] k={k:g}, St={st:g}")
                records.append(empty_record(k, st))
                stats["missing"] += 1
                continue

            try:
                df = read_case(path)

                period_specification = get_period_specification(
                    k,
                    st,
                )

                selected_periods = resolve_periods(
                    df,
                    period_specification,
                    k,
                    st,
                )

                selected_rows = rows_for_periods(
                    df,
                    selected_periods,
                )

                cl_mean = mean_or_nan(
                    selected_rows,
                    Config.CL_COL,
                )
                ct_mean = mean_or_nan(
                    selected_rows,
                    Config.CT_COL,
                )
                cp_mean = mean_or_nan(
                    selected_rows,
                    Config.CP_COL,
                )
                eta_t = calculate_eta_t(
                    ct_mean,
                    cp_mean,
                )

                periodicity = assess_five_cycle_periodicity(
                    selected_rows,
                )

                record = {
                    "k": k,
                    "st": st,
                    "source_file": str(path),
                    "mean_periods": period_text(selected_periods),
                    "selected_period_count": len(selected_periods),
                    "cl_mean": cl_mean,
                    "ct_mean": ct_mean,
                    "cp_mean": cp_mean,
                    "eta_t": eta_t,
                }
                record.update(periodicity)
                records.append(record)

                eta_text = (
                    f"{eta_t:.6f}"
                    if np.isfinite(eta_t)
                    else "NaN"
                )
                if periodicity["periodicity_valid"]:
                    cl_state = (
                        "aperiodic"
                        if periodicity["cl_nonperiodic"]
                        else "periodic"
                    )
                    ct_state = (
                        "aperiodic"
                        if periodicity["ct_nonperiodic"]
                        else "periodic"
                    )
                    cp_state = (
                        "aperiodic"
                        if periodicity["cp_nonperiodic"]
                        else "periodic"
                    )
                    periodicity_text = (
                        f"CL={cl_state} "
                        f"(Delta={periodicity['cl_cycle_max_difference']:.4f}); "
                        f"CT={ct_state} "
                        f"(Delta={periodicity['ct_cycle_max_difference']:.4f}); "
                        f"CP={cp_state} "
                        f"(Delta={periodicity['cp_cycle_max_difference']:.4f})"
                    )
                else:
                    periodicity_text = "periodicity not assessed"

                print(
                    f"[OK] k={k:g}, St={st:g}; "
                    f"periods={period_text(selected_periods)}; "
                    f"eta_T={eta_text}; "
                    f"{periodicity_text}"
                )

                stats["success"] += 1

            except Exception as error:
                print(f"[Error] {path.name}: {error}")
                records.append(
                    empty_record(k, st, str(path))
                )
                stats["error"] += 1

    return pd.DataFrame(records), stats


# ============================================================
# Matrix construction and plotting
# ============================================================
def make_matrix(result_df, column):
    matrix = np.full(
        (len(Config.ST_LIST), len(Config.K_LIST)),
        np.nan,
    )

    for i, st in enumerate(Config.ST_LIST):
        for j, k in enumerate(Config.K_LIST):
            mask = (
                np.isclose(result_df["st"], st)
                & np.isclose(result_df["k"], k)
            )

            if mask.any():
                matrix[i, j] = pd.to_numeric(
                    result_df.loc[mask, column].iloc[0],
                    errors="coerce",
                )

    return matrix


def make_flag_matrix(result_df, column):
    """Build a Boolean k-St matrix for flags such as nonperiodic."""
    matrix = np.zeros(
        (len(Config.ST_LIST), len(Config.K_LIST)),
        dtype=bool,
    )

    for i, st in enumerate(Config.ST_LIST):
        for j, k in enumerate(Config.K_LIST):
            mask = (
                np.isclose(result_df["st"], st)
                & np.isclose(result_df["k"], k)
            )

            if not mask.any():
                continue

            value = result_df.loc[mask, column].iloc[0]

            if pd.isna(value):
                continue

            if isinstance(value, str):
                matrix[i, j] = (
                    value.strip().lower()
                    in {"true", "1", "yes", "y"}
                )
            else:
                matrix[i, j] = bool(value)

    return matrix


def make_norm(matrix, setting):
    valid = matrix[np.isfinite(matrix)]

    vmin = setting["vmin"]
    vmax = setting["vmax"]

    if vmin is None:
        vmin = float(valid.min()) if valid.size else 0.0

    if vmax is None:
        vmax = float(valid.max()) if valid.size else 1.0

    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0e-6

    center = setting["vcenter"]

    if (
        center is not None
        and vmin < center < vmax
    ):
        return TwoSlopeNorm(
            vmin=vmin,
            vcenter=center,
            vmax=vmax,
        )

    return Normalize(
        vmin=vmin,
        vmax=vmax,
    )



class DiagonalLineLegendHandle:
    """Dummy handle used by the custom diagonal-line legend handler."""


class HandlerDiagonalLines(HandlerBase):
    """
    Draw several ordinary diagonal Line2D paths inside the legend box.

    This avoids SVG <pattern> elements and therefore improves compatibility
    with Adobe Illustrator.
    """

    def create_artists(
        self,
        legend,
        original_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        transform,
    ):
        artists = []

        # Use a compact set of visually parallel '/' lines.
        for fraction in (-0.35, 0.0, 0.35, 0.70):
            x0 = xdescent + max(0.0, fraction) * width
            y0 = ydescent + max(0.0, -fraction) * height
            x1 = xdescent + min(1.0, 1.0 + fraction) * width
            y1 = ydescent + min(1.0, 1.0 - fraction) * height

            if x1 <= x0 or y1 <= y0:
                continue

            artists.append(
                Line2D(
                    [x0, x1],
                    [y0, y1],
                    color=Config.NONPERIODIC_LINE_COLOUR,
                    linewidth=Config.NONPERIODIC_LINEWIDTH,
                    solid_capstyle="butt",
                    transform=transform,
                )
            )

        return artists


def diagonal_segments_for_cell(cell_x, cell_y, spacing):
    """
    Return explicit line segments for one 1 x 1 heatmap cell.

    The line orientation is selected so that the displayed result resembles
    '/' even when the St axis is inverted. Every segment terminates exactly
    on a cell boundary, so no SVG clipping pattern is required.
    """
    if spacing <= 0.0:
        raise ValueError("NONPERIODIC_LINE_SPACING must be positive")

    segments = []
    tolerance = 1.0e-12

    if Config.INVERT_ST_AXIS:
        # With an inverted y-axis, y = -x + intercept appears as '/'.
        intercepts = np.arange(0.0, 2.0 + 0.5 * spacing, spacing)

        for intercept in intercepts:
            candidates = []

            # Intersections with x = 0 and x = 1.
            for x_local in (0.0, 1.0):
                y_local = -x_local + intercept
                if -tolerance <= y_local <= 1.0 + tolerance:
                    candidates.append((x_local, np.clip(y_local, 0.0, 1.0)))

            # Intersections with y = 0 and y = 1.
            for y_local in (0.0, 1.0):
                x_local = intercept - y_local
                if -tolerance <= x_local <= 1.0 + tolerance:
                    candidates.append((np.clip(x_local, 0.0, 1.0), y_local))

            unique_points = []
            for point in candidates:
                if not any(
                    np.allclose(point, existing, atol=tolerance, rtol=0.0)
                    for existing in unique_points
                ):
                    unique_points.append(point)

            if len(unique_points) >= 2:
                point_1, point_2 = unique_points[:2]
                if not np.allclose(point_1, point_2, atol=tolerance, rtol=0.0):
                    segments.append([
                        (cell_x + point_1[0], cell_y + point_1[1]),
                        (cell_x + point_2[0], cell_y + point_2[1]),
                    ])

    else:
        # Without y-axis inversion, y = x + offset appears as '/'.
        offsets = np.arange(-1.0, 1.0 + 0.5 * spacing, spacing)

        for offset in offsets:
            if offset < 0.0:
                x_start = -offset
                y_start = 0.0
                x_end = 1.0
                y_end = 1.0 + offset
            else:
                x_start = 0.0
                y_start = offset
                x_end = 1.0 - offset
                y_end = 1.0

            if x_end - x_start <= tolerance and y_end - y_start <= tolerance:
                continue

            segments.append([
                (cell_x + x_start, cell_y + y_start),
                (cell_x + x_end, cell_y + y_end),
            ])

    return segments


def draw_nonperiodic_diagonal_lines(ax, cell_x, cell_y):
    """
    Mark one non-repeatable cell with explicit diagonal vector lines.

    LineCollection writes ordinary SVG path elements rather than hatch
    patterns. This is more reliable in Adobe Illustrator and keeps each
    marking as vector geometry.
    """
    segments = diagonal_segments_for_cell(
        cell_x=cell_x,
        cell_y=cell_y,
        spacing=Config.NONPERIODIC_LINE_SPACING,
    )

    if not segments:
        return

    collection = LineCollection(
        segments,
        colors=Config.NONPERIODIC_LINE_COLOUR,
        linewidths=Config.NONPERIODIC_LINEWIDTH,
        linestyles="solid",
        capstyle="butt",
        zorder=2,
    )
    collection.set_clip_on(True)
    collection.set_clip_path(ax.patch)
    ax.add_collection(collection)


def annotation_colour(image, value):
    red, green, blue = image.cmap(
        image.norm(value)
    )[:3]

    luminance = (
        0.2126 * red
        + 0.7152 * green
        + 0.0722 * blue
    )

    return "black" if luminance > 0.55 else "white"


def draw_panel(ax, matrix, nonperiodic_matrix, setting, letter):
    image = ax.pcolormesh(
        np.arange(len(Config.K_LIST) + 1),
        np.arange(len(Config.ST_LIST) + 1),
        matrix,
        cmap=setting["cmap"],
        norm=make_norm(matrix, setting),
        shading="flat",
        edgecolors="white",
        linewidth=0.5,
    )

    # Draw explicit vector lines before numerical annotations so that values
    # remain legible. These are ordinary SVG paths, not hatch patterns.
    for i in range(len(Config.ST_LIST)):
        for j in range(len(Config.K_LIST)):
            if not nonperiodic_matrix[i, j]:
                continue

            draw_nonperiodic_diagonal_lines(
                ax=ax,
                cell_x=j,
                cell_y=i,
            )

    for i in range(len(Config.ST_LIST)):
        for j in range(len(Config.K_LIST)):
            value = matrix[i, j]

            if np.isfinite(value):
                text = setting["fmt"].format(value)
                colour = annotation_colour(image, value)
            else:
                text = "--"
                colour = "0.45"

            ax.text(
                j + 0.5,
                i + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=Config.VALUE_FONT_SIZE,
                color=colour,
                zorder=3,
            )

    ax.set_xticks(
        np.arange(len(Config.K_LIST)) + 0.5
    )
    ax.set_yticks(
        np.arange(len(Config.ST_LIST)) + 0.5
    )

    ax.set_xticklabels([
        f"{k:g}" for k in Config.K_LIST
    ])
    ax.set_yticklabels([
        f"{st:g}" for st in Config.ST_LIST
    ])

    ax.set_xlim(0, len(Config.K_LIST))
    ax.set_ylim(0, len(Config.ST_LIST))
    ax.set_aspect("equal")

    if Config.INVERT_ST_AXIS:
        ax.invert_yaxis()

    # Keep the k and St values, but hide all x- and y-axis tick marks.
    ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=True,
        length=0,
    )
    ax.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=True,
        length=0,
    )

    ax.set_xlabel(r"$k$", labelpad=3)
    ax.set_ylabel(
        r"$St$",
        rotation=0,
        labelpad=14,
        ha="right",
        va="center",
    )

    ax.set_title(
        setting["title"],
        pad=5,
    )

    ax.text(
        -0.02,
        1.04,
        rf"($\mathit{{{letter}}}$)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontweight="normal",
        clip_on=False,
    )

    colour_bar = ax.figure.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.025,
    )

    # Only numerical values are displayed beside the colour bar.
    cbar_ticks = setting.get("cbar_ticks")

    if cbar_ticks is not None:
        colour_bar.set_ticks(cbar_ticks)

    colour_bar.ax.tick_params(
        direction="in",
        labelsize=Config.FONT_SIZE - 1,
    )


def plot_figure(result_df):
    # The third item is the panel-specific periodicity flag.
    # None means that no hatching is drawn for that panel.
    panel_order = [
        ("cl_mean", "cl_nonperiodic", "a"),
        ("ct_mean", "ct_nonperiodic", "b"),
        ("cp_mean", "cp_nonperiodic", "c"),
        ("eta_t", None, "d"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=Config.FIG_SIZE,
        dpi=Config.DPI,
    )

    empty_flag_matrix = np.zeros(
        (len(Config.ST_LIST), len(Config.K_LIST)),
        dtype=bool,
    )

    for ax, (column, flag_column, letter) in zip(
        axes.flat,
        panel_order,
    ):
        if flag_column is None:
            # eta_T is a derived ratio and is not marked as aperiodic.
            nonperiodic_matrix = empty_flag_matrix
        else:
            nonperiodic_matrix = make_flag_matrix(
                result_df,
                flag_column,
            )

        draw_panel(
            ax,
            make_matrix(result_df, column),
            nonperiodic_matrix,
            Config.PANELS[column],
            letter,
        )

    if Config.SHOW_NONPERIODIC_LEGEND:
        legend_handle = DiagonalLineLegendHandle()
        fig.legend(
            handles=[legend_handle],
            labels=["Non-repeatable cycle-averaged coefficient"],
            handler_map={
                DiagonalLineLegendHandle: HandlerDiagonalLines(),
            },
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            frameon=False,
            handlelength=1.8,
            handleheight=1.0,
        )

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.13,
        top=0.95,
        wspace=0.3,
        hspace=0.1,
    )

    Config.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for suffix in Config.OUTPUT_FORMATS:
        path = (
            Config.OUTPUT_DIR
            / f"force_power_etaT_heatmaps.{suffix}"
        )

        save_kwargs = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }

        if suffix.lower() in {
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff",
        }:
            save_kwargs["dpi"] = Config.DPI

        fig.savefig(
            path,
            **save_kwargs,
        )
        print(f"Saved: {path}")

    plt.close(fig)


def main():
    configure_matplotlib()

    if not Config.DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found:\n{Config.DATA_DIR}"
        )

    Config.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df, stats = process_all_cases()

    if stats["success"] == 0:
        raise RuntimeError(
            "No valid cases were read. "
            "No blank summary or heatmap was saved."
        )

    summary_path = (
        Config.OUTPUT_DIR
        / "force_power_etaT_summary.csv"
    )

    result_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.8e",
    )

    print(f"Saved: {summary_path}")

    plot_figure(result_df)

    print(
        "\nCompleted: "
        f"{stats['success']} successful, "
        f"{stats['missing']} missing, "
        f"{stats['error']} errors."
    )


if __name__ == "__main__":
    main()