"""
Create one detector-response map per selected FERS signal channel.

Main behavior
-------------

1. Detect FERS channels that contain signal.
2. Ask which channel to analyze, or "all".
3. Use the same synchronization logic as Plot 04:
       startup counts
       -> sustained zero-count waiting region
       -> first non-zero measurement
       -> real scan start = t = 0
4. Align that FERS time with the first configured stationary
   motor scan point.
5. Match FERS Rate_Hz measurements to each stationary motor point.
6. Calculate the mean counting rate at each position.
7. Automatically choose a useful color scale for each channel/run.

Channels are always analyzed separately.
They are never summed or combined.
"""

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm


# ============================================================
# Shared utilities
# ============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import analysis_utils as au


# ============================================================
# Analysis settings
# ============================================================

POSITION_TOLERANCE_CM = 0.001

MIN_STOP_DURATION_S = 1.0

POINT_SIZE = 50

PLOT_PADDING_CM = 0.5

MOTOR_COUNTER_MODULUS_US = float(2**32)

# Number of consecutive zero-count measurements required
# before the next non-zero Channel 0 measurement is accepted
# as the real scan start.
MIN_QUIET_EVENTS = 5

# Percentile used for the visual color scale.
#
# This prevents one or two extreme values from destroying
# the contrast of the whole detector map.
COLOR_PERCENTILE = 98.0

# Number of color bands in the spectrum.
NUMBER_OF_COLOR_LEVELS = 24


# ============================================================
# Motor-file patterns
# ============================================================

_POSITION_PATTERN = re.compile(
    r"^\[(\d+)\]\s+"
    r"([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*$"
)


_FIRST_POINT_PATTERN = re.compile(
    r"scan first point.*?=\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)"
)


# ============================================================
# Run-number handling
# ============================================================

def normalize_run_number(
    value: str,
) -> str:

    value = value.strip()

    if value.lower().startswith("run"):
        value = value[3:]

    if not value.isdigit():
        raise ValueError(
            "Enter a run number such as 63 or Run63."
        )

    return value


def ask_run_number(
    prompt: str,
) -> str:

    while True:

        try:
            return normalize_run_number(
                input(prompt)
            )

        except ValueError as error:
            print(error)


# ============================================================
# Read motor data
# ============================================================

def read_motor_data(
    motor_file: Path,
) -> tuple[pd.DataFrame, dict]:

    if not motor_file.exists():
        raise FileNotFoundError(
            f"Motor file not found: {motor_file}"
        )

    rows = []

    first_scan_point = None

    with motor_file.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            if first_scan_point is None:

                first_match = (
                    _FIRST_POINT_PATTERN.search(
                        line
                    )
                )

                if first_match:

                    first_scan_point = {
                        "X_cm": float(
                            first_match.group(1)
                        ),
                        "Y_cm": float(
                            first_match.group(2)
                        ),
                        "Dwell_s": float(
                            first_match.group(3)
                        ),
                    }

            position_match = (
                _POSITION_PATTERN.match(
                    line.strip()
                )
            )

            if position_match:

                (
                    epoch_s,
                    timestamp_us,
                    x_cm,
                    y_cm,
                ) = position_match.groups()

                rows.append(
                    {
                        "Epoch_s": int(
                            epoch_s
                        ),
                        "Motor_Timestamp_us": float(
                            timestamp_us
                        ),
                        "X_cm": float(
                            x_cm
                        ),
                        "Y_cm": float(
                            y_cm
                        ),
                        "Source_Line": line_number,
                    }
                )

    if not rows:
        raise ValueError(
            "No motor-position rows were found."
        )

    if first_scan_point is None:
        raise ValueError(
            "Configured first scan point was not found."
        )

    return (
        pd.DataFrame(rows),
        first_scan_point,
    )


# ============================================================
# Correct motor timestamps
# ============================================================

def unwrap_motor_timestamps(
    motor_data: pd.DataFrame,
) -> pd.DataFrame:

    retained = []

    previous = None

    offset_us = 0.0

    rollovers = 0

    for row_index, row in motor_data.iterrows():

        raw = float(
            row["Motor_Timestamp_us"]
        )

        if (
            previous is not None
            and raw < previous
        ):

            is_rollover = (
                previous
                > 0.8 * MOTOR_COUNTER_MODULUS_US
                and raw
                < 0.2 * MOTOR_COUNTER_MODULUS_US
            )

            if is_rollover:

                offset_us += (
                    MOTOR_COUNTER_MODULUS_US
                )

                rollovers += 1

                print(
                    "Motor timestamp rollover "
                    f"detected at motor row {row_index}."
                )

            else:

                print(
                    "Motor timestamp reset "
                    f"detected at motor row {row_index}. "
                    "Parking data removed."
                )

                break

        item = row.to_dict()

        item[
            "Unwrapped_Time_us"
        ] = (
            raw + offset_us
        )

        retained.append(
            item
        )

        previous = raw

    if not retained:
        raise ValueError(
            "No motor rows remained after "
            "timestamp processing."
        )

    result = (
        pd.DataFrame(retained)
        .reset_index(drop=True)
    )

    first_us = float(
        result[
            "Unwrapped_Time_us"
        ].iloc[0]
    )

    result[
        "Motor_Relative_Time_s"
    ] = (
        result[
            "Unwrapped_Time_us"
        ]
        - first_us
    ) / 1_000_000.0

    print(
        "Motor timestamp rollovers corrected: "
        f"{rollovers}"
    )

    return result


# ============================================================
# Stationary motor intervals
# ============================================================

def append_stationary_interval(
    group,
    intervals,
    minimum_duration_s,
) -> None:

    if group.empty:
        return

    start = float(
        group[
            "Motor_Relative_Time_s"
        ].iloc[0]
    )

    end = float(
        group[
            "Motor_Relative_Time_s"
        ].iloc[-1]
    )

    duration = (
        end - start
    )

    if duration < minimum_duration_s:
        return

    intervals.append(
        {
            "X_cm": float(
                group["X_cm"].iloc[0]
            ),

            "Y_cm": float(
                group["Y_cm"].iloc[0]
            ),

            "Motor_Start_Time_s": start,

            "Motor_End_Time_s": end,

            "Duration_s": duration,

            "Motor_Rows": len(group),
        }
    )


def extract_stationary_intervals(
    motor_timeline: pd.DataFrame,
    minimum_duration_s: float = (
        MIN_STOP_DURATION_S
    ),
    position_tolerance_cm: float = (
        POSITION_TOLERANCE_CM
    ),
) -> pd.DataFrame:

    intervals = []

    group_start = 0

    for row_index in range(
        1,
        len(motor_timeline),
    ):

        previous = (
            motor_timeline.iloc[
                row_index - 1
            ]
        )

        current = (
            motor_timeline.iloc[
                row_index
            ]
        )

        changed = (
            abs(
                current["X_cm"]
                - previous["X_cm"]
            )
            > position_tolerance_cm

            or

            abs(
                current["Y_cm"]
                - previous["Y_cm"]
            )
            > position_tolerance_cm
        )

        if changed:

            append_stationary_interval(
                motor_timeline.iloc[
                    group_start:
                    row_index
                ],
                intervals,
                minimum_duration_s,
            )

            group_start = row_index

    append_stationary_interval(
        motor_timeline.iloc[
            group_start:
        ],
        intervals,
        minimum_duration_s,
    )

    if not intervals:
        raise ValueError(
            "No stationary motor scan "
            "points were detected."
        )

    table = pd.DataFrame(
        intervals
    )

    table.insert(
        0,
        "Interval_Number",
        range(
            1,
            len(table) + 1,
        ),
    )

    return table


# ============================================================
# Select real scan intervals
# ============================================================

def select_scan_intervals(
    interval_table: pd.DataFrame,
    first_scan_point: dict,
    tolerance: float = (
        POSITION_TOLERANCE_CM
    ),
) -> pd.DataFrame:

    matches = interval_table[
        (
            interval_table["X_cm"]
            - first_scan_point["X_cm"]
        ).abs()
        <= tolerance
    ]

    matches = matches[
        (
            matches["Y_cm"]
            - first_scan_point["Y_cm"]
        ).abs()
        <= tolerance
    ]

    if matches.empty:
        raise ValueError(
            "No stationary interval matched "
            "the configured first scan point."
        )

    first_index = int(
        matches.index[0]
    )

    result = (
        interval_table
        .loc[first_index:]
        .copy()
        .reset_index(drop=True)
    )

    result[
        "Scan_Point"
    ] = range(
        1,
        len(result) + 1,
    )

    return result


# ============================================================
# FERS synchronization
# ============================================================

def find_real_fers_scan_start(
    fers_file: Path,
    active_channels: list[int],
    min_quiet_events: int = (
        MIN_QUIET_EVENTS
    ),
) -> dict:
    """
    Find the real FERS scan start.

    Channel 0 is preferred.

    Logic:

        startup non-zero measurements
                ->
        sustained zero-count waiting region
                ->
        first non-zero measurement
                ->
        real scan start
    """

    raw_data = (
        au.read_fers_raw(
            fers_file
        )
        .copy()
    )

    if 0 in active_channels:

        reference_channel = 0

    else:

        reference_channel = (
            active_channels[0]
        )

        print(
            "Channel 0 is not active. "
            "Using Channel "
            f"{reference_channel} "
            "for synchronization."
        )

    reference_data = (
        raw_data[
            raw_data["CH_Id"]
            == reference_channel
        ]
        .sort_values(
            "TStamp_us"
        )
        .reset_index(drop=True)
        .copy()
    )

    if reference_data.empty:
        raise ValueError(
            "No synchronization reference "
            "data were found."
        )

    quiet_run = 0

    quiet_region_seen = False

    synchronization_index = None

    for index, row in (
        reference_data.iterrows()
    ):

        counts = float(
            row["Counts"]
        )

        if counts == 0:

            quiet_run += 1

            if (
                quiet_run
                >= min_quiet_events
            ):
                quiet_region_seen = True

            continue

        if quiet_region_seen:

            synchronization_index = index

            break

        quiet_run = 0

    if synchronization_index is None:
        raise ValueError(
            "Could not determine the real "
            "FERS scan start."
        )

    sync_row = (
        reference_data.iloc[
            synchronization_index
        ]
    )

    fers_sync_us = float(
        sync_row[
            "TStamp_us"
        ]
    )

    sync_counts = float(
        sync_row[
            "Counts"
        ]
    )

    print()
    print(
        "FERS scan-start detection"
    )

    print(
        "-------------------------"
    )

    print(
        "Reference channel: "
        f"{reference_channel}"
    )

    print(
        "Method:"
    )

    print(
        "startup counts -> zero-count "
        "waiting region -> first "
        "non-zero measurement"
    )

    print(
        "Zero measurements before scan: "
        f"{quiet_run}"
    )

    print(
        "FERS scan-start timestamp: "
        f"{fers_sync_us:.3f} us"
    )

    print(
        "Counts at scan start: "
        f"{sync_counts:.0f}"
    )

    print()

    return {
        "fers_sync_us":
            fers_sync_us,

        "reference_channel":
            reference_channel,

        "sync_counts":
            sync_counts,
    }


# ============================================================
# Synchronize motor intervals
# ============================================================

def synchronize_scan_intervals(
    scan_intervals: pd.DataFrame,
) -> pd.DataFrame:
    """
    First stationary motor scan point = t = 0.
    """

    result = (
        scan_intervals.copy()
    )

    first_motor_start = float(
        result[
            "Motor_Start_Time_s"
        ].iloc[0]
    )

    result[
        "Start_Time_s"
    ] = (
        result[
            "Motor_Start_Time_s"
        ]
        - first_motor_start
    )

    result[
        "End_Time_s"
    ] = (
        result[
            "Motor_End_Time_s"
        ]
        - first_motor_start
    )

    return result


# ============================================================
# Calculate detector response
# ============================================================

def compute_detector_response(
    scan_intervals: pd.DataFrame,
    fers_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match synchronized FERS rate samples to each
    stationary motor interval.
    """

    fers_data = (
        fers_data
        .dropna(
            subset=[
                "Rate_Hz",
                "Synchronized_Time_s",
            ]
        )
        .copy()
    )

    rows = []

    for _, interval in (
        scan_intervals.iterrows()
    ):

        start = float(
            interval[
                "Start_Time_s"
            ]
        )

        end = float(
            interval[
                "End_Time_s"
            ]
        )

        samples = fers_data[
            (
                fers_data[
                    "Synchronized_Time_s"
                ]
                >= start
            )
            &
            (
                fers_data[
                    "Synchronized_Time_s"
                ]
                <= end
            )
        ]

        n = len(samples)

        if n:

            mean_rate = float(
                samples[
                    "Rate_Hz"
                ].mean()
            )

            median_rate = float(
                samples[
                    "Rate_Hz"
                ].median()
            )

        else:

            mean_rate = np.nan

            median_rate = np.nan

        if n > 1:

            std_rate = float(
                samples[
                    "Rate_Hz"
                ].std(
                    ddof=1
                )
            )

        else:

            std_rate = np.nan

        rows.append(
            {
                "Scan_Point":
                    int(
                        interval[
                            "Scan_Point"
                        ]
                    ),

                "X_cm":
                    float(
                        interval[
                            "X_cm"
                        ]
                    ),

                "Y_cm":
                    float(
                        interval[
                            "Y_cm"
                        ]
                    ),

                "Start_Time_s":
                    start,

                "End_Time_s":
                    end,

                "Duration_s":
                    float(
                        interval[
                            "Duration_s"
                        ]
                    ),

                "Mean_Rate_Hz":
                    mean_rate,

                "Median_Rate_Hz":
                    median_rate,

                "Std_Rate_Hz":
                    std_rate,

                "FERS_Samples":
                    n,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Plot geometry
# ============================================================

def plot_limits(
    scan_intervals,
):

    return (
        float(
            scan_intervals[
                "X_cm"
            ].min()
        ),

        float(
            scan_intervals[
                "X_cm"
            ].max()
        ),

        float(
            scan_intervals[
                "Y_cm"
            ].min()
        ),

        float(
            scan_intervals[
                "Y_cm"
            ].max()
        ),
    )


def figure_size(
    x_min,
    x_max,
    y_min,
    y_max,
):

    horizontal = max(
        y_max - y_min,
        1.0,
    )

    vertical = max(
        x_max - x_min,
        1.0,
    )

    height = 7.0

    width = min(
        max(
            height
            * horizontal
            / vertical,
            8.0,
        ),
        16.0,
    )

    return (
        width,
        height,
    )


# ============================================================
# Automatic color scale
# ============================================================

def calculate_color_scale(
    rates: pd.Series,
) -> tuple[float, float]:
    """
    Automatically determine a useful color range.

    The 98th percentile is used instead of the absolute
    maximum so isolated extreme points do not compress
    the entire map into a small part of the spectrum.

    The underlying detector-response values are NOT changed.
    This only affects visualization.
    """

    values = (
        pd.to_numeric(
            rates,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    values = (
        values[
            values >= 0
        ]
    )

    if values.empty:

        return (
            1.0,
            0.1,
        )

    actual_max = float(
        values.max()
    )

    robust_max = float(
        np.percentile(
            values,
            COLOR_PERCENTILE,
        )
    )

    if (
        not np.isfinite(
            robust_max
        )
        or robust_max <= 0
    ):

        robust_max = (
            actual_max
        )

    if robust_max <= 0:

        return (
            1.0,
            0.1,
        )

    # --------------------------------------------------------
    # Aim for approximately 8 major colorbar intervals.
    # --------------------------------------------------------

    target_step = (
        robust_max
        / 8.0
    )

    magnitude = (
        10.0
        ** np.floor(
            np.log10(
                target_step
            )
        )
    )

    scaled_step = (
        target_step
        / magnitude
    )

    if scaled_step <= 1:

        factor = 1.0

    elif scaled_step <= 2:

        factor = 2.0

    elif scaled_step <= 2.5:

        factor = 2.5

    elif scaled_step <= 5:

        factor = 5.0

    else:

        factor = 10.0

    tick_step = (
        factor
        * magnitude
    )

    color_max = (
        np.ceil(
            robust_max
            / tick_step
        )
        * tick_step
    )

    return (
        float(color_max),
        float(tick_step),
    )


# ============================================================
# Detector-response map
# ============================================================

def create_detector_response_plot(
    response_table: pd.DataFrame,
    scan_intervals: pd.DataFrame,
    motor_run: str,
    fers_run: str,
    channel: int,
    output_plot: Path,
) -> None:

    plot_data = (
        response_table
        .dropna(
            subset=[
                "Mean_Rate_Hz"
            ]
        )
        .copy()
    )

    if plot_data.empty:

        raise ValueError(
            f"Channel {channel} has no valid "
            "detector-response points."
        )

    # ========================================================
    # Physical motor dimensions
    # ========================================================

    (
        x_min,
        x_max,
        y_min,
        y_max,
    ) = plot_limits(
        scan_intervals
    )

    (
        width,
        height,
    ) = figure_size(
        x_min,
        x_max,
        y_min,
        y_max,
    )

    # ========================================================
    # Adaptive frequency range
    # ========================================================

    actual_max_rate = float(
        plot_data[
            "Mean_Rate_Hz"
        ].max()
    )

    (
        color_max_hz,
        tick_step_hz,
    ) = calculate_color_scale(
        plot_data[
            "Mean_Rate_Hz"
        ]
    )

    # ========================================================
    # Spectrum
    # ========================================================

    color_boundaries = (
        np.linspace(
            0.0,
            color_max_hz,
            NUMBER_OF_COLOR_LEVELS + 1,
        )
    )

    colorbar_ticks = (
        np.arange(
            0.0,
            color_max_hz
            + tick_step_hz,
            tick_step_hz,
        )
    )

    cmap = plt.get_cmap(
        "turbo",
        NUMBER_OF_COLOR_LEVELS,
    )

    norm = BoundaryNorm(
        color_boundaries,
        cmap.N,
        clip=True,
    )

    # ========================================================
    # Plot
    # ========================================================

    fig, axis = (
        plt.subplots(
            figsize=(
                width,
                height,
            )
        )
    )

    scatter = axis.scatter(
        plot_data[
            "Y_cm"
        ],
        plot_data[
            "X_cm"
        ],
        c=plot_data[
            "Mean_Rate_Hz"
        ],
        cmap=cmap,
        norm=norm,
        s=POINT_SIZE,
        marker="o",
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )

    colorbar = fig.colorbar(
        scatter,
        ax=axis,
        boundaries=color_boundaries,
        ticks=colorbar_ticks,
        pad=0.03,
    )

    colorbar.set_label(
        "Mean Counting Rate (Hz)"
    )

    axis.set_title(
        "Detector Response Map\n"
        f"Motor Run{motor_run} | "
        f"FERS Run{fers_run} | "
        f"Channel {channel}"
    )

    # Motor Y horizontally.
    axis.set_xlabel(
        "Motor Y Position (cm)"
    )

    # Motor X vertically.
    axis.set_ylabel(
        "Motor X Position (cm)"
    )

    axis.set_xlim(
        y_min
        - PLOT_PADDING_CM,
        y_max
        + PLOT_PADDING_CM,
    )

    # Laboratory orientation:
    # lower X values are shown at the top.
    axis.set_ylim(
        x_max
        + PLOT_PADDING_CM,
        x_min
        - PLOT_PADDING_CM,
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    axis.grid(
        True,
        linestyle=":",
        alpha=0.35,
        zorder=0,
    )

    fig.tight_layout()

    fig.savefig(
        output_plot,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # Color-scale diagnostics
    # ========================================================

    print()
    print(
        "Color-scale information"
    )

    print(
        "-----------------------"
    )

    print(
        "Maximum measured mean response: "
        f"{actual_max_rate:.2f} Hz"
    )

    print(
        f"{COLOR_PERCENTILE:.0f}th-percentile "
        "adaptive color range: "
        f"0 to {color_max_hz:.2f} Hz"
    )

    print(
        "Colorbar tick spacing: "
        f"{tick_step_hz:.2f} Hz"
    )

    if (
        actual_max_rate
        > color_max_hz
    ):

        print(
            "Note: extreme high-rate points use the "
            "top spectrum color so that the rest of "
            "the map keeps useful visual contrast."
        )


# ============================================================
# Main detector-response analysis
# ============================================================

def run_detector_response_analysis(
    motor_run: str,
    fers_run: str,
    channel_selection: str | None = None,
) -> None:

    motor_run = (
        normalize_run_number(
            motor_run
        )
    )

    fers_run = (
        normalize_run_number(
            fers_run
        )
    )

    motor_file = (
        au.DATA_DIR
        / f"Run{motor_run}_conf.txt"
    )

    fers_file = (
        au.DATA_DIR
        / f"Run{fers_run}_list.csv"
    )

    au.ensure_output_dirs()

    # ========================================================
    # Detect active FERS channels
    # ========================================================

    active_channels = (
        au.detect_active_channels(
            fers_file
        )
    )

    print()
    print(
        "Detected FERS channels with signal: "
        + ", ".join(
            map(
                str,
                active_channels,
            )
        )
    )

    # ========================================================
    # Channel selection
    # ========================================================

    if channel_selection is None:

        (
            selected_channels,
            _,
        ) = (
            au.ask_channel_selection(
                active_channels
            )
        )

    else:

        (
            selected_channels,
            _,
        ) = (
            au.normalize_channel_selection(
                channel_selection,
                active_channels,
            )
        )

    print(
        "Selected FERS channels: "
        + ", ".join(
            map(
                str,
                selected_channels,
            )
        )
    )

    # ========================================================
    # Prepare FERS data
    # ========================================================

    (
        all_tables,
        _,
    ) = (
        au.prepare_fers_channels(
            fers_file,
            active_channels,
        )
    )

    # ========================================================
    # Synchronization
    # ========================================================

    sync = (
        find_real_fers_scan_start(
            fers_file,
            active_channels,
            MIN_QUIET_EVENTS,
        )
    )

    fers_sync_us = float(
        sync[
            "fers_sync_us"
        ]
    )

    print(
        "Synchronization reference channel: "
        f"{sync['reference_channel']}"
    )

    # ========================================================
    # Convert FERS channels to synchronized time
    # ========================================================

    synchronized_fers_tables = {}

    for channel in (
        active_channels
    ):

        channel_data = (
            all_tables[
                channel
            ]
            .copy()
        )

        # Ignore FERS information before real scan.
        channel_data = (
            channel_data[
                channel_data[
                    "TStamp_us"
                ]
                >= fers_sync_us
            ]
            .copy()
        )

        # Real FERS scan start = t = 0.
        channel_data[
            "Synchronized_Time_s"
        ] = (
            channel_data[
                "TStamp_us"
            ]
            - fers_sync_us
        ) / 1_000_000.0

        synchronized_fers_tables[
            channel
        ] = (
            channel_data
            .reset_index(
                drop=True
            )
        )

    # ========================================================
    # Motor
    # ========================================================

    (
        motor_data,
        first_scan_point,
    ) = (
        read_motor_data(
            motor_file
        )
    )

    motor_timeline = (
        unwrap_motor_timestamps(
            motor_data
        )
    )

    intervals = (
        extract_stationary_intervals(
            motor_timeline
        )
    )

    scan_intervals = (
        select_scan_intervals(
            intervals,
            first_scan_point,
        )
    )

    synchronized_intervals = (
        synchronize_scan_intervals(
            scan_intervals
        )
    )

    print()
    print(
        "Motor scan range:"
    )

    print(
        "X: "
        f"{scan_intervals['X_cm'].min():.2f} "
        "to "
        f"{scan_intervals['X_cm'].max():.2f} cm"
    )

    print(
        "Y: "
        f"{scan_intervals['Y_cm'].min():.2f} "
        "to "
        f"{scan_intervals['Y_cm'].max():.2f} cm"
    )

    # ========================================================
    # Analyze each selected channel separately
    # ========================================================

    for channel in (
        selected_channels
    ):

        print()
        print(
            "========================================"
        )

        print(
            f"Channel {channel}"
        )

        print(
            "========================================"
        )

        response = (
            compute_detector_response(
                synchronized_intervals,
                synchronized_fers_tables[
                    channel
                ],
            )
        )

        output_csv = (
            au.RESULTS_DIR
            / (
                f"Run{motor_run}_config_"
                f"Run{fers_run}_CH{channel}_"
                "detector_response.csv"
            )
        )

        output_plot = (
            au.PLOTS_DIR
            / (
                f"Run{motor_run}_config_"
                f"Run{fers_run}_CH{channel}_"
                "detector_response_map.png"
            )
        )

        response.to_csv(
            output_csv,
            index=False,
        )

        create_detector_response_plot(
            response,
            synchronized_intervals,
            motor_run,
            fers_run,
            channel,
            output_plot,
        )

        valid = (
            response[
                response[
                    "FERS_Samples"
                ]
                > 0
            ]
        )

        missing = (
            response[
                response[
                    "FERS_Samples"
                ]
                == 0
            ]
        )

        print()
        print(
            "Synchronization time: "
            "0.000 s"
        )

        print(
            "Points with FERS data: "
            f"{len(valid)} / "
            f"{len(response)}"
        )

        print(
            "Points without FERS data: "
            f"{len(missing)}"
        )

        if not valid.empty:

            print(
                "Mean samples per point: "
                f"{valid['FERS_Samples'].mean():.1f}"
            )

            print(
                "Minimum mean rate: "
                f"{valid['Mean_Rate_Hz'].min():.2f} Hz"
            )

            print(
                "Maximum mean rate: "
                f"{valid['Mean_Rate_Hz'].max():.2f} Hz"
            )

        print(
            f"Results: "
            f"{output_csv}"
        )

        print(
            f"Plot:    "
            f"{output_plot}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print(
        "Detector Response Map"
    )

    print(
        "---------------------"
    )

    if len(sys.argv) == 1:

        motor_run = (
            ask_run_number(
                "Enter the motor configuration "
                "run number: "
            )
        )

        fers_run = (
            ask_run_number(
                "Enter the FERS run number: "
            )
        )

        run_detector_response_analysis(
            motor_run,
            fers_run,
        )

    elif len(sys.argv) == 4:

        run_detector_response_analysis(
            sys.argv[1],
            sys.argv[2],
            sys.argv[3],
        )

    else:

        sys.exit(
            "Usage:\n"
            "  python3 "
            "05_detector_response_map.py\n"
            "or\n"
            "  python3 "
            "05_detector_response_map.py "
            "MOTOR_RUN FERS_RUN CHANNEL|all"
        )


if __name__ == "__main__":
    main()
