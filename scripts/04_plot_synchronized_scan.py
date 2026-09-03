"""
Create one synchronized motor/FERS plot per selected FERS channel.

Synchronization logic
---------------------

The FERS board may record some non-zero measurements immediately
after it is enabled, before the motor scan actually starts.

These startup measurements must NOT be used for synchronization.

The synchronization procedure is:

    startup non-zero measurements
                |
                v
            IGNORE THEM

    sustained zero-count waiting region on Channel 0
                |
                v
    first non-zero Channel 0 measurement
                |
                v
        REAL FERS SCAN START

That FERS timestamp is aligned with the configured first motor
scan position.

The same synchronization timestamp is used for every FERS channel.

If "all" is selected, every active channel is plotted separately.
Channels are never summed or combined.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import analysis_utils as au


# ============================================================
# Analysis settings
# ============================================================

POSITION_TOLERANCE_CM = 0.001

# Number of consecutive zero-count Channel 0 measurements
# required before accepting the next non-zero measurement
# as the beginning of the real motor scan.
MIN_QUIET_EVENTS = 5


# ============================================================
# User input
# ============================================================

def ask_run_name(
    prompt: str,
) -> str:
    """
    Ask for a run number and convert it to RunNN.
    """

    while True:

        value = input(
            prompt
        ).strip()

        try:

            return au.normalize_run_name(
                value
            )

        except ValueError as error:

            print(error)


# ============================================================
# Find first configured motor scan point
# ============================================================

def find_first_motor_scan_index(
    raw_motion: pd.DataFrame,
    first_scan_point: dict,
    tolerance_cm: float = POSITION_TOLERANCE_CM,
) -> int:
    """
    Find the first motor row corresponding to the
    configured first scan position.
    """

    matches = raw_motion[
        (
            raw_motion["X_cm"]
            - first_scan_point["X_cm"]
        ).abs()
        <= tolerance_cm
    ]

    matches = matches[
        (
            matches["Y_cm"]
            - first_scan_point["Y_cm"]
        ).abs()
        <= tolerance_cm
    ]

    if matches.empty:

        raise ValueError(
            "The configured first motor position was not found: "
            f"X = {first_scan_point['X_cm']:.2f} cm, "
            f"Y = {first_scan_point['Y_cm']:.2f} cm."
        )

    return int(
        matches.index[0]
    )


# ============================================================
# Find the real FERS scan start
# ============================================================

def find_real_fers_scan_start(
    fers_file: Path,
    active_channels: list[int],
    min_quiet_events: int = MIN_QUIET_EVENTS,
) -> dict:
    """
    Find the real beginning of the motor scan in the FERS data.

    Experimental sequence:

        Board enabled
            |
            v
        startup non-zero measurements
            |
            | ignored
            v
        sustained zero-count waiting region
            |
            v
        first non-zero measurement
            |
            v
        REAL SCAN START


    Channel 0 is used as the synchronization reference when
    it is available.

    Importantly, the resulting timestamp is then used for ALL
    selected FERS channels.
    """

    raw_data = (
        au.read_fers_raw(
            fers_file
        )
        .copy()
    )

    # ========================================================
    # Select synchronization reference channel
    # ========================================================

    if 0 in active_channels:

        reference_channel = 0

    else:

        reference_channel = (
            active_channels[0]
        )

        print()
        print(
            "Warning:"
        )

        print(
            "Channel 0 is not active in this run."
        )

        print(
            "Using Channel "
            f"{reference_channel} "
            "as the synchronization reference."
        )

    # ========================================================
    # Extract reference-channel measurements
    # ========================================================

    reference_data = (
        raw_data[
            raw_data["CH_Id"]
            == reference_channel
        ]
        .sort_values(
            "TStamp_us"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    if reference_data.empty:

        raise ValueError(
            "No FERS measurements were found for "
            f"reference Channel {reference_channel}."
        )

    # ========================================================
    # Search for:
    #
    # startup signal
    #      ->
    # sustained zero-count region
    #      ->
    # first non-zero measurement
    #
    # ========================================================

    quiet_run = 0

    quiet_region_seen = False

    synchronization_index = None

    quiet_region_start_index = None

    for index, row in (
        reference_data.iterrows()
    ):

        counts = float(
            row["Counts"]
        )

        # ----------------------------------------------------
        # Zero-count measurement
        # ----------------------------------------------------

        if counts == 0:

            if quiet_run == 0:

                quiet_region_start_index = (
                    index
                )

            quiet_run += 1

            if (
                quiet_run
                >= min_quiet_events
            ):

                quiet_region_seen = True

            continue

        # ----------------------------------------------------
        # Non-zero measurement AFTER quiet region
        # ----------------------------------------------------

        if quiet_region_seen:

            synchronization_index = (
                index
            )

            break

        # ----------------------------------------------------
        # Non-zero measurement BEFORE quiet region
        #
        # This belongs to startup and is ignored.
        # ----------------------------------------------------

        quiet_run = 0

        quiet_region_start_index = None

    # ========================================================
    # Check synchronization was found
    # ========================================================

    if synchronization_index is None:

        raise ValueError(
            "Could not determine the real FERS scan start "
            f"using Channel {reference_channel}. "
            "No non-zero measurement was found after "
            f"at least {min_quiet_events} consecutive "
            "zero-count measurements."
        )

    # ========================================================
    # Synchronization row
    # ========================================================

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

    # Number of zero measurements immediately preceding sync.
    zero_measurements_before_sync = (
        quiet_run
    )

    # ========================================================
    # Diagnostics
    # ========================================================

    print()
    print(
        "FERS scan-start detection"
    )

    print(
        "-------------------------"
    )

    print(
        "Synchronization reference channel: "
        f"{reference_channel}"
    )

    print(
        "Method:"
    )

    print(
        "startup counts -> sustained zero-count region "
        "-> first non-zero measurement"
    )

    print(
        "Required consecutive zero measurements: "
        f"{min_quiet_events}"
    )

    print(
        "Zero measurements before scan start: "
        f"{zero_measurements_before_sync}"
    )

    print(
        "First real FERS scan timestamp: "
        f"{fers_sync_us:.3f} us"
    )

    print(
        "Counts at first real scan point: "
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

        "event_index":
            synchronization_index,

        "zero_measurements_before_sync":
            zero_measurements_before_sync,
    }


# ============================================================
# Main synchronized analysis
# ============================================================

def run_synchronized_analysis(
    config_run: str,
    fers_run: str,
    channel_selection: str | None = None,
) -> None:

    motor_file = (
        au.DATA_DIR
        / f"{config_run}_conf.txt"
    )

    fers_file = (
        au.DATA_DIR
        / f"{fers_run}_list.csv"
    )

    au.ensure_output_dirs()

    # ========================================================
    # Read FERS data
    # ========================================================

    print()
    print(
        "Reading FERS data..."
    )

    active_channels = (
        au.detect_active_channels(
            fers_file
        )
    )

    print(
        "Detected active FERS channels: "
        + ", ".join(
            str(channel)
            for channel
            in active_channels
        )
    )

    # ========================================================
    # Select channels to analyze
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
            str(channel)
            for channel
            in selected_channels
        )
    )

    # ========================================================
    # Prepare all active channels
    #
    # They all use the same FERS acquisition time origin.
    # ========================================================

    (
        all_channel_tables,
        fers_origin_us,
    ) = (
        au.prepare_fers_channels(
            fers_file,
            active_channels,
        )
    )

    # ========================================================
    # NEW synchronization logic
    # ========================================================

    sync_result = (
        find_real_fers_scan_start(
            fers_file,
            active_channels,
            MIN_QUIET_EVENTS,
        )
    )

    fers_sync_us = float(
        sync_result[
            "fers_sync_us"
        ]
    )

    sync_reference_channel = (
        sync_result[
            "reference_channel"
        ]
    )

    sync_counts = float(
        sync_result[
            "sync_counts"
        ]
    )

    # Original location of the real scan start
    # within the full FERS acquisition.
    synchronization_time_s = (
        fers_sync_us
        - fers_origin_us
    ) / 1_000_000.0

    # ========================================================
    # Read motor data
    # ========================================================

    print()
    print(
        "Reading motor data..."
    )

    raw_motion = (
        au.read_motor_positions(
            motor_file
        )
    )

    first_scan_point = (
        au.read_configured_first_point(
            motor_file
        )
    )

    first_scan_index = (
        find_first_motor_scan_index(
            raw_motion,
            first_scan_point,
        )
    )

    # ========================================================
    # Remove motor records before first scan point
    # ========================================================

    scan_candidate = (
        raw_motion
        .loc[
            first_scan_index:
        ]
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Correct motor timestamp rollover/reset
    # ========================================================

    motor_scan = (
        au.unwrap_motor_timestamps(
            scan_candidate
        )
    )

    # ========================================================
    # Set first motor scan point to synchronized t = 0
    # ========================================================

    motor_scan[
        "Aligned_Time_s"
    ] = (
        motor_scan[
            "Motor_Time_s"
        ]
    )

    motor_duration_s = float(
        motor_scan[
            "Motor_Time_s"
        ].iloc[-1]
    )

    # ========================================================
    # Synchronize every FERS channel
    # ========================================================

    synchronized_channel_tables = {}

    for channel in active_channels:

        channel_data = (
            all_channel_tables[
                channel
            ]
            .copy()
        )

        # ----------------------------------------------------
        # Remove ALL FERS data before the real scan start.
        # ----------------------------------------------------

        channel_data = (
            channel_data[
                channel_data[
                    "TStamp_us"
                ]
                >= fers_sync_us
            ]
            .copy()
        )

        # ----------------------------------------------------
        # Define scan start as synchronized t = 0.
        # ----------------------------------------------------

        channel_data[
            "Synchronized_Time_s"
        ] = (
            channel_data[
                "TStamp_us"
            ]
            - fers_sync_us
        ) / 1_000_000.0

        synchronized_channel_tables[
            channel
        ] = (
            channel_data
            .reset_index(
                drop=True
            )
        )

    # ========================================================
    # Synchronization summary
    # ========================================================

    print()
    print(
        "Synchronization"
    )

    print(
        "---------------------------"
    )

    print(
        "Reference channel: "
        f"{sync_reference_channel}"
    )

    print(
        "FERS synchronization timestamp: "
        f"{fers_sync_us:.3f} us"
    )

    print(
        "Counts at synchronization: "
        f"{sync_counts:.0f}"
    )

    print(
        "Original FERS scan-start time: "
        f"{synchronization_time_s:.3f} s"
    )

    print(
        "Synchronized time zero: "
        "0.000 s"
    )

    print()
    print(
        "First motor scan point:"
    )

    print(
        f"X = "
        f"{motor_scan['X_cm'].iloc[0]:.2f} cm, "
        f"Y = "
        f"{motor_scan['Y_cm'].iloc[0]:.2f} cm"
    )

    print(
        "Last motor scan point:"
    )

    print(
        f"X = "
        f"{motor_scan['X_cm'].iloc[-1]:.2f} cm, "
        f"Y = "
        f"{motor_scan['Y_cm'].iloc[-1]:.2f} cm"
    )

    print(
        f"Motor duration: "
        f"{motor_duration_s:.3f} s"
    )

    print(
        f"Motor rows used: "
        f"{len(motor_scan)}"
    )

    # ========================================================
    # Create one separate synchronized graph per channel
    # ========================================================

    for channel in selected_channels:

        channel_data = (
            synchronized_channel_tables[
                channel
            ]
        )

        fers_plot_data = (
            channel_data
            .dropna(
                subset=[
                    "Rate_Hz"
                ]
            )
            .copy()
        )

        # ----------------------------------------------------
        # create_synchronized_plot() expects the FERS X-axis
        # in Relative_Time_s.
        #
        # Replace it with the synchronized scan time.
        # ----------------------------------------------------

        fers_plot_data[
            "Relative_Time_s"
        ] = (
            fers_plot_data[
                "Synchronized_Time_s"
            ]
        )

        # ----------------------------------------------------
        # Keep only data corresponding to actual motor scan.
        # ----------------------------------------------------

        fers_plot_data = (
            fers_plot_data[
                (
                    fers_plot_data[
                        "Relative_Time_s"
                    ]
                    >= 0.0
                )
                &
                (
                    fers_plot_data[
                        "Relative_Time_s"
                    ]
                    <= motor_duration_s
                )
            ]
            .copy()
        )

        # ====================================================
        # Output files
        # ====================================================

        plot_file = (
            au.PLOTS_DIR
            / (
                f"{config_run}_config_"
                f"{fers_run}_CH{channel}_"
                "fers_synchronized.png"
            )
        )

        result_file = (
            au.RESULTS_DIR
            / (
                f"{config_run}_config_"
                f"{fers_run}_CH{channel}_"
                "synchronization.csv"
            )
        )

        # ====================================================
        # Create synchronized plot
        #
        # Synchronization line is at t = 0.
        # ====================================================

        au.create_synchronized_plot(
            motor_scan,
            fers_plot_data,
            0.0,
            config_run,
            fers_run,
            plot_file,
            channel,
        )

        # ====================================================
        # Save synchronization metadata
        # ====================================================

        result_df = pd.DataFrame(
            [
                {
                    "Configuration_Run":
                        config_run,

                    "FERS_Run":
                        fers_run,

                    "Analysis_Channel":
                        channel,

                    "Synchronization_Reference_Channel":
                        sync_reference_channel,

                    "Synchronization_Method":
                        (
                            "first_nonzero_after_"
                            "sustained_zero_region"
                        ),

                    "FERS_Sync_TStamp_us":
                        fers_sync_us,

                    "FERS_Sync_Counts":
                        sync_counts,

                    "Original_FERS_Synchronization_Time_s":
                        synchronization_time_s,

                    "Synchronized_Time_Zero_s":
                        0.0,

                    "Motor_Sync_TStamp_us":
                        float(
                            motor_scan[
                                "Motor_Timestamp_us"
                            ].iloc[0]
                        ),

                    "Motor_Sync_X_cm":
                        float(
                            motor_scan[
                                "X_cm"
                            ].iloc[0]
                        ),

                    "Motor_Sync_Y_cm":
                        float(
                            motor_scan[
                                "Y_cm"
                            ].iloc[0]
                        ),

                    "Motor_Duration_s":
                        motor_duration_s,

                    "Motor_Rows_Used":
                        len(
                            motor_scan
                        ),

                    "FERS_Rows_Used":
                        len(
                            fers_plot_data
                        ),
                }
            ]
        )

        result_df.to_csv(
            result_file,
            index=False,
        )

        # ====================================================
        # Print channel result
        # ====================================================

        print()
        print(
            f"Channel {channel} finished."
        )

        print(
            "FERS points used after synchronization: "
            f"{len(fers_plot_data)}"
        )

        print(
            f"Plot saved:    "
            f"{plot_file}"
        )

        print(
            f"Results saved: "
            f"{result_file}"
        )


# ============================================================
# Main program
# ============================================================

def main() -> None:

    print()
    print(
        "Synchronized Motor and FERS Plot"
    )

    print(
        "---------------------------------"
    )

    # ========================================================
    # Interactive mode
    # ========================================================

    if len(sys.argv) == 1:

        config_run = (
            ask_run_name(
                "Enter the motor configuration "
                "run number: "
            )
        )

        fers_run = (
            ask_run_name(
                "Enter the FERS run number: "
            )
        )

        channel_selection = None

    # ========================================================
    # Command-line mode
    #
    # Example:
    #
    # python3 04_plot_synchronized_scan.py 63 63 0
    #
    # or:
    #
    # python3 04_plot_synchronized_scan.py 63 63 all
    # ========================================================

    elif len(sys.argv) == 4:

        try:

            config_run = (
                au.normalize_run_name(
                    sys.argv[1]
                )
            )

            fers_run = (
                au.normalize_run_name(
                    sys.argv[2]
                )
            )

        except ValueError as error:

            sys.exit(
                f"\nError:\n{error}"
            )

        channel_selection = (
            sys.argv[3]
        )

    else:

        sys.exit(
            "Usage:\n"
            "  python3 "
            "04_plot_synchronized_scan.py\n"
            "or\n"
            "  python3 "
            "04_plot_synchronized_scan.py "
            "MOTOR_RUN FERS_RUN CHANNEL|all"
        )

    # ========================================================
    # Run analysis
    # ========================================================

    try:

        run_synchronized_analysis(
            config_run,
            fers_run,
            channel_selection,
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:

        sys.exit(
            f"\nError:\n{error}"
        )


if __name__ == "__main__":
    main()
