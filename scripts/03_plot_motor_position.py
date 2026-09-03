"""
Plot motor position versus time for one source scan.

Time zero is the configured FIRST SCAN POINT,
not the first line written to the motor log.

Usage:
    python3 scripts/03_plot_motor_position.py
"""

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import analysis_utils as au


POSITION_TOLERANCE_CM = 0.001


def ask_run_name() -> str:
    while True:
        value = input(
            "Enter the motor run number: "
        )

        try:
            return au.normalize_run_name(value)

        except ValueError as error:
            print(error)


def find_first_scan_index(
    motion,
    first_scan_point,
) -> int:
    """
    Find the first motor row corresponding to the
    configured first source-scan position.
    """

    matches = motion[
        (
            motion["X_cm"]
            - first_scan_point["X_cm"]
        ).abs()
        <= POSITION_TOLERANCE_CM
    ]

    matches = matches[
        (
            matches["Y_cm"]
            - first_scan_point["Y_cm"]
        ).abs()
        <= POSITION_TOLERANCE_CM
    ]

    if matches.empty:
        raise ValueError(
            "Configured first scan position was not "
            "found in the motor data: "
            f"X = {first_scan_point['X_cm']:.2f} cm, "
            f"Y = {first_scan_point['Y_cm']:.2f} cm."
        )

    return int(matches.index[0])


def run_motor_analysis(
    run_name: str,
) -> None:

    run_name = au.normalize_run_name(
        run_name
    )

    motor_file = (
        au.DATA_DIR
        / f"{run_name}_conf.txt"
    )

    output_csv = (
        au.RESULTS_DIR
        / f"{run_name}_motor_positions.csv"
    )

    output_plot = (
        au.PLOTS_DIR
        / f"{run_name}_motor_position_vs_time.png"
    )

    au.ensure_output_dirs()

    print()
    print("Reading motor data...")

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
        find_first_scan_index(
            raw_motion,
            first_scan_point,
        )
    )

    # Remove everything before the real scan begins.
    scan_motion = (
        raw_motion
        .loc[first_scan_index:]
        .reset_index(drop=True)
    )

    # Correct counter rollovers and remove parking.
    motion = (
        au.unwrap_motor_timestamps(
            scan_motion
        )
    )

    columns = [
        "Motor_Timestamp_us",
        "Unwrapped_Timestamp_us",
        "Motor_Time_s",
        "X_cm",
        "Y_cm",
        "Source_Line",
    ]

    motion[
        columns
    ].to_csv(
        output_csv,
        index=False,
    )

    au.create_motor_plot(
        motion,
        run_name,
        output_plot,
    )

    duration_s = float(
        motion[
            "Motor_Time_s"
        ].iloc[-1]
    )

    print()
    print("Motor scan")
    print("-----------------------")

    print(
        "First scan point:"
    )

    print(
        f"X = {motion['X_cm'].iloc[0]:.2f} cm, "
        f"Y = {motion['Y_cm'].iloc[0]:.2f} cm"
    )

    print(
        "Last scan point:"
    )

    print(
        f"X = {motion['X_cm'].iloc[-1]:.2f} cm, "
        f"Y = {motion['Y_cm'].iloc[-1]:.2f} cm"
    )

    print(
        f"Duration: {duration_s:.3f} s"
    )

    print(
        f"Motor rows used: {len(motion)}"
    )

    print(
        f"Results: {output_csv}"
    )

    print(
        f"Plot:    {output_plot}"
    )


def main() -> None:

    print()
    print("Motor Position Plot")
    print("-------------------")

    if len(sys.argv) == 1:

        run_name = ask_run_name()

    elif len(sys.argv) == 2:

        run_name = sys.argv[1]

    else:

        sys.exit(
            "Usage: python3 "
            "scripts/03_plot_motor_position.py [RUN]"
        )

    try:

        run_motor_analysis(
            run_name
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
