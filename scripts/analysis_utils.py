"""Shared utilities for CERN SiPM source-scan analysis."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
PLOTS_DIR = PROJECT_DIR / "plots"
RESULTS_DIR = PROJECT_DIR / "results"

MOTOR_COUNTER_MODULUS_US = float(2**32)
PREFERRED_SYNC_CHANNEL = 0

BACKGROUND_WINDOW_S = 300.0
BACKGROUND_SIGMA_FACTOR = 5.0
SYNC_MIN_RATE_INCREASE_HZ = 50.0
SYNC_SMOOTHING_WINDOW_S = 5.0
SYNC_MIN_SUSTAINED_S = 5.0
MIN_BACKGROUND_SAMPLES = 3


def ensure_output_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_run_name(value: str) -> str:
    value = value.strip()
    if value.lower().startswith("run"):
        value = value[3:].strip()
    if not value.isdigit():
        raise ValueError("Enter a run number such as 63 or Run63.")
    return f"Run{value}"


def read_fers_raw(fers_file: Path) -> pd.DataFrame:
    if not fers_file.exists():
        raise FileNotFoundError(f"FERS file not found: {fers_file}")

    data = pd.read_csv(fers_file, comment="/")
    required = {"TStamp_us", "CH_Id", "Counts"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"{fers_file.name} is missing required columns: {sorted(missing)}"
        )

    data = data.dropna(subset=["TStamp_us", "CH_Id", "Counts"]).copy()
    if data.empty:
        raise ValueError(f"No usable FERS rows found in {fers_file.name}.")

    data["CH_Id"] = data["CH_Id"].astype(int)
    return data.sort_values("TStamp_us").reset_index(drop=True)


def detect_active_channels(fers_file: Path) -> list[int]:
    """Channels with at least one real signal row: Counts > 0."""
    data = read_fers_raw(fers_file)
    channels = sorted(
        data.loc[data["Counts"] > 0, "CH_Id"].astype(int).unique().tolist()
    )
    if not channels:
        raise ValueError("No FERS channels with Counts > 0 were found.")
    return channels


def ask_channel_selection(active_channels: list[int]) -> tuple[list[int], bool]:
    if not active_channels:
        raise ValueError("No active FERS channels are available.")

    while True:
        value = input(
            "Enter FERS channel to analyze, or type 'all': "
        ).strip().lower()

        if value == "all":
            return list(active_channels), True

        if value.isdigit():
            channel = int(value)
            if channel in active_channels:
                return [channel], False
            print(f"Channel {channel} does not have a detected signal in this run.")
        else:
            print("Enter a channel number or 'all'.")


def normalize_channel_selection(
    selection: str,
    active_channels: list[int],
) -> tuple[list[int], bool]:
    value = selection.strip().lower()
    if value == "all":
        return list(active_channels), True
    if not value.isdigit():
        raise ValueError("Enter a channel number or 'all'.")

    channel = int(value)
    if channel not in active_channels:
        raise ValueError(
            f"Channel {channel} does not have a detected signal in this run."
        )
    return [channel], False


def get_common_fers_origin_us(fers_file: Path) -> float:
    data = read_fers_raw(fers_file)
    return float(data["TStamp_us"].min())


def read_fers_data(fers_file: Path, channel: int) -> pd.DataFrame:
    data = read_fers_raw(fers_file)
    channel_data = (
        data[data["CH_Id"] == int(channel)]
        .sort_values("TStamp_us")
        .reset_index(drop=True)
        .copy()
    )
    if channel_data.empty:
        raise ValueError(f"No Channel {channel} rows found in {fers_file.name}.")
    return channel_data


def calculate_fers_rate(
    channel_data: pd.DataFrame,
    origin_us: float | None = None,
) -> pd.DataFrame:
    data = channel_data.sort_values("TStamp_us").reset_index(drop=True).copy()
    if data.empty:
        raise ValueError("Cannot calculate FERS rate from an empty table.")

    if origin_us is None:
        origin_us = float(data["TStamp_us"].iloc[0])

    data["Relative_Time_s"] = (data["TStamp_us"] - origin_us) / 1_000_000.0
    data["Delta_Time_s"] = data["TStamp_us"].diff() / 1_000_000.0
    data["Rate_Hz"] = np.nan

    valid = data["Delta_Time_s"] > 0
    data.loc[valid, "Rate_Hz"] = (
        data.loc[valid, "Counts"] / data.loc[valid, "Delta_Time_s"]
    )
    return data


def prepare_fers_channels(
    fers_file: Path,
    channels: list[int],
) -> tuple[dict[int, pd.DataFrame], float]:
    origin_us = get_common_fers_origin_us(fers_file)
    prepared = {}
    for channel in channels:
        prepared[channel] = calculate_fers_rate(
            read_fers_data(fers_file, channel),
            origin_us=origin_us,
        )
    return prepared, origin_us


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


def read_motor_positions(motor_file: Path) -> pd.DataFrame:
    if not motor_file.exists():
        raise FileNotFoundError(f"Motor file not found: {motor_file}")

    rows = []
    with motor_file.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            match = _POSITION_PATTERN.match(line.strip())
            if not match:
                continue
            epoch_s, timestamp_us, x_cm, y_cm = match.groups()
            rows.append(
                {
                    "Epoch_s": int(epoch_s),
                    "Motor_Timestamp_us": float(timestamp_us),
                    "X_cm": float(x_cm),
                    "Y_cm": float(y_cm),
                    "Source_Line": line_number,
                }
            )

    if not rows:
        raise ValueError(f"No motor-position rows found in {motor_file.name}.")
    return pd.DataFrame(rows)


def read_configured_first_point(motor_file: Path) -> dict:
    if not motor_file.exists():
        raise FileNotFoundError(f"Motor file not found: {motor_file}")

    with motor_file.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            match = _FIRST_POINT_PATTERN.search(line)
            if match:
                x_cm, y_cm, dwell_s = match.groups()
                return {
                    "X_cm": float(x_cm),
                    "Y_cm": float(y_cm),
                    "Dwell_s": float(dwell_s),
                }

    raise ValueError(
        f"The configured 'scan first point' was not found in {motor_file.name}."
    )


def unwrap_motor_timestamps(
    motion: pd.DataFrame,
    timestamp_column: str = "Motor_Timestamp_us",
) -> pd.DataFrame:
    retained = []
    previous = None
    offset_us = 0.0
    rollover_count = 0

    for row_number, row in motion.reset_index(drop=True).iterrows():
        raw = float(row[timestamp_column])

        if previous is not None and raw < previous:
            is_rollover = (
                previous > 0.8 * MOTOR_COUNTER_MODULUS_US
                and raw < 0.2 * MOTOR_COUNTER_MODULUS_US
            )
            if is_rollover:
                offset_us += MOTOR_COUNTER_MODULUS_US
                rollover_count += 1
                print(
                    f"  Motor timestamp rollover detected at row {row_number}. "
                    "Scan continues."
                )
            else:
                print(
                    f"  Motor timestamp reset detected at row {row_number}. "
                    "Treating this as the start of parking; later rows removed."
                )
                break

        item = row.to_dict()
        item["Unwrapped_Timestamp_us"] = raw + offset_us
        retained.append(item)
        previous = raw

    if not retained:
        raise ValueError("No motor rows remained after timestamp processing.")

    result = pd.DataFrame(retained).reset_index(drop=True)
    first_us = float(result["Unwrapped_Timestamp_us"].iloc[0])
    result["Motor_Time_s"] = (
        result["Unwrapped_Timestamp_us"] - first_us
    ) / 1_000_000.0

    print(f"  Motor counter rollovers corrected: {rollover_count}")
    return result


def find_fers_synchronization_point(
    fers_data: pd.DataFrame,
    background_window_s: float = BACKGROUND_WINDOW_S,
    background_sigma_factor: float = BACKGROUND_SIGMA_FACTOR,
    min_rate_increase_hz: float = SYNC_MIN_RATE_INCREASE_HZ,
    smoothing_window_s: float = SYNC_SMOOTHING_WINDOW_S,
    min_sustained_s: float = SYNC_MIN_SUSTAINED_S,
    min_background_samples: int = MIN_BACKGROUND_SAMPLES,
    print_diagnostics: bool = True,
) -> dict:
    data = fers_data.reset_index(drop=True)
    valid = data[
        data["Rate_Hz"].notna() & (data["Delta_Time_s"] > 0)
    ].copy()

    if len(valid) < min_background_samples:
        raise ValueError("Not enough valid FERS rate samples for synchronization.")

    median_dt = float(valid["Delta_Time_s"].median())
    if not np.isfinite(median_dt) or median_dt <= 0:
        raise ValueError("Could not determine a valid FERS acquisition interval.")

    first_valid_time_s = float(valid["Relative_Time_s"].iloc[0])
    background = valid[
        valid["Relative_Time_s"] <= first_valid_time_s + background_window_s
    ]

    if len(background) < min_background_samples:
        raise ValueError("Not enough samples in the FERS background window.")

    background_mean_hz = float(background["Rate_Hz"].mean())
    background_std_hz = float(background["Rate_Hz"].std(ddof=1))
    if not np.isfinite(background_std_hz):
        background_std_hz = 0.0

    threshold_hz = max(
        background_mean_hz + background_sigma_factor * background_std_hz,
        background_mean_hz + min_rate_increase_hz,
    )

    smoothing_samples = max(1, round(smoothing_window_s / median_dt))
    sustained_samples = max(1, round(min_sustained_s / median_dt))

    smoothed = valid["Rate_Hz"].rolling(
        window=smoothing_samples,
        min_periods=1,
    ).mean()

    run_length = 0
    run_start = None
    sync_position = None

    for position, above in enumerate((smoothed > threshold_hz).to_numpy()):
        if above:
            if run_length == 0:
                run_start = position
            run_length += 1
            if run_length >= sustained_samples:
                sync_position = run_start
                break
        else:
            run_length = 0
            run_start = None

    diagnostics = {
        "background_window_s": background_window_s,
        "background_samples": len(background),
        "background_mean_hz": background_mean_hz,
        "background_std_hz": background_std_hz,
        "sync_threshold_hz": threshold_hz,
        "median_delta_time_s": median_dt,
        "smoothing_window_s": smoothing_window_s,
        "smoothing_samples": smoothing_samples,
        "min_sustained_s": min_sustained_s,
        "sustained_samples": sustained_samples,
    }

    if sync_position is None:
        if print_diagnostics:
            _print_sync_diagnostics(diagnostics, False)
        raise ValueError("No sustained rise above the FERS background was found.")

    sync_row = valid.iloc[sync_position]
    diagnostics["detected_sync_time_s"] = float(sync_row["Relative_Time_s"])
    diagnostics["detected_sync_rate_hz"] = float(sync_row["Rate_Hz"])

    if print_diagnostics:
        _print_sync_diagnostics(diagnostics, True)

    return {
        "fers_sync_row": sync_row,
        "fers_sync_index": int(sync_row.name),
        "diagnostics": diagnostics,
    }


def _print_sync_diagnostics(diagnostics: dict, detected: bool) -> None:
    print("\nFERS synchronization")
    print("--------------------")
    print(f"Background window        : {diagnostics['background_window_s']:.1f} s")
    print(f"Background samples       : {diagnostics['background_samples']}")
    print(f"Background mean          : {diagnostics['background_mean_hz']:.3f} Hz")
    print(f"Background std           : {diagnostics['background_std_hz']:.3f} Hz")
    print(f"Synchronization threshold: {diagnostics['sync_threshold_hz']:.3f} Hz")
    print(
        f"Smoothing window         : {diagnostics['smoothing_window_s']:.1f} s "
        f"({diagnostics['smoothing_samples']} samples)"
    )
    print(
        f"Required sustained rise  : {diagnostics['min_sustained_s']:.1f} s "
        f"({diagnostics['sustained_samples']} samples)"
    )
    if detected:
        print(f"Detected sync time       : {diagnostics['detected_sync_time_s']:.3f} s")
        print(f"Detected sync Rate_Hz    : {diagnostics['detected_sync_rate_hz']:.3f} Hz")
    else:
        print("Detected sync time       : NOT FOUND")
        print("Detected sync Rate_Hz    : NOT FOUND")


def find_common_fers_synchronization(
    processed_channels: dict[int, pd.DataFrame],
    preferred_channel: int = PREFERRED_SYNC_CHANNEL,
) -> dict:
    channels = sorted(processed_channels)
    if not channels:
        raise ValueError("No FERS channels were supplied for synchronization.")

    order = channels
    if preferred_channel in channels:
        order = [preferred_channel] + [c for c in channels if c != preferred_channel]

    errors = []
    for channel in order:
        print(f"Trying synchronization with Channel {channel}...")
        try:
            result = find_fers_synchronization_point(
                processed_channels[channel],
                print_diagnostics=False,
            )
        except ValueError as error:
            errors.append(f"CH{channel}: {error}")
            continue

        result["reference_channel"] = channel
        print(f"Synchronization reference channel: {channel}")
        _print_sync_diagnostics(result["diagnostics"], True)
        return result

    raise ValueError(
        "No active FERS channel produced a valid synchronization point. "
        + "; ".join(errors)
    )


def create_fers_plot(
    fers_plot_data: pd.DataFrame,
    run_name: str,
    output_plot: Path,
    channel: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        fers_plot_data["Relative_Time_s"],
        fers_plot_data["Rate_Hz"],
        color="tab:green",
        linestyle="None",
        marker=".",
        markersize=1.5,
        label=f"FERS Channel {channel} Rate",
    )
    ax.set_title(f"{run_name}: FERS Channel {channel} Counting Rate vs Time")
    ax.set_xlabel("Time from Start of FERS Acquisition (s)")
    ax.set_ylabel("Counting Rate (Hz)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_motor_plot(
    motion: pd.DataFrame,
    run_name: str,
    output_plot: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        motion["Motor_Time_s"], motion["X_cm"],
        color="tab:blue", linestyle="--", linewidth=1.8, label="X position"
    )
    ax.plot(
        motion["Motor_Time_s"], motion["Y_cm"],
        color="tab:orange", linestyle="--", linewidth=1.8, label="Y position"
    )
    ax.set_title(f"{run_name}: Motor Position vs Time")
    ax.set_xlabel("Time from Start of Motor Recording (s)")
    ax.set_ylabel("Position (cm)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_synchronized_plot(
    motor_scan: pd.DataFrame,
    fers_plot_data: pd.DataFrame,
    synchronization_time_s: float,
    config_run: str,
    fers_run: str,
    output_plot: Path,
    channel: int,
) -> None:
    fig, motor_axis = plt.subplots(figsize=(14, 7))
    fers_axis = motor_axis.twinx()

    x_line = motor_axis.plot(
        motor_scan["Aligned_Time_s"], motor_scan["X_cm"],
        color="tab:blue", linestyle="--", linewidth=2.0, label="X Position"
    )
    y_line = motor_axis.plot(
        motor_scan["Aligned_Time_s"], motor_scan["Y_cm"],
        color="tab:orange", linestyle="--", linewidth=2.0, label="Y Position"
    )
    fers_line = fers_axis.plot(
        fers_plot_data["Relative_Time_s"], fers_plot_data["Rate_Hz"],
        color="tab:green", linestyle="None", marker=".", markersize=1.5,
        label=f"FERS Channel {channel} Rate"
    )
    sync_line = motor_axis.axvline(
        synchronization_time_s,
        color="black", linestyle=":", linewidth=1.2,
        label="Synchronization Point"
    )

    motor_axis.set_xlabel("Synchronized Time (s)")
    motor_axis.set_ylabel("Motor Position (cm)")
    fers_axis.set_ylabel(f"FERS Channel {channel} Rate (Hz)", color="tab:green")
    fers_axis.tick_params(axis="y", labelcolor="tab:green")
    motor_axis.grid(True, linestyle=":", alpha=0.4)
    motor_axis.set_title(
        "Synchronized Motor Position and FERS Counting Rate\n"
        f"Configuration {config_run} | FERS {fers_run} | Channel {channel}"
    )

    lines = x_line + y_line + fers_line + [sync_line]
    motor_axis.legend(lines, [line.get_label() for line in lines], loc="upper right")
    fig.tight_layout()
    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)
