import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Project folders
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
PLOTS_DIR = PROJECT_DIR / "plots"

RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ============================================================
# Ask which run should be analyzed
# ============================================================

def ask_for_run() -> str:
    """Ask the user for a run number and return a name such as Run10."""

    user_input = input(
        "Which run do you want to analyze? "
        "Enter a number such as 10 or a name such as Run10: "
    ).strip()

    if not user_input:
        print("ERROR: No run was entered.")
        sys.exit(1)

    if user_input.lower().startswith("run"):
        number_part = user_input[3:]
    else:
        number_part = user_input

    if not number_part.isdigit():
        print("ERROR: The run must be entered as 10 or Run10.")
        sys.exit(1)

    return f"Run{number_part}"


# ============================================================
# Locate and check the files belonging to the run
# ============================================================

def get_run_files(run_name: str) -> dict[str, Path]:
    """Return the expected files for the selected run."""

    return {
        "CSV data file": DATA_DIR / f"{run_name}_list.csv",
        "Info file": DATA_DIR / f"{run_name}_Info.txt",
        "Service info file": DATA_DIR / f"{run_name}_ServiceInfo.txt",
        "DAT file": DATA_DIR / f"{run_name}_list.dat",
    }


def check_run_files(run_files: dict[str, Path]) -> None:
    """Print whether each expected run file exists."""

    print("\nFile check")
    print("-" * 65)

    for description, file_path in run_files.items():
        status = "FOUND" if file_path.exists() else "MISSING"
        print(f"{description:<22}: {status:<7} {file_path}")


# ============================================================
# Read the RunXX_Info.txt file
# ============================================================

def read_info_file(info_file: Path) -> dict[str, str]:
    """Read selected acquisition settings from the Info file."""

    run_info = {
        "Start Time": "Not found",
        "Stop Time": "Not found",
        "Elapsed time": "Not found",
        "PtrgPeriod": "Not found",
        "CountingMode": "Not found",
        "BunchTrgSource": "Not found",
    }

    if not info_file.exists():
        print(f"\nWarning: Info file not found: {info_file}")
        return run_info

    with info_file.open("r", encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = raw_line.strip()

            if line.startswith("Start Time:"):
                run_info["Start Time"] = line.split(":", 1)[1].strip()

            elif line.startswith("Stop Time:"):
                run_info["Stop Time"] = line.split(":", 1)[1].strip()

            elif line.startswith("Elapsed time") and "=" in line:
                run_info["Elapsed time"] = line.split("=", 1)[1].strip()

            elif line.startswith("PtrgPeriod"):
                run_info["PtrgPeriod"] = (
                    line.split("#", 1)[0]
                    .replace("PtrgPeriod", "", 1)
                    .strip()
                )

            elif line.startswith("CountingMode"):
                run_info["CountingMode"] = (
                    line.split("#", 1)[0]
                    .replace("CountingMode", "", 1)
                    .strip()
                )

            elif line.startswith("BunchTrgSource"):
                run_info["BunchTrgSource"] = (
                    line.split("#", 1)[0]
                    .replace("BunchTrgSource", "", 1)
                    .strip()
                )

    return run_info


# ============================================================
# Read and validate the CSV file
# ============================================================

def read_csv_file(csv_file: Path) -> pd.DataFrame:
    """Read the Janus CSV and verify that required columns exist."""

    if not csv_file.exists():
        print(f"\nERROR: Main CSV file is missing: {csv_file}")
        sys.exit(1)

    # Janus metadata lines begin with //.
    # comment="/" causes pandas to skip those lines.
    data = pd.read_csv(csv_file, comment="/")

    required_columns = {"TStamp_us", "CH_Id", "Counts"}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        print(
            "\nERROR: The CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
        sys.exit(1)

    if data.empty:
        print("\nERROR: The CSV contains no data rows.")
        sys.exit(1)

    return data


# ============================================================
# Analyze channels
# ============================================================

def analyze_channels(data: pd.DataFrame) -> dict:
    """Calculate totals and statistics for detector channels."""

    counts_per_channel = (
        data.groupby("CH_Id", sort=True)["Counts"]
        .sum()
        .sort_index()
    )

    max_channel = int(counts_per_channel.idxmax())
    max_channel_counts = int(counts_per_channel.max())

    return {
        "counts_per_channel": counts_per_channel,
        "number_of_channels": int(data["CH_Id"].nunique()),
        "max_channel": max_channel,
        "max_channel_counts": max_channel_counts,
    }


# ============================================================
# Analyze triggers
# ============================================================

def analyze_triggers(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Treat each unique timestamp as one periodic trigger read.

    For every trigger:
    - sum Counts over all channels;
    - calculate absolute time;
    - calculate delta time from the previous trigger;
    - calculate rate = counts / delta time.

    For the first trigger, delta time is the interval from acquisition
    start to the first trigger.
    """

    counts_per_trigger = (
        data.groupby("TStamp_us", sort=True)["Counts"]
        .sum()
        .reset_index()
    )

    counts_per_trigger.columns = ["TStamp_us", "Total_Counts"]

    counts_per_trigger["Absolute_Time_s"] = (
        counts_per_trigger["TStamp_us"] / 1_000_000
    )

    counts_per_trigger["Delta_Time_s"] = (
        counts_per_trigger["Absolute_Time_s"].diff()
    )

    # The first timestamp is measured from acquisition start.
    counts_per_trigger.loc[0, "Delta_Time_s"] = (
        counts_per_trigger.loc[0, "Absolute_Time_s"]
    )

    # Protect against invalid or zero time differences.
    valid_delta = counts_per_trigger["Delta_Time_s"] > 0

    counts_per_trigger["Rate_Hz"] = float("nan")
    counts_per_trigger.loc[valid_delta, "Rate_Hz"] = (
        counts_per_trigger.loc[valid_delta, "Total_Counts"]
        / counts_per_trigger.loc[valid_delta, "Delta_Time_s"]
    )

    calculated_trigger_period_s = float(
        counts_per_trigger["Delta_Time_s"].mean()
    )

    first_trigger_time_us = float(
        counts_per_trigger["TStamp_us"].iloc[0]
    )

    last_trigger_time_us = float(
        counts_per_trigger["TStamp_us"].iloc[-1]
    )

    duration_first_to_last_s = (
        last_trigger_time_us - first_trigger_time_us
    ) / 1_000_000

    min_row = counts_per_trigger.loc[
        counts_per_trigger["Total_Counts"].idxmin()
    ]

    max_row = counts_per_trigger.loc[
        counts_per_trigger["Total_Counts"].idxmax()
    ]

    trigger_statistics = {
        "number_of_triggers": len(counts_per_trigger),
        "calculated_trigger_period_s": calculated_trigger_period_s,
        "first_trigger_time_us": first_trigger_time_us,
        "last_trigger_time_us": last_trigger_time_us,
        "time_before_first_trigger_s": first_trigger_time_us / 1_000_000,
        "duration_first_to_last_s": duration_first_to_last_s,
        "total_pulses": int(counts_per_trigger["Total_Counts"].sum()),
        "average_pulses_per_trigger": float(
            counts_per_trigger["Total_Counts"].mean()
        ),
        "minimum_count": int(min_row["Total_Counts"]),
        "minimum_count_time_us": float(min_row["TStamp_us"]),
        "maximum_count": int(max_row["Total_Counts"]),
        "maximum_count_time_us": float(max_row["TStamp_us"]),
        "average_rate_hz": float(
            counts_per_trigger["Rate_Hz"].mean()
        ),
    }

    return counts_per_trigger, trigger_statistics


# ============================================================
# Analyze zero and nonzero rows
# ============================================================

def analyze_zero_counts(data: pd.DataFrame) -> dict:
    """Separate rows containing zero counts from nonzero rows."""

    zero_mask = data["Counts"] == 0

    return {
        "zero_count_rows": int(zero_mask.sum()),
        "nonzero_count_rows": int((~zero_mask).sum()),
        "total_pulses_excluding_zero_rows": int(
            data.loc[~zero_mask, "Counts"].sum()
        ),
    }


# ============================================================
# Save output tables
# ============================================================

def save_tables(
    run_name: str,
    counts_per_channel: pd.Series,
    trigger_table: pd.DataFrame,
) -> dict[str, Path]:
    """Save channel and trigger tables as CSV files."""

    channel_csv = RESULTS_DIR / f"{run_name}_counts_per_channel.csv"
    trigger_csv = RESULTS_DIR / f"{run_name}_counts_per_trigger.csv"

    counts_per_channel.to_csv(
        channel_csv,
        header=["Total_Counts"],
    )

    trigger_table.to_csv(
        trigger_csv,
        index=False,
        columns=[
            "TStamp_us",
            "Absolute_Time_s",
            "Delta_Time_s",
            "Total_Counts",
            "Rate_Hz",
        ],
    )

    return {
        "channel_csv": channel_csv,
        "trigger_csv": trigger_csv,
    }


# ============================================================
# Save trigger-by-trigger report
# ============================================================

def save_trigger_report(
    run_name: str,
    run_info: dict[str, str],
    trigger_table: pd.DataFrame,
    trigger_statistics: dict,
) -> Path:
    """Save one row for every trigger in a readable text report."""

    report_file = RESULTS_DIR / f"{run_name}_trigger_counts.txt"

    with report_file.open("w", encoding="utf-8") as file:
        file.write("=" * 88 + "\n")
        file.write("                     Trigger-by-Trigger Report\n")
        file.write("=" * 88 + "\n\n")

        file.write(f"Run Name                  : {run_name}\n")
        file.write(
            f"Trigger Period from Info  : {run_info['PtrgPeriod']}\n"
        )
        file.write(
            "Calculated Trigger Period : "
            f"{trigger_statistics['calculated_trigger_period_s']:.6f} s\n\n"
        )

        file.write(
            f"{'Trigger':<10}"
            f"{'Absolute Time (s)':<22}"
            f"{'Delta Time (s)':<20}"
            f"{'Counts':<16}"
            f"{'Rate (Hz)':<16}\n"
        )

        file.write("-" * 88 + "\n")

        for trigger_number, row in enumerate(
            trigger_table.itertuples(index=False),
            start=1,
        ):
            file.write(
                f"{trigger_number:<10}"
                f"{row.Absolute_Time_s:<22.6f}"
                f"{row.Delta_Time_s:<20.6f}"
                f"{int(row.Total_Counts):<16}"
                f"{row.Rate_Hz:<16.2f}\n"
            )

    return report_file


# ============================================================
# Save run summary
# ============================================================

def save_summary(
    run_name: str,
    csv_file: Path,
    run_files: dict[str, Path],
    run_info: dict[str, str],
    data: pd.DataFrame,
    channel_statistics: dict,
    trigger_statistics: dict,
    zero_statistics: dict,
) -> Path:
    """Save the main human-readable run summary."""

    summary_file = RESULTS_DIR / f"{run_name}_summary.txt"

    with summary_file.open("w", encoding="utf-8") as file:
        file.write("=" * 52 + "\n")
        file.write("              FERS / Janus Run Summary\n")
        file.write("=" * 52 + "\n\n")

        file.write(f"Run Name                 : {run_name}\n")
        file.write(f"Input File               : {csv_file}\n\n")

        file.write("----------- File Check -----------\n\n")

        for description, file_path in run_files.items():
            status = "FOUND" if file_path.exists() else "MISSING"
            file.write(
                f"{description:<22}: {status:<7} {file_path}\n"
            )

        file.write("\n----------- Run Information -----------\n\n")
        file.write(
            f"Start Time               : {run_info['Start Time']}\n"
        )
        file.write(
            f"Stop Time                : {run_info['Stop Time']}\n"
        )
        file.write(
            f"Elapsed Time             : {run_info['Elapsed time']}\n"
        )
        file.write(
            f"Trigger Period from Info : {run_info['PtrgPeriod']}\n"
        )
        file.write(
            f"Calculated Trigger Period: "
            f"{trigger_statistics['calculated_trigger_period_s']:.6f} s\n"
        )
        file.write(
            f"Counting Mode            : {run_info['CountingMode']}\n"
        )
        file.write(
            f"Bunch Trigger Source     : {run_info['BunchTrgSource']}\n"
        )

        file.write("\n----------- Acquisition Information -----------\n\n")
        file.write(f"Number of Rows           : {len(data)}\n")
        file.write(
            f"Number of Channels       : "
            f"{channel_statistics['number_of_channels']}\n"
        )
        file.write(
            f"Number of Triggers       : "
            f"{trigger_statistics['number_of_triggers']}\n"
        )

        file.write("\n----------- Counting Statistics -----------\n\n")
        file.write(
            f"Total Pulses             : "
            f"{trigger_statistics['total_pulses']:,}\n"
        )
        file.write(
            f"Average Pulses/Trigger   : "
            f"{trigger_statistics['average_pulses_per_trigger']:,.2f}\n"
        )
        file.write(
            f"Highest-count Channel    : "
            f"{channel_statistics['max_channel']}\n"
        )
        file.write(
            f"Highest Channel Counts   : "
            f"{channel_statistics['max_channel_counts']:,}\n"
        )
        file.write(
            f"Minimum Trigger Count    : "
            f"{trigger_statistics['minimum_count']:,}\n"
        )
        file.write(
            f"Minimum Count Time       : "
            f"{trigger_statistics['minimum_count_time_us']:.1f} us\n"
        )
        file.write(
            f"Maximum Trigger Count    : "
            f"{trigger_statistics['maximum_count']:,}\n"
        )
        file.write(
            f"Maximum Count Time       : "
            f"{trigger_statistics['maximum_count_time_us']:.1f} us\n"
        )

        file.write("\n----------- Timing -----------\n\n")
        file.write(
            f"First Trigger Time       : "
            f"{trigger_statistics['first_trigger_time_us']:.1f} us\n"
        )
        file.write(
            f"Last Trigger Time        : "
            f"{trigger_statistics['last_trigger_time_us']:.1f} us\n"
        )
        file.write(
            f"Time Before First Trigger: "
            f"{trigger_statistics['time_before_first_trigger_s']:.6f} s\n"
        )
        file.write(
            f"Duration First-to-Last   : "
            f"{trigger_statistics['duration_first_to_last_s']:.6f} s\n"
        )
        file.write(
            f"Average Trigger Rate     : "
            f"{trigger_statistics['average_rate_hz']:,.2f} Hz\n"
        )

        file.write("\n----------- Zero Count Discrimination -----------\n\n")
        file.write(
            f"Zero-count Rows          : "
            f"{zero_statistics['zero_count_rows']}\n"
        )
        file.write(
            f"Nonzero-count Rows       : "
            f"{zero_statistics['nonzero_count_rows']}\n"
        )
        file.write(
            f"Total Pulses Excl. Zeros : "
            f"{zero_statistics['total_pulses_excluding_zero_rows']:,}\n"
        )

    return summary_file


# ============================================================
# Generate plots
# ============================================================

def create_plots(
    run_name: str,
    counts_per_channel: pd.Series,
    trigger_table: pd.DataFrame,
) -> dict[str, Path]:
    """Generate channel, pulse-time, and rate-time plots."""

    channel_plot = PLOTS_DIR / f"{run_name}_counts_per_channel.png"
    pulse_plot = PLOTS_DIR / f"{run_name}_pulses_vs_time.png"
    rate_plot = PLOTS_DIR / f"{run_name}_rate_vs_time.png"

    # Plot 1: total counts per channel
    plt.figure(figsize=(11, 6))
    counts_per_channel.plot(kind="bar")
    plt.xlabel("Channel ID")
    plt.ylabel("Total Counts")
    plt.title(f"{run_name}: Total Counts per SiPM Channel")
    plt.tight_layout()
    plt.savefig(channel_plot)
    plt.close()

    # Plot 2: pulses at each trigger, displayed as dots
    plt.figure(figsize=(11, 6))
    plt.plot(
    trigger_table["Absolute_Time_s"],
    trigger_table["Total_Counts"],
    linestyle="None",
    marker=".",
    markersize=0.2,
    color="tab:blue",
    )

    plt.xlabel("Time from Acquisition Start (s)")
    plt.ylabel("Number of Pulses")
    plt.title(f"{run_name}: Number of Pulses vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pulse_plot)
    plt.close()

    # Plot 3: rate at each trigger, displayed as dots
    plt.figure(figsize=(11, 6))
    plt.plot(
    trigger_table["Absolute_Time_s"],
    trigger_table["Rate_Hz"],
    linestyle="None",
    marker=".",
    markersize=2,
    color="tab:green",
    )

    plt.yscale("log")
    plt.xlabel("Time from Acquisition Start (s)")
    plt.ylabel("Rate (Hz)")
    plt.title(f"{run_name}: Counting Rate vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(rate_plot)
    plt.close()

    return {
        "channel_plot": channel_plot,
        "pulse_plot": pulse_plot,
        "rate_plot": rate_plot,
    }


# ============================================================
# Main program
# ============================================================

def main() -> None:
    """Run the full analysis workflow."""

    run_name = ask_for_run()
    run_files = get_run_files(run_name)

    check_run_files(run_files)

    csv_file = run_files["CSV data file"]
    info_file = run_files["Info file"]

    run_info = read_info_file(info_file)
    data = read_csv_file(csv_file)

    print(f"\nSuccessfully loaded {run_name}")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {list(data.columns)}")

    channel_statistics = analyze_channels(data)

    trigger_table, trigger_statistics = analyze_triggers(data)

    zero_statistics = analyze_zero_counts(data)

    saved_tables = save_tables(
        run_name,
        channel_statistics["counts_per_channel"],
        trigger_table,
    )

    trigger_report = save_trigger_report(
        run_name,
        run_info,
        trigger_table,
        trigger_statistics,
    )

    summary_file = save_summary(
        run_name,
        csv_file,
        run_files,
        run_info,
        data,
        channel_statistics,
        trigger_statistics,
        zero_statistics,
    )

    plot_files = create_plots(
        run_name,
        channel_statistics["counts_per_channel"],
        trigger_table,
    )

    print("\n" + "=" * 65)
    print("Analysis completed successfully")
    print("=" * 65)
    print(f"Run analyzed       : {run_name}")
    print(f"Summary            : {summary_file}")
    print(f"Trigger report     : {trigger_report}")
    print(f"Trigger CSV        : {saved_tables['trigger_csv']}")
    print(f"Channel CSV        : {saved_tables['channel_csv']}")
    print(f"Channel plot       : {plot_files['channel_plot']}")
    print(f"Pulses-time plot   : {plot_files['pulse_plot']}")
    print(f"Rate-time plot     : {plot_files['rate_plot']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
