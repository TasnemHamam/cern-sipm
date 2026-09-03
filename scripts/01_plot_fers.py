"""Plot FERS counting rate for one selected channel or all signal channels separately."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_utils as au


def ask_run_name() -> str:
    while True:
        try:
            return au.normalize_run_name(input("Enter the FERS run number: "))
        except ValueError as error:
            print(error)


def run_fers_analysis(run_name: str, channel_selection: str | None = None) -> None:
    run_name = au.normalize_run_name(run_name)
    fers_file = au.DATA_DIR / f"{run_name}_list.csv"
    au.ensure_output_dirs()

    active_channels = au.detect_active_channels(fers_file)
    print("Detected FERS channels with signal: " + ", ".join(map(str, active_channels)))

    if channel_selection is None:
        selected_channels, _ = au.ask_channel_selection(active_channels)
    else:
        selected_channels, _ = au.normalize_channel_selection(
            channel_selection, active_channels
        )

    tables, _ = au.prepare_fers_channels(fers_file, selected_channels)

    for channel in selected_channels:
        plot_data = tables[channel].dropna(subset=["Rate_Hz"]).copy()
        output_csv = au.RESULTS_DIR / f"{run_name}_CH{channel}_fers_rate.csv"
        output_plot = au.PLOTS_DIR / f"{run_name}_CH{channel}_fers_rate_vs_time.png"

        columns = [
            c for c in [
                "Trg_Id", "TStamp_us", "Relative_Time_s",
                "Delta_Time_s", "Counts", "Rate_Hz"
            ] if c in plot_data.columns
        ]
        plot_data[columns].to_csv(output_csv, index=False)
        au.create_fers_plot(plot_data, run_name, output_plot, channel)

        print(f"\nChannel {channel} completed.")
        print(f"Results: {output_csv}")
        print(f"Plot:    {output_plot}")


def main() -> None:
    if len(sys.argv) == 1:
        run_fers_analysis(ask_run_name())
    elif len(sys.argv) == 2:
        run_fers_analysis(sys.argv[1])
    elif len(sys.argv) == 3:
        run_fers_analysis(sys.argv[1], sys.argv[2])
    else:
        sys.exit("Usage: python3 scripts/01_plot_fers.py [RUN] [CHANNEL|all]")


if __name__ == "__main__":
    main()

