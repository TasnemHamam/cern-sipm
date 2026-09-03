"""Run the complete source-scan analysis with one channel choice."""

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_utils as au

SCRIPT_DIR = Path(__file__).resolve().parent


def ask_run_name(prompt: str) -> str:
    while True:
        try:
            return au.normalize_run_name(input(prompt))
        except ValueError as error:
            print(error)


def run_script(script_name: str, *arguments: str) -> None:
    command = [sys.executable, str(SCRIPT_DIR / script_name), *arguments]
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} stopped with exit code {result.returncode}.")


def main() -> None:
    print("\nComplete Source-Scan Analysis")
    print("=============================")

    motor_run = ask_run_name("Enter the motor configuration run number: ")
    fers_run = ask_run_name("Enter the FERS run number: ")

    fers_file = au.DATA_DIR / f"{fers_run}_list.csv"

    try:
        active_channels = au.detect_active_channels(fers_file)
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"\nError:\n{error}")

    print("\nDetected FERS channels with signal: " + ", ".join(map(str, active_channels)))
    selected_channels, all_mode = au.ask_channel_selection(active_channels)

    channel_argument = "all" if all_mode else str(selected_channels[0])

    print("\n[1/4] FERS counting-rate plot(s)")
    run_script("01_plot_fers.py", fers_run, channel_argument)

    print("\n[2/4] Motor-position plot")
    run_script("03_plot_motor_position.py", motor_run)

    print("\n[3/4] Synchronized motor/FERS plot(s)")
    run_script(
        "04_plot_synchronized_scan.py",
        motor_run,
        fers_run,
        channel_argument,
    )

    print("\n[4/4] Detector-response map(s)")
    run_script(
        "05_detector_response_map.py",
        motor_run,
        fers_run,
        channel_argument,
    )

    print("\nComplete analysis finished.")


if __name__ == "__main__":
    main()

