#!/usr/bin/env python3
"""Measure vehicle speeds with predefined lines and save a PDF figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trafficintelligence import moving, storage


SITE_NAME = ""
SQLITE_PATTERN = "*.sqlite"

# Enter the two endpoints of each speed line in world coordinates (metres).
# Example: LINE1 = ((12.34, 5.67), (18.90, 5.67))
LINE1 = ((),())
LINE2 = ((),())

FPS = 30.0
DISTANCE_M = None
MOTOR_VEHICLE_TYPES = {1, 3, 5, 6}

MIN_VALID_SPEED_KMH = 1.0
MAX_VALID_SPEED_KMH = 120.0

USER_TYPE_NAMES = {
    1: "car",
    3: "motorcyclist",
    5: "bus",
    6: "truck",
}


def get_speed_line_points() -> np.ndarray:
    """Return the predefined speed-line endpoints as a 4-by-2 array."""
    if LINE1 is None or LINE2 is None:
        raise ValueError(
            "Set LINE1 and LINE2 near the top of the script before running it."
        )

    world_points = np.asarray([*LINE1, *LINE2], dtype=float)
    if world_points.shape != (4, 2):
        raise ValueError(
            "LINE1 and LINE2 must each contain two (x, y) endpoints."
        )
    return world_points


def load_motor_vehicles(sqlite_paths: list[Path]):
    """Load motor-vehicle trajectories from one or more SQLite files."""
    motor_vehicles = []

    for sqlite_path in sqlite_paths:
        if not sqlite_path.is_file():
            raise FileNotFoundError(
                f"SQLite file not found: {sqlite_path}"
            )

        objects = storage.loadTrajectoriesFromSqlite(
            str(sqlite_path),
            "object",
        )

        motor_vehicles.extend(
            obj
            for obj in objects
            if obj.getUserType() in MOTOR_VEHICLE_TYPES
        )

        print(f"Loaded: {sqlite_path}")

    print("Motor-vehicle trajectories:", len(motor_vehicles))
    return motor_vehicles


def measure_speeds(
    motor_vehicles,
    world_points: np.ndarray,
) -> pd.DataFrame:
    """Calculate speeds for vehicles that cross both predefined lines."""
    line1_p1 = moving.Point(*world_points[0])
    line1_p2 = moving.Point(*world_points[1])
    line2_p1 = moving.Point(*world_points[2])
    line2_p2 = moving.Point(*world_points[3])

    results = []

    for obj in motor_vehicles:
        instants_1, _, orientations_1 = (
            obj.getInstantsCrossingSegment(
                line1_p1,
                line1_p2,
                True,
            )
        )

        instants_2, _, orientations_2 = (
            obj.getInstantsCrossingSegment(
                line2_p1,
                line2_p2,
                True,
            )
        )

        if len(instants_1) == 0 or len(instants_2) == 0:
            continue

        candidates = []

        for i, instant_1 in enumerate(instants_1):
            for j, instant_2 in enumerate(instants_2):
                if bool(orientations_1[i]) != bool(orientations_2[j]):
                    continue

                frame_difference = abs(
                    float(instant_2) - float(instant_1)
                )

                if frame_difference > 0:
                    candidates.append(
                        (
                            frame_difference,
                            float(instant_1),
                            float(instant_2),
                        )
                    )

        if not candidates:
            continue

        # If noise creates several crossings, use the closest valid pair.
        frame_difference, instant_1, instant_2 = min(
            candidates,
            key=lambda item: item[0],
        )

        travel_time_s = frame_difference / FPS
        speed_kmh = DISTANCE_M / travel_time_s * 3.6

        if not MIN_VALID_SPEED_KMH <= speed_kmh <= MAX_VALID_SPEED_KMH:
            continue

        results.append(
            {
                "object_id": obj.getNum(),
                "road_user_type": obj.getUserType(),
                "road_user_name": USER_TYPE_NAMES.get(
                    obj.getUserType(),
                    "other",
                ),
                "instant_line1": instant_1,
                "instant_line2": instant_2,
                "frame_difference": frame_difference,
                "travel_time_s": travel_time_s,
                "distance_m": DISTANCE_M,
                "speed_kmh": speed_kmh,
                "crossing_order": (
                    "line1_to_line2"
                    if instant_1 < instant_2
                    else "line2_to_line1"
                ),
            }
        )

    return pd.DataFrame(results)


def save_speed_figure(
    speed_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the speed distribution and save it as a PDF."""
    speeds = (
        speed_results["speed_kmh"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if speeds.empty:
        raise RuntimeError("No valid numeric speed values were produced.")

    mean_speed = speeds.mean()
    median_speed = speeds.median()
    percentile_85 = speeds.quantile(0.85)

    bin_width = 5.0
    lower_limit = np.floor(speeds.min() / bin_width) * bin_width
    upper_limit = np.ceil(speeds.max() / bin_width) * bin_width

    if upper_limit <= lower_limit:
        upper_limit = lower_limit + bin_width

    bins = np.arange(
        lower_limit,
        upper_limit + bin_width,
        bin_width,
    )

    plt.figure(figsize=(8, 5))
    plt.hist(
        speeds,
        bins=bins,
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
    )
    plt.axvline(
        mean_speed,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_speed:.1f} km/h",
    )
    plt.axvline(
        median_speed,
        color="darkorange",
        linestyle=":",
        linewidth=2,
        label=f"Median = {median_speed:.1f} km/h",
    )
    plt.axvline(
        percentile_85,
        color="green",
        linestyle="-.",
        linewidth=2,
        label=f"85th percentile = {percentile_85:.1f} km/h",
    )
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Number of vehicles")
    plt.title(
        f"Vehicle speed distribution at {SITE_NAME}\nn = {len(speeds)}"
    )
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Measure vehicle speeds in all matching SQLite files."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path.cwd(),
        help="Folder containing the SQLite files (default: current folder).",
    )
    parser.add_argument(
        "--pattern",
        default=SQLITE_PATTERN,
        help=f"SQLite filename pattern (default: {SQLITE_PATTERN}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output PDF path (default: inside the input folder)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_folder = args.folder.resolve()
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {input_folder}")

    sqlite_paths = sorted(input_folder.glob(args.pattern))
    if not sqlite_paths:
        raise FileNotFoundError(
            f"No files matching '{args.pattern}' were found in {input_folder}"
        )

    print(f"Found {len(sqlite_paths)} SQLite file(s) in {input_folder}")

    output_path = args.output
    if output_path is None:
        output_path = input_folder / (
            ".pdf"
        )

    world_points = get_speed_line_points()
    motor_vehicles = load_motor_vehicles(sqlite_paths)
    speed_results = measure_speeds(motor_vehicles, world_points)

    print("\nVehicles crossing both lines:", len(speed_results))

    if speed_results.empty:
        raise RuntimeError(
            "No vehicle crossed both lines. Check LINE1 and LINE2."
        )

    print("\nSpeed summary:")
    print(speed_results["speed_kmh"].describe())

    save_speed_figure(speed_results, output_path)
    print(f"\nPDF saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
