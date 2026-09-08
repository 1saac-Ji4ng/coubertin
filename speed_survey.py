#!/usr/bin/env python3
"""Select two lines, measure Viau vehicle speeds, and save a PDF figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Traffic Intelligence is provided by the activated Python environment.
from trafficintelligence import moving, storage


FPS = 30.0
DISTANCE_M = 19.96
UNITS_PER_PIXEL = 0.0520833333
MOTOR_VEHICLE_TYPES = {1, 3, 5, 6}

USER_TYPE_NAMES = {
    1: "car",
    3: "motorcyclist",
    5: "bus",
    6: "truck",
}


def select_speed_lines(world_image_path: Path) -> np.ndarray:
    """Click line 1 endpoints, followed by line 2 endpoints."""
    if not world_image_path.is_file():
        raise FileNotFoundError(
            f"World image not found: {world_image_path}"
        )

    image = plt.imread(world_image_path)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image)
    ax.set_title(
        "Click 4 points: Line 1 endpoint 1, Line 1 endpoint 2, "
        "Line 2 endpoint 1, Line 2 endpoint 2"
    )

    print("Click four points in world.png.")
    pixel_points = np.asarray(plt.ginput(4, timeout=-1), dtype=float)

    if pixel_points.shape != (4, 2):
        raise RuntimeError("Exactly four points must be selected.")

    ax.plot(pixel_points[:2, 0], pixel_points[:2, 1], "b-", linewidth=3)
    ax.plot(pixel_points[2:, 0], pixel_points[2:, 1], "r-", linewidth=3)
    fig.canvas.draw()

    print("The selected lines are displayed. Close the window to continue.")
    plt.show()

    world_points = pixel_points * UNITS_PER_PIXEL
    print("\nWorld coordinates:")
    print(world_points)
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
    """Calculate speeds for vehicles that cross both selected lines."""
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
    plt.title(f"Vehicle speed distribution at Viau\nn = {len(speeds)}")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Select two Viau speed lines, calculate vehicle speeds, "
            "and save the speed-distribution PDF."
        )
    )
    parser.add_argument(
        "--world-image",
        required=True,
        type=Path,
        help="Path to world.png.",
    )
    parser.add_argument(
        "--sqlite",
        required=True,
        nargs="+",
        type=Path,
        help="One or more trajectory SQLite files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output PDF path. By default, it is saved beside the first "
            "SQLite file."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    output_path = args.output
    if output_path is None:
        output_path = args.sqlite[0].with_name(
            "viau_speed_distribution.pdf"
        )

    world_points = select_speed_lines(args.world_image)
    motor_vehicles = load_motor_vehicles(args.sqlite)
    speed_results = measure_speeds(motor_vehicles, world_points)

    print("\nVehicles crossing both lines:", len(speed_results))

    if speed_results.empty:
        raise RuntimeError(
            "No vehicle crossed both lines. Check the selected lines."
        )

    print("\nSpeed summary:")
    print(speed_results["speed_kmh"].describe())

    save_speed_figure(speed_results, output_path)
    print(f"\nPDF saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
