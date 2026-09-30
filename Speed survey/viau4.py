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


SITE_NAME = "viau"
SQLITE_PATTERN = "*.sqlite"

# Enter the two endpoints of each speed line in world coordinates (metres).
# Example: LINE1 = ((12.34, 5.67), (18.90, 5.67))
LINE1 = ((22.8145, 23.3973),(37.5596, 29.4688))
LINE2 = ((21.0798, 32.2680),(34.6816, 38.0635))

FPS = 30.0
DISTANCE_M = 8.93
MOTOR_VEHICLE_TYPES = {1, 3, 5, 6}

MIN_VALID_SPEED_KMH = 1.0
MAX_VALID_SPEED_KMH = 120.0

USER_TYPE_NAMES = {
    1: "car",
    3: "motorcyclist",
    5: "bus",
    6: "truck",
}

# LINE1 is the upper screenline and LINE2 is the lower screenline in world.png.
# Change only these labels if the lane/order correspondence is later reversed.
DIRECTION_LABELS = {
    "line1_to_line2": "West",
    "line2_to_line1": "East",
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
    point_results = []

    # Diagnostics for checking where trajectories are lost.
    line1_crossers = 0
    line2_crossers = 0
    both_line_crossers = 0
    compatible_pair_crossers = 0

    paired_by_order = {
        "line1_to_line2": 0,
        "line2_to_line1": 0,
    }
    rejected_by_speed = {
        "line1_to_line2": 0,
        "line2_to_line1": 0,
    }

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

        if len(instants_1) > 0:
            line1_crossers += 1

        if len(instants_2) > 0:
            line2_crossers += 1

        if len(instants_1) == 0 or len(instants_2) == 0:
            continue

        both_line_crossers += 1

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

        compatible_pair_crossers += 1

        # If noise creates several crossings, use the closest valid pair.
        frame_difference, instant_1, instant_2 = min(
            candidates,
            key=lambda item: item[0],
        )

        travel_time_s = frame_difference / FPS
        speed_kmh = DISTANCE_M / travel_time_s * 3.6

        crossing_order = (
            "line1_to_line2"
            if instant_1 < instant_2
            else "line2_to_line1"
        )
        paired_by_order[crossing_order] += 1

        if not MIN_VALID_SPEED_KMH <= speed_kmh <= MAX_VALID_SPEED_KMH:
            rejected_by_speed[crossing_order] += 1
            continue

        direction_label = DIRECTION_LABELS[crossing_order]

        start_instant = int(np.ceil(min(instant_1, instant_2)))
        end_instant = int(np.floor(max(instant_1, instant_2)))

        for instant in range(start_instant, end_instant + 1):
            position = obj.getPositionAtInstant(instant)
            velocity = obj.getVelocityAtInstant(instant)

            instantaneous_speed_kmh = velocity.norm2() * FPS * 3.6

            if (
                MIN_VALID_SPEED_KMH
                <= instantaneous_speed_kmh
                <= MAX_VALID_SPEED_KMH
            ):
                point_results.append(
                    {
                        "object_id": obj.getNum(),
                        "instant": instant,
                        "x": position.x,
                        "y": position.y,
                        "speed_kmh": instantaneous_speed_kmh,
                        "crossing_order": crossing_order,
                        "direction": direction_label,
                    }
                )

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
                "crossing_order": crossing_order,
                "direction": direction_label,
            }
        )

    print("\nScreenline diagnostics:")
    print(f"Vehicles crossing LINE1: {line1_crossers}")
    print(f"Vehicles crossing LINE2: {line2_crossers}")
    print(f"Vehicles crossing both lines: {both_line_crossers}")
    print(
        "Vehicles with compatible crossings:",
        compatible_pair_crossers,
    )

    print("\nDirection counts before speed filtering:")
    for order, count in paired_by_order.items():
        print(f"  {order}: {count}")

    print("\nRejected by speed limits:")
    for order, count in rejected_by_speed.items():
        print(f"  {order}: {count}")

    return (
        pd.DataFrame(results), 
        pd.DataFrame(point_results)
    )


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

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    colors = ["steelblue", "darkorange"]

    for ax, (crossing_order, direction_label), color in zip(
        axes,
        DIRECTION_LABELS.items(),
        colors,
    ):
        direction_speeds = (
            speed_results.loc[
                speed_results["crossing_order"] == crossing_order,
                "speed_kmh",
            ]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if direction_speeds.empty:
            ax.text(
                0.5,
                0.5,
                "No vehicles detected",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            mean_speed = direction_speeds.mean()
            median_speed = direction_speeds.median()
            percentile_85 = direction_speeds.quantile(0.85)

            ax.hist(
                direction_speeds,
                bins=bins,
                color=color,
                edgecolor="black",
                alpha=0.8,
            )
            ax.axvline(
                mean_speed,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean = {mean_speed:.1f} km/h",
            )
            ax.axvline(
                median_speed,
                color="darkorange",
                linestyle=":",
                linewidth=2,
                label=f"Median = {median_speed:.1f} km/h",
            )
            ax.axvline(
                percentile_85,
                color="green",
                linestyle="-.",
                linewidth=2,
                label=f"85th percentile = {percentile_85:.1f} km/h",
            )
            ax.legend(fontsize=9)

        ax.set_title(f"{direction_label}\nn = {len(direction_speeds)}")
        ax.set_xlabel("Distance-based speed (km/h)")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Number of vehicles")
    fig.suptitle(f"Vehicle speed distributions by direction at {SITE_NAME}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

def save_speed_heatmap(
    point_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot spatial mean speed using hexagonal bins."""

    if point_results.empty:
        raise RuntimeError("No trajectory points available for heatmap.")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    x_min, x_max = point_results["x"].min(), point_results["x"].max()
    y_min, y_max = point_results["y"].min(), point_results["y"].max()
    speed_min = point_results["speed_kmh"].min()
    speed_max = point_results["speed_kmh"].max()
    last_hexbin = None

    for ax, (crossing_order, direction_label) in zip(
        axes,
        DIRECTION_LABELS.items(),
    ):
        direction_points = point_results.loc[
            point_results["crossing_order"] == crossing_order
        ]

        if direction_points.empty:
            ax.text(
                0.5,
                0.5,
                "No trajectory points detected",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            last_hexbin = ax.hexbin(
                direction_points["x"],
                direction_points["y"],
                C=direction_points["speed_kmh"],
                gridsize=40,
                reduce_C_function=np.mean,
                mincnt=5,
                cmap="turbo",
                edgecolors="none",
                extent=(x_min, x_max, y_min, y_max),
                vmin=speed_min,
                vmax=speed_max,
            )

        ax.set_xlabel("World X (m)")
        ax.set_title(direction_label)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel("World Y (m)")
    fig.suptitle(f"Vehicle speed heatmaps by direction at {SITE_NAME}")

    if last_hexbin is not None:
        colorbar = fig.colorbar(last_hexbin, ax=axes)
        colorbar.set_label("Mean instantaneous speed (km/h)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


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
            "viau_speed_distribution.pdf"
        )

    world_points = get_speed_line_points()
    motor_vehicles = load_motor_vehicles(sqlite_paths)
    speed_results, point_results = measure_speeds(
         motor_vehicles,
         world_points,
    )

    print("\nVehicles crossing both lines:", len(speed_results))

    if speed_results.empty:
        raise RuntimeError(
            "No vehicle crossed both lines. Check LINE1 and LINE2."
        )

    print("\nSpeed summary:")
    print(speed_results["speed_kmh"].describe())

    print("\nSpeed summary by direction:")
    print(
        speed_results.groupby("direction")["speed_kmh"].agg(
            vehicles="count",
            mean="mean",
            median="median",
            percentile_85=lambda values: values.quantile(0.85),
        )
    )

    save_speed_figure(speed_results, output_path)
    print(f"\nPDF saved to: {output_path.resolve()}")

    heatmap_output_path = (
        input_folder / "viau_speed_heatmap.pdf"
    )

    save_speed_heatmap(
        point_results,
        heatmap_output_path,
    )

    print(
        f"Heatmap saved to: {heatmap_output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
