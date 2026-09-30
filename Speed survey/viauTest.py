#!/usr/bin/env python3
"""Diagnose which Viau trajectories cross each predefined screenline."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trafficintelligence import moving, storage


SQLITE_PATTERN = "*.sqlite"

LINE1 = (
    (22.8145, 23.3973),
    (37.5596, 29.4688),
)

LINE2 = (
    (21.0798, 32.2680),
    (34.6816, 38.0635),
)

MOTOR_VEHICLE_TYPES = {1, 3, 5, 6}


def load_motor_vehicles(sqlite_paths: list[Path]) -> list[dict]:
    """Load motor-vehicle trajectories and retain their source filenames."""
    records = []

    for sqlite_path in sqlite_paths:
        objects = storage.loadTrajectoriesFromSqlite(
            str(sqlite_path),
            "object",
        )

        vehicles_in_file = 0
        for obj in objects:
            if obj.getUserType() not in MOTOR_VEHICLE_TYPES:
                continue

            records.append(
                {
                    "source_file": sqlite_path.name,
                    "object_id": obj.getNum(),
                    "object": obj,
                }
            )
            vehicles_in_file += 1

        print(
            f"Loaded {sqlite_path.name}: "
            f"{vehicles_in_file} motor-vehicle trajectories"
        )

    return records


def classify_trajectories(records: list[dict]) -> dict:
    """Classify trajectories by their intersections with LINE1 and LINE2."""
    line1_p1 = moving.Point(*LINE1[0])
    line1_p2 = moving.Point(*LINE1[1])
    line2_p1 = moving.Point(*LINE2[0])
    line2_p2 = moving.Point(*LINE2[1])

    groups = {
        "line1_only": [],
        "line2_only": [],
        "both_lines": [],
        "neither_line": [],
    }

    line1_orientation_events = Counter()
    line2_orientation_events = Counter()
    paired_direction_records = []

    for record in records:
        obj = record["object"]

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

        crosses_line1 = len(instants_1) > 0
        crosses_line2 = len(instants_2) > 0

        line1_orientation_events.update(
            bool(value) for value in orientations_1
        )
        line2_orientation_events.update(
            bool(value) for value in orientations_2
        )

        diagnostic_record = {
            **record,
            "instants_line1": instants_1,
            "instants_line2": instants_2,
            "orientations_line1": orientations_1,
            "orientations_line2": orientations_2,
        }

        if crosses_line1 and crosses_line2:
            groups["both_lines"].append(diagnostic_record)

            candidates = []
            for i, instant_1 in enumerate(instants_1):
                for j, instant_2 in enumerate(instants_2):
                    orientation_1 = bool(orientations_1[i])
                    orientation_2 = bool(orientations_2[j])

                    if orientation_1 != orientation_2:
                        continue

                    frame_difference = abs(
                        float(instant_2) - float(instant_1)
                    )

                    if frame_difference > 0:
                        candidates.append(
                            {
                                "frame_difference": frame_difference,
                                "instant_line1": float(instant_1),
                                "instant_line2": float(instant_2),
                                "orientation": orientation_1,
                            }
                        )

            if candidates:
                best_pair = min(
                    candidates,
                    key=lambda item: item["frame_difference"],
                )

                crossing_order = (
                    "line1_to_line2"
                    if best_pair["instant_line1"]
                    < best_pair["instant_line2"]
                    else "line2_to_line1"
                )

                paired_direction_records.append(
                    {
                        "source_file": record["source_file"],
                        "object_id": record["object_id"],
                        "orientation": best_pair["orientation"],
                        "crossing_order": crossing_order,
                        "frame_difference": best_pair[
                            "frame_difference"
                        ],
                    }
                )

        elif crosses_line1:
            groups["line1_only"].append(diagnostic_record)
        elif crosses_line2:
            groups["line2_only"].append(diagnostic_record)
        else:
            groups["neither_line"].append(diagnostic_record)

    groups["line1_orientation_events"] = line1_orientation_events
    groups["line2_orientation_events"] = line2_orientation_events
    groups["paired_direction_records"] = paired_direction_records
    return groups


def get_trajectory_xy(obj) -> tuple[np.ndarray, np.ndarray]:
    """Return all available world-coordinate positions for one trajectory."""
    xs = []
    ys = []

    first_instant = int(obj.getFirstInstant())
    last_instant = int(obj.getLastInstant())

    for instant in range(first_instant, last_instant + 1):
        try:
            position = obj.getPositionAtInstant(instant)
        except Exception:
            continue

        if position is None:
            continue

        if np.isfinite(position.x) and np.isfinite(position.y):
            xs.append(position.x)
            ys.append(position.y)

    return np.asarray(xs), np.asarray(ys)


def sample_records(
    records: list[dict],
    maximum: int,
    seed: int = 42,
) -> list[dict]:
    """Take a deterministic sample to keep the PDF readable."""
    if len(records) <= maximum:
        return records

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(records),
        size=maximum,
        replace=False,
    )
    return [records[index] for index in indices]


def print_diagnostics(groups: dict) -> None:
    """Print counts and the orientation-to-direction relationship."""
    line1_only = groups["line1_only"]
    line2_only = groups["line2_only"]
    both_lines = groups["both_lines"]
    neither_line = groups["neither_line"]

    print("\nTrajectory classification:")
    print(f"LINE1 only: {len(line1_only)}")
    print(f"LINE2 only: {len(line2_only)}")
    print(f"Both lines: {len(both_lines)}")
    print(f"Neither line: {len(neither_line)}")
    print(
        "Vehicles crossing LINE1:",
        len(line1_only) + len(both_lines),
    )
    print(
        "Vehicles crossing LINE2:",
        len(line2_only) + len(both_lines),
    )

    print("\nLINE1 orientation events:")
    print(dict(groups["line1_orientation_events"]))

    print("\nLINE2 orientation events:")
    print(dict(groups["line2_orientation_events"]))

    paired_df = pd.DataFrame(groups["paired_direction_records"])
    if paired_df.empty:
        print("\nNo compatible two-line crossing pairs were found.")
        return

    table = pd.crosstab(
        paired_df["orientation"],
        paired_df["crossing_order"],
        margins=True,
    )

    print("\nOrientation versus crossing order:")
    print(table.to_string())


def save_trajectory_figure(
    groups: dict,
    output_path: Path,
    maximum: int,
) -> None:
    """Save LINE1-only, LINE2-only, and both-line trajectories to PDF."""
    plot_groups = [
        (
            "LINE1 only",
            groups["line1_only"],
            "tab:red",
        ),
        (
            "LINE2 only",
            groups["line2_only"],
            "tab:orange",
        ),
        (
            "Both lines",
            groups["both_lines"],
            "tab:blue",
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, (title, records, color) in zip(axes, plot_groups):
        selected_records = sample_records(records, maximum)

        for record in selected_records:
            xs, ys = get_trajectory_xy(record["object"])

            if len(xs) < 2:
                continue

            ax.plot(
                xs,
                ys,
                color=color,
                alpha=0.18,
                linewidth=0.8,
            )

        ax.plot(
            [LINE1[0][0], LINE1[1][0]],
            [LINE1[0][1], LINE1[1][1]],
            color="black",
            linewidth=3,
            label="LINE1",
        )
        ax.plot(
            [LINE2[0][0], LINE2[1][0]],
            [LINE2[0][1], LINE2[1][1]],
            color="green",
            linewidth=3,
            label="LINE2",
        )

        ax.set_title(
            f"{title}\n"
            f"n={len(records)}, displayed={len(selected_records)}"
        )
        ax.set_xlabel("World X (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.legend()

    axes[0].set_ylabel("World Y (m)")
    fig.suptitle("Viau screenline trajectory diagnostics")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Classify and plot trajectories that cross Viau screenlines."
        )
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path.cwd(),
        help="Folder containing SQLite files (default: current folder).",
    )
    parser.add_argument(
        "--pattern",
        default=SQLITE_PATTERN,
        help=f"SQLite filename pattern (default: {SQLITE_PATTERN}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PDF path (default: inside the input folder).",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=500,
        help="Maximum trajectories displayed per panel (default: 500).",
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
            f"No files matching '{args.pattern}' in {input_folder}"
        )

    output_path = args.output
    if output_path is None:
        output_path = input_folder / "viau_screenline_diagnostics.pdf"

    print(f"Found {len(sqlite_paths)} SQLite file(s)")
    records = load_motor_vehicles(sqlite_paths)
    print("Total motor-vehicle trajectories:", len(records))

    groups = classify_trajectories(records)
    print_diagnostics(groups)

    save_trajectory_figure(
        groups,
        output_path,
        maximum=args.max_trajectories,
    )

    print(f"\nDiagnostic PDF saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
