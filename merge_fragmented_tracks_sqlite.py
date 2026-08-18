#!/usr/bin/env python3
"""
Merge fragmented dltrack.py trajectories and write a NEW SQLite database.

Final merge rule
----------------
A pair of trajectories is merged when all of the following are true:

1. Same road_user_type
2. Trajectory B starts after trajectory A ends
3. B starts within SEARCH_WINDOW_FRAMES of A
4. A and B are mutual nearest continuation/predecessor candidates
5. frame_gap <= MERGE_MAX_FRAME_GAP
6. spatial_distance <= MERGE_MAX_SPATIAL_DISTANCE

Current working thresholds:
- MERGE_MAX_FRAME_GAP = 10 frames
- MERGE_MAX_SPATIAL_DISTANCE = 0.50 world-coordinate units
- FPS = 30, so 10 frames is approximately 0.33 s

Output
------
A new SQLite file is created with the SAME main table names:

- objects
- objects_features
- positions
- velocities

The merged SQLite can therefore be used as the input for the next
trajectory-cleaning / denoising step.

Two additional provenance tables are also saved:

- merge_map   : original IDs -> merged IDs
- merge_pairs : the trajectory pairs selected for merging
- merge_parameters : parameters used for this run

The original SQLite file is never modified.
Missing frames between trajectory fragments are NOT interpolated.
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

FPS = 30

# Broad search window used only to find plausible continuations.
SEARCH_WINDOW_FRAMES = 30

# Final merge thresholds.
MERGE_MAX_FRAME_GAP = 10
MERGE_MAX_SPATIAL_DISTANCE = 0.50


# ---------------------------------------------------------------------
# 1. Load the raw dltrack tables
# ---------------------------------------------------------------------

def load_raw_tables(conn):
    objects = pd.read_sql_query("SELECT * FROM objects;", conn)
    objects_features = pd.read_sql_query(
        "SELECT * FROM objects_features;", conn
    )
    positions = pd.read_sql_query("SELECT * FROM positions;", conn)
    velocities = pd.read_sql_query("SELECT * FROM velocities;", conn)

    for col in ["object_id", "road_user_type"]:
        if col in objects.columns:
            objects[col] = objects[col].astype(int)

    for col in ["object_id", "trajectory_id"]:
        objects_features[col] = objects_features[col].astype(int)

    for df in [positions, velocities]:
        df["trajectory_id"] = df["trajectory_id"].astype(int)
        df["frame_number"] = df["frame_number"].astype(int)

    return objects, objects_features, positions, velocities


# ---------------------------------------------------------------------
# 2. Create one frame-level tracking dataframe
# ---------------------------------------------------------------------

def build_tracks(objects, objects_features, positions):
    tracks = (
        objects[["object_id", "road_user_type"]]
        .merge(
            objects_features[["object_id", "trajectory_id"]],
            on="object_id",
            how="inner",
        )
        .merge(
            positions,
            on="trajectory_id",
            how="inner",
        )
        .sort_values(["object_id", "frame_number"])
        .reset_index(drop=True)
    )

    return tracks


# ---------------------------------------------------------------------
# 3. Confirm object_id <-> trajectory_id is one-to-one
# ---------------------------------------------------------------------

def validate_mapping(tracks):
    object_to_traj = tracks.groupby("object_id")["trajectory_id"].nunique()
    traj_to_object = tracks.groupby("trajectory_id")["object_id"].nunique()

    if (object_to_traj > 1).any():
        raise ValueError(
            "At least one object_id maps to multiple trajectory_id values."
        )

    if (traj_to_object > 1).any():
        raise ValueError(
            "At least one trajectory_id maps to multiple object_id values."
        )


# ---------------------------------------------------------------------
# 4. Summarize each original trajectory
# ---------------------------------------------------------------------

def make_track_summary(tracks):
    return (
        tracks
        .sort_values(["object_id", "frame_number"])
        .groupby(["object_id", "road_user_type", "trajectory_id"])
        .agg(
            start_frame=("frame_number", "min"),
            end_frame=("frame_number", "max"),
            start_x=("x_coordinate", "first"),
            start_y=("y_coordinate", "first"),
            end_x=("x_coordinate", "last"),
            end_y=("y_coordinate", "last"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------
# 5. Generate possible continuation pairs
# ---------------------------------------------------------------------

def build_candidate_pairs(track_summary):
    candidates = []
    rows = list(track_summary.itertuples(index=False))

    for a in rows:
        for b in rows:

            # Different original objects only.
            if a.object_id == b.object_id:
                continue

            # Only merge the same road-user type.
            if a.road_user_type != b.road_user_type:
                continue

            # B must start after A ends.
            if b.start_frame <= a.end_frame:
                continue

            frame_gap = int(b.start_frame - a.end_frame)

            # Broad temporal search window.
            if frame_gap > SEARCH_WINDOW_FRAMES:
                continue

            spatial_distance = float(
                np.hypot(
                    b.start_x - a.end_x,
                    b.start_y - a.end_y,
                )
            )

            candidates.append(
                {
                    "object_A": int(a.object_id),
                    "object_B": int(b.object_id),
                    "road_user_type": int(a.road_user_type),
                    "A_end_frame": int(a.end_frame),
                    "B_start_frame": int(b.start_frame),
                    "frame_gap": frame_gap,
                    "gap_seconds": frame_gap / FPS,
                    "spatial_distance": spatial_distance,
                }
            )

    return pd.DataFrame(candidates)


# ---------------------------------------------------------------------
# 6. Keep mutual nearest matches
# ---------------------------------------------------------------------

def keep_mutual_best_matches(candidate_pairs):
    """
    A -> B is retained only if:
      - B is the spatially closest later candidate for A, AND
      - A is the spatially closest earlier candidate for B.

    frame_gap is used as the tie-breaker.
    """
    if candidate_pairs.empty:
        return candidate_pairs.copy()

    best_successor = (
        candidate_pairs
        .sort_values(["object_A", "spatial_distance", "frame_gap"])
        .groupby("object_A", as_index=False)
        .first()
    )

    best_predecessor = (
        candidate_pairs
        .sort_values(["object_B", "spatial_distance", "frame_gap"])
        .groupby("object_B", as_index=False)
        .first()[["object_B", "object_A"]]
        .rename(columns={"object_A": "best_predecessor_A"})
    )

    mutual = best_successor.merge(
        best_predecessor,
        on="object_B",
        how="left",
    )

    mutual = mutual[
        mutual["object_A"] == mutual["best_predecessor_A"]
    ].copy()

    return (
        mutual
        .drop(columns=["best_predecessor_A"])
        .sort_values(["frame_gap", "spatial_distance"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# 7. Apply the final merge rule
# ---------------------------------------------------------------------

def select_merge_pairs(mutual_candidates):
    merge_pairs = mutual_candidates[
        (mutual_candidates["frame_gap"] <= MERGE_MAX_FRAME_GAP)
        & (
            mutual_candidates["spatial_distance"]
            <= MERGE_MAX_SPATIAL_DISTANCE
        )
    ].copy()

    return merge_pairs.reset_index(drop=True)


# ---------------------------------------------------------------------
# 8. Build merged object groups
# ---------------------------------------------------------------------

def build_object_merge_map(tracks, merge_pairs):
    """
    Connected pairs are merged into one group.

    Example:
        3834 -> 3856
        3856 -> 3860

    becomes one group:
        3834, 3856, 3860

    The smallest original object_id is used as the merged object_id.
    """
    object_ids = sorted(tracks["object_id"].astype(int).unique())
    parent = {obj: obj for obj in object_ids}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a == root_b:
            return

        smaller = min(root_a, root_b)
        larger = max(root_a, root_b)
        parent[larger] = smaller

    for row in merge_pairs.itertuples(index=False):
        union(int(row.object_A), int(row.object_B))

    mapping = pd.DataFrame(
        {
            "original_object_id": object_ids,
            "merged_object_id": [find(obj) for obj in object_ids],
        }
    )

    return mapping


# ---------------------------------------------------------------------
# 9. Build merged trajectory IDs
# ---------------------------------------------------------------------

def build_full_merge_map(
    objects,
    objects_features,
    object_merge_map,
):
    """
    Each merged object group also receives one merged trajectory_id.

    The smallest original trajectory_id in that merged group is used.
    """
    mapping = (
        objects[["object_id", "road_user_type"]]
        .merge(
            objects_features[["object_id", "trajectory_id"]],
            on="object_id",
            how="inner",
        )
        .merge(
            object_merge_map,
            left_on="object_id",
            right_on="original_object_id",
            how="inner",
        )
    )

    group_trajectory_id = (
        mapping
        .groupby("merged_object_id")["trajectory_id"]
        .min()
        .rename("merged_trajectory_id")
        .reset_index()
    )

    mapping = mapping.merge(
        group_trajectory_id,
        on="merged_object_id",
        how="left",
    )

    mapping = mapping.rename(
        columns={
            "object_id": "source_object_id",
            "trajectory_id": "original_trajectory_id",
        }
    )

    return mapping[
        [
            "source_object_id",
            "road_user_type",
            "original_trajectory_id",
            "merged_object_id",
            "merged_trajectory_id",
        ]
    ].sort_values("source_object_id")


# ---------------------------------------------------------------------
# 10. Create the corrected main tables
# ---------------------------------------------------------------------

def build_merged_tables(
    objects,
    positions,
    velocities,
    merge_map,
):
    # ---- objects ----
    merged_objects = (
        merge_map[
            ["merged_object_id", "road_user_type"]
        ]
        .drop_duplicates()
        .rename(columns={"merged_object_id": "object_id"})
        .sort_values("object_id")
        .reset_index(drop=True)
    )

    # Keep the original schema.
    merged_objects["n_objects"] = 1

    # ---- objects_features ----
    merged_objects_features = (
        merge_map[
            ["merged_object_id", "merged_trajectory_id"]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "merged_object_id": "object_id",
                "merged_trajectory_id": "trajectory_id",
            }
        )
        .sort_values("object_id")
        .reset_index(drop=True)
    )

    trajectory_map = (
        merge_map[
            ["original_trajectory_id", "merged_trajectory_id"]
        ]
        .drop_duplicates()
    )

    # ---- positions ----
    merged_positions = positions.merge(
        trajectory_map,
        left_on="trajectory_id",
        right_on="original_trajectory_id",
        how="inner",
    )

    merged_positions["trajectory_id"] = (
        merged_positions["merged_trajectory_id"].astype(int)
    )

    merged_positions = (
        merged_positions[
            [
                "trajectory_id",
                "frame_number",
                "x_coordinate",
                "y_coordinate",
            ]
        ]
        # Defensive handling in case two fragments unexpectedly share a frame.
        .groupby(["trajectory_id", "frame_number"], as_index=False)
        .agg(
            x_coordinate=("x_coordinate", "mean"),
            y_coordinate=("y_coordinate", "mean"),
        )
        .sort_values(["trajectory_id", "frame_number"])
        .reset_index(drop=True)
    )

    # ---- velocities ----
    merged_velocities = velocities.merge(
        trajectory_map,
        left_on="trajectory_id",
        right_on="original_trajectory_id",
        how="inner",
    )

    merged_velocities["trajectory_id"] = (
        merged_velocities["merged_trajectory_id"].astype(int)
    )

    merged_velocities = (
        merged_velocities[
            [
                "trajectory_id",
                "frame_number",
                "x_coordinate",
                "y_coordinate",
            ]
        ]
        .groupby(["trajectory_id", "frame_number"], as_index=False)
        .agg(
            x_coordinate=("x_coordinate", "mean"),
            y_coordinate=("y_coordinate", "mean"),
        )
        .sort_values(["trajectory_id", "frame_number"])
        .reset_index(drop=True)
    )

    return (
        merged_objects,
        merged_objects_features,
        merged_positions,
        merged_velocities,
    )


# ---------------------------------------------------------------------
# 11. Save a new SQLite database
# ---------------------------------------------------------------------

def write_new_sqlite(
    output_path,
    merged_objects,
    merged_objects_features,
    merged_positions,
    merged_velocities,
    merge_map,
    merge_pairs,
):
    if output_path.exists():
        output_path.unlink()

    conn = sqlite3.connect(output_path)
    cur = conn.cursor()

    # Same main schemas as the original dltrack SQLite.
    cur.execute(
        """
        CREATE TABLE objects (
            object_id INTEGER PRIMARY KEY,
            road_user_type INTEGER,
            n_objects INTEGER
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE objects_features (
            object_id INTEGER,
            trajectory_id INTEGER,
            PRIMARY KEY (object_id, trajectory_id)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE positions (
            trajectory_id INTEGER,
            frame_number INTEGER,
            x_coordinate REAL,
            y_coordinate REAL,
            PRIMARY KEY (trajectory_id, frame_number)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE velocities (
            trajectory_id INTEGER,
            frame_number INTEGER,
            x_coordinate REAL,
            y_coordinate REAL,
            PRIMARY KEY (trajectory_id, frame_number)
        );
        """
    )

    conn.commit()

    merged_objects.to_sql(
        "objects",
        conn,
        if_exists="append",
        index=False,
    )

    merged_objects_features.to_sql(
        "objects_features",
        conn,
        if_exists="append",
        index=False,
    )

    merged_positions.to_sql(
        "positions",
        conn,
        if_exists="append",
        index=False,
    )

    merged_velocities.to_sql(
        "velocities",
        conn,
        if_exists="append",
        index=False,
    )

    # Extra provenance tables.
    merge_map.to_sql(
        "merge_map",
        conn,
        if_exists="replace",
        index=False,
    )

    merge_pairs.to_sql(
        "merge_pairs",
        conn,
        if_exists="replace",
        index=False,
    )

    parameters = pd.DataFrame(
        {
            "parameter": [
                "fps",
                "search_window_frames",
                "merge_max_frame_gap",
                "merge_max_gap_seconds",
                "merge_max_spatial_distance",
            ],
            "value": [
                FPS,
                SEARCH_WINDOW_FRAMES,
                MERGE_MAX_FRAME_GAP,
                MERGE_MAX_FRAME_GAP / FPS,
                MERGE_MAX_SPATIAL_DISTANCE,
            ],
        }
    )

    parameters.to_sql(
        "merge_parameters",
        conn,
        if_exists="replace",
        index=False,
    )

    conn.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge fragmented dltrack trajectories and create "
            "a new SQLite database."
        )
    )

    parser.add_argument(
        "--db",
        required=True,
        help="Path to the original dltrack SQLite file.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path to the merged SQLite file. "
            "Default: <input_stem>_merged.sqlite"
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.db).expanduser().resolve()

    if args.output is None:
        output_path = input_path.with_name(
            input_path.stem + "_merged.sqlite"
        )
    else:
        output_path = Path(args.output).expanduser().resolve()

    # Load.
    with sqlite3.connect(input_path) as conn:
        (
            objects,
            objects_features,
            positions,
            velocities,
        ) = load_raw_tables(conn)

    # Build raw tracking table.
    tracks = build_tracks(
        objects,
        objects_features,
        positions,
    )

    validate_mapping(tracks)

    # Find fragmented pairs.
    track_summary = make_track_summary(tracks)
    candidate_pairs = build_candidate_pairs(track_summary)
    mutual_candidates = keep_mutual_best_matches(candidate_pairs)
    merge_pairs = select_merge_pairs(mutual_candidates)

    # Create merged ID mapping.
    object_merge_map = build_object_merge_map(
        tracks,
        merge_pairs,
    )

    merge_map = build_full_merge_map(
        objects,
        objects_features,
        object_merge_map,
    )

    # Create corrected SQLite tables.
    (
        merged_objects,
        merged_objects_features,
        merged_positions,
        merged_velocities,
    ) = build_merged_tables(
        objects,
        positions,
        velocities,
        merge_map,
    )

    # Save new SQLite.
    write_new_sqlite(
        output_path,
        merged_objects,
        merged_objects_features,
        merged_positions,
        merged_velocities,
        merge_map,
        merge_pairs,
    )

    raw_count = int(objects["object_id"].nunique())
    merged_count = int(merged_objects["object_id"].nunique())

    print("\n=== Merge parameters ===")
    print(f"FPS: {FPS}")
    print(
        f"Search window: <= {SEARCH_WINDOW_FRAMES} frames "
        f"({SEARCH_WINDOW_FRAMES / FPS:.2f} s)"
    )
    print(
        f"Final frame gap: <= {MERGE_MAX_FRAME_GAP} frames "
        f"({MERGE_MAX_FRAME_GAP / FPS:.2f} s)"
    )
    print(
        "Final spatial distance: "
        f"<= {MERGE_MAX_SPATIAL_DISTANCE} world-coordinate units"
    )

    print("\n=== Results ===")
    print(f"Raw object count: {raw_count}")
    print(f"Mutual candidates: {len(mutual_candidates)}")
    print(f"Selected merge pairs: {len(merge_pairs)}")
    print(f"Merged road-user count: {merged_count}")
    print(f"Count reduction: {raw_count - merged_count}")

    print("\n=== Merge pairs ===")
    if len(merge_pairs) == 0:
        print("No trajectories were merged.")
    else:
        print(
            merge_pairs[
                [
                    "object_A",
                    "object_B",
                    "road_user_type",
                    "frame_gap",
                    "gap_seconds",
                    "spatial_distance",
                ]
            ].to_string(index=False)
        )

    print("\nNew SQLite database:")
    print(output_path)
    print(
        "\nUse this merged SQLite as the input for the next "
        "trajectory denoising / cleaning step."
    )


if __name__ == "__main__":
    main()
