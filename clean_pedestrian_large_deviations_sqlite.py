#!/usr/bin/env python3
"""
Conservative large-deviation cleaning for pedestrian trajectories stored in a
Traffic Intelligence-style SQLite database.

The script DOES NOT overwrite the input database and DOES NOT modify the
original Traffic Intelligence tables in the output copy.

Instead, it:
  1. copies the merged SQLite database to a new output file;
  2. reconstructs one object-level pedestrian position per frame by averaging
     all feature-trajectory positions belonging to the same object;
  3. detects large-deviation candidates using BOTH:
       - an absolute displacement threshold; and
       - a local displacement-ratio threshold relative to the previous
         movements of the same pedestrian;
  4. conservatively classifies candidates as:
       - isolated_spike: the object jumps away and the next frame returns close
         to the previous local path;
       - persistent_break: the object jumps and remains spatially separated,
         suggesting a trajectory discontinuity;
       - uncertain: insufficient evidence for automatic correction;
  5. interpolates only high-confidence isolated spikes;
  6. starts a new motion segment at high-confidence persistent breaks while
     preserving the original object_id;
  7. writes derived cleaned trajectories and a full audit log into NEW tables.

Important:
  - Small-scale jitter smoothing is intentionally NOT performed here.
  - The output database keeps all original tables unchanged.
  - Downstream motion analysis should use cleaned_x_coordinate,
    cleaned_y_coordinate and avoid computing speed/acceleration across different
    segment_id values.

Default thresholds are the conservative values diagnostically inspected for the
current dataset:
  - 0.5 m per frame absolute displacement;
  - >3x the median of the previous 5 valid movements;
  - minimum 3 previous movements for a local baseline.

At 30 fps, 0.5 m/frame corresponds to 15 m/s, which is intentionally far above
normal pedestrian motion and therefore targets only severe discontinuities.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CLEANED_TABLE = "pedestrian_cleaned_positions"
LOG_TABLE = "trajectory_cleaning_log"
RUN_INFO_TABLE = "trajectory_cleaning_run_info"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect and conservatively clean large pedestrian trajectory "
            "deviations in a Traffic Intelligence-style SQLite database."
        )
    )
    parser.add_argument("input", type=Path, help="Merged input SQLite database")
    parser.add_argument("output", type=Path, help="Output cleaned SQLite database")
    parser.add_argument(
        "--road-user-type",
        type=int,
        default=2,
        help="Pedestrian road_user_type value (default: 2)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video frame rate, used only for reporting (default: 30)",
    )
    parser.add_argument(
        "--absolute-threshold",
        type=float,
        default=0.5,
        help="Candidate absolute displacement threshold in m/frame (default: 0.5)",
    )
    parser.add_argument(
        "--local-ratio-threshold",
        type=float,
        default=3.0,
        help="Candidate/local-continuity ratio threshold (default: 3.0)",
    )
    parser.add_argument(
        "--local-window",
        type=int,
        default=5,
        help="Number of previous movements for local median baseline (default: 5)",
    )
    parser.add_argument(
        "--min-local-points",
        type=int,
        default=3,
        help="Minimum previous movements required for local median (default: 3)",
    )
    parser.add_argument(
        "--break-cooldown",
        type=int,
        default=3,
        help=(
            "Minimum frame separation between automatic break starts for the same "
            "object (default: 3)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output database",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(f"Input database not found: {args.input}")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output must be different files.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}\n"
            "Use --overwrite if you intentionally want to replace it."
        )
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.absolute_threshold <= 0:
        raise ValueError("--absolute-threshold must be > 0")
    if args.local_ratio_threshold <= 1:
        raise ValueError("--local-ratio-threshold should be > 1")
    if args.local_window < 1:
        raise ValueError("--local-window must be >= 1")
    if not (1 <= args.min_local_points <= args.local_window):
        raise ValueError("--min-local-points must be between 1 and --local-window")
    if args.break_cooldown < 0:
        raise ValueError("--break-cooldown must be >= 0")


def require_tables(conn: sqlite3.Connection, required: Iterable[str]) -> None:
    existing = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    missing = [name for name in required if name not in existing]
    if missing:
        raise RuntimeError(
            "Required SQLite table(s) missing: " + ", ".join(sorted(missing))
        )


def load_object_level_pedestrians(
    conn: sqlite3.Connection,
    road_user_type: int,
) -> pd.DataFrame:
    """Reconstruct one object-level position per pedestrian per frame."""
    require_tables(conn, ["objects", "objects_features", "positions"])

    query = """
        SELECT
            o.object_id,
            p.frame_number,
            p.x_coordinate,
            p.y_coordinate,
            o.road_user_type
        FROM objects AS o
        JOIN objects_features AS f
            ON o.object_id = f.object_id
        JOIN positions AS p
            ON f.trajectory_id = p.trajectory_id
        WHERE o.road_user_type = ?
    """
    raw = pd.read_sql_query(query, conn, params=(road_user_type,))

    if raw.empty:
        raise RuntimeError(
            f"No rows found for road_user_type={road_user_type}. "
            "Check the road-user type mapping."
        )

    for col in ["object_id", "frame_number", "road_user_type"]:
        raw[col] = pd.to_numeric(raw[col], errors="raise").astype(int)
    for col in ["x_coordinate", "y_coordinate"]:
        raw[col] = pd.to_numeric(raw[col], errors="raise").astype(float)

    # An object may contain several feature trajectories. The diagnostic pipeline
    # used throughout this project represents the object position at each frame by
    # the mean of all available feature positions for that object/frame.
    df = (
        raw.groupby(["object_id", "frame_number", "road_user_type"], as_index=False)
        .agg(
            raw_x_coordinate=("x_coordinate", "mean"),
            raw_y_coordinate=("y_coordinate", "mean"),
            n_feature_positions=("x_coordinate", "size"),
        )
        .sort_values(["object_id", "frame_number"])
        .reset_index(drop=True)
    )

    return df


def euclidean(dx: pd.Series, dy: pd.Series) -> pd.Series:
    return np.sqrt(dx.astype(float) ** 2 + dy.astype(float) ** 2)


def compute_detection_metrics(
    df: pd.DataFrame,
    absolute_threshold: float,
    local_ratio_threshold: float,
    local_window: int,
    min_local_points: int,
) -> pd.DataFrame:
    df = df.copy().sort_values(["object_id", "frame_number"]).reset_index(drop=True)
    g = df.groupby("object_id", sort=False)

    df["prev_frame"] = g["frame_number"].shift(1)
    df["next_frame"] = g["frame_number"].shift(-1)

    df["prev_x"] = g["raw_x_coordinate"].shift(1)
    df["prev_y"] = g["raw_y_coordinate"].shift(1)
    df["next_x"] = g["raw_x_coordinate"].shift(-1)
    df["next_y"] = g["raw_y_coordinate"].shift(-1)

    df["gap_from_prev"] = df["frame_number"] - df["prev_frame"]
    df["gap_to_next"] = df["next_frame"] - df["frame_number"]

    step = euclidean(
        df["raw_x_coordinate"] - df["prev_x"],
        df["raw_y_coordinate"] - df["prev_y"],
    )
    df["step_displacement"] = np.where(df["gap_from_prev"].eq(1), step, np.nan)

    # Break local-baseline computation whenever the raw data itself has a frame gap.
    df["raw_segment_seed"] = df["gap_from_prev"].ne(1).astype(int)
    df["raw_segment_id"] = g["raw_segment_seed"].cumsum().astype(int)

    df["local_median"] = (
        df.groupby(["object_id", "raw_segment_id"], sort=False)["step_displacement"]
        .transform(
            lambda s: s.shift(1)
            .rolling(window=local_window, min_periods=min_local_points)
            .median()
        )
    )

    eps = 1e-9
    df["local_jump_ratio"] = df["step_displacement"] / df["local_median"].clip(lower=eps)

    df["is_large_jump_candidate"] = (
        df["step_displacement"].gt(absolute_threshold)
        & df["local_jump_ratio"].gt(local_ratio_threshold)
    )

    # Candidate geometry using t-1, t, t+1.
    jump_out = euclidean(
        df["next_x"] - df["raw_x_coordinate"],
        df["next_y"] - df["raw_y_coordinate"],
    )
    df["jump_out"] = np.where(df["gap_to_next"].eq(1), jump_out, np.nan)

    skip = euclidean(df["next_x"] - df["prev_x"], df["next_y"] - df["prev_y"])
    both_adjacent = df["gap_from_prev"].eq(1) & df["gap_to_next"].eq(1)
    df["skip_distance"] = np.where(both_adjacent, skip, np.nan)
    df["skip_ratio"] = df["skip_distance"] / df["local_median"].clip(lower=eps)

    # A high-confidence isolated spike is a two-step pattern:
    #
    #        t-1 ----> t (bad point) ----> t+1
    #          \_______________________/
    #                 locally plausible
    #
    # Therefore the incoming movement to t and the outgoing movement from t are
    # both large, but skipping t makes t-1 -> t+1 locally plausible. The return
    # movement at t+1 may itself satisfy the candidate rule, so it must not be
    # mistaken for a second break.
    has_context = both_adjacent & df["local_median"].notna()
    skip_is_locally_plausible = (
        df["skip_distance"].le(absolute_threshold)
        & df["skip_ratio"].le(local_ratio_threshold)
    )
    skip_remains_discontinuous = (
        df["skip_distance"].gt(absolute_threshold)
        & df["skip_ratio"].gt(local_ratio_threshold)
    )

    df["classification"] = "none"

    spike_mask = (
        df["is_large_jump_candidate"]
        & has_context
        & df["jump_out"].gt(absolute_threshold)
        & skip_is_locally_plausible
    )
    df.loc[spike_mask, "classification"] = "isolated_spike"

    # Mark the immediately following candidate, when present, as the return leg
    # of the same isolated spike. Its coordinates are kept; only the bad point t
    # is interpolated.
    spike_return_mask = (
        df.groupby("object_id", sort=False)["classification"]
        .shift(1)
        .eq("isolated_spike")
        & df["gap_from_prev"].eq(1)
        & df["is_large_jump_candidate"]
    )
    df.loc[spike_return_mask, "classification"] = "spike_return"

    break_mask = (
        df["is_large_jump_candidate"]
        & has_context
        & df["classification"].eq("none")
        & skip_remains_discontinuous
    )
    df.loc[break_mask, "classification"] = "persistent_break"

    uncertain_mask = df["is_large_jump_candidate"] & df["classification"].eq("none")
    df.loc[uncertain_mask, "classification"] = "uncertain"

    return df


def apply_conservative_cleaning(
    df: pd.DataFrame,
    break_cooldown: int,
) -> pd.DataFrame:
    """Interpolate only high-confidence spikes and create motion segment IDs."""
    df = df.copy().sort_values(["object_id", "frame_number"]).reset_index(drop=True)
    df["cleaned_x_coordinate"] = df["raw_x_coordinate"]
    df["cleaned_y_coordinate"] = df["raw_y_coordinate"]
    df["cleaning_action"] = "none"
    df["is_segment_break_start"] = False

    # Interpolate isolated spikes using t-1 and t+1. Classification guarantees
    # that both adjacent frames exist and are consecutive.
    spike_mask = df["classification"].eq("isolated_spike")
    df.loc[spike_mask, "cleaned_x_coordinate"] = (
        df.loc[spike_mask, "prev_x"] + df.loc[spike_mask, "next_x"]
    ) / 2.0
    df.loc[spike_mask, "cleaned_y_coordinate"] = (
        df.loc[spike_mask, "prev_y"] + df.loc[spike_mask, "next_y"]
    ) / 2.0
    df.loc[spike_mask, "cleaning_action"] = "interpolated_spike"

    # Collapse a cluster of persistent-break candidates into one segment boundary.
    for object_id, idx in df.groupby("object_id", sort=False).groups.items():
        last_break_frame: int | None = None
        for i in idx:
            if df.at[i, "classification"] != "persistent_break":
                continue
            frame = int(df.at[i, "frame_number"])
            if last_break_frame is None or frame - last_break_frame > break_cooldown:
                df.at[i, "is_segment_break_start"] = True
                df.at[i, "cleaning_action"] = "segment_break"
                last_break_frame = frame
            else:
                # Keep the candidate in the log, but do not create several tiny
                # segments for one short burst of instability.
                df.at[i, "classification"] = "break_cluster_member"
                df.at[i, "cleaning_action"] = "kept_raw"

    df.loc[df["classification"].eq("uncertain"), "cleaning_action"] = "kept_raw"

    # Start a new segment after raw frame gaps as well as at detected persistent
    # breaks. object_id itself is preserved to avoid re-introducing fragmentation.
    df["new_motion_segment"] = (
        df["gap_from_prev"].ne(1) | df["is_segment_break_start"]
    ).astype(int)
    df["segment_id"] = (
        df.groupby("object_id", sort=False)["new_motion_segment"].cumsum().astype(int)
    )

    return df


def write_output_tables(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    cleaned_columns = [
        "object_id",
        "frame_number",
        "road_user_type",
        "raw_x_coordinate",
        "raw_y_coordinate",
        "cleaned_x_coordinate",
        "cleaned_y_coordinate",
        "segment_id",
        "n_feature_positions",
        "is_large_jump_candidate",
        "classification",
        "cleaning_action",
        "is_segment_break_start",
        "step_displacement",
        "local_median",
        "local_jump_ratio",
        "jump_out",
        "skip_distance",
        "skip_ratio",
    ]

    cleaned = df[cleaned_columns].copy()
    for col in ["is_large_jump_candidate", "is_segment_break_start"]:
        cleaned[col] = cleaned[col].astype(int)

    cleaned.to_sql(CLEANED_TABLE, conn, if_exists="replace", index=False)

    log = cleaned[cleaned["is_large_jump_candidate"].eq(1)].copy()
    log.to_sql(LOG_TABLE, conn, if_exists="replace", index=False)

    info_rows = [
        ("input_database", str(args.input)),
        ("output_database", str(args.output)),
        ("road_user_type", str(args.road_user_type)),
        ("fps", str(args.fps)),
        ("absolute_threshold_m_per_frame", str(args.absolute_threshold)),
        ("absolute_threshold_m_per_s", str(args.absolute_threshold * args.fps)),
        ("local_ratio_threshold", str(args.local_ratio_threshold)),
        ("local_window", str(args.local_window)),
        ("min_local_points", str(args.min_local_points)),
        ("break_cooldown_frames", str(args.break_cooldown)),
        ("pedestrian_objects", str(df["object_id"].nunique())),
        ("pedestrian_points", str(len(df))),
        ("large_jump_candidates", str(int(df["is_large_jump_candidate"].sum()))),
        (
            "objects_with_candidates",
            str(df.loc[df["is_large_jump_candidate"], "object_id"].nunique()),
        ),
        ("isolated_spikes", str(int(df["classification"].eq("isolated_spike").sum()))),
        ("spike_return_candidates", str(int(df["classification"].eq("spike_return").sum()))),
        ("segment_breaks", str(int(df["is_segment_break_start"].sum()))),
        ("uncertain_candidates", str(int(df["classification"].eq("uncertain").sum()))),
    ]
    pd.DataFrame(info_rows, columns=["key", "value"]).to_sql(
        RUN_INFO_TABLE, conn, if_exists="replace", index=False
    )

    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{CLEANED_TABLE}_object_frame "
        f"ON {CLEANED_TABLE}(object_id, frame_number)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{CLEANED_TABLE}_segment "
        f"ON {CLEANED_TABLE}(object_id, segment_id, frame_number)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{LOG_TABLE}_object_frame "
        f"ON {LOG_TABLE}(object_id, frame_number)"
    )
    conn.commit()


def print_summary(df: pd.DataFrame, args: argparse.Namespace) -> None:
    candidates = df["is_large_jump_candidate"]
    print("\n=== Pedestrian large-deviation cleaning summary ===")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Pedestrian objects: {df['object_id'].nunique()}")
    print(f"Pedestrian points:  {len(df)}")
    print(f"Candidate threshold: > {args.absolute_threshold:.3f} m/frame "
          f"and > {args.local_ratio_threshold:.2f}x local median")
    print(f"Equivalent absolute speed at {args.fps:g} fps: "
          f"> {args.absolute_threshold * args.fps:.1f} m/s")
    print(f"Large-jump candidates: {int(candidates.sum())}")
    print(
        "Objects with candidates: "
        f"{df.loc[candidates, 'object_id'].nunique()}"
    )
    print(f"Interpolated isolated spikes: {int(df['classification'].eq('isolated_spike').sum())}")
    print(f"Spike-return candidates kept as valid observations: {int(df['classification'].eq('spike_return').sum())}")
    print(f"Motion segment breaks created: {int(df['is_segment_break_start'].sum())}")
    print(f"Uncertain candidates kept raw: {int(df['classification'].eq('uncertain').sum())}")
    print("\nOriginal Traffic Intelligence tables were preserved unchanged.")
    print(f"Use table '{CLEANED_TABLE}' for cleaned pedestrian trajectories.")
    print(f"Review table '{LOG_TABLE}' for every candidate and cleaning action.")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists() and args.overwrite:
            args.output.unlink()

        # Read the input before copying so an invalid DB cannot overwrite/create an
        # apparently successful output.
        with sqlite3.connect(args.input) as input_conn:
            pedestrians = load_object_level_pedestrians(
                input_conn,
                road_user_type=args.road_user_type,
            )

        analysed = compute_detection_metrics(
            pedestrians,
            absolute_threshold=args.absolute_threshold,
            local_ratio_threshold=args.local_ratio_threshold,
            local_window=args.local_window,
            min_local_points=args.min_local_points,
        )
        cleaned = apply_conservative_cleaning(
            analysed,
            break_cooldown=args.break_cooldown,
        )

        shutil.copy2(args.input, args.output)
        with sqlite3.connect(args.output) as output_conn:
            write_output_tables(output_conn, cleaned, args)

        print_summary(cleaned, args)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
