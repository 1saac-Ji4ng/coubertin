#!/usr/bin/env python3
"""
Conservative all-in-one cleaning for pedestrian trajectories stored in a
Traffic Intelligence-style SQLite database.

Pipeline
--------
1. Copy the merged input SQLite database to a NEW output database.
2. Reconstruct one object-level pedestrian position per frame by averaging all
   feature-trajectory positions belonging to the same object.
3. Detect severe large-deviation candidates using BOTH:
      - absolute per-frame displacement; and
      - displacement relative to the recent local median of the same object.
4. Classify high-confidence large deviations as:
      - isolated_spike: jump away and return to the previous local path;
      - persistent_break: jump and remain spatially separated;
      - uncertain: insufficient evidence for automatic correction.
5. Interpolate only high-confidence isolated spikes.
6. Create a new motion segment only for persistent breaks that pass minimum
   segment-length protection. object_id is preserved.
7. Smooth remaining small-scale jitter with a centered moving average applied
   ONLY within each final motion segment, never across a frame gap or accepted
   trajectory break.
8. Write cleaned positions, diagnostics, and an audit log into NEW SQLite tables.

The original Traffic Intelligence tables in the output copy are left unchanged.

Default parameters used for the current dataset
-----------------------------------------------
- FPS: 30
- severe-jump absolute threshold: 0.5 m/frame (= 15 m/s at 30 fps)
- severe-jump local ratio threshold: 3x recent local median
- local baseline: previous 5 movements, at least 3 available
- minimum segment length created by cleaning: 10 points
- jitter smoothing half-width: 5 frames (11-frame centered window)
- minimum points required before smoothing a segment: 11

These defaults are intentionally conservative. Every threshold is exposed as a
command-line option so sensitivity analyses can be reproduced.
"""

from __future__ import annotations

import argparse
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
            "Clean pedestrian trajectory large deviations and small jitter in a "
            "Traffic Intelligence-style SQLite database."
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
        help="Video frame rate, used for reporting (default: 30)",
    )
    parser.add_argument(
        "--absolute-threshold",
        type=float,
        default=0.5,
        help="Large-jump absolute displacement threshold in m/frame (default: 0.5)",
    )
    parser.add_argument(
        "--local-ratio-threshold",
        type=float,
        default=3.0,
        help="Large-jump/local-median ratio threshold (default: 3.0)",
    )
    parser.add_argument(
        "--local-window",
        type=int,
        default=5,
        help="Previous movements used for local median baseline (default: 5)",
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
        help="Minimum frame separation between break starts in one raw segment (default: 3)",
    )
    parser.add_argument(
        "--min-segment-points",
        type=int,
        default=10,
        help=(
            "Minimum number of points allowed on BOTH sides of a cleaning-created "
            "segment break (default: 10)"
        ),
    )
    parser.add_argument(
        "--smoothing-halfwidth",
        type=int,
        default=5,
        help=(
            "Centered moving-average half-width for residual jitter. A half-width "
            "of 5 gives an 11-frame window (default: 5)"
        ),
    )
    parser.add_argument(
        "--min-smoothing-points",
        type=int,
        default=11,
        help="Minimum final-segment length required for smoothing (default: 11)",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable small-jitter smoothing but keep large-deviation cleaning",
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
    if args.min_segment_points < 2:
        raise ValueError("--min-segment-points must be >= 2")
    if args.smoothing_halfwidth < 0:
        raise ValueError("--smoothing-halfwidth must be >= 0")
    if args.min_smoothing_points < 2:
        raise ValueError("--min-smoothing-points must be >= 2")


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
    """Reconstruct one object-level pedestrian position per frame."""
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

    # Several feature trajectories can belong to one road-user object. The same
    # object-level representation used during diagnosis is retained here: mean
    # world position of all available feature positions for an object/frame.
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
    """Compute severe-jump diagnostics without modifying positions."""
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

    # Raw segment = frame-continuous data before any cleaning-created break.
    df["raw_segment_seed"] = df["gap_from_prev"].ne(1).astype(int)
    df["raw_segment_id"] = (
        df.groupby("object_id", sort=False)["raw_segment_seed"].cumsum().astype(int)
    )

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

    jump_out = euclidean(
        df["next_x"] - df["raw_x_coordinate"],
        df["next_y"] - df["raw_y_coordinate"],
    )
    df["jump_out"] = np.where(df["gap_to_next"].eq(1), jump_out, np.nan)

    skip = euclidean(df["next_x"] - df["prev_x"], df["next_y"] - df["prev_y"])
    both_adjacent = df["gap_from_prev"].eq(1) & df["gap_to_next"].eq(1)
    df["skip_distance"] = np.where(both_adjacent, skip, np.nan)
    df["skip_ratio"] = df["skip_distance"] / df["local_median"].clip(lower=eps)

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

    # The following candidate can simply be the return leg from the spike.
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


def apply_large_deviation_cleaning(
    df: pd.DataFrame,
    break_cooldown: int,
    min_segment_points: int,
) -> pd.DataFrame:
    """Interpolate spikes and accept only well-supported trajectory breaks."""
    df = df.copy().sort_values(["object_id", "frame_number"]).reset_index(drop=True)

    # Stage-1 coordinates after severe-deviation correction, before smoothing.
    df["pre_smooth_x_coordinate"] = df["raw_x_coordinate"]
    df["pre_smooth_y_coordinate"] = df["raw_y_coordinate"]
    df["cleaning_action"] = "none"
    df["is_segment_break_start"] = False

    spike_mask = df["classification"].eq("isolated_spike")
    df.loc[spike_mask, "pre_smooth_x_coordinate"] = (
        df.loc[spike_mask, "prev_x"] + df.loc[spike_mask, "next_x"]
    ) / 2.0
    df.loc[spike_mask, "pre_smooth_y_coordinate"] = (
        df.loc[spike_mask, "prev_y"] + df.loc[spike_mask, "next_y"]
    ) / 2.0
    df.loc[spike_mask, "cleaning_action"] = "interpolated_spike"

    # Candidates are processed inside each ORIGINAL frame-continuous segment.
    # A cleaning-created break is accepted only if the segment on its left and
    # the final remainder on its right both contain at least min_segment_points.
    # This prevents cleaning from re-introducing severe fragmentation.
    grouped = df.groupby(["object_id", "raw_segment_id"], sort=False)

    for (_, _), idx in grouped.groups.items():
        ordered_idx = list(idx)
        if len(ordered_idx) < 2 * min_segment_points:
            # This raw segment is too short to split safely at all.
            for i in ordered_idx:
                if df.at[i, "classification"] == "persistent_break":
                    df.at[i, "classification"] = "short_segment_risk"
                    df.at[i, "cleaning_action"] = "kept_raw"
            continue

        pos_by_index = {row_idx: pos for pos, row_idx in enumerate(ordered_idx)}
        persistent = [
            i for i in ordered_idx if df.at[i, "classification"] == "persistent_break"
        ]

        # First collapse nearby candidates into one candidate per instability burst.
        cooldown_filtered: list[int] = []
        last_candidate_frame: int | None = None
        for i in persistent:
            frame = int(df.at[i, "frame_number"])
            if (
                last_candidate_frame is None
                or frame - last_candidate_frame > break_cooldown
            ):
                cooldown_filtered.append(i)
                last_candidate_frame = frame
            else:
                df.at[i, "classification"] = "break_cluster_member"
                df.at[i, "cleaning_action"] = "kept_raw"

        # Then protect the minimum length of every cleaning-created segment.
        last_boundary_pos = 0
        n_total = len(ordered_idx)

        for i in cooldown_filtered:
            pos = pos_by_index[i]
            left_len = pos - last_boundary_pos
            right_len = n_total - pos

            if left_len < min_segment_points or right_len < min_segment_points:
                df.at[i, "classification"] = "short_segment_risk"
                df.at[i, "cleaning_action"] = "kept_raw"
                continue

            df.at[i, "is_segment_break_start"] = True
            df.at[i, "cleaning_action"] = "segment_break"
            last_boundary_pos = pos

    df.loc[df["classification"].eq("uncertain"), "cleaning_action"] = "kept_raw"

    # Start a final motion segment at either an original frame gap or an accepted
    # large-deviation break. object_id remains unchanged.
    df["new_motion_segment"] = (
        df["gap_from_prev"].ne(1) | df["is_segment_break_start"]
    ).astype(int)
    df["segment_id"] = (
        df.groupby("object_id", sort=False)["new_motion_segment"].cumsum().astype(int)
    )

    return df


def apply_segmentwise_smoothing(
    df: pd.DataFrame,
    smoothing_halfwidth: int,
    min_smoothing_points: int,
    enabled: bool,
) -> pd.DataFrame:
    """Smooth residual small jitter without crossing suspicious discontinuities."""
    df = df.copy().sort_values(["object_id", "frame_number"]).reset_index(drop=True)

    # Even when a persistent break is NOT promoted to an official segment break
    # because it would create a very short segment, smoothing must never average
    # across that severe jump. The same applies to uncertain severe candidates.
    # Therefore smoothing uses a stricter continuity partition than segment_id.
    unsafe_candidate = (
        df["is_large_jump_candidate"]
        & ~df["classification"].isin(["isolated_spike", "spike_return"])
    )
    df["valid_motion_step"] = (
        df["gap_from_prev"].eq(1) & ~unsafe_candidate
    )
    df["new_smoothing_segment"] = (
        df["gap_from_prev"].ne(1)
        | df["is_segment_break_start"]
        | unsafe_candidate
    ).astype(int)
    df["smoothing_segment_id"] = (
        df.groupby("object_id", sort=False)["new_smoothing_segment"]
        .cumsum()
        .astype(int)
    )

    df["cleaned_x_coordinate"] = df["pre_smooth_x_coordinate"]
    df["cleaned_y_coordinate"] = df["pre_smooth_y_coordinate"]
    df["smoothing_applied"] = False
    df["smoothing_window"] = 0

    if not enabled or smoothing_halfwidth == 0:
        df["smoothing_delta"] = 0.0
        return df

    window = 2 * smoothing_halfwidth + 1

    for (_, _), idx in df.groupby(
        ["object_id", "smoothing_segment_id"], sort=False
    ).groups.items():
        idx = list(idx)
        if len(idx) < max(min_smoothing_points, window):
            continue

        x = df.loc[idx, "pre_smooth_x_coordinate"].astype(float)
        y = df.loc[idx, "pre_smooth_y_coordinate"].astype(float)

        # Conservative centered moving average: only positions with a COMPLETE
        # +/- half-width neighborhood are smoothed. Boundary points remain
        # unchanged, preventing endpoint drift and avoiding averaging across any
        # raw gap or severe-jump boundary.
        smooth_x = x.rolling(window=window, center=True, min_periods=window).mean()
        smooth_y = y.rolling(window=window, center=True, min_periods=window).mean()
        full_window = smooth_x.notna() & smooth_y.notna()

        if not full_window.any():
            continue

        smooth_idx = list(x.index[full_window])
        df.loc[smooth_idx, "cleaned_x_coordinate"] = smooth_x.loc[full_window].to_numpy()
        df.loc[smooth_idx, "cleaned_y_coordinate"] = smooth_y.loc[full_window].to_numpy()
        df.loc[smooth_idx, "smoothing_applied"] = True
        df.loc[smooth_idx, "smoothing_window"] = window

    df["smoothing_delta"] = np.sqrt(
        (df["cleaned_x_coordinate"] - df["pre_smooth_x_coordinate"]) ** 2
        + (df["cleaned_y_coordinate"] - df["pre_smooth_y_coordinate"]) ** 2
    )
    return df

def segment_summary(df: pd.DataFrame, use_final_segments: bool) -> pd.DataFrame:
    if use_final_segments:
        return (
            df.groupby(["object_id", "segment_id"], sort=False)
            .size()
            .reset_index(name="n_points")
        )
    return (
        df.groupby(["object_id", "raw_segment_id"], sort=False)
        .size()
        .reset_index(name="n_points")
    )


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
        "pre_smooth_x_coordinate",
        "pre_smooth_y_coordinate",
        "cleaned_x_coordinate",
        "cleaned_y_coordinate",
        "segment_id",
        "smoothing_segment_id",
        "valid_motion_step",
        "raw_segment_id",
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
        "smoothing_applied",
        "smoothing_window",
        "smoothing_delta",
    ]

    cleaned = df[cleaned_columns].copy()
    for col in ["is_large_jump_candidate", "is_segment_break_start", "smoothing_applied", "valid_motion_step"]:
        cleaned[col] = cleaned[col].astype(int)

    cleaned.to_sql(CLEANED_TABLE, conn, if_exists="replace", index=False)

    # Large-deviation audit log. Smoothing is auditable in the cleaned table via
    # pre_smooth_* / cleaned_* / smoothing_delta, avoiding a 28k-row extra log.
    log = cleaned[cleaned["is_large_jump_candidate"].eq(1)].copy()
    log.to_sql(LOG_TABLE, conn, if_exists="replace", index=False)

    before_segments = segment_summary(df, use_final_segments=False)
    after_segments = segment_summary(df, use_final_segments=True)

    smoothed_points = int(df["smoothing_applied"].sum())
    smoothed_segments = int(
        df.loc[df["smoothing_applied"], ["object_id", "segment_id"]]
        .drop_duplicates()
        .shape[0]
    )

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
        ("min_segment_points", str(args.min_segment_points)),
        ("smoothing_enabled", str(not args.no_smoothing)),
        ("smoothing_halfwidth", str(args.smoothing_halfwidth)),
        ("smoothing_window", str(2 * args.smoothing_halfwidth + 1)),
        ("min_smoothing_points", str(args.min_smoothing_points)),
        ("pedestrian_objects", str(df["object_id"].nunique())),
        ("pedestrian_points", str(len(df))),
        ("large_jump_candidates", str(int(df["is_large_jump_candidate"].sum()))),
        (
            "objects_with_candidates",
            str(df.loc[df["is_large_jump_candidate"], "object_id"].nunique()),
        ),
        ("isolated_spikes", str(int(df["classification"].eq("isolated_spike").sum()))),
        ("spike_return_candidates", str(int(df["classification"].eq("spike_return").sum()))),
        ("segment_breaks_accepted", str(int(df["is_segment_break_start"].sum()))),
        ("break_cluster_members", str(int(df["classification"].eq("break_cluster_member").sum()))),
        ("short_segment_risk_kept_raw", str(int(df["classification"].eq("short_segment_risk").sum()))),
        ("uncertain_candidates", str(int(df["classification"].eq("uncertain").sum()))),
        ("raw_continuous_segments", str(len(before_segments))),
        ("final_motion_segments", str(len(after_segments))),
        ("smoothed_segments", str(smoothed_segments)),
        ("smoothed_points", str(smoothed_points)),
        ("invalid_motion_steps", str(int((~df["valid_motion_step"]).sum()))),
        ("median_smoothing_delta_m", str(float(df["smoothing_delta"].median()))),
        ("max_smoothing_delta_m", str(float(df["smoothing_delta"].max()))),
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
    before_segments = segment_summary(df, use_final_segments=False)
    after_segments = segment_summary(df, use_final_segments=True)

    print("\n=== Pedestrian trajectory cleaning summary ===")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Pedestrian objects: {df['object_id'].nunique()}")
    print(f"Pedestrian points:  {len(df)}")

    print("\n--- Large-deviation stage ---")
    print(
        f"Candidate rule: displacement > {args.absolute_threshold:.3f} m/frame "
        f"AND > {args.local_ratio_threshold:.2f}x local median"
    )
    print(
        f"Equivalent absolute speed at {args.fps:g} fps: "
        f"> {args.absolute_threshold * args.fps:.1f} m/s"
    )
    print(f"Large-jump candidates: {int(candidates.sum())}")
    print(f"Objects with candidates: {df.loc[candidates, 'object_id'].nunique()}")
    print(f"Interpolated isolated spikes: {int(df['classification'].eq('isolated_spike').sum())}")
    print(f"Accepted segment breaks: {int(df['is_segment_break_start'].sum())}")
    print(f"Nearby break-cluster candidates kept raw: {int(df['classification'].eq('break_cluster_member').sum())}")
    print(f"Breaks rejected by minimum-segment protection: {int(df['classification'].eq('short_segment_risk').sum())}")
    print(f"Uncertain candidates kept raw: {int(df['classification'].eq('uncertain').sum())}")
    print(f"Raw continuous segments: {len(before_segments)}")
    print(f"Final motion segments:   {len(after_segments)}")

    print("\n--- Small-jitter smoothing stage ---")
    if args.no_smoothing or args.smoothing_halfwidth == 0:
        print("Smoothing disabled.")
    else:
        smoothed_points = int(df["smoothing_applied"].sum())
        smoothed_segments = int(
            df.loc[df["smoothing_applied"], ["object_id", "segment_id"]]
            .drop_duplicates()
            .shape[0]
        )
        print(
            f"Centered moving-average window: "
            f"{2 * args.smoothing_halfwidth + 1} frames "
            f"(half-width={args.smoothing_halfwidth})"
        )
        print(f"Minimum segment length for smoothing: {args.min_smoothing_points} points")
        print(f"Smoothed motion segments: {smoothed_segments}")
        print(f"Smoothed points: {smoothed_points}")
        print(f"Median smoothing adjustment: {df['smoothing_delta'].median():.4f} m")
        print(f"Maximum smoothing adjustment: {df['smoothing_delta'].max():.4f} m")
        print(f"Motion steps quarantined from downstream kinematics: {int((~df['valid_motion_step']).sum())}")

    print("\nOriginal Traffic Intelligence tables were preserved unchanged.")
    print(f"Use table '{CLEANED_TABLE}' for final cleaned pedestrian trajectories.")
    print(f"Review table '{LOG_TABLE}' for every severe-jump candidate and action.")
    print(
        "For downstream speed/acceleration/TTC/PET calculations, never compute "
        "motion across different segment_id values and only use rows where "
        "valid_motion_step = 1 for step-based kinematics."
    )


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists() and args.overwrite:
            args.output.unlink()

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

        severe_cleaned = apply_large_deviation_cleaning(
            analysed,
            break_cooldown=args.break_cooldown,
            min_segment_points=args.min_segment_points,
        )

        final_cleaned = apply_segmentwise_smoothing(
            severe_cleaned,
            smoothing_halfwidth=args.smoothing_halfwidth,
            min_smoothing_points=args.min_smoothing_points,
            enabled=not args.no_smoothing,
        )

        shutil.copy2(args.input, args.output)
        with sqlite3.connect(args.output) as output_conn:
            write_output_tables(output_conn, final_cleaned, args)

        print_summary(final_cleaned, args)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
