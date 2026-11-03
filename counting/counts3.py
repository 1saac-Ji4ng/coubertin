#!/usr/bin/env python3
"""Count road-user trajectory crossings across one SQLite file or a folder.

This script is tailored to the supplied dltrack SQLite schema:
  positions(trajectory_id, frame_number, x_coordinate, y_coordinate)
  objects(object_id, road_user_type, n_objects)
  objects_features(object_id, trajectory_id)

It counts the first valid crossing of one finite virtual line per trajectory.
For a folder, each SQLite file is one observation. It creates two one-page
PDFs: an all-file summary and a per-file heatmap. Screenlines are extended 2 m
at each endpoint by default; set --extend-line-m to adjust the extension.

The database stores numeric road_user_type codes, mapped below to readable
road-user types. Coordinates are metres in the calibrated world coordinate
system. The supplied site convention is LINE1-to-LINE2 = West and
LINE2-to-LINE1 = East. For LINE1, negative_to_positive is West and
positive_to_negative is East.
"""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


FPS_DEFAULT = 30.0
INTERVAL_MINUTES_DEFAULT = 15
LINE_COORDS = {
    "LINE1": ((1.1859, 9.2850), (41.4056, 28.6423)),
    "LINE2": ((0.1358, 27.6972), (36.7850, 45.0242)),
}
TYPE_LABELS = {
    1: "car",
    2: "pedestrian",
    3: "motorcyclist",
    4: "cyclist",
    5: "bus",
    6: "truck",
}


def extend_line(line, extension_m):
    """Extend both ends of a screenline by extension_m, along its own axis."""
    (ax, ay), (bx, by) = line
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length == 0 or extension_m <= 0:
        return line
    ux, uy = dx / length, dy / length
    return ((ax - ux * extension_m, ay - uy * extension_m),
            (bx + ux * extension_m, by + uy * extension_m))


def natural_sort_key(path: Path):
    return [(0, int(part)) if part.isdigit() else (1, part.lower())
            for part in re.split(r"(\d+)", str(path))]


def line_intersection_fraction(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
    eps: float = 1e-10,
) -> Optional[Tuple[float, int]]:
    """Return (fraction along p1->p2, direction sign) for a segment crossing.

    direction sign is +1 for positive-side to negative-side, -1 for the
    reverse. Only intersections with the finite measurement line are accepted.
    """
    lx, ly = b[0] - a[0], b[1] - a[1]
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    s1 = lx * (p1[1] - a[1]) - ly * (p1[0] - a[0])
    s2 = lx * (p2[1] - a[1]) - ly * (p2[0] - a[0])
    if s1 * s2 >= 0 or abs(s1 - s2) < eps:
        return None
    t = s1 / (s1 - s2)
    ix, iy = p1[0] + t * dx, p1[1] + t * dy
    denom = lx * lx + ly * ly
    u = ((ix - a[0]) * lx + (iy - a[1]) * ly) / denom
    if -eps <= u <= 1 + eps:
        return t, (1 if s1 > 0 else -1)
    return None


def load_tracks(db_path: Path):
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    query = """
        SELECT p.trajectory_id, o.road_user_type, p.frame_number,
               p.x_coordinate, p.y_coordinate
        FROM positions AS p
        JOIN objects_features AS f ON f.trajectory_id = p.trajectory_id
        JOIN objects AS o ON o.object_id = f.object_id
        ORDER BY p.trajectory_id, p.frame_number
    """
    tracks = defaultdict(list)
    track_types = {}
    for row in con.execute(query):
        tid = int(row["trajectory_id"])
        tracks[tid].append(
            (int(row["frame_number"]), float(row["x_coordinate"]), float(row["y_coordinate"]))
        )
        track_types[tid] = int(row["road_user_type"])
    max_frame = con.execute("SELECT MAX(frame_number) FROM positions").fetchone()[0]
    con.close()
    return tracks, track_types, int(max_frame or 0)


def first_crossing(
    points: List[Tuple[int, float, float]],
    line_a: Tuple[float, float],
    line_b: Tuple[float, float],
    max_gap_frames: int,
):
    for (f1, x1, y1), (f2, x2, y2) in zip(points, points[1:]):
        if f2 <= f1 or f2 - f1 > max_gap_frames:
            continue
        hit = line_intersection_fraction((x1, y1), (x2, y2), line_a, line_b)
        if hit is not None:
            frac, direction = hit
            return f1 + frac * (f2 - f1), direction
    return None


def process_database(db_path, site, line_name, fps, interval_minutes, max_gap_frames, type_map, extension_m):
    tracks, track_types, max_frame = load_tracks(db_path)
    line_a, line_b = extend_line(LINE_COORDS[line_name], extension_m)
    events = []
    for tid, points in tracks.items():
        hit = first_crossing(points, line_a, line_b, max_gap_frames)
        if hit is None:
            continue
        crossing_frame, direction_sign = hit
        elapsed_sec = crossing_frame / fps
        bin_index = int(elapsed_sec // (interval_minutes * 60))
        type_code = track_types[tid]
        events.append({
            "site": site,
            "source_file": db_path.name,
            "line": line_name,
            "trajectory_id": tid,
            "road_user_type_code": type_code,
            "road_user_type": type_map.get(type_code, f"type_{type_code}"),
            "direction": "positive_to_negative" if direction_sign > 0 else "negative_to_positive",
            "crossing_frame": round(crossing_frame, 3),
            "elapsed_seconds": round(elapsed_sec, 3),
            "elapsed_time": f"{int(elapsed_sec // 3600):02d}:{int(elapsed_sec % 3600 // 60):02d}:{int(elapsed_sec % 60):02d}",
            "interval_index": bin_index,
        })

    duration_sec = (max_frame + 1) / fps
    bin_seconds = interval_minutes * 60
    n_bins = max(1, math.ceil(duration_sec / bin_seconds))
    type_codes = sorted(set(track_types.values()))
    directions = ["positive_to_negative", "negative_to_positive"]
    counts = Counter((e["interval_index"], e["road_user_type_code"], e["direction"]) for e in events)
    count_rows = []
    modal_rows = []
    type_totals = Counter((e["interval_index"], e["road_user_type_code"]) for e in events)
    for bi in range(n_bins):
        start = bi * bin_seconds
        end = min((bi + 1) * bin_seconds, duration_sec)
        interval_info = {
            "site": site,
            "source_file": db_path.name,
            "line": line_name,
            "interval_index": bi,
            "interval_start": f"{int(start // 3600):02d}:{int(start % 3600 // 60):02d}:{int(start % 60):02d}",
            "interval_end": f"{int(end // 3600):02d}:{int(end % 3600 // 60):02d}:{int(end % 60):02d}",
            "interval_duration_minutes": round((end - start) / 60, 3),
            "is_partial_interval": (end - start) < bin_seconds,
        }
        total = sum(type_totals[(bi, code)] for code in type_codes)
        for code in type_codes:
            label = type_map.get(code, f"type_{code}")
            for direction in directions:
                count_rows.append({
                    **interval_info,
                    "road_user_type_code": code,
                    "road_user_type": label,
                    "direction": direction,
                    "count": counts[(bi, code, direction)],
                })
            n = type_totals[(bi, code)]
            modal_rows.append({
                **interval_info,
                "road_user_type_code": code,
                "road_user_type": label,
                "count": n,
                "modal_share_percent": round(100 * n / total, 2) if total else 0.0,
                "all_types_total": total,
            })
    summary = {
        "site": site,
        "source_file": db_path.name,
        "line": line_name,
        "tracks_loaded": len(tracks),
        "recording_duration_minutes": round(duration_sec / 60, 3),
        "interval_minutes": interval_minutes,
        "crossing_events": len(events),
    }
    return events, count_rows, modal_rows, summary


def make_summary_page(pdf, events, summary, type_map):
    type_codes = sorted(type_map)
    directions = ["positive_to_negative", "negative_to_positive"]
    direction_labels = {"positive_to_negative": "East", "negative_to_positive": "West"}
    colors = {"positive_to_negative": "#2878B5", "negative_to_positive": "#E07A2D"}
    totals = Counter((e["road_user_type_code"], e["direction"]) for e in events)
    totals_by_type = [sum(totals[(code, direction)] for direction in directions)
                      for code in type_codes]
    total_events = sum(totals_by_type)
    labels = [type_map[code] for code in type_codes]

    fig = plt.figure(figsize=(11.7, 8.3), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
    ax_count = fig.add_subplot(grid[0, 0])
    ax_share = fig.add_subplot(grid[0, 1])

    x = np.arange(len(type_codes))
    width = 0.36
    for j, direction in enumerate(directions):
        vals = [totals[(code, direction)] for code in type_codes]
        bars = ax_count.bar(x + (j - 0.5) * width, vals, width,
                            color=colors[direction],
                            label=direction_labels[direction])
        ax_count.bar_label(bars, padding=2, fontsize=8)
    ax_count.set_xticks(x, labels, rotation=25, ha="right")
    ax_count.set_ylabel("Crossing events")
    ax_count.set_title("Total crossings by road-user type and direction")
    ax_count.legend(title="Direction", frameon=False, fontsize=8)
    ax_count.grid(axis="y", alpha=0.2)

    shares = [100 * n / total_events if total_events else 0 for n in totals_by_type]
    bars = ax_share.barh(labels, shares, color="#4C956C")
    for bar, n, pct in zip(bars, totals_by_type, shares):
        y = bar.get_y() + bar.get_height() / 2
        if pct >= 78:
            ax_share.text(pct - 1.5, y, f"{n} ({pct:.1f}%)", va="center",
                          ha="right", color="white", fontsize=8, fontweight="bold")
        else:
            ax_share.text(pct + 1.2, y, f"{n} ({pct:.1f}%)", va="center",
                          ha="left", fontsize=8)
    ax_share.set_xlim(0, 100)
    ax_share.invert_yaxis()
    ax_share.set_xlabel("Share of counted crossings (%)")
    ax_share.set_title(f"Road-user composition (N = {total_events})")
    ax_share.grid(axis="x", alpha=0.2)

    fig.suptitle(
        f"Road-user count summary — {summary['site']} | {summary['line']}\n"
        f"SQLite files: {summary['files_processed']} · "
        f"Recorded time: {summary['recording_duration_minutes']:.1f} min · "
        f"LINE crossings: {summary['crossing_events']}",
        fontsize=13, fontweight="bold"
    )
    fig.text(0.5, 0.015, "East = LINE2 → LINE1; West = LINE1 → LINE2",
             ha="center", fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def make_heatmap_page(pdf, events, summary, type_map):
    type_codes = sorted(type_map)
    directions = ["positive_to_negative", "negative_to_positive"]
    direction_labels = {"positive_to_negative": "East", "negative_to_positive": "West"}
    file_labels = summary["file_labels"]
    by_file = Counter((e["source_file"], e["road_user_type_code"], e["direction"])
                      for e in events)

    column_labels = []
    for code in type_codes:
        for direction in directions:
            column_labels.append(f"{type_map[code]}\n{direction_labels[direction]}")
    column_labels.append("Total")
    row_labels = list(file_labels) + ["TOTAL"]

    matrix = []
    for file_label in file_labels:
        row = [by_file[(file_label, code, direction)]
               for code in type_codes for direction in directions]
        matrix.append(row + [sum(row)])
    column_totals = [sum(row[i] for row in matrix) for i in range(len(column_labels) - 1)]
    matrix.append(column_totals + [sum(column_totals)])
    arr = np.asarray(matrix, dtype=float)

    fig_height = max(8.3, min(16.5, 4.0 + 0.23 * len(row_labels)))
    fig, ax = plt.subplots(figsize=(16.5, fig_height), constrained_layout=True)
    im = ax.imshow(arr, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_xticks(np.arange(len(column_labels)), column_labels, fontsize=8)
    y_font = max(5, min(8, 170 // max(1, len(row_labels))))
    ax.set_yticks(np.arange(len(row_labels)), row_labels, fontsize=y_font)
    ax.set_title("LINE1 crossings by SQLite file, road-user type, and direction", pad=14)
    ax.set_xlabel("East = LINE2 → LINE1; West = LINE1 → LINE2")
    ax.tick_params(axis="both", length=0)

    vmax = arr.max() if arr.size else 0
    annotation_size = max(5, min(8, 220 // max(1, len(row_labels))))
    for ri in range(arr.shape[0]):
        for ci in range(arr.shape[1]):
            value = int(arr[ri, ci])
            ax.text(ci, ri, str(value), ha="center", va="center",
                    color="white" if value > vmax * 0.55 and vmax > 0 else "#222222",
                    fontsize=annotation_size,
                    fontweight="bold" if ri == len(row_labels) - 1 or ci == len(column_labels) - 1 else "normal")
    ax.axhline(len(file_labels) - 0.5, color="#111111", linewidth=1.4)
    ax.axvline(len(column_labels) - 1.5, color="#111111", linewidth=1.4)
    fig.colorbar(im, ax=ax, label="Crossing events", fraction=0.025, pad=0.015)
    fig.suptitle(
        f"Road-user counts — {summary['site']} | {summary['line']} | "
        f"Files: {summary['files_processed']} · Recorded time: "
        f"{summary['recording_duration_minutes']:.1f} min · "
        f"LINE crossings: {summary['crossing_events']}",
        fontsize=13, fontweight="bold"
    )
    pdf.savefig(fig)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A SQLite file or a folder containing SQLite files")
    parser.add_argument("--site", default=None, help="Site label (default: input folder name)")
    parser.add_argument("--line", choices=LINE_COORDS, default="LINE1", help="Count one line per run")
    parser.add_argument("--fps", type=float, default=FPS_DEFAULT)
    parser.add_argument("--extend-line-m", type=float, default=2.0,
                        help="Extend both ends of each measurement line by this many metres (default: 2)")
    parser.add_argument("--max-gap-frames", type=int, default=5,
                        help="Ignore sparse consecutive points farther apart than this")
    parser.add_argument("--no-recursive", action="store_true", help="Do not search subfolders")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input path not found: {args.input}")
    if args.fps <= 0 or args.extend_line_m < 0:
        parser.error("FPS must be positive and line extension cannot be negative")
    if args.input.is_file():
        if args.input.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
            parser.error("Input file must have a .sqlite, .sqlite3, or .db extension")
        db_files = [args.input]
        default_site = args.input.parent.name
    else:
        globber = args.input.rglob if not args.no_recursive else args.input.glob
        db_files = [p for pattern in ("*.sqlite", "*.sqlite3", "*.db") for p in globber(pattern) if p.is_file()]
        db_files.sort(key=natural_sort_key)
        default_site = args.input.name
    if not db_files:
        parser.error("No .sqlite, .sqlite3, or .db files found in the input path")

    site = args.site or default_site or "site"
    all_events, summaries, failures = [], [], []
    input_root = args.input if args.input.is_dir() else args.input.parent
    for db_path in db_files:
        try:
            events, _, _, summary = process_database(
                db_path, site, args.line, args.fps, 15,
                args.max_gap_frames, TYPE_LABELS, args.extend_line_m
            )
            display_file = db_path.relative_to(input_root).as_posix()
            for event in events:
                event["source_file"] = display_file
                all_events.append(event)
            summary["source_file"] = display_file
            summaries.append(summary)
            print(f"OK {db_path}: {summary['tracks_loaded']} tracks, {summary['crossing_events']} crossings")
        except (sqlite3.Error, KeyError, ValueError) as exc:
            failures.append({"source_file": str(db_path), "error": str(exc)})
            print(f"SKIP {db_path}: {exc}")

    if not summaries:
        parser.error("No readable SQLite files were processed")
    safe_site = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in site)
    summary_pdf_path = Path.cwd() / f"road_user_counts_{safe_site}_{args.line}_summary.pdf"
    heatmap_pdf_path = Path.cwd() / f"road_user_counts_{safe_site}_{args.line}_heatmap.pdf"
    report_summary = {
        "site": site,
        "line": args.line,
        "tracks_loaded": sum(s["tracks_loaded"] for s in summaries),
        "recording_duration_minutes": sum(s["recording_duration_minutes"] for s in summaries),
        "crossing_events": len(all_events),
        "files_processed": len(summaries),
        "file_labels": [s["source_file"] for s in summaries],
    }
    with PdfPages(summary_pdf_path) as pdf:
        make_summary_page(pdf, all_events, report_summary, TYPE_LABELS)
    with PdfPages(heatmap_pdf_path) as pdf:
        make_heatmap_page(pdf, all_events, report_summary, TYPE_LABELS)

    print(f"Processed successfully: {len(summaries)} / {len(db_files)} file(s)")
    print(f"Total crossing events across processed files (may include overlapping clips): {len(all_events)}")
    print(f"Summary PDF saved in the current folder: {summary_pdf_path}")
    print(f"Per-file heatmap PDF saved in the current folder: {heatmap_pdf_path}")
    print(f"Each line endpoint extended by {args.extend_line_m:g} m along the line direction.")
    print("NOTE: each SQLite file is one heatmap row; columns split road-user type by East/West direction.")
    print("NOTE: for official flow totals, count one of the two measurement lines; do not add both.")
    if failures:
        print(f"WARNING: {len(failures)} file(s) could not be read; see the messages above.")


if __name__ == "__main__":
    main()
