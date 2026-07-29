#!/usr/bin/env python3
"""Build the runtime occupancy grid from the annotated office floor plan.

The generated NPZ is intentionally committed so robot deployments only need
NumPy.  Pillow is required solely when regenerating the map after a floor-plan
change.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from PIL import Image


def _long_runs(mask: np.ndarray, dr: int, dc: int, minimum: int) -> np.ndarray:
    """Keep dark pixels belonging to a long horizontal/vertical/diagonal run."""
    rows, cols = mask.shape
    kept = np.zeros_like(mask)
    starts = []
    for row in range(rows):
        for col in range(cols):
            previous_row, previous_col = row - dr, col - dc
            if not (0 <= previous_row < rows and 0 <= previous_col < cols):
                starts.append((row, col))

    for start_row, start_col in starts:
        run = []
        row, col = start_row, start_col
        while 0 <= row < rows and 0 <= col < cols:
            if mask[row, col]:
                run.append((row, col))
            else:
                if len(run) >= minimum:
                    rr, cc = zip(*run)
                    kept[rr, cc] = True
                run = []
            row += dr
            col += dc
        if len(run) >= minimum:
            rr, cc = zip(*run)
            kept[rr, cc] = True
    return kept


def _inside_polygon(
    x: np.ndarray,
    y: np.ndarray,
    points: Sequence[Sequence[float]],
) -> np.ndarray:
    """Vectorized even/odd polygon fill at cell centers."""
    inside = np.zeros_like(x, dtype=bool)
    x1, y1 = points[-1]
    for x2, y2 in points:
        crosses = ((y1 > y) != (y2 > y)) & (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1
        )
        inside ^= crosses
        x1, y1 = x2, y2
    return inside


def build_occupancy(
    map_json: Path,
    floor_plan: Path,
    *,
    resolution_m: float,
    gray_threshold: int,
    minimum_structure_length_m: float,
) -> Tuple[np.ndarray, dict]:
    data = json.loads(map_json.read_text(encoding="utf-8"))
    coordinates = data["coordinate_system"]
    bbox = coordinates["source_floor_bbox_px"]
    dimensions = coordinates["floor_dimensions_m"]

    image = np.asarray(Image.open(floor_plan).convert("RGB"), dtype=np.uint8)
    crop = image[
        int(bbox["top"]):int(bbox["bottom"]),
        int(bbox["left"]):int(bbox["right"]),
    ].copy()

    # Red circles and labels are annotations, not physical obstacles.  Their
    # black borders are removed later by the minimum structural-run filter.
    red = (
        (crop[:, :, 0] >= 145)
        & (crop[:, :, 0] >= crop[:, :, 1].astype(np.int16) * 1.35)
        & (crop[:, :, 0] >= crop[:, :, 2].astype(np.int16) * 1.35)
    )
    crop[red] = 255

    width_cells = int(math.ceil(float(dimensions["east_west"]) / resolution_m))
    height_cells = int(math.ceil(float(dimensions["north_south"]) / resolution_m))
    # Reverse both source axes and transpose: source-up is world +x and
    # source-left is world +y. Rows in the result are world y; columns are x.
    transformed = np.transpose(crop[::-1, ::-1], (1, 0, 2))
    resized = np.asarray(
        Image.fromarray(transformed).resize(
            (width_cells, height_cells),
            resample=Image.Resampling.BOX,
        ),
        dtype=np.float32,
    )
    gray = 0.299 * resized[:, :, 0] + 0.587 * resized[:, :, 1] + 0.114 * resized[:, :, 2]
    dark = gray < gray_threshold

    minimum_cells = max(2, int(math.ceil(minimum_structure_length_m / resolution_m)))
    occupied = np.zeros_like(dark)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        occupied |= _long_runs(dark, dr, dc, minimum_cells)

    yy, xx = np.indices(occupied.shape, dtype=np.float32)
    world_x = (xx + 0.5) * resolution_m
    world_y = (yy + 0.5) * resolution_m
    navigable_boundary = data.get("navigable_boundary", data["floor_boundary"])
    occupied |= ~_inside_polygon(world_x, world_y, navigable_boundary)

    for zone in data.get("exclusion_zones", []):
        if zone.get("type") != "polygon" or len(zone.get("points", [])) < 3:
            continue
        occupied |= _inside_polygon(world_x, world_y, zone["points"])

    metadata = {
        "resolution_m": resolution_m,
        "origin_x_m": 0.0,
        "origin_y_m": 0.0,
        "robot_clearance_m": float(data["occupancy_map"]["robot_clearance_m"]),
        "preferred_clearance_m": float(data["occupancy_map"]["preferred_clearance_m"]),
    }
    return occupied, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, default=Path(__file__).with_name("cail_lab.json"))
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).with_name("535-mission-map.png"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("cail_lab_occupancy.npz"),
    )
    parser.add_argument("--resolution", type=float, default=0.10)
    parser.add_argument("--gray-threshold", type=int, default=195)
    parser.add_argument("--minimum-structure-length", type=float, default=0.70)
    args = parser.parse_args()

    occupied, metadata = build_occupancy(
        args.map,
        args.image,
        resolution_m=args.resolution,
        gray_threshold=args.gray_threshold,
        minimum_structure_length_m=args.minimum_structure_length,
    )
    np.savez_compressed(
        args.output,
        occupied=occupied.astype(np.uint8),
        **{key: np.asarray(value) for key, value in metadata.items()},
    )
    print(
        f"wrote {args.output}: {occupied.shape[1]}x{occupied.shape[0]} cells, "
        f"{100.0 * occupied.mean():.1f}% occupied"
    )


if __name__ == "__main__":
    main()
