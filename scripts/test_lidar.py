#!/usr/bin/env python3
"""
Check and calibrate the Unitree LiDAR before letting it steer the robot.

Navigation can sense obstacles with the depth camera, the LiDAR, or both. The
camera sees a narrow cone in front; the LiDAR sees all round. That matters most
in free navigation, moving without a map, where nothing but live sensing keeps
the robot off the furniture.

Two things go wrong here and neither raises an error.

  1. No data. DDS is not up, the topic is wrong, or the dome is not spinning.
     The robot then navigates on depth alone and nobody notices until it clips
     something outside the camera's cone.

  2. Data arriving rotated. The LiDAR reports points in its own frame, and the
     sensor is bolted to the head at an angle to the robot's nose.
     NAV_LIDAR_POINTCLOUD_YAW_OFFSET_RAD rotates them into the robot's frame.
     Coverage is 360 degrees either way, so nothing goes missing: what changes
     is the direction every object is filed under. With the wrong offset a wall
     in front is recorded as a wall to the left, the forward corridor check
     reads a strip of the map pointing somewhere else, and the robot walks into
     something it can see perfectly well.

Usage:
    python scripts/test_lidar.py                 # check the feed, save a picture
    python scripts/test_lidar.py --calibrate     # measure the mounting angle
    python scripts/test_lidar.py --seconds 10    # merge scans for longer

Exit status is 0 only when usable LiDAR data arrived, so this works as a
bring-up gate.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Where snapshots are written, and how many are kept. Old ones are pruned so a
# robot checked often does not slowly fill its disk.
CHECK_DIR = Path(os.environ.get("NAV_LIDAR_CHECK_DIR", "~/lidar_checks"))
KEEP_SNAPSHOTS = 10

# Metres across a saved picture. Wide enough to show a room, tight enough that
# something a metre away is obvious.
VIEW_SPAN_M = 6.0

# How long to gather scans for. One rotation lays down a thin scatter of points,
# which the obstacle grid copes with and a person cannot read.
DEFAULT_GATHER_S = 4.0


def accumulate(service, seconds: float):
    """
    Merge scans over a few seconds into one picture.

    Returns (merged cells, distinct samples seen, last grid). Holding the
    maximum fills walls in, so the shape of the room appears rather than a
    sparse dusting of returns.
    """
    import numpy as np

    merged = None
    last = None
    seen_at = set()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        grid = service.get_obstacle_grid()
        if grid is not None:
            last = grid
            if grid.timestamp not in seen_at:
                seen_at.add(grid.timestamp)
                merged = (grid.grid.copy() if merged is None
                          else np.maximum(merged, grid.grid))
        time.sleep(0.05)
    return merged, len(seen_at), last


def render_png(cells, grid, title: str, size: int = 520):
    """
    Draw merged scans as an image, robot centred and facing up.

    The obstacle grid stores forward as decreasing row and left as decreasing
    column, so the array is drawn as it stands.
    """
    import cv2
    import numpy as np

    half_cells = int(VIEW_SPAN_M / 2 / grid.resolution)
    top = grid.origin_row - half_cells
    left = grid.origin_col - half_cells
    window = np.zeros((half_cells * 2, half_cells * 2), dtype=np.float32)
    src_top, src_left = max(0, top), max(0, left)
    src = cells[src_top:top + half_cells * 2, src_left:left + half_cells * 2]
    if src.size:
        window[src_top - top:src_top - top + src.shape[0],
               src_left - left:src_left - left + src.shape[1]] = src

    image = np.zeros((*window.shape, 3), dtype=np.uint8)
    image[:] = (28, 24, 20)
    image[window > 0] = (255, 232, 120)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)

    px_per_m = size / VIEW_SPAN_M
    centre = size // 2
    for metres in (1, 2):
        radius = int(metres * px_per_m)
        cv2.circle(image, (centre, centre), radius, (70, 70, 70), 1)
        cv2.putText(image, f"{metres}m", (centre + radius - 24, centre - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (125, 125, 125), 1)

    cv2.arrowedLine(image, (centre, centre),
                    (centre, centre - int(0.9 * px_per_m)),
                    (120, 255, 120), 2, tipLength=0.25)
    cv2.putText(image, "FRONT", (centre + 10, centre - int(0.9 * px_per_m) + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 255, 120), 1)
    cv2.circle(image, (centre, centre), 5, (80, 80, 255), -1)

    banner = np.zeros((26, size, 3), dtype=np.uint8)
    cv2.putText(banner, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (235, 235, 235), 1)
    return np.vstack([banner, image])


def save_png(image, label: str) -> Path:
    """Write a snapshot, prune the oldest beyond KEEP_SNAPSHOTS, return the path."""
    import cv2

    directory = CHECK_DIR.expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    # Milliseconds because two runs in the same second would otherwise write to
    # the same name, and the older picture would be lost rather than pruned.
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    path = directory / f"lidar-{label}-{stamp}.png"
    cv2.imwrite(str(path), image)

    snapshots = sorted(directory.glob("lidar-*.png"), key=lambda p: p.stat().st_mtime)
    for stale in snapshots[:-KEEP_SNAPSHOTS]:
        stale.unlink(missing_ok=True)
    return path


# Only look for the calibration target in this band. Anything nearer is the
# robot seeing its own body: the self mask is a box around the chassis, and
# returns sitting right on its edge survive it. Measuring off those gives a
# bearing to whichever part of the robot leaked through, which changes between
# rotations. Further out is furniture and walls, which would win on a clear
# floor and give an angle measured off the wrong thing entirely.
TARGET_FAR_M = 2.0

# Clear of the self mask by a margin, derived from the mask so the two cannot
# drift apart.
SELF_MASK_MARGIN_M = 0.15


def target_near_m() -> float:
    """Closest a calibration target may be and still be the robot's own body."""
    mask = max(
        float(os.environ.get("NAV_LIDAR_SELF_MASK_FORWARD", 0.45)),
        float(os.environ.get("NAV_LIDAR_SELF_MASK_REAR", 0.35)),
        float(os.environ.get("NAV_LIDAR_SELF_MASK_HALF_WIDTH", 0.25)),
    )
    return mask + SELF_MASK_MARGIN_M

# Cells within this radius of the closest one are treated as the same object,
# so the bearing is to the middle of the target rather than to whichever corner
# of it happened to return first.
TARGET_RADIUS_M = 0.25

# Independent measurements taken per calibration run. The median of several is
# used, because any single one can be thrown by a passing reflection.
CALIBRATION_ROUNDS = 5


def occupied_xy(cells, grid):
    """Occupied cells as (x forward, y left) metres from the robot."""
    import numpy as np

    rows, cols = np.nonzero(cells > 0)
    if rows.size == 0:
        return None, None
    # build_obstacle_grid stores row = origin_row - x/res and
    # col = origin_col - y/res, so both invert the same way.
    x = (grid.origin_row - rows).astype(np.float32) * grid.resolution
    y = (grid.origin_col - cols).astype(np.float32) * grid.resolution
    return x, y


def measure_bearing(cells, grid):
    """
    Bearing to the middle of the nearest object, in degrees.

    Taken over merged scans rather than one rotation, and over the whole object
    rather than its nearest cell. A single cell moves between rotations, which
    made repeated calibrations disagree by tens of degrees.

    Returns (degrees, distance) or (None, None) when nothing is in the band.
    """
    import numpy as np

    x, y = occupied_xy(cells, grid)
    if x is None:
        return None, None

    distance = np.hypot(x, y)
    in_band = (distance >= target_near_m()) & (distance <= TARGET_FAR_M)
    if not np.any(in_band):
        return None, None
    x, y, distance = x[in_band], y[in_band], distance[in_band]

    closest = int(np.argmin(distance))
    same_object = np.hypot(x - x[closest], y - y[closest]) <= TARGET_RADIUS_M

    # Circular mean, so an object straddling the back of the robot does not
    # average its bearings to the front.
    bearings = np.arctan2(y[same_object], x[same_object])
    mean = math.atan2(float(np.mean(np.sin(bearings))),
                      float(np.mean(np.cos(bearings))))
    return math.degrees(mean), float(np.mean(distance[same_object]))


def describe(grid) -> str:
    """One line on the nearest obstacle, in degrees rather than radians."""
    if grid is None or grid.nearest_obstacle_m == float("inf"):
        return "nothing within range"
    degrees = math.degrees(grid.nearest_obstacle_bearing)
    side = "ahead" if abs(degrees) < 10 else ("left" if degrees > 0 else "right")
    return f"nearest {grid.nearest_obstacle_m:.2f} m at {degrees:+.1f} deg ({side})"


def start_service(yaw_offset_deg=None):
    """Build and start a LiDAR service, optionally overriding the mounting angle."""
    from coded_tools.unigo2.lidar_processor import LidarPerimeterService

    if yaw_offset_deg is not None:
        os.environ["NAV_LIDAR_POINTCLOUD_YAW_OFFSET_RAD"] = str(
            math.radians(yaw_offset_deg))
    # The service reads the environment, so build it after any override.
    service = LidarPerimeterService()
    service.start()
    return service


def report_no_data(service, topic: str, timeout: float) -> int:
    """Explain a dead feed, in the order worth checking."""
    reason = service.subscriber_error
    if reason:
        print(f"reason    : {reason}")
    sys.stdout.flush()
    print(f"\nNo usable LiDAR data within {timeout:.0f}s. Check, in order:",
          file=sys.stderr)
    print("  - the dome spins freely and the robot is powered up", file=sys.stderr)
    print("  - setmyenv.sh has been sourced (CYCLONEDDS_HOME, CYCLONEDDS_URI)",
          file=sys.stderr)
    print("  - GO2_NETWORK_INTERFACE names the interface facing the robot",
          file=sys.stderr)
    print("  - NAV_USE_LIDAR is not set to 0", file=sys.stderr)
    print(f"  - something is publishing on {topic}", file=sys.stderr)
    return 1


def run_calibrate(gather_s: float, topic: str) -> int:
    """
    Measure the mounting angle rather than guessing at pictures.

    With the rotation switched off, whatever sits directly in front of the nose
    is reported at the sensor's own bearing. That bearing is the mounting angle,
    so the offset that corrects it is its negative.

    Several measurements are taken and the median used, because one rotation can
    be thrown by a reflection and a single reading is not worth trusting.
    """
    import numpy as np

    print("Put one unmistakable object a metre directly in front of the nose,")
    print("closer than anything else within two metres, then stand clear.")
    print(f"Returns nearer than {target_near_m():.2f} m are ignored: that close"
          " is the robot's own body.\n")

    # Measure with the configured offset applied rather than with no rotation.
    # The self mask is a box in the robot's frame and is applied after the
    # rotation, so un-rotating the cloud moves the robot's body out from under
    # its own mask: the chassis and rear legs then survive as obstacles half a
    # metre away, and the bearing measured is to whichever part of the robot
    # leaked through. Keeping the current offset leaves the mask aligned, and
    # the answer comes out as a correction to it.
    current_deg = math.degrees(
        float(os.environ.get("NAV_LIDAR_POINTCLOUD_YAW_OFFSET_RAD",
                             math.radians(70.0))))
    print(f"measuring with the configured {current_deg:.1f} deg applied, so the"
          " robot stays masked\n")

    service = start_service(yaw_offset_deg=current_deg)
    if service.backend in {"none", "disabled"}:
        status = report_no_data(service, topic, gather_s)
        service.stop()
        return status

    window = max(0.8, gather_s / CALIBRATION_ROUNDS)
    readings, distances, total = [], [], 0
    for round_index in range(CALIBRATION_ROUNDS):
        cells, samples, grid = accumulate(service, window)
        total += samples
        if cells is None or grid is None:
            continue
        degrees, distance = measure_bearing(cells, grid)
        if degrees is None:
            continue
        readings.append(degrees)
        distances.append(distance)
        print(f"  round {round_index + 1}: {degrees:+.1f} deg "
              f"at {distance:.2f} m ({samples} scans)")
    service.stop()

    if not readings:
        print(f"\nNothing found between {target_near_m():.2f} m and "
              f"{TARGET_FAR_M:.1f} m. Place an object in that band and retry.",
              file=sys.stderr)
        return 1

    measured_deg = float(np.median(readings))
    spread = max(readings) - min(readings)
    # The target is physically straight ahead, so whatever bearing it comes back
    # at is how far the configured offset is out. Subtracting corrects it.
    offset_deg = current_deg - measured_deg
    offset_rad = math.radians(offset_deg)

    print(f"\nsamples   : {total} scans over {CALIBRATION_ROUNDS} rounds")
    print(f"target    : {np.mean(distances):.2f} m away")
    print(f"target sits at: {measured_deg:+.1f} deg median, "
          f"{spread:.1f} deg spread")
    if abs(measured_deg) < 5.0:
        print(f"That is close to straight ahead, so {current_deg:.1f} deg is"
              " already right for this robot.")

    if spread > 15.0:
        # Readings this inconsistent are measuring different things, so the
        # median is not meaningful and should not be pasted into a config.
        print(f"\nReadings disagree by {spread:.0f} degrees, so this is not a"
              " reliable measurement.", file=sys.stderr)
        print("Something other than the target is the closest thing, or the"
              " target moved.", file=sys.stderr)
        print("Clear the floor around the robot, leave one object about a metre"
              " ahead, and retry.", file=sys.stderr)
        return 1

    print(f"\nMounting angle is {offset_deg:+.1f} deg "
          f"({current_deg:.1f} measured minus {measured_deg:+.1f} residual).")
    print("Put this in setmyenv.sh:")
    print(f'    export NAV_LIDAR_POINTCLOUD_YAW_OFFSET_RAD="{offset_rad:.4f}"'
          f'   # {offset_deg:.1f} degrees')

    # Confirmation. The same surroundings redrawn with the measured offset
    # applied: the object should now sit straight up from the robot.
    confirm = start_service(yaw_offset_deg=offset_deg)
    cells, _, grid = accumulate(confirm, gather_s)
    confirm.stop()
    if cells is not None and grid is not None:
        after_deg, after_m = measure_bearing(cells, grid)
        if after_deg is not None:
            print(f"\nafter     : target now at {after_deg:+.1f} deg, "
                  f"{after_m:.2f} m")
        title = f"calibrated {offset_deg:+.1f} deg | spread {spread:.1f} deg"
        print(f"snapshot  : {save_png(render_png(cells, grid, title), 'calibrate')}")
        print("The object should be straight up from the robot in that picture.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the Unitree LiDAR feed, or measure its mounting angle")
    parser.add_argument("--calibrate", action="store_true",
                        help="measure NAV_LIDAR_POINTCLOUD_YAW_OFFSET_RAD")
    parser.add_argument("--seconds", type=float, default=DEFAULT_GATHER_S,
                        help=f"seconds of scans to merge (default: {DEFAULT_GATHER_S:.0f})")
    parser.add_argument("--yaw-offset", type=float, default=None, metavar="DEG",
                        help="try this mounting angle without editing the env")
    parser.add_argument("--no-png", action="store_true", help="skip the snapshot")
    args = parser.parse_args()

    topic = os.environ.get("NAV_LIDAR_TOPIC", "rt/utlidar/cloud")
    interface = (os.environ.get("GO2_NETWORK_INTERFACE")
                 or os.environ.get("CYCLONEDDS_NETWORK_INTERFACE") or "(default)")
    print(f"topic     : {topic}")
    print(f"interface : {interface}")

    if args.calibrate:
        return run_calibrate(args.seconds, topic)

    service = start_service(args.yaw_offset)
    print(f"backend   : {service.backend}")
    if service.backend in {"none", "disabled"}:
        status = report_no_data(service, topic, args.seconds)
        service.stop()
        return status

    offset_rad = float(os.environ.get("NAV_LIDAR_POINTCLOUD_YAW_OFFSET_RAD",
                                      math.radians(70.0)))
    print(f"mounting  : {math.degrees(offset_rad):.1f} deg ({offset_rad:.4f} rad)")

    cells, samples, grid = accumulate(service, args.seconds)
    service.stop()
    if cells is None or grid is None:
        return report_no_data(service, topic, args.seconds)

    print(f"samples   : {samples} scans merged over {args.seconds:.0f}s")
    print(f"nearest   : {describe(grid)}")

    if not args.no_png:
        title = (f"{math.degrees(offset_rad):.1f} deg | {samples} scans | "
                 f"{describe(grid)}")
        print(f"snapshot  : {save_png(render_png(cells, grid, title), 'check')}")
        print("\nIf the walls in that picture do not match the room, the mounting")
        print("angle is wrong. Measure it with:")
        print("    python scripts/test_lidar.py --calibrate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
