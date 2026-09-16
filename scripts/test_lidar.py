#!/usr/bin/env python3
"""
Check the Unitree LiDAR before letting it steer the robot.

Navigation can sense obstacles with the depth camera, the LiDAR, or both. The
LiDAR sees all round rather than through the camera's narrow cone, which is
what makes it worth having for free navigation -- moving without a map, where
nothing but live sensing keeps the robot off the furniture.

Two things can be wrong, and they look identical from across the room:

  1. No data. DDS is not up, the topic is wrong, or the dome is not spinning.
     The robot then navigates on depth alone and nobody notices until it
     clips something outside the camera's cone.

  2. Data arriving rotated. The point cloud is rotated into the robot's frame
     by NAV_LIDAR_POINTCLOUD_YAW_OFFSET_RAD, which describes how the unit is
     mounted. Get it wrong and the map is turned: the robot swerves around
     empty floor and walks into a real wall. Nothing errors.

This reports the first and lets you see the second, by drawing what the robot
believes is around it. Stand somewhere specific and check the picture agrees.

Usage:
    python scripts/test_lidar.py                  # watch until Ctrl-C
    python scripts/test_lidar.py --seconds 5      # one look and exit
    python scripts/test_lidar.py --sweep          # try candidate mounting angles
    python scripts/test_lidar.py --yaw-offset 70  # check one angle, in degrees

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

# Angles worth trying when the picture looks rotated. 70 degrees is the mounting
# on the robots this repo was developed against and is the default.
SWEEP_DEGREES = (0.0, 45.0, 70.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)

# Sectors used to describe what is around the robot, as (label, centre degrees).
# Positive bearings are to the left, matching the obstacle grid.
SECTORS = (
    ("ahead", 0.0),
    ("left", 90.0),
    ("behind", 180.0),
    ("right", -90.0),
)

# Everything shown, kept so --out can write it to a file. A sweep prints nine
# drawings; over ssh they scroll away exactly when you need to compare them.
_REPORT: list[str] = []


def emit(line: str = "") -> None:
    """Show a line and keep it for the report file."""
    print(line)
    _REPORT.append(line)


def write_report(path: Path) -> None:
    """Write everything shown so far, and say where it went."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_REPORT) + "\n", encoding="utf-8")
    print(f"\nreport written to {path}")


def _bearing_label(degrees: float) -> str:
    """Name the sector a bearing falls in, for a one-line summary."""
    best = min(SECTORS, key=lambda s: abs(_wrap_degrees(degrees - s[1])))
    return best[0]


def _wrap_degrees(degrees: float) -> float:
    """Fold an angle into -180..180 so comparisons behave near the wrap."""
    return (degrees + 180.0) % 360.0 - 180.0


def render(grid, width: int = 31, span_m: float = 3.0) -> str:
    """
    Draw a top-down view of the obstacle grid, robot at the centre.

    Text rather than a plot because this runs over ssh on the robot, which is
    where the answer is needed. Up is straight ahead.
    """
    import numpy as np

    half = width // 2
    cells_per_char = max(1, int(round(span_m / grid.resolution / width)))
    rows = []
    for screen_row in range(-half, half + 1):
        line = []
        for screen_col in range(-half, half + 1):
            if screen_row == 0 and screen_col == 0:
                line.append("R")
                continue
            # Screen row grows downward; the grid's forward axis grows upward.
            top = grid.origin_row + screen_row * cells_per_char
            left = grid.origin_col - screen_col * cells_per_char
            block = grid.grid[
                max(0, top): max(0, top) + cells_per_char,
                max(0, left): max(0, left) + cells_per_char,
            ]
            if block.size == 0:
                line.append(" ")
            elif float(np.max(block)) > 0:
                line.append("#")
            else:
                line.append(".")
        rows.append("".join(line))

    metres = (width // 2) * cells_per_char * grid.resolution
    header = f"   forward is up, {metres:.1f} m to an edge, R is the robot"
    return header + "\n" + "\n".join("   " + row for row in rows)


def describe(grid) -> str:
    """One line on the nearest obstacle, in words rather than radians."""
    if grid.nearest_obstacle_m == float("inf"):
        return "nothing within range"
    degrees = math.degrees(grid.nearest_obstacle_bearing)
    return (
        f"nearest {grid.nearest_obstacle_m:.2f} m at {degrees:+.0f} deg "
        f"({_bearing_label(degrees)})"
    )


def sample_once(service, timeout_s: float):
    """Wait for one fresh grid, or None if none arrives in time."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        grid = service.get_obstacle_grid()
        if grid is not None:
            return grid
        time.sleep(0.1)
    return None


def start_service(yaw_offset_deg=None):
    """Build and start a LiDAR service, optionally overriding the mounting."""
    from coded_tools.unigo2.lidar_processor import LidarPerimeterService

    if yaw_offset_deg is not None:
        os.environ["NAV_LIDAR_POINTCLOUD_YAW_OFFSET_RAD"] = str(
            math.radians(yaw_offset_deg)
        )
    # This reads the environment, so it has to be built after the override.
    service = LidarPerimeterService()
    service.start()
    return service


def run_sweep(timeout_s: float) -> int:
    """
    Draw the same surroundings at each candidate mounting angle.

    Only one of them will match the room you are standing in. That is the value
    for NAV_LIDAR_POINTCLOUD_YAW_OFFSET_RAD, in radians.
    """
    emit("Stand somewhere unmistakable -- a metre in front of the robot, or in")
    emit("a doorway -- and pick the angle whose picture matches the room.\n")
    for degrees in SWEEP_DEGREES:
        service = start_service(degrees)
        grid = sample_once(service, timeout_s)
        service.stop()
        radians = math.radians(degrees)
        if grid is None:
            emit(f"--- {degrees:5.1f} deg ({radians:.4f} rad): no data")
            continue
        emit(f"--- {degrees:5.1f} deg ({radians:.4f} rad): {describe(grid)}")
        emit(render(grid))
        emit()
    emit("Set the winner in setmyenv.sh, in radians:")
    emit('    export NAV_LIDAR_POINTCLOUD_YAW_OFFSET_RAD="1.2217"   # 70 degrees')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the Unitree LiDAR feed")
    parser.add_argument("--seconds", type=float, default=None,
                        help="watch for this long then exit (default: until Ctrl-C)")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="how long to wait for the first sample (default: 5)")
    parser.add_argument("--yaw-offset", type=float, default=None, metavar="DEG",
                        help="try this mounting angle in degrees, without editing the env")
    parser.add_argument("--sweep", action="store_true",
                        help="draw the surroundings at each candidate mounting angle")
    parser.add_argument("--quiet", action="store_true",
                        help="summary only, no picture")
    parser.add_argument("--out", type=Path, default=None, metavar="PATH",
                        help="also write the report here, for sharing or comparing")
    args = parser.parse_args()

    topic = os.environ.get("NAV_LIDAR_TOPIC", "rt/utlidar/cloud")
    interface = (os.environ.get("GO2_NETWORK_INTERFACE")
                 or os.environ.get("CYCLONEDDS_NETWORK_INTERFACE") or "(default)")
    emit(f"topic     : {topic}")
    emit(f"interface : {interface}")

    if args.sweep:
        status = run_sweep(args.timeout)
        if args.out:
            write_report(args.out)
        return status

    service = start_service(args.yaw_offset)
    backend = service.backend
    emit(f"backend   : {backend}")
    if backend in {"none", "disabled"}:
        # Started but not subscribed: DDS never came up, or LiDAR is switched off.
        reason = service.subscriber_error
        if reason:
            emit(f"reason    : {reason}")
        # Flush first so the report above is not interleaved with the advice
        # below when both are going to the same terminal.
        sys.stdout.flush()
        print("\nNo LiDAR subscription. Things to check, in order:", file=sys.stderr)
        print("  - the dome spins freely and the robot is powered up", file=sys.stderr)
        print("  - setmyenv.sh has been sourced (CYCLONEDDS_HOME, CYCLONEDDS_URI)",
              file=sys.stderr)
        print("  - GO2_NETWORK_INTERFACE names the interface facing the robot",
              file=sys.stderr)
        print('  - NAV_USE_LIDAR is not set to 0', file=sys.stderr)
        service.stop()
        if args.out:
            write_report(args.out)
        return 1

    grid = sample_once(service, args.timeout)
    if grid is None:
        sys.stdout.flush()
        print(f"\nSubscribed to {topic} but no sample arrived within "
              f"{args.timeout:.0f}s.", file=sys.stderr)
        print("The topic exists but nothing is publishing -- check the dome is "
              "spinning.", file=sys.stderr)
        service.stop()
        if args.out:
            write_report(args.out)
        return 1

    offset_rad = float(os.environ.get("NAV_LIDAR_POINTCLOUD_YAW_OFFSET_RAD",
                                      math.radians(70.0)))
    emit(f"mounting  : {math.degrees(offset_rad):.1f} deg "
          f"({offset_rad:.4f} rad) -- NAV_LIDAR_POINTCLOUD_YAW_OFFSET_RAD")
    emit(f"first look: {describe(grid)}")
    if not args.quiet:
        emit(render(grid))
    emit("\nIf the picture does not match the room, the mounting angle is wrong."
          "\nRun with --sweep to find the right one.")

    deadline = None if args.seconds is None else time.monotonic() + args.seconds
    try:
        while deadline is None or time.monotonic() < deadline:
            time.sleep(1.0)
            grid = service.get_obstacle_grid()
            if grid is None:
                emit("  stale: no fresh sample in the last second")
                continue
            emit(f"  {describe(grid)}")
    except KeyboardInterrupt:
        print()
    finally:
        service.stop()
        if args.out:
            write_report(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
