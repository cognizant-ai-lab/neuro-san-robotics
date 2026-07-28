"""Shared temporal confirmation for noisy obstacle readings."""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ObstacleConfirmationTracker:
    """Confirm repeated obstacle readings before they affect navigation state."""

    min_seconds: float
    min_readings: int
    distance_tolerance_m: Optional[float] = None
    bearing_tolerance_rad: Optional[float] = None
    first_seen_at: Optional[float] = None
    count: int = 0
    distance_m: float = float("inf")
    bearing_rad: float = 0.0

    def update(
        self,
        distance_m: float,
        bearing_rad: float,
        now: Optional[float] = None,
    ) -> Tuple[bool, bool]:
        """Record one obstacle reading and return (confirmed, started_new_track)."""
        now = time.monotonic() if now is None else now
        started_new_track = not self._matches_track(distance_m, bearing_rad)

        if started_new_track:
            self.first_seen_at = now
            self.count = 1
            self.distance_m = distance_m
            self.bearing_rad = bearing_rad
        else:
            self.count += 1
            self.distance_m = min(self.distance_m, distance_m)
            self.bearing_rad = bearing_rad

        confirmed = (
            now - self.first_seen_at >= self.min_seconds
            and self.count >= self.min_readings
        )
        return confirmed, started_new_track

    def reset(self) -> None:
        """Forget the pending obstacle track."""
        self.first_seen_at = None
        self.count = 0
        self.distance_m = float("inf")
        self.bearing_rad = 0.0

    def _matches_track(self, distance_m: float, bearing_rad: float) -> bool:
        """Return True when a reading belongs to the current pending track."""
        if self.first_seen_at is None:
            return False

        if (
            self.distance_tolerance_m is not None
            and abs(distance_m - self.distance_m) > self.distance_tolerance_m
        ):
            return False

        if self.bearing_tolerance_rad is not None:
            bearing_delta = abs(
                math.atan2(
                    math.sin(bearing_rad - self.bearing_rad),
                    math.cos(bearing_rad - self.bearing_rad),
                )
            )
            if bearing_delta > self.bearing_tolerance_rad:
                return False

        return True
