"""Shot launcher pieces: slider panel reader, validated scripted shots, and goal tracking."""

import math
from dataclasses import dataclass

from .config import FieldDims

SHOT_FIELDS = ("x", "y", "speed", "direction_deg", "elevation_deg")
MAX_SHOT_SPEED = 15.0
MAX_ELEVATION_DEG = 80.0


@dataclass(frozen=True)
class Shot:
    x: float
    y: float
    speed: float
    direction_deg: float  # field frame; 180 = toward our goal (-x)
    elevation_deg: float

    @staticmethod
    def from_panel(values: dict[str, float]) -> "Shot":
        return Shot(
            values["shot_origin_x"],
            values["shot_origin_y"],
            values["shot_speed"],
            values["shot_direction_deg"],
            values["shot_elevation_deg"],
        )


def shot_velocity(speed: float, direction_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    d, e = math.radians(direction_deg), math.radians(elevation_deg)
    return speed * math.cos(e) * math.cos(d), speed * math.cos(e) * math.sin(d), speed * math.sin(e)


def shot_from_dict(d: dict) -> Shot:
    unknown = set(d) - set(SHOT_FIELDS)
    if unknown:
        raise ValueError(f"unknown shot fields: {sorted(unknown)}")
    values = {}
    for key in SHOT_FIELDS:
        v = float(d[key])
        if not math.isfinite(v):
            raise ValueError(f"{key} must be finite")
        values[key] = v
    if not 0.0 <= values["speed"] <= MAX_SHOT_SPEED:
        raise ValueError(f"speed must be in [0, {MAX_SHOT_SPEED}] m/s")
    if not 0.0 <= values["elevation_deg"] <= MAX_ELEVATION_DEG:
        raise ValueError(f"elevation_deg must be in [0, {MAX_ELEVATION_DEG}]")
    return Shot(**values)


class ShotPanel:
    """Edge-triggered reader of the viewer sliders: one shot per drag of shot_fire past 0.5."""

    def __init__(self):
        self._armed = True

    def read(self, values: dict[str, float]) -> Shot | None:
        firing = values["shot_fire"] > 0.5
        if not firing:
            self._armed = True
            return None
        if self._armed:
            self._armed = False
            return Shot.from_panel(values)
        return None


class ShotTracker:
    """Classifies the latest shot against our goal (-x): 'goal' or 'no_goal'."""

    def __init__(self, fd: FieldDims, goal_height: float, timeout_s: float = 8.0, rest_speed: float = 0.05,
                 grace_s: float = 0.5):
        self.fd = fd
        self.goal_height = goal_height
        self.timeout_s = timeout_s
        self.rest_speed = rest_speed
        self.grace_s = grace_s
        self.tally = {"goal": 0, "no_goal": 0}
        self._started: float | None = None

    def start(self, now: float) -> None:
        self._started = now

    def update(self, now: float, ball_pos, ball_vel) -> str | None:
        if self._started is None:
            return None
        x, y, z = (float(v) for v in ball_pos)
        if x < -self.fd.length / 2:
            inside = abs(y) < self.fd.goal_width / 2 and z < self.goal_height
            return self._finish("goal" if inside else "no_goal")
        elapsed = now - self._started
        resting = math.hypot(float(ball_vel[0]), float(ball_vel[1])) < self.rest_speed
        if elapsed > self.timeout_s or (elapsed > self.grace_s and resting):
            return self._finish("no_goal")
        return None

    def _finish(self, outcome: str) -> str:
        self._started = None
        self.tally[outcome] += 1
        return outcome
