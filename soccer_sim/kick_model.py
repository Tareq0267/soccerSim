"""Stand-in for Booster's closed RL visual kick (SDK topic rt/kick_ball).

On the robot, brain sends kVisualKick start and then streams /kick_ball references
(ball position relative to the robot, desired kick direction, power); the firmware
approaches and kicks. This model approaches with the walk policy and, once aligned
with the real ball within reach, launches the ball. Replace it if Booster publishes
the T2 visual-kick policy.
"""

import math
from dataclasses import dataclass

from .loco_rpc import KICK_V2


@dataclass(frozen=True)
class KickRef:
    ball_x: float  # ball position in robot frame (brain::msg::Kick.x)
    ball_y: float  # brain::msg::Kick.y
    dir: float  # desired kick direction relative to robot, rad (Kick.dir)
    power: float  # desired kick range, m (Kick.power)
    stamp: float  # seconds, caller's clock


@dataclass(frozen=True)
class KickParams:
    # ponytail: hand-tuned approximation of the firmware kick; calibrate against real kick videos.
    # Tuned in sim: at 0.25 m the launched ball hits the T2's swinging foot (1.1 m/s, deflected);
    # 0.35-0.45 m all launch cleanly, 0.40 is mid-band.
    strike_ball_x: float = 0.40  # ball distance in front of the trunk at contact
    approach_gain: float = 1.5
    yaw_gain: float = 1.5
    max_vx: float = 0.8
    max_vy: float = 0.4
    max_vyaw: float = 1.0
    strike_tol_x: float = 0.08
    strike_tol_y: float = 0.06
    strike_tol_dir: float = 0.25
    reach: float = 0.55  # the real ball must be this close for a strike
    ball_decel: float = 0.8  # m/s^2 rolling deceleration used to turn "range" into launch speed
    v2_power_scale: float = 1.3  # VisualKickVersion.kV2 is documented as "stronger kicking force"
    max_ball_speed: float = 8.0
    ref_timeout: float = 0.5
    cooldown: float = 0.8


@dataclass(frozen=True)
class KickOutput:
    vx: float
    vy: float
    vyaw: float
    strike_speed: float = 0.0
    strike_dir: float = 0.0


def ball_speed_for_power(power: float, version: int, p: KickParams) -> float:
    """Launch speed that rolls the ball `power` metres under constant deceleration."""
    speed = math.sqrt(2.0 * p.ball_decel * max(0.0, power))
    if version == KICK_V2:
        speed *= p.v2_power_scale
    return min(p.max_ball_speed, speed)


def _clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))


STOP = KickOutput(0.0, 0.0, 0.0)


class VisualKickModel:
    def __init__(self, params: KickParams = KickParams()):
        self.p = params
        self._cooldown_until = -math.inf

    def reset(self) -> None:
        self._cooldown_until = -math.inf

    def step(self, ref: KickRef | None, now: float, true_ball_robot: tuple[float, float], version: int) -> KickOutput:
        p = self.p
        if ref is None or now - ref.stamp > p.ref_timeout or now < self._cooldown_until:
            return STOP
        ex = ref.ball_x - p.strike_ball_x
        ey = ref.ball_y
        edir = math.atan2(math.sin(ref.dir), math.cos(ref.dir))
        aligned = abs(ex) < p.strike_tol_x and abs(ey) < p.strike_tol_y and abs(edir) < p.strike_tol_dir
        if aligned and math.hypot(*true_ball_robot) < p.reach:
            self._cooldown_until = now + p.cooldown
            return KickOutput(0.0, 0.0, 0.0, ball_speed_for_power(ref.power, version, p), ref.dir)
        return KickOutput(
            _clamp(p.approach_gain * ex, p.max_vx),
            _clamp(p.approach_gain * ey, p.max_vy),
            _clamp(p.yaw_gain * edir, p.max_vyaw),
        )
