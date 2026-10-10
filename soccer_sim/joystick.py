"""Booster-remote sticks -> velocity, the way the robot's firmware lets the remote override brain."""

from dataclasses import dataclass

DEADZONE = 0.1  # brain's go_manual threshold in Brain::joystickCallback


@dataclass(frozen=True)
class ManualLimits:
    # ponytail: comfortable driving speeds; the real remote's scaling lives in Booster firmware.
    vx: float = 1.0
    vy: float = 0.5
    vyaw: float = 1.0


def manual_velocity(lx: float, ly: float, rx: float, ry: float,
                    limits: ManualLimits = ManualLimits()) -> tuple[float, float, float] | None:
    """None = sticks centred (brain keeps control). Axes follow RemoteControllerState:
    ly = -1 is push front, lx/rx = -1 is push left."""
    if max(abs(lx), abs(ly), abs(rx), abs(ry)) <= DEADZONE:
        return None
    return -ly * limits.vx, -lx * limits.vy, -rx * limits.vyaw


class FirmwareRemote:
    """Robot-side remote combo used by the boot sequence: LT+Up stands the T2 up into soccer mode.

    ponytail: assumed meaning of LT+Up from the team's match procedure; Booster's firmware remote
    map is not public. brain also receives the press (it nudges vxFactor), exactly as on the robot.
    """

    def __init__(self):
        self._was_pressed = False

    def update(self, lt: bool, hat_u: bool) -> bool:
        pressed = lt and hat_u
        edge = pressed and not self._was_pressed
        self._was_pressed = pressed
        return edge
