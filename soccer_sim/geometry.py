import math


def wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    w = math.atan2(math.sin(a), math.cos(a))
    return math.pi if w == -math.pi else w


def yaw_from_quat_wxyz(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rpy_from_quat_wxyz(q) -> tuple[float, float, float]:
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch, yaw_from_quat_wxyz(q)


def world_to_robot_2d(px: float, py: float, rx: float, ry: float, ryaw: float) -> tuple[float, float]:
    dx, dy = px - rx, py - ry
    c, s = math.cos(ryaw), math.sin(ryaw)
    return c * dx + s * dy, -s * dx + c * dy
