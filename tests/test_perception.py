import math

import numpy as np
import pytest

from soccer_sim.config import CameraCfg
from soccer_sim.geometry import rpy_from_quat_wxyz, world_to_robot_2d, wrap_pi
from soccer_sim.odometry import OdometryTracker
from soccer_sim.perception import (
    Camera, Pose2D, detect_ball, detect_goalposts, detect_markings, visible_lines,
)

CFG = CameraCfg()
ORIGIN = Pose2D(0.0, 0.0, 0.0)


def level_camera(x=0.0, y=0.0, z=1.2, yaw=0.0):
    """Camera looking horizontally along the robot heading; optical axes in world."""
    right = [math.sin(yaw), -math.cos(yaw), 0.0]
    down = [0.0, 0.0, -1.0]
    forward = [math.cos(yaw), math.sin(yaw), 0.0]
    return Camera(CFG, np.array([x, y, z]), np.column_stack([right, down, forward]))


def test_world_to_robot_2d_rotates_into_robot_frame():
    x, y = world_to_robot_2d(1.0, 4.0, 1.0, 1.0, math.pi / 2)
    assert (x, y) == pytest.approx((3.0, 0.0), abs=1e-9)


def test_rpy_from_quat_roll_only():
    roll = 0.3
    q = (math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0)
    assert rpy_from_quat_wxyz(q) == pytest.approx((0.3, 0.0, 0.0), abs=1e-9)


def test_odometry_is_spawn_relative():
    odom = OdometryTracker(Pose2D(-6.0, -7.5, math.pi / 2))
    assert odom.update(Pose2D(-6.0, -5.5, math.pi / 2)) == pytest.approx((2.0, 0.0, 0.0), abs=1e-9)


def test_odometry_theta_wraps():
    odom = OdometryTracker(Pose2D(0.0, 0.0, 3.0))
    _, _, theta = odom.update(Pose2D(0.0, 0.0, -3.0))
    assert -math.pi < theta <= math.pi
    assert theta == pytest.approx(wrap_pi(-6.0))


def test_ball_in_front_is_detected_in_robot_frame():
    det = detect_ball(level_camera(), ORIGIN, (3.0, 0.0, 0.11), 0.11, 8.0)
    assert det is not None and det.label == "Ball"
    assert det.pos_robot == pytest.approx((3.0, 0.0), abs=1e-9)
    xmin, ymin, xmax, ymax = det.bbox
    assert xmin < CFG.cx < xmax
    assert (ymin + ymax) / 2 > CFG.cy  # ball is below the camera's optical axis


def test_ball_with_rotated_robot():
    cam = level_camera(x=1.0, y=1.0, yaw=math.pi / 2)
    det = detect_ball(cam, Pose2D(1.0, 1.0, math.pi / 2), (1.0, 4.0, 0.11), 0.11, 8.0)
    assert det is not None
    assert det.pos_robot == pytest.approx((3.0, 0.0), abs=1e-9)


def test_ball_behind_camera_is_not_detected():
    assert detect_ball(level_camera(), ORIGIN, (-3.0, 0.0, 0.11), 0.11, 8.0) is None


def test_ball_outside_horizontal_fov_is_not_detected():
    assert detect_ball(level_camera(), ORIGIN, (1.0, 5.0, 0.11), 0.11, 8.0) is None


def test_ball_beyond_range_is_not_detected():
    assert detect_ball(level_camera(), ORIGIN, (9.0, 0.0, 0.11), 0.11, 8.0) is None


def test_point_exactly_on_right_border_is_outside():
    cam = level_camera()
    # u = fx * right / forward + cx == width  ->  right = (width - cx) * forward / fx; right = -y in world
    forward = 3.0
    right = (CFG.width - cam.cx) * forward / cam.fx
    assert cam.project((forward, -right, 1.2)) is None


def test_penalty_point_has_pixel_target():
    cam = level_camera()
    dets = detect_markings(cam, ORIGIN, [("PenaltyPoint", 3.0, 0.0)], 5.0)
    assert len(dets) == 1 and dets[0].label == "PenaltyPoint"
    u, v = dets[0].target_uv
    assert u == pytest.approx(cam.cx)
    assert v == pytest.approx(cam.cy + cam.fy * 1.2 / 3.0)


def test_goalpost_uses_ground_point():
    dets = detect_goalposts(level_camera(), ORIGIN, [(6.0, 1.3)], 1.2, 0.05, 12.0)
    assert len(dets) == 1 and dets[0].label == "Goalpost"
    assert dets[0].pos_robot == pytest.approx((6.0, 1.3), abs=1e-9)


def test_line_crossing_view_is_clipped_to_fov():
    cam = level_camera()
    lines = visible_lines(cam, ORIGIN, [((2.0, -10.0), (2.0, 10.0))], 5.0)
    assert len(lines) == 1
    x0, y0, x1, y1 = lines[0].robot
    half_width_at_2m = 2.0 * cam.cx / cam.fx
    assert x0 == pytest.approx(2.0) and x1 == pytest.approx(2.0)
    assert abs(y0) <= half_width_at_2m + 1e-6 and abs(y1) <= half_width_at_2m + 1e-6
    assert abs(y1 - y0) > 1.0


def test_line_behind_robot_is_dropped():
    assert visible_lines(level_camera(), ORIGIN, [((-2.0, -3.0), (-2.0, 3.0))], 5.0) == []


def test_all_outputs_finite():
    cam = level_camera()
    dets = [detect_ball(cam, ORIGIN, (2.0, 0.3, 0.11), 0.11, 8.0)]
    dets += detect_markings(cam, ORIGIN, [("LCross", 3.0, 1.0), ("TCross", 0.0, 0.0)], 5.0)
    lines = visible_lines(cam, ORIGIN, [((2.0, -1.0), (4.0, 1.0)), ((0.0, 0.0), (0.0, 0.0))], 5.0)
    for d in dets:
        assert d is not None
        assert all(math.isfinite(v) for v in (*d.bbox, *d.pos_robot))
    for line in lines:
        assert all(math.isfinite(v) for v in (*line.robot, *line.uv))
