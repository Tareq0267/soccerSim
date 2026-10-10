import math

import mujoco
import numpy as np
import pytest

from soccer_sim.config import SHOT_SLIDERS, SimConfig
from soccer_sim.geometry import world_to_robot_2d
from soccer_sim.kick_model import KickRef
from soccer_sim.loco_rpc import LocoState
from soccer_sim.shots import Shot
from soccer_sim.sim import SoccerController

pytestmark = pytest.mark.slow
DT = 0.02


class Clock:
    def __init__(self):
        self.t = 0.0


def run(ctrl, loco, seconds, clock, kick_ref_fn=None, manual=None):
    for _ in range(int(round(seconds / DT))):
        ref = kick_ref_fn(clock.t) if kick_ref_fn else None
        ctrl.apply(loco, clock.t, ref, manual)
        ctrl.step()
        clock.t += DT


@pytest.fixture
def ctrl():
    c = SoccerController(SimConfig())
    c.start()
    return c


def test_stands_still_without_falling(ctrl):
    start = ctrl.robot_pose()
    run(ctrl, LocoState(), 3.0, Clock())
    end = ctrl.robot_pose()
    assert not ctrl.fallen()
    assert math.hypot(end.x - start.x, end.y - start.y) < 0.3


def test_walks_forward_on_move_command(ctrl):
    run(ctrl, LocoState(vx=0.5), 4.0, Clock())
    x, y, _ = ctrl.odometer()
    assert not ctrl.fallen()
    assert x > 1.0  # odometer x = forward in the spawn frame
    assert abs(y) < 0.5


def test_crab_walks_sideways(ctrl):
    run(ctrl, LocoState(vy=0.4), 4.0, Clock())
    x, y, _ = ctrl.odometer()
    assert not ctrl.fallen()
    assert y > 0.3
    assert abs(x) < 0.5


def test_head_tracks_rotate_head_targets(ctrl):
    run(ctrl, LocoState(head_pitch=0.3, head_yaw=0.5), 1.0, Clock())
    q, _, _ = ctrl.joint_state()
    assert q[1] == pytest.approx(0.3, abs=0.05)  # head_pitch_joint
    assert q[0] == pytest.approx(0.5, abs=0.05)  # aa_head_yaw_joint


def test_ball_rests_on_floor(ctrl):
    run(ctrl, LocoState(), 1.0, Clock())
    assert ctrl.ball_position()[2] == pytest.approx(SimConfig().ball_radius, abs=0.01)


def test_head_pose_is_in_ground_base_frame(ctrl):
    run(ctrl, LocoState(), 1.0, Clock())
    pos, quat = ctrl.head_pose_in_base()
    assert 1.0 < pos[2] < 1.6  # head height above the ground
    assert abs(pos[0]) < 0.2 and abs(pos[1]) < 0.2
    assert np.linalg.norm(quat) == pytest.approx(1.0)


def test_fall_is_reported_and_getup_restores_upright(ctrl):
    clock = Clock()
    run(ctrl, LocoState(), 1.0, clock)
    pose = ctrl.robot_pose()
    roll = 1.3
    ctrl.mj_data.qpos[3:7] = [math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]
    loco = LocoState()
    run(ctrl, loco, 1.0, clock)  # must keep stepping, no exception, no controller stop
    assert ctrl.fallen()
    assert ctrl.is_running
    rpy, _, _ = ctrl.imu()
    assert abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8
    loco.getup_requested = True
    run(ctrl, loco, 1.0, clock)
    assert not loco.getup_requested
    assert not ctrl.fallen()
    end = ctrl.robot_pose()
    assert math.hypot(end.x - pose.x, end.y - pose.y) < 0.5


def test_odometry_continuous_after_getup(ctrl):
    clock = Clock()
    run(ctrl, LocoState(vx=0.5), 2.0, clock)
    before = ctrl.odometer()
    loco = LocoState(getup_requested=True)
    ctrl.mj_data.qpos[3:7] = [math.cos(0.65), math.sin(0.65), 0.0, 0.0]
    mujoco.mj_forward(ctrl.mj_model, ctrl.mj_data)  # refresh xmat so fallen() sees the tilt on the first tick
    run(ctrl, loco, 0.2, clock)
    after = ctrl.odometer()
    assert math.hypot(after[0] - before[0], after[1] - before[1]) < 0.5


def test_perceives_ball_placed_in_front():
    c = SoccerController(SimConfig(spawn_xy_yaw=(0.0, 0.0, 0.0), ball_xy=(2.0, 0.0)))
    c.start()
    run(c, LocoState(), 1.0, Clock())
    dets, lines = c.perceive()
    balls = [d for d in dets if d.label == "Ball"]
    assert len(balls) == 1
    assert balls[0].pos_robot[0] == pytest.approx(2.0, abs=0.3)
    assert all(np.isfinite(v) for line in lines for v in (*line.robot, *line.uv))


def test_visual_kick_walks_onto_ball_and_launches_it():
    c = SoccerController(SimConfig(spawn_xy_yaw=(0.0, 0.0, 0.0), ball_xy=(0.7, 0.05)))
    c.start()
    clock = Clock()
    run(c, LocoState(), 1.0, clock)  # settle
    loco = LocoState(visual_kick_active=True)

    def truth_ref(now):
        pose, ball = c.robot_pose(), c.ball_position()
        bx, by = world_to_robot_2d(ball[0], ball[1], pose.x, pose.y, pose.yaw)
        return KickRef(ball_x=bx, ball_y=by, dir=0.0, power=3.0, stamp=now)

    start_x = c.ball_position()[0]
    max_speed = 0.0
    for _ in range(int(6.0 / DT)):
        c.apply(loco, clock.t, truth_ref(clock.t))
        c.step()
        clock.t += DT
        max_speed = max(max_speed, float(np.linalg.norm(c.ball_velocity()[:2])))
    assert max_speed > 1.0
    assert c.ball_position()[0] - start_x > 1.0
    assert not c.fallen()


def test_fallen_robot_never_kicks():
    c = SoccerController(SimConfig(spawn_xy_yaw=(0.0, 0.0, 0.0), ball_xy=(0.4, 0.0), ball_detector="ground_truth"))
    c.start()
    clock = Clock()
    run(c, LocoState(), 0.5, clock)
    c.mj_data.qpos[3:7] = [math.cos(0.75), math.sin(0.75), 0.0, 0.0]  # roll 1.5 rad: lying down
    mujoco.mj_forward(c.mj_model, c.mj_data)
    assert c.fallen()
    loco = LocoState(visual_kick_active=True)
    aligned = KickRef(ball_x=0.40, ball_y=0.0, dir=0.0, power=3.0, stamp=0.0)
    for _ in range(25):
        c.apply(loco, clock.t, KickRef(**{**aligned.__dict__, "stamp": clock.t}))
        c.step()
        clock.t += DT
        assert np.linalg.norm(c.ball_velocity()[:2]) < 1.0  # never launched


def test_yolo_detects_the_rendered_ball_at_the_right_range():
    c = SoccerController(SimConfig(spawn_xy_yaw=(0.0, 0.0, 0.0), ball_xy=(2.0, 0.2)))
    c.start()
    run(c, LocoState(), 1.0, Clock())
    rgb, _ = c.render()
    dets, _ = c.perceive(rgb=rgb)
    balls = [d for d in dets if d.label == "Ball"]
    assert len(balls) == 1, f"YOLO found {len(balls)} balls"
    pose, ball = c.robot_pose(), c.ball_position()
    true_x, true_y = world_to_robot_2d(ball[0], ball[1], pose.x, pose.y, pose.yaw)
    assert balls[0].confidence >= 0.4
    assert math.hypot(balls[0].pos_robot[0] - true_x, balls[0].pos_robot[1] - true_y) < 0.3


def test_ground_truth_ball_mode_skips_yolo():
    c = SoccerController(SimConfig(spawn_xy_yaw=(0.0, 0.0, 0.0), ball_xy=(2.0, 0.0), ball_detector="ground_truth"))
    c.start()
    assert c.yolo is None
    run(c, LocoState(), 0.5, Clock())
    rgb, _ = c.render()
    dets, _ = c.perceive(rgb=rgb)
    assert [d.confidence for d in dets if d.label == "Ball"] == [pytest.approx(0.9)]  # ground-truth confidence


def test_shot_panel_starts_at_defaults_and_survives_stepping(ctrl):
    defaults = {name: default for name, _lo, _hi, default in SHOT_SLIDERS}
    assert ctrl.shot_panel_values() == pytest.approx(defaults)
    run(ctrl, LocoState(vx=0.3), 0.5, Clock())  # the robot PD loop must not overwrite slider values
    assert ctrl.shot_panel_values() == pytest.approx(defaults)


def test_shoot_teleports_and_launches_the_ball_toward_our_goal(ctrl):
    ctrl.shoot(Shot(x=-8.0, y=0.0, speed=6.0, direction_deg=180.0, elevation_deg=0.0))
    assert ctrl.ball_position() == pytest.approx([-8.0, 0.0, SimConfig().ball_radius], abs=1e-6)
    assert ctrl.ball_velocity() == pytest.approx([-6.0, 0.0, 0.0], abs=1e-6)
    run(ctrl, LocoState(), 1.0, Clock())
    assert ctrl.ball_position()[0] < -9.5


def test_gamepad_overrides_brain_move(ctrl):
    # brain asks to walk backwards; the held stick (forward) wins, like the robot's firmware remote
    run(ctrl, LocoState(vx=-0.5), 3.0, Clock(), manual=(0.5, 0.0, 0.0))
    x, _, _ = ctrl.odometer()
    assert x > 0.6


def test_render_produces_rgb_and_depth_of_the_floor(ctrl):
    run(ctrl, LocoState(), 0.5, Clock())
    rgb, depth = ctrl.render()
    assert rgb.shape == (448, 544, 3) and rgb.dtype == np.uint8
    assert depth.shape == (448, 544)
    assert rgb.std() > 1.0  # not a blank frame
    centre = float(depth[224, 272])
    # camera ~1.2 m up, optical axis ~35 deg down -> floor ~2 m along the axis
    assert 1.0 < centre < 4.0
