import math

import mujoco
import numpy as np
import pytest

from soccer_sim.config import SHOT_SLIDERS
from soccer_sim.geometry import world_to_robot_2d
from soccer_sim.kick_model import KickRef
from soccer_sim.loco_rpc import LocoState
from soccer_sim.z01 import CommandLimits, Z01SoccerController, Z01WalkPolicy, z01_sim_config

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
    c = Z01SoccerController(z01_sim_config())
    c.start()
    return c


def test_policy_metadata_matches_the_29_joint_robot():
    p = Z01WalkPolicy()
    assert len(p.joint_names) == len(p.default_joint_pos) == len(p.stiffness) == len(p.damping) == 29
    assert p.joint_names[-2:] == ["head_pitch", "head_yaw"]
    assert np.all(p.action_scale == 0.25)
    p.reset()
    targets = p.act(np.zeros(3), np.zeros(3), np.array([0.0, 0.0, -1.0]), p.default_joint_pos, np.zeros(29),
                    np.zeros(3))
    assert targets.shape == (29,) and np.all(np.isfinite(targets))


def test_commands_are_clamped_to_the_trained_range():
    assert CommandLimits().clamp(3.0, -2.0, 5.0) == (1.0, -0.5, 1.0)
    assert CommandLimits().clamp(-3.0, 0.2, -0.3) == (-1.0, 0.2, -0.3)


def test_z01_world_has_29_position_motors_then_the_shot_sliders():
    c = Z01SoccerController(z01_sim_config(ball_detector="ground_truth"))
    m = c.mj_model
    assert m.nu == 29 + len(SHOT_SLIDERS)
    assert sorted(c._motors) == list(range(29))
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert m.opt.timestep == pytest.approx(0.005) and c.policy_dt == pytest.approx(0.02)
    knee = m.actuator("left_knee_motor")
    assert knee.gainprm[0] == pytest.approx(100.0) and knee.biasprm[2] == pytest.approx(-4.0)
    assert knee.forcerange[1] == pytest.approx(140.0)


@pytest.mark.slow
def test_stands_still_without_falling(ctrl):
    start = ctrl.robot_pose()
    run(ctrl, LocoState(), 3.0, Clock())
    end = ctrl.robot_pose()
    assert not ctrl.fallen()
    assert math.hypot(end.x - start.x, end.y - start.y) < 0.3


@pytest.mark.slow
def test_walks_forward_on_move_command(ctrl):
    run(ctrl, LocoState(vx=0.5), 4.0, Clock())
    x, y, _ = ctrl.odometer()
    assert not ctrl.fallen()
    assert x > 1.0
    assert abs(y) < 0.5


@pytest.mark.slow
def test_crab_walks_sideways(ctrl):
    run(ctrl, LocoState(vy=0.4), 4.0, Clock())
    x, y, _ = ctrl.odometer()
    assert not ctrl.fallen()
    assert y > 0.3
    assert abs(x) < 0.7


@pytest.mark.slow
def test_turns_in_place(ctrl):
    run(ctrl, LocoState(vyaw=0.8), 3.0, Clock())
    _, _, theta = ctrl.odometer()
    assert not ctrl.fallen()
    assert theta > 1.5


@pytest.mark.slow
def test_head_tracks_brain_targets_not_the_policy(ctrl):
    run(ctrl, LocoState(head_pitch=0.3, head_yaw=0.5), 1.0, Clock())
    q, _, _ = ctrl.joint_state()  # policy joint order: ..., head_pitch, head_yaw
    assert q[-2] == pytest.approx(0.3, abs=0.05)
    assert q[-1] == pytest.approx(0.5, abs=0.05)


@pytest.mark.slow
def test_fall_is_reported_and_getup_restores_upright(ctrl):
    clock = Clock()
    run(ctrl, LocoState(), 1.0, clock)
    pose = ctrl.robot_pose()
    roll = 1.3
    ctrl.mj_data.qpos[3:7] = [math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]
    loco = LocoState()
    run(ctrl, loco, 1.0, clock)
    assert ctrl.fallen()
    loco.getup_requested = True
    run(ctrl, loco, 1.0, clock)
    assert not ctrl.fallen()
    end = ctrl.robot_pose()
    assert math.hypot(end.x - pose.x, end.y - pose.y) < 0.5


@pytest.mark.slow
def test_visual_kick_walks_onto_ball_and_launches_it():
    c = Z01SoccerController(z01_sim_config(spawn_xy_yaw=(0.0, 0.0, 0.0), ball_xy=(0.7, 0.05)))
    c.start()
    clock = Clock()
    run(c, LocoState(), 1.0, clock)
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


@pytest.mark.slow
def test_yolo_detects_the_rendered_ball_at_the_right_range():
    c = Z01SoccerController(z01_sim_config(spawn_xy_yaw=(0.0, 0.0, 0.0), ball_xy=(2.0, 0.2)))
    c.start()
    run(c, LocoState(head_pitch=0.4), 1.0, Clock())
    rgb, _ = c.render()
    dets, _ = c.perceive(rgb=rgb)
    balls = [d for d in dets if d.label == "Ball"]
    assert len(balls) == 1, f"YOLO found {len(balls)} balls"
    pose, ball = c.robot_pose(), c.ball_position()
    true_x, true_y = world_to_robot_2d(ball[0], ball[1], pose.x, pose.y, pose.yaw)
    assert math.hypot(balls[0].pos_robot[0] - true_x, balls[0].pos_robot[1] - true_y) < 0.3


@pytest.mark.slow
def test_head_camera_sees_neither_own_body_nor_foot_capsules():
    c = Z01SoccerController(z01_sim_config(ball_xy=(6.0, 4.0)))  # ball far out of view
    c.start()
    run(c, LocoState(head_pitch=1.2), 1.0, Clock())  # looking at its own feet
    rgb, _ = c.render()
    dets, _ = c.perceive(rgb=rgb)
    assert [d.bbox for d in dets if d.label == "Ball"] == []
    green = rgb[..., 1].astype(int) - np.maximum(rgb[..., 0], rgb[..., 2]).astype(int) > 20
    assert green.mean() > 0.95  # nothing but turf (and maybe a line) under the robot
