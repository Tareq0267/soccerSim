import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from booster_deploy.robots.t2 import T2_31DOF_CFG
from soccer_sim.config import SHOT_SLIDERS, CameraCfg, FieldDims, SimConfig
from soccer_sim.field import (
    OPTICAL_FROM_MUJOCO_CAM, ROBOT_ACTUATOR_GROUP, build_world, field_lines, goalposts, map_markings,
)


def test_map_markings_match_brain_calc_map_markings():
    fd = FieldDims()
    marks = map_markings(fd)
    labels = [m[0] for m in marks]
    assert len(marks) == 26
    assert labels.count("LCross") == 12
    assert labels.count("TCross") == 10
    assert labels.count("XCross") == 2
    assert labels.count("PenaltyPoint") == 2
    penalty = sorted(m[1] for m in marks if m[0] == "PenaltyPoint")
    assert penalty[0] == pytest.approx(-(fd.length / 2 - fd.penalty_dist))
    assert penalty[1] == pytest.approx(fd.length / 2 - fd.penalty_dist)
    assert ("XCross", 0.0, fd.circle_radius) in marks
    assert ("LCross", -fd.length / 2, fd.width / 2) in marks


def test_field_lines_cover_boundary_boxes_and_circle():
    fd = FieldDims()
    segs = field_lines(fd, circle_segments=24)
    # 5 boundary/halfway + 2 halves * 2 boxes * 3 sides + 24 circle chords
    assert len(segs) == 5 + 12 + 24
    for (x0, y0), (x1, y1) in segs:
        assert abs(x0) <= fd.length / 2 + 1e-9 and abs(x1) <= fd.length / 2 + 1e-9
        assert abs(y0) <= fd.width / 2 + 1e-9 and abs(y1) <= fd.width / 2 + 1e-9


def test_goalposts_are_on_goal_lines():
    fd = FieldDims()
    posts = goalposts(fd)
    assert len(posts) == 4
    assert all(abs(abs(x) - fd.length / 2) < 1e-9 for x, _ in posts)
    assert all(abs(abs(y) - fd.goal_width / 2) < 1e-9 for _, y in posts)


def test_world_has_robot_ball_camera_and_imu():
    model = build_world(SimConfig())
    assert model.nu == 31 + len(SHOT_SLIDERS)
    assert [model.actuator(i).name for i in range(31)] == list(T2_31DOF_CFG.joint_names)
    assert model.joint(T2_31DOF_CFG.joint_names[0]).qposadr[0] == 7
    ball = model.joint("ball_joint")
    assert ball.type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert ball.qposadr[0] == 7 + 31
    assert model.camera("head_rgb").bodyid[0] == model.body("head_pitch_link").id
    assert model.sensor("imu_gyro").dim[0] == 3
    assert model.sensor("imu_acc").dim[0] == 3


def test_shot_sliders_are_zero_force_panel_controls():
    model = build_world(SimConfig())
    for i in range(31):
        assert model.actuator_group[i] == ROBOT_ACTUATOR_GROUP  # hidden from the viewer's Control panel
    for offset, (name, lo, hi, _default) in enumerate(SHOT_SLIDERS):
        act = model.actuator(name)
        assert act.id == 31 + offset
        assert model.actuator_group[act.id] == 0
        assert tuple(model.actuator_ctrlrange[act.id]) == pytest.approx((lo, hi))
        assert not np.any(model.actuator_gainprm[act.id]) and not np.any(model.actuator_biasprm[act.id])


def test_ball_and_turf_are_textured_for_the_detector():
    model = build_world(SimConfig())
    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    for geom in ("ball_geom", "floor"):
        mat = model.geom_matid[model.geom(geom).id]
        assert mat >= 0, f"{geom} has no material"
        assert model.mat_texid[mat][rgb_role] >= 0, f"{geom} material has no texture"
    assert model.geom_size[model.geom("ball_geom").id][0] == pytest.approx(0.105)


def test_robot_mass_matches_real_t2():
    model = build_world(SimConfig())
    ball_id = model.body("ball").id
    robot_mass = sum(model.body_mass[i] for i in range(1, model.nbody) if i != ball_id)
    assert 40.0 < robot_mass < 45.0  # Booster spec: ~43 kg


def test_camera_is_placed_with_team_calibration():
    cfg = SimConfig()
    model = build_world(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = model.camera("head_rgb").id
    head = model.body("head_pitch_link").id
    t = cfg.camera.cam_to_head_matrix()
    # vertical FOV from fy: 2*atan(224/208.883) ~= 94.0 deg
    assert model.cam_fovy[cam] == pytest.approx(94.0, abs=0.1)
    r_head = data.xmat[head].reshape(3, 3)
    expected_pos = data.xpos[head] + r_head @ t[:3, 3]
    assert np.allclose(data.cam_xpos[cam], expected_pos, atol=1e-6)
    r_opt = data.cam_xmat[cam].reshape(3, 3) @ OPTICAL_FROM_MUJOCO_CAM
    assert np.allclose(r_opt, r_head @ t[:3, :3], atol=1e-4)


def test_camera_defaults_match_team_vision_yaml():
    root = Path(os.environ.get("ROBOTEDGE_ROOT", "/Users/johnnytan5/Robotedge-5v5-RCAP2026"))
    path = root / "src/vision/config/vision.yaml"
    if not path.exists():
        pytest.skip(f"{path} not found")
    cam = yaml.safe_load(path.read_text())["camera"]
    cfg = CameraCfg()
    assert cam.get("extrin_direction", "camera_to_head") == "camera_to_head"
    intrin = cam["intrin"]
    assert (cfg.fx, cfg.fy, cfg.cx, cfg.cy) == pytest.approx((intrin["fx"], intrin["fy"], intrin["cx"], intrin["cy"]))
    assert np.allclose(cfg.cam_to_head_matrix(), np.array(cam["extrin"], dtype=float))
