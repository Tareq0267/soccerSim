import math
from pathlib import Path

import mujoco
import numpy as np

from .config import SHOT_SLIDERS, SimConfig
from .field import OPTICAL_FROM_MUJOCO_CAM, build_world, field_lines, goalposts, map_markings
from .geometry import rpy_from_quat_wxyz, world_to_robot_2d, yaw_from_quat_wxyz
from .kick_model import KickRef, VisualKickModel
from .loco_rpc import MOVING_MODES, LocoState
from .odometry import OdometryTracker
from .perception import (
    Camera, Detection, LineObs, Pose2D, detect_ball, detect_goalposts, detect_markings, visible_lines,
)
from .robots import RobotSpec
from .shots import Shot, shot_velocity
from .yolo_ball import YoloBallDetector

UPRIGHT_MIN_ZZ = 0.5  # trunk z-axis . world z; same threshold as booster_deploy's safety fallback
ROBOT_VISUAL_GROUP = 1  # both robot MJCFs put visual meshes in group 1 (field and ball are in 0)


def _make_ball_detector(cfg: SimConfig) -> YoloBallDetector | None:
    if cfg.ball_detector == "ground_truth":
        return None
    if cfg.ball_detector != "yolo":
        raise ValueError(f"ball_detector must be 'yolo' or 'ground_truth', got {cfg.ball_detector!r}")
    if not Path(cfg.yolo_model_path).exists():
        raise FileNotFoundError(f"{cfg.yolo_model_path} missing; run `pixi run fetch-models`")
    return YoloBallDetector(cfg.yolo_model_path, cfg.yolo_ball_conf, cfg.yolo_nms_iou)


class SoccerWorld:
    """The robot-independent half of a soccer controller: field, ball, head camera, perception, shots, kick.

    Subclasses own the walking policy. They call _init_world() once the policy's physics step is known,
    and implement reset_robot() and _set_velocity_command().
    """

    def _init_world(self, sim_cfg: SimConfig, robot: RobotSpec, physics_dt: float) -> None:
        self.sim_cfg = sim_cfg
        self.robot_spec = robot
        self.mj_model = build_world(sim_cfg, robot)
        self.mj_model.opt.timestep = physics_dt
        # Elliptic cone: friction opposes the actual slip direction; pyramidal bends diagonal ball rolls toward 45°.
        self.mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.mj_data = mujoco.MjData(self.mj_model)

        self._sliders = {name: self.mj_model.actuator(name).id for name, *_ in SHOT_SLIDERS}
        for name, _lo, _hi, default in SHOT_SLIDERS:
            self.mj_data.ctrl[self._sliders[name]] = default
        ball_joint = self.mj_model.joint("ball_joint")
        self._ball_q = int(ball_joint.qposadr[0])
        self._ball_dq = int(ball_joint.dofadr[0])
        self._cam = self.mj_model.camera("head_rgb").id
        self._head_body = self.mj_model.body(robot.head_body).id
        self._trunk_body = self.mj_model.body(robot.root_body).id
        self._gyro = int(self.mj_model.sensor("imu_gyro").adr[0])
        self._acc = int(self.mj_model.sensor("imu_acc").adr[0])
        self._head_target = (0.0, 0.0)  # (pitch, yaw)

        self._markings = map_markings(sim_cfg.field)
        self._lines = field_lines(sim_cfg.field)
        self._posts = goalposts(sim_cfg.field)

        self.kick_model = VisualKickModel()
        self._was_kicking = False
        self._renderer: mujoco.Renderer | None = None
        self.yolo = _make_ball_detector(sim_cfg)
        self._rng = np.random.default_rng()

        sx, sy, syaw = sim_cfg.spawn_xy_yaw
        self.odometry = OdometryTracker(Pose2D(sx, sy, syaw))

    # ---- resets -------------------------------------------------------------
    def reset_robot(self, x: float, y: float, yaw: float) -> None:
        raise NotImplementedError

    def reset_ball(self, x: float, y: float) -> None:
        d = self.mj_data
        d.qpos[self._ball_q:self._ball_q + 7] = [x, y, self.sim_cfg.ball_radius, 1.0, 0.0, 0.0, 0.0]
        d.qvel[self._ball_dq:self._ball_dq + 6] = 0.0
        mujoco.mj_forward(self.mj_model, d)

    # ---- inputs from brain -------------------------------------------------
    def _set_velocity_command(self, vx: float, vy: float, vyaw: float) -> None:
        raise NotImplementedError

    def apply(self, loco: LocoState, now: float, kick_ref: KickRef | None = None,
              manual: tuple[float, float, float] | None = None) -> None:
        if manual is not None and loco.mode in MOVING_MODES:  # held stick: firmware remote beats brain and kick
            vx, vy, vyaw = manual
            self.kick_model.reset()
        elif loco.kicking() and not self.fallen():  # a robot on the ground cannot kick
            if not self._was_kicking:
                self.kick_model.reset()
            pose, ball = self.robot_pose(), self.ball_position()
            true_ball = world_to_robot_2d(ball[0], ball[1], pose.x, pose.y, pose.yaw)
            out = self.kick_model.step(kick_ref, now, true_ball, loco.visual_kick_version)
            vx, vy, vyaw = out.vx, out.vy, out.vyaw
            if out.strike_speed > 0.0:
                self._launch_ball(out.strike_speed, pose.yaw + out.strike_dir)
        else:
            vx, vy, vyaw = loco.effective_velocity()
        self._was_kicking = loco.kicking()

        self._set_velocity_command(vx, vy, vyaw)
        self._head_target = (loco.head_pitch, loco.head_yaw)
        if loco.getup_requested:
            loco.getup_requested = False
            if self.fallen():
                pose = self.robot_pose()
                self.reset_robot(pose.x, pose.y, pose.yaw)

    def shot_panel_values(self) -> dict[str, float]:
        return {name: float(self.mj_data.ctrl[i]) for name, i in self._sliders.items()}

    def shoot(self, shot: Shot) -> None:
        d = self.mj_data
        d.qpos[self._ball_q:self._ball_q + 7] = [shot.x, shot.y, self.sim_cfg.ball_radius, 1.0, 0.0, 0.0, 0.0]
        d.qvel[self._ball_dq:self._ball_dq + 3] = shot_velocity(shot.speed, shot.direction_deg, shot.elevation_deg)
        d.qvel[self._ball_dq + 3:self._ball_dq + 6] = 0.0
        mujoco.mj_forward(self.mj_model, d)

    def _launch_ball(self, speed: float, world_dir: float) -> None:
        d = self.mj_data
        d.qvel[self._ball_dq:self._ball_dq + 3] = [speed * math.cos(world_dir), speed * math.sin(world_dir), 0.0]
        d.qvel[self._ball_dq + 3:self._ball_dq + 6] = 0.0

    # ---- outputs to brain --------------------------------------------------
    def fallen(self) -> bool:
        trunk_zz = self.mj_data.xmat[self._trunk_body].reshape(3, 3)[2, 2]
        return bool(trunk_zz < UPRIGHT_MIN_ZZ)

    def robot_pose(self) -> Pose2D:
        q = self.mj_data.qpos
        return Pose2D(float(q[0]), float(q[1]), yaw_from_quat_wxyz(q[3:7]))

    def ball_position(self) -> np.ndarray:
        return self.mj_data.qpos[self._ball_q:self._ball_q + 3].copy()

    def ball_velocity(self) -> np.ndarray:
        return self.mj_data.qvel[self._ball_dq:self._ball_dq + 3].copy()

    def odometer(self) -> tuple[float, float, float]:
        return self.odometry.update(self.robot_pose())

    def imu(self):
        d = self.mj_data
        rpy = rpy_from_quat_wxyz(d.qpos[3:7])
        gyro = tuple(float(v) for v in d.sensordata[self._gyro:self._gyro + 3])
        acc = tuple(float(v) for v in d.sensordata[self._acc:self._acc + 3])
        return rpy, gyro, acc

    def joint_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self.mj_data
        return d.qpos[self._q].copy(), d.qvel[self._dq].copy(), d.qfrc_actuator[self._dq].copy()

    def _base(self) -> tuple[np.ndarray, np.ndarray]:
        """Ground base frame: trunk xy at z = 0, trunk yaw only (brain's ground_height = 0)."""
        pose = self.robot_pose()
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        return np.array([pose.x, pose.y, 0.0]), np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def head_pose_in_base(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.mj_data
        origin, r_base = self._base()
        pos = r_base.T @ (d.xpos[self._head_body] - origin)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (r_base.T @ d.xmat[self._head_body].reshape(3, 3)).reshape(9))
        return pos, quat

    def camera(self) -> Camera:
        d = self.mj_data
        r_opt = d.cam_xmat[self._cam].reshape(3, 3) @ OPTICAL_FROM_MUJOCO_CAM
        return Camera(self.sim_cfg.camera, d.cam_xpos[self._cam], r_opt)

    def perceive(self, rng=None, rgb: np.ndarray | None = None) -> tuple[list[Detection], list[LineObs]]:
        c = self.sim_cfg
        rng = rng if rng is not None else self._rng
        cam, robot = self.camera(), self.robot_pose()
        dets: list[Detection] = []
        if self.yolo is not None and rgb is not None:
            dets += self.yolo.detect(rgb, cam, robot, c.ball_radius)
        else:
            ball = detect_ball(
                cam, robot, self.ball_position(), c.ball_radius, c.max_ball_range, c.detection_noise_std, rng
            )
            if ball is not None:
                dets.append(ball)
        dets += detect_goalposts(
            cam, robot, self._posts, c.goal_height, c.goalpost_radius, c.max_goalpost_range, c.detection_noise_std, rng
        )
        dets += detect_markings(cam, robot, self._markings, c.max_marking_range, c.detection_noise_std, rng)
        return dets, visible_lines(cam, robot, self._lines, c.max_line_range)

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        cam_cfg = self.sim_cfg.camera
        if self._renderer is None:  # created lazily on the stepping thread (OpenGL context affinity)
            self._renderer = mujoco.Renderer(self.mj_model, height=cam_cfg.height, width=cam_cfg.width)
        r = self._renderer
        # ponytail: own body hidden from the head camera — YOLO scores the chrome chest/hand meshes up to 0.89 as
        # "Ball" and brain has no self-filter. Costs self-occlusion; revisit if the real robot shows the same FP.
        opt = mujoco.MjvOption()
        opt.geomgroup[ROBOT_VISUAL_GROUP] = 0
        r.disable_depth_rendering()
        r.update_scene(self.mj_data, camera=self._cam, scene_option=opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.sim_cfg.render_shadows
        r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self.sim_cfg.render_shadows
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(self.mj_data, camera=self._cam, scene_option=opt)
        depth = r.render().astype(np.float32)
        r.disable_depth_rendering()
        return rgb, depth
