import math

import mujoco
import numpy as np
import torch
from booster_deploy.controllers.base_controller import BaseController
from tasks.locomotion.robots.t2 import T2WalkTaskCfg

from .config import SimConfig
from .robots import t2_spec
from .world import SoccerWorld

HEAD_YAW_IDX, HEAD_PITCH_IDX = 0, 1  # T2_31DOF_CFG.joint_names order


class SoccerController(BaseController, SoccerWorld):
    """Runs Booster's T2 walk policy in the full soccer world (robot + field + ball + camera).

    Mirrors booster_deploy's MujocoController PD loop, but slices robot joints out of a
    model that also contains a free-floating ball, overrides head targets with brain's
    RotateHead command, hands the chassis to VisualKickModel while a visual kick is
    active, renders the head camera, and never stops on a fall.
    """

    def __init__(self, sim_cfg: SimConfig):
        cfg = T2WalkTaskCfg()
        cfg.policy.enable_safety_fallback = False
        super().__init__(cfg)
        self._init_world(sim_cfg, t2_spec(), cfg.mujoco.physics_dt)
        self.decimation = cfg.mujoco.decimation

        nj = self.robot.num_joints
        self._nu_robot = nj  # robot motors are actuators 0..30; the shot sliders follow them
        self._q = slice(7, 7 + nj)
        self._dq = slice(6, 6 + nj)

        self._kp = self.robot.joint_stiffness.numpy()
        self._kd = self.robot.joint_damping.numpy()
        self._effort = self.robot.effort_limit.numpy()

        sx, sy, syaw = sim_cfg.spawn_xy_yaw
        self.reset_robot(sx, sy, syaw)
        self.reset_ball(*sim_cfg.ball_xy)

    def reset_robot(self, x: float, y: float, yaw: float) -> None:
        d = self.mj_data
        d.qpos[0:3] = [x, y, self.cfg.mujoco.init_pos[2]]
        d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        d.qpos[self._q] = self.robot.default_joint_pos.numpy()
        d.qvel[0:6] = 0.0
        d.qvel[self._dq] = 0.0
        mujoco.mj_forward(self.mj_model, d)
        self.update_state()
        if self.is_running:
            self.policy.reset()

    def _set_velocity_command(self, vx: float, vy: float, vyaw: float) -> None:
        self.vel_command.lin_vel_x = vx
        self.vel_command.lin_vel_y = vy
        self.vel_command.ang_vel_yaw = vyaw

    # ---- booster_deploy controller contract ---------------------------------
    def update_state(self) -> None:
        d, data = self.mj_data, self.robot.data
        data.joint_pos = torch.from_numpy(d.qpos[self._q].astype(np.float32))
        data.joint_vel = torch.from_numpy(d.qvel[self._dq].astype(np.float32))
        data.feedback_torque = torch.from_numpy(d.qfrc_actuator[self._dq].astype(np.float32))
        data.root_pos_w = torch.from_numpy(d.qpos[0:3].astype(np.float32))
        data.root_quat_w = torch.from_numpy(d.qpos[3:7].astype(np.float32))
        data.root_lin_vel_b = torch.from_numpy(d.qvel[0:3].astype(np.float32))
        data.root_ang_vel_b = torch.from_numpy(d.qvel[3:6].astype(np.float32))

    def ctrl_step(self, dof_targets: torch.Tensor) -> None:
        targets = dof_targets.detach().cpu().numpy().astype(np.float64)
        targets[HEAD_PITCH_IDX], targets[HEAD_YAW_IDX] = self._head_target
        d = self.mj_data
        for _ in range(self.decimation):
            q, dq = d.qpos[self._q], d.qvel[self._dq]
            # Only the robot motors; ctrl[31:] are the viewer's shot sliders.
            d.ctrl[: self._nu_robot] = np.clip(self._kp * (targets - q) - self._kd * dq, -self._effort, self._effort)
            mujoco.mj_step(self.mj_model, d)

    def step(self) -> None:
        self.update_state()
        self.ctrl_step(self.policy_step())
