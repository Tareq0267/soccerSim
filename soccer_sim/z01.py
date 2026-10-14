"""Zoomlion Z01 in the soccer world, walked by Zvalley's Zlab_Gym velocity policy (ONNX, no torch).

The policy was trained in mjlab on Zlab_Gym's Z01_std_add_head MJCF, so the sim loads that same file and
rebuilds what mjlab adds at load time: <position> motors with the policy's gains, the collision setup, and
the 200 Hz implicitfast physics with the policy at 50 Hz.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from .config import CameraCfg, SimConfig
from .kick_model import KickParams, VisualKickModel
from .robots import RobotSpec
from .world import SoccerWorld

ZLAB_GYM = Path(__file__).resolve().parents[1] / "vendor" / "Zlab_Gym"
Z01_MJCF = ZLAB_GYM / "src" / "assets" / "robots" / "Z01_std_add_head" / "xmls" / "Z01_std_add_head.xml"
Z01_WALK_POLICY = ZLAB_GYM / "src" / "assets" / "motions" / "z01_std_add_head" / "demo_act" / "z01_velocity_flat_base.onnx"

# mjlab's velocity task: MujocoCfg(timestep=0.005), decimation=4, integrator implicitfast.
PHYSICS_DT = 0.005
DECIMATION = 4
# Torso height for the policy's default pose (knee 0.5): Zlab_Gym's rough-task spawn, feet ~7 mm above ground.
SPAWN_HEIGHT = 0.83
HEAD_PITCH_LIMIT = 1.309  # Z01_std_add_head.xml joint ranges
HEAD_YAW_LIMIT = 1.57
# The released policy barely steps for |(vx, vy)| below ~0.33 m/s or |vyaw| below ~0.4 rad/s (measured in this
# sim with both friction cones; mjlab's own posture threshold is 0.05, so it's learned, not configured).
POLICY_MIN_LINEAR_SPEED = 0.35
_OBS_NAMES = ("base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command")

# Placeholder until the team calibrates a Z01: brain's T2 intrinsics (vision.yaml), mounted where Zvalley's MJCF
# puts depth_cam on head_yaw_link, looking straight ahead (optical z = head x, x = -head y, y = -head z).
Z01_CAMERA = CameraCfg(
    cam_to_head=(
        (0.0, 0.0, 1.0, 0.069),
        (-1.0, 0.0, 0.0, 0.01),
        (0.0, -1.0, 0.0, 0.068),
        (0.0, 0.0, 0.0, 1.0),
    ),
)


@dataclass(frozen=True)
class CommandLimits:
    """The last stage of Zlab_Gym's velocity curriculum; the policy hasn't seen commands outside it."""

    vx: tuple[float, float] = (-1.0, 1.0)
    vy: float = 0.5
    vyaw: float = 1.0

    def clamp(self, vx: float, vy: float, vyaw: float) -> tuple[float, float, float]:
        return (
            float(np.clip(vx, *self.vx)),
            float(np.clip(vy, -self.vy, self.vy)),
            float(np.clip(vyaw, -self.vyaw, self.vyaw)),
        )


def _floats(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split(",")], dtype=np.float64)


class Z01WalkPolicy:
    """The exported actor plus everything mjlab stored next to it: joint order, default pose, gains, action scale."""

    def __init__(self, path: Path = Z01_WALK_POLICY):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `git submodule update --init vendor/Zlab_Gym`")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        if tuple(meta["observation_names"].split(",")) != _OBS_NAMES:
            raise ValueError(f"{path.name}: unexpected observation layout {meta['observation_names']}")
        self.joint_names: list[str] = meta["joint_names"].split(",")
        self.default_joint_pos = _floats(meta["default_joint_pos"])
        self.stiffness = _floats(meta["joint_stiffness"])
        self.damping = _floats(meta["joint_damping"])
        self.action_scale = _floats(meta["action_scale"])
        self._input = self.session.get_inputs()[0].name
        self.last_action = np.zeros(len(self.joint_names))

    def reset(self) -> None:
        self.last_action[:] = 0.0

    def act(self, lin_vel_b, ang_vel_b, gravity_b, joint_pos, joint_vel, command) -> np.ndarray:
        """One 50 Hz step: observation in, joint position targets (policy joint order) out."""
        obs = np.concatenate([
            lin_vel_b, ang_vel_b, gravity_b, joint_pos - self.default_joint_pos, joint_vel, self.last_action, command,
        ]).astype(np.float32)[None, :]
        self.last_action = self.session.run(None, {self._input: obs})[0][0].astype(np.float64)
        return self.default_joint_pos + self.action_scale * self.last_action


def _add_motors(policy: Z01WalkPolicy):
    """mjlab rebuilds the robot this way at load time; without it the policy meets a different robot."""

    def prepare(spec: mujoco.MjSpec) -> None:
        # Collision proxies are the geoms named *_collision; feet get condim 3, priority and friction 0.6
        # (Zlab_Gym FULL_COLLISION). They move to group 3 so the cameras don't draw the foot capsules.
        for geom in spec.geoms:
            if not geom.name.endswith("_collision"):
                continue
            foot = geom.name.startswith(("left_foot", "right_foot"))
            geom.condim = 3 if foot else 1
            if foot:
                geom.priority = 1
                geom.friction = [0.6, geom.friction[1], geom.friction[2]]
            geom.group = 3
        for name, kp, kd in zip(policy.joint_names, policy.stiffness, policy.damping):
            effort = spec.joint(name).actfrcrange[1]
            spec.add_actuator(
                name=f"{name}_motor",
                target=name,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                gainprm=[kp] + [0.0] * 9,
                biasprm=[0.0, -kp, -kd] + [0.0] * 7,
                forcelimited=mujoco.mjtLimited.mjLIMITED_TRUE,
                forcerange=[-effort, effort],
            )

    return prepare


def z01_spec(policy: Z01WalkPolicy) -> RobotSpec:
    return RobotSpec(
        name="z01",
        mjcf_path=str(Z01_MJCF),
        root_body="torso_link",
        head_body="head_yaw_link",
        slider_joint="head_yaw",
        prepare=_add_motors(policy),
    )


def z01_sim_config(**overrides) -> SimConfig:
    """SimConfig for the Z01: the placeholder head camera, everything else as for the T2 unless overridden."""
    return SimConfig(**{"camera": Z01_CAMERA, **overrides})


class Z01SoccerController(SoccerWorld):
    """Zvalley's Z01 velocity policy in the full soccer world; same interface as the T2's SoccerController.

    Like the T2 path it overrides the policy's head targets with brain's head command, hands the chassis to
    the kick stand-in while a visual kick is active, and never stops on a fall. Commands are clamped to the
    range the policy was trained on (CommandLimits).
    """

    def __init__(self, sim_cfg: SimConfig, limits: CommandLimits = CommandLimits(),
                 policy_path: Path = Z01_WALK_POLICY):
        self.policy = Z01WalkPolicy(policy_path)
        self._init_world(sim_cfg, z01_spec(self.policy), PHYSICS_DT)
        self.mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.decimation = DECIMATION
        self.policy_dt = PHYSICS_DT * DECIMATION
        self.limits = limits
        self.kick_model = VisualKickModel(KickParams(min_approach_speed=POLICY_MIN_LINEAR_SPEED))

        m, names = self.mj_model, self.policy.joint_names
        # Index arrays in the policy's joint order (the MJCF order differs).
        self._q = np.array([m.joint(n).qposadr[0] for n in names])
        self._dq = np.array([m.joint(n).dofadr[0] for n in names])
        self._motors = np.array([m.actuator(f"{n}_motor").id for n in names])
        self._nu_robot = len(names)
        self._kp, self._kd = self.policy.stiffness, self.policy.damping
        self._head_pitch, self._head_yaw = names.index("head_pitch"), names.index("head_yaw")
        self._lin_vel = int(m.sensor("imu_lin_vel").adr[0])
        self._ang_vel = int(m.sensor("imu_ang_vel").adr[0])
        self.command = np.zeros(3)
        self.is_running = False

        sx, sy, syaw = sim_cfg.spawn_xy_yaw
        self.reset_robot(sx, sy, syaw)
        self.reset_ball(*sim_cfg.ball_xy)

    def start(self) -> None:
        self.policy.reset()
        self.is_running = True

    def reset_robot(self, x: float, y: float, yaw: float) -> None:
        d = self.mj_data
        d.qpos[0:3] = [x, y, SPAWN_HEIGHT]
        d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        d.qpos[self._q] = self.policy.default_joint_pos
        d.qvel[0:6] = 0.0
        d.qvel[self._dq] = 0.0
        d.ctrl[self._motors] = self.policy.default_joint_pos
        mujoco.mj_forward(self.mj_model, d)
        self.policy.reset()

    def _set_velocity_command(self, vx: float, vy: float, vyaw: float) -> None:
        self.command[:] = self.limits.clamp(vx, vy, vyaw)

    def observation(self) -> tuple[np.ndarray, ...]:
        d = self.mj_data
        r_trunk = d.xmat[self._trunk_body].reshape(3, 3)
        return (
            d.sensordata[self._lin_vel:self._lin_vel + 3].copy(),
            d.sensordata[self._ang_vel:self._ang_vel + 3].copy(),
            r_trunk.T @ np.array([0.0, 0.0, -1.0]),
            d.qpos[self._q].copy(),
            d.qvel[self._dq].copy(),
            self.command.copy(),
        )

    def ctrl_step(self, targets: np.ndarray) -> None:
        targets = np.array(targets, dtype=np.float64)
        pitch, yaw = self._head_target
        targets[self._head_pitch] = np.clip(pitch, -HEAD_PITCH_LIMIT, HEAD_PITCH_LIMIT)
        targets[self._head_yaw] = np.clip(yaw, -HEAD_YAW_LIMIT, HEAD_YAW_LIMIT)
        d = self.mj_data
        d.ctrl[self._motors] = targets  # position motors: the PD runs inside MuJoCo, as in mjlab
        for _ in range(self.decimation):
            mujoco.mj_step(self.mj_model, d)

    def step(self) -> None:
        self.ctrl_step(self.policy.act(*self.observation()))
