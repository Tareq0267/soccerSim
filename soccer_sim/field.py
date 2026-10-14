import math

import mujoco
import numpy as np

from .config import SHOT_SLIDERS, FieldDims, SimConfig
from .robots import RobotSpec, t2_spec

Segment = tuple[tuple[float, float], tuple[float, float]]

# MuJoCo cameras look along -z with y up; the optical frame looks along +z with y down.
OPTICAL_FROM_MUJOCO_CAM = np.diag([1.0, -1.0, -1.0])
# Robot motors live in this actuator group; the viewer hides it so its Control panel shows only shot sliders.
ROBOT_ACTUATOR_GROUP = 3


def map_markings(fd: FieldDims) -> list[tuple[str, float, float]]:
    """Same 26 markings, same coordinates, as BrainConfig::calcMapMarkings()."""
    out: list[tuple[str, float, float]] = []
    for xs in (-1.0, 1.0):
        for ys in (1.0, -1.0):
            out += [
                ("LCross", xs * fd.length / 2, ys * fd.width / 2),
                ("TCross", xs * fd.length / 2, ys * fd.penalty_area_width / 2),
                ("TCross", xs * fd.length / 2, ys * fd.goal_area_width / 2),
                ("LCross", xs * (fd.length / 2 - fd.penalty_area_length), ys * fd.penalty_area_width / 2),
                ("LCross", xs * (fd.length / 2 - fd.goal_area_length), ys * fd.goal_area_width / 2),
            ]
    for ys in (1.0, -1.0):
        out += [("TCross", 0.0, ys * fd.width / 2), ("XCross", 0.0, ys * fd.circle_radius)]
    for xs in (-1.0, 1.0):
        out.append(("PenaltyPoint", xs * (fd.length / 2 - fd.penalty_dist), 0.0))
    return out


def field_lines(fd: FieldDims, circle_segments: int = 24) -> list[Segment]:
    half_l, half_w = fd.length / 2, fd.width / 2
    segs: list[Segment] = [
        ((-half_l, -half_w), (half_l, -half_w)),
        ((-half_l, half_w), (half_l, half_w)),
        ((-half_l, -half_w), (-half_l, half_w)),
        ((half_l, -half_w), (half_l, half_w)),
        ((0.0, -half_w), (0.0, half_w)),
    ]
    for xs in (-1.0, 1.0):
        for depth, box_half_w in (
            (fd.penalty_area_length, fd.penalty_area_width / 2),
            (fd.goal_area_length, fd.goal_area_width / 2),
        ):
            x_in = xs * (half_l - depth)
            segs += [
                ((xs * half_l, -box_half_w), (x_in, -box_half_w)),
                ((xs * half_l, box_half_w), (x_in, box_half_w)),
                ((x_in, -box_half_w), (x_in, box_half_w)),
            ]
    r = fd.circle_radius
    for i in range(circle_segments):
        a0 = 2 * math.pi * i / circle_segments
        a1 = 2 * math.pi * (i + 1) / circle_segments
        segs.append(((r * math.cos(a0), r * math.sin(a0)), (r * math.cos(a1), r * math.sin(a1))))
    return segs


def goalposts(fd: FieldDims) -> list[tuple[float, float]]:
    return [(xs * fd.length / 2, ys * fd.goal_width / 2) for xs in (-1.0, 1.0) for ys in (1.0, -1.0)]


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def build_world(cfg: SimConfig, robot: RobotSpec | None = None) -> mujoco.MjModel:
    robot = robot if robot is not None else t2_spec()
    spec = mujoco.MjSpec.from_file(robot.mjcf_path)
    if robot.prepare is not None:
        robot.prepare(spec)
    world = spec.worldbody
    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)

    def _material(name: str, texture: str, **kwargs):
        mat = spec.add_material(name=name, **kwargs)
        textures = [""] * len(mat.textures)
        textures[rgb_role] = texture
        mat.textures = textures

    # The YOLO model was trained on sim imagery with a patterned ball and textured grass;
    # a plain white sphere on flat green is never detected (2026-10-10 spike).
    spec.add_texture(
        name="turf_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        mark=mujoco.mjtMark.mjMARK_RANDOM, random=0.25, rgb1=[0.13, 0.5, 0.18], markrgb=[0.2, 0.62, 0.25],
        width=512, height=512,
    )
    _material("turf_mat", "turf_tex", texrepeat=[8.0, 6.0])
    spec.add_texture(
        name="ball_tex", type=mujoco.mjtTexture.mjTEXTURE_CUBE, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[1.0, 1.0, 1.0], rgb2=[0.05, 0.05, 0.05], width=64, height=64,
    )
    _material("ball_mat", "ball_tex")

    floor = spec.geom("floor")
    if floor is None:  # a bare robot model (e.g. one made for mjlab, which supplies its own terrain)
        floor = world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE)
        world.add_light(name="sun", pos=[0.0, 0.0, 20.0], dir=[0.0, 0.0, -1.0],
                        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                        diffuse=[0.8, 0.8, 0.8], ambient=[0.3, 0.3, 0.3])
    floor.material = "turf_mat"
    # Finite render size so the texture repeat is defined; planes still collide infinitely.
    floor.size = [cfg.field.length / 2 + 3.0, cfg.field.width / 2 + 3.0, 0.1]

    for i, ((x0, y0), (x1, y1)) in enumerate(field_lines(cfg.field)):
        length = math.hypot(x1 - x0, y1 - y0)
        world.add_geom(
            name=f"line_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[length / 2, cfg.line_width / 2, 0.0005],
            pos=[(x0 + x1) / 2, (y0 + y1) / 2, 0.0005],
            quat=_yaw_quat(math.atan2(y1 - y0, x1 - x0)),
            rgba=[1.0, 1.0, 1.0, 1.0],
            contype=0,
            conaffinity=0,
        )

    for i, (gx, gy) in enumerate(goalposts(cfg.field)):
        world.add_geom(
            name=f"goalpost_{i}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[cfg.goalpost_radius, cfg.goal_height / 2, 0.0],
            pos=[gx, gy, cfg.goal_height / 2],
            rgba=[1.0, 1.0, 1.0, 1.0],
        )
    for xs, name in ((-1.0, "own"), (1.0, "opp")):
        world.add_geom(
            name=f"crossbar_{name}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[cfg.goalpost_radius, cfg.field.goal_width / 2, 0.0],
            pos=[xs * cfg.field.length / 2, 0.0, cfg.goal_height],
            quat=[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],  # cylinder axis z -> y
            rgba=[1.0, 1.0, 1.0, 1.0],
        )

    ball = world.add_body(name="ball", pos=[cfg.ball_xy[0], cfg.ball_xy[1], cfg.ball_radius])
    ball.add_freejoint(name="ball_joint")
    ball.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[cfg.ball_radius, 0.0, 0.0],
        mass=cfg.ball_mass,
        material="ball_mat",
        condim=6,
        # ponytail: rolling friction guessed for artificial turf; tune from real ball roll-out distance.
        friction=[0.8, 0.02, 0.01],
    )

    cam_to_head = cfg.camera.cam_to_head_matrix()
    cam_quat = np.zeros(4)
    mujoco.mju_mat2Quat(cam_quat, (cam_to_head[:3, :3] @ OPTICAL_FROM_MUJOCO_CAM).reshape(9))
    spec.body(robot.head_body).add_camera(
        name="head_rgb",
        pos=cam_to_head[:3, 3].tolist(),
        quat=cam_quat.tolist(),
        fovy=math.degrees(2.0 * math.atan(cfg.camera.height / 2 / cfg.camera.fy)),
    )

    # Both robot MJCFs have an "imu" site at the trunk origin; add noise-free sensors on it.
    spec.add_sensor(name="imu_gyro", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(
        name="imu_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu"
    )

    for act in spec.actuators:  # only the robot motors exist at this point
        act.group = ROBOT_ACTUATOR_GROUP
    for name, lo, hi, _default in SHOT_SLIDERS:
        # ponytail: zero-gain actuators used purely as viewer sliders; they never apply force.
        slider = spec.add_actuator(
            name=name,
            target=robot.slider_joint,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE,
            ctrlrange=[lo, hi],
            group=0,
        )
        slider.gainprm = [0.0] * len(slider.gainprm)
        slider.biasprm = [0.0] * len(slider.biasprm)
    return spec.compile()
