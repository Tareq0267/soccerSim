# Zoomlion Z01 port: what the vendor repos give us

Survey of the three Zvalley repos (cloned 2026-10-14) and what each means for running soccerSim
with the Zoomlion Z01 instead of the Booster T2. English translations of the Chinese-only docs are
next to this file:

- [zv_robot_sdk_README_en.md](zv_robot_sdk_README_en.md): the SDK's README
- [Z01_std_robocup_README_en.md](Z01_std_robocup_README_en.md): the 6-robot RoboCup scene README

`Zlab_Gym` and `zv_robot_assets` already have English READMEs.

## The robot

From the product brochure (`zv_robot_assets/Z01 Robot Product Manual.pdf`, two marketing pages):

| | Z01 |
|---|---|
| Height | 138 cm |
| Mass with battery | 38.55 kg |
| Joint motors | 31 |
| Arm DoF | 7 per arm |
| Peak joint torque | 130 Nm |
| Sensors | 1 camera + microphone (lidar optional) |
| Compute | 10-core CPU, optional NVIDIA Jetson AGX Orin |
| Battery | 9600 mAh |
| Top speed | 2 m/s |

## The three repos

| Repo | What it is | Licence | Useful to soccerSim for |
|---|---|---|---|
| `zv_robot_sdk` | Cyclone DDS 0.10.5 wrapper (`ChannelFactory` / `ChannelPublisher` / `ChannelSubscriber`), IDL-generated message headers, prebuilt **Linux x86_64-only** libs, a pybind11 module for CPython 3.9–3.12 | Apache-2.0 | The robot-side interface brain would talk to: topics and message layouts |
| `Zlab_Gym` | mjlab 1.2.0 + patched rsl_rl training framework for `Z01_std_add_head`: one motion-tracking task, flat and rough velocity tasks | Apache-2.0 | **The walk policy** (`z01_velocity_flat_base.onnx`) and its robot config |
| `zv_robot_assets` | URDF + generated MJCF for five Z01 variants, a RoboCup field scene, a 6-robot scene | not stated | Robot MJCF, meshes, head camera mount, field |

### Robot variants (`zv_robot_assets`)

| Variant | Joints | Notes |
|---|---|---|
| `Z01_std` | 29 | Baseline: 7-DoF arms, 3-DoF waist, 6-DoF legs, no head |
| `Z01_std_add_head` | 29 | + `head_pitch`/`head_yaw` and a `depth_cam`; waist pitch/roll welded. **This is what Zlab_Gym trains on.** |
| `Z01_std_parallel_ankle` | 33 | + parallel ankle rods coupled by tendon equality constraints |
| `Z01_std_football_player` | 33 | parallel_ankle + soccer field scene |
| `Z01_std_robocup` | 6 × 33 | Six parallel_ankle robots on one field |

Head chain: `torso_link → head_pitch_link (pitch) → head_yaw_link (yaw) → depth_cam`. Pitch is the
parent joint, the opposite order to the T2. Limits: pitch ±1.309 rad, yaw ±1.57 rad. The camera in
the MJCF is a placeholder (`fovy 58.29`, `64×36`), not a calibrated intrinsic.

## The walk policy (`Zlab_Gym/.../demo_act/z01_velocity_flat_base.onnx`)

This replaces Booster's `t2_walk` in `soccer_sim/sim.py`. Read from the ONNX metadata:

- Input `obs[1, 99]`, output `actions[1, 29]`.
- Observation order: `base_lin_vel(3), base_ang_vel(3), projected_gravity(3), joint_pos(29) relative
  to default, joint_vel(29), last actions(29), command(3: vx, vy, wz)`.
- Joint order: `left arm ×7, right arm ×7, waist_yaw, left leg ×6, right leg ×6, head_pitch, head_yaw`.
- `target = default_joint_pos + 0.25 × action` for every joint, then PD with the stored
  `joint_stiffness` / `joint_damping`.
- Timing (checked in mjlab 1.2.0's `velocity_env_cfg.py`): 0.005 s physics step, decimation 4, so the
  policy runs at 50 Hz; integrator `implicitfast`. Motors are MuJoCo `<position>` actuators with
  `forcerange` = the joint's effort limit. `base_lin_vel` / `base_ang_vel` come from the `imu_lin_vel`
  velocimeter and `imu_ang_vel` gyro on the torso `imu` site. No observation scaling.
- **Dead zone (measured in soccerSim, 2026-10-14):** the policy barely steps when `|(vx, vy)|` is below
  about 0.33 m/s or `|vyaw|` below about 0.4 rad/s; it shuffles at 1–2 cm/s instead. Same with
  pyramidal and elliptic friction. mjlab's posture reward switches to "standing" only below 0.05, so this
  is learned behaviour of this checkpoint, not a config threshold. Above it, tracking is good
  (0.5 → 0.42, 1.0 → 0.96 m/s averaged including acceleration; 0.8 rad/s → 0.77). Retraining with
  Zlab_Gym and more small commands should fix it; it needs a CUDA GPU.

Things to watch:

- **The ONNX metadata doesn't match the "flat" config in the repo.** Its default pose (knee 0.5,
  hip_pitch −0.25) and gains (legs kp 100, ankle pitch 40, ankle roll 20, arms 20) are the
  **rough-terrain** config. The flat config in `env_cfgs.py` (knee 0.3, legs kp 50/100, arms 40)
  is not what this file was trained with. Trust the ONNX metadata.
- `base_lin_vel` is in the observation. In sim we have ground truth; on the robot it comes from an
  estimator we can't see.
- The policy outputs head targets too, but in training the head is just held near zero. soccerSim
  should override the two head targets with brain's head command, as it does for the T2.
- Training commands: vx −1…2 m/s, vy −1…1 m/s, wz −1…1 rad/s (play mode vx up to 2.0,
  wz ±0.7).

There's no public kick policy. The `kick` motion model is only reachable on the robot via
`/change_cmd model=kick`, so the sim's kick stays a stand-in, as with Booster.

## The robot's control interface (`zv_robot_sdk`)

| Purpose | Topic | Type |
|---|---|---|
| Walk velocity | `/motion_cmd` | `robot_msgs/RemoteControlMotion` (`token, vx, vy, vyaw, gait_step, is_continous`) |
| State / motion model / identity | `/change_cmd` | `robot_msgs/RemoteControlChange` (`key, val`), keys `status`, `model`, `name` |
| Joint commands | `rt/all_joint_cmd` | `robot_msgs AllJointCmd_` |
| Joint states | `rt/all_joint_state` | `robot_msgs AllJointState_` |
| IMU + navigation | `rt/nav_all` | `yesense_interface NavAll_` |
| Head | undocumented | `RemoteControlHead_` (`pos_roll, pos_pitch, is_absolute`) or `HeadControl_` exist |
| Robot state | undocumented | `RobotState_` (`INITIALIZING … CUSTOM_RULE`) |

State machine: `INIT → READY → RL_GAIT → (RL_RULE) → STOP → INIT`. Walking happens in `rl_gait`.

The DDS topic and type names follow the ROS 2 convention (`rt/` prefix, `pkg::msg::dds_::Type_`),
so a ROS 2 node on Cyclone DDS can talk to the robot directly if it has `robot_msgs` `.msg` files
with the same layout. Zvalley hasn't published those `.msg` files, but they can be rebuilt from the
IDL headers. That means **soccerSim doesn't need the Linux-only SDK binaries**.

Gaps and inconsistencies to raise with Zvalley:

1. The `/motion_cmd` example in their README publishes to `/change_cmd`.
2. The lowcmd example sends 23 joints (and a 25-entry kd list) to a 29/33-joint robot; the joint
   order of `rt/all_joint_cmd` is not documented.
3. No documented topics for the head, robot state, camera or odometry.
4. What `gait_step` and `token` do.

## How this maps onto soccerSim

| soccerSim piece | Today (T2) | Z01 |
|---|---|---|
| Robot MJCF | `booster_assets` T2_31dof | `zv_robot_assets/Z01_std_add_head` (matches the policy's 29 joints) |
| Walk policy | `booster_deploy` `t2_walk` | `z01_velocity_flat_base.onnx` via onnxruntime (no `booster_deploy` needed) |
| Head camera pose | Robotedge `vision.yaml` | Needs a Z01 calibration; MJCF mount as a first guess |
| Field | Robotedge `FD_ROBOLEAGUE` 22 × 14 m | Zvalley scene is 14 × 9 m (adult size). Brain's field config decides which. |
| Brain ↔ robot | Booster loco RPC (`booster_msgs`), `booster_interface` LowState / Odometer / remote | `/motion_cmd`, `/change_cmd`, `rt/all_joint_state`, `rt/nav_all` |
| Kick | stand-in | stand-in |
| Build platform | osx-arm64 pixi | unchanged; no SDK binaries required |

The biggest open question is the brain side: whether the team's brain on the Z01 still speaks the
Booster interfaces (through an adapter on the robot) or is being ported to the Zvalley topics. The
bridge's job in soccerSim depends entirely on that.

## Status of the port

Done (step 1, Python core, runs on Windows):

- `soccer_sim/robots.py`: `RobotSpec`, what the world needs from a body. `build_world` takes one; T2 stays
  the default and Booster is imported only for it.
- `soccer_sim/world.py`: `SoccerWorld`, the robot-independent half of the old `SoccerController`
  (field, ball, camera, perception, shots, kick).
- `soccer_sim/sim.py`: T2 `SoccerController` on top of it, behaviour unchanged.
- `soccer_sim/z01.py`: `Z01WalkPolicy` (ONNX + its metadata) and `Z01SoccerController`.
- `scripts/view_z01.py`: MuJoCo viewer demo without ROS.
- `tests/test_z01.py`: standing, walking, turning, head, fall/get-up, kick stand-in, YOLO on rendered
  frames, own body hidden from the camera.
- `KickParams.min_approach_speed`: the kick stand-in's approach no longer stalls in the Z01's dead zone.

Next:

- Step 2: ROS bridge speaking the Z01's topics (`/motion_cmd`, `/change_cmd`, `rt/all_joint_state`,
  `rt/nav_all`), with `robot_msgs` `.msg` files rebuilt from the SDK headers; run it under WSL2.
- A Z01 head-camera calibration to replace `Z01_CAMERA`.
- Z01 head limits in `loco_rpc` (it still clamps to the T2's), once the bridge drives the Z01.
