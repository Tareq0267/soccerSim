# SoccerSim (Booster T2 + Robotedge brain in MuJoCo) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the unmodified Robotedge `brain_node` (behavior tree, localization, goalie/striker strategy, RL visual kick flow) against a physically simulated Booster T2 on a RoboCup field on macOS, so strategy changes can be tested without the real robot.

**Architecture:** A pure-Python core (`soccer_sim`) builds a MuJoCo world (Booster's real T2 MJCF + field lines + goals + ball + the head camera placed with the team's own calibration) and drives the T2 with Booster's trained walking policy (`booster_deploy` task `t2_walk`). A ROS 2 node (`soccer_sim_bridge`) impersonates the robot's SDK and vision stack on the exact topics `brain_node` already uses: it turns `brain`'s loco RPC requests into policy velocity commands, runs a stand-in for Booster's closed RL visual kick driven by `brain`'s `/kick_ball` references, renders RGB and depth images from the head camera, detects the ball in those images with a real YOLOv8 model (Booster's sim-trained `sim_data_det_0126.onnx`), and publishes odometry, low state, head pose and vision detections (YOLO ball + ground-truth goalposts, markings, lines). For testing, a shot launcher is driven from sliders in the MuJoCo viewer's top-right panel, and a browser "controller tab" provides a virtual (or physical) gamepad, a referee panel with set plays, and live status. `brain_node` is built from the Robotedge repo and launched with a sim parameter overlay; no Robotedge source changes.

**Tech Stack:** macOS (Apple Silicon), pixi + RoboStack ROS 2 Kilted, colcon, Python 3.12, MuJoCo ≥ 3.2.6 (Python bindings, `MjSpec`, `Renderer`), PyTorch (CPU), `booster_deploy` @ `7bb1462`, `booster_assets`, pytest.

**Spec:** No separate spec file. The design was agreed in conversation on 2026-10-08..10 and is captured in "Design Summary" below; executors should read that section before any task.

## Design Summary

Data flow (every arrow is an existing `brain_node` interface — nothing in `brain` changes):

```
brain_node ──LocoApiTopicReq (booster_msgs/RpcReqMsg)──▶ bridge ──▶ LocoState ──┬─▶ T2 walk policy (vx, vy, vyaw) ──▶ MuJoCo
brain_node ──/kick_ball (brain/msg/Kick)───────────────▶ bridge ──▶ VisualKickModel ┘   (while kVisualKick is active)
brain_node ◀─LocoApiTopicResp (booster_msgs/RpcRespMsg)── bridge
brain_node ◀─/odometer_state, /low_state, /head_pose_stamped────────────── simulator robot state
brain_node ◀─/boostercamera/head/{rgb,depth,rgb/camera_info}───────────── MuJoCo render of the calibrated head camera
brain_node ◀─/booster_vision/detection, /booster_vision/line_segments──── Ball: YOLOv8 on the rendered RGB; goalposts/markings/lines: ground truth through the same camera
brain_node ◀─/robocup/game_controller──────────────────────────────────── Referee (schedule, overridable from the controller tab, incl. set plays)
brain_node ◀─/remote_controller_state──────────────────────────────────── controller tab gamepad (virtual sticks/buttons or a physical pad)

controller tab (http://127.0.0.1:8765) ──▶ bridge: gamepad state, referee overrides, "shoot" button, status
MuJoCo viewer right panel sliders ───────▶ bridge: shot origin x/y, speed, direction, elevation, fire
/sim/shoot (std_msgs/Float64MultiArray) ─▶ bridge: scripted shots; results on /sim/shot_result (std_msgs/String)
```

Decisions:
- **Locomotion realism** comes from Booster's trained `t2_walk.pt` with Booster's PD gains/effort limits (`T2WalkControllerCfg`). The policy accepts `(vx, vy, vyaw)` — the same three numbers `brain` sends in its `kMove` RPC.
- **RL visual kick.** On the robot, `brain` sends `kVisualKick {"start": true}` and then publishes `/kick_ball` (ball position relative to robot, desired kick direction, power) every tick; Booster's closed onboard policy (SDK topic `rt/kick_ball`) does the approach and kick. No T2 kick policy is public (checked all `booster_deploy` commits and Booster's other repos on 2026-10-10; Booster Studio has no virtual T2). The sim replaces it with `VisualKickModel`: while the kick is active it owns the chassis, walks the T2 policy onto the ball using `brain`'s `/kick_ball` references, and when aligned *and* the real ball is within reach it launches the ball in the commanded direction with a speed derived from `power`. This exercises `brain`'s whole `RLVisionKick` flow (slow down, start, head tilt, ball-motion confirmation, aborts, exit) with an approximate kick. It is clearly labelled as a stand-in and is swappable if Booster publishes the real policy.
- **Camera.** The head camera is placed in the MuJoCo model using the team's calibration from `$ROBOTEDGE_ROOT/src/vision/config/vision.yaml` (`camera.intrin`, `camera.extrin`, camera→head). The sim renders RGB (`rgb8`) and depth (`16UC1`, millimetres) from it; `brain` uses depth for obstacle detection.
- **Ball detection is real; everything else is ground truth.** The team's own models exist only as Jetson TensorRT engines, so the ball is detected with Booster's public sim-trained YOLOv8 detector (`robocup_demo/src/vision/model/sim_data_det_0126.onnx`, Apache-2.0 repo, AGPL-3.0 Ultralytics weights — fine for internal testing), run with onnxruntime on every rendered RGB frame. Ranging copies Robotedge vision (`BallPoseEstimator::EstimateByColor`): the bbox-centre ray is intersected with the plane z = ball radius (0.105 m, size 5); Ball confidence ≥ 0.4, NMS IoU 0.4. A 2026-10-10 spike showed the model needs a patterned ball (plain white: 0.00; black/white checker + textured turf: 0.72–0.84 at 1.2–3.5 m) and does not reliably detect primitive goalposts or line crossings, so goalposts, field markings and field lines stay ground truth (projected through the same camera model). `SimConfig.ball_detector = "ground_truth"` switches the ball back to ground truth for comparison.
- **Startup and the real match procedure.** `game.xml` runs `control_state=3` once at boot, so `brain` needs no button press to play. The team's real procedure is still reproducible: with `boot_sequence` on (the `run_sim.sh` default), the T2 boots in prepare mode at the sideline (`brain`'s Move is ignored) until **LT+Up** on the gamepad stands it into soccer mode — the firmware step, modelled by `FirmwareRemote` (assumed meaning of LT+Up; the press also reaches `brain`, as on the robot). **LT+A** / **LT+B** are `brain`'s own recalibrate / autonomous commands and are forwarded unchanged. The match preset holds the referee in INITIAL (`referee_schedule = "0:0"`) until READY / SET / PLAYING are pressed in the controller tab. `run_sim.sh --quick` skips all of this (soccer mode at boot, scripted referee).
- **Gamepad = firmware-level remote.** On the robot, the Booster remote's sticks drive locomotion in firmware; `brain` only notices stick deflection (`go_manual`, threshold 0.1) and pauses strategy — no `brain` code converts sticks to velocity. The sim mirrors that: a deflected stick (|axis| > 0.1) overrides `brain`'s Move and the visual kick (left stick = vx/vy, right stick x = vyaw). Buttons are forwarded to `brain` unchanged on `/remote_controller_state`, so LT+B / LT+X / LT+A work exactly as on the robot. The controller tab is a plain HTML page served by the bridge on 127.0.0.1 only; it also reads any physical gamepad through the browser Gamepad API. If the tab stops sending for 0.5 s, the gamepad reverts to neutral.
- **Shot launcher.** MuJoCo's viewer "Control" panel (top-right) draws a slider per actuator. The world gets six zero-force actuators used purely as sliders: `shot_origin_x`, `shot_origin_y`, `shot_speed`, `shot_direction_deg`, `shot_elevation_deg`, `shot_fire`; the robot's 31 motors are put in actuator group 3, which the viewer hides from the panel. Dragging `shot_fire` above 0.5 teleports the ball to the origin and launches it; it re-arms when dragged back below 0.5. The same shot can be fired from the controller tab's Shoot button or scripted on `/sim/shoot`. After each shot a tracker reports `goal` (ball crosses our goal line between the posts under the bar) or `no_goal` (stopped, went wide, or 8 s timeout) on `/sim/shot_result`, and keeps a tally.
- **Set plays.** The referee follows the `referee_schedule` parameter until the controller tab overrides it; then it holds whatever was set. A set play (direct/indirect free kick, penalty, throw-in, goal kick, corner) is sent with `kicking_team` = us or the opponent, the `stopped` flag (ball being placed), and a 30 s `secondary_time` countdown, after which the set play clears — the fields `brain` reads in `gameControlCallback`.
- **Frames.** `brain`'s robot ("base") frame has its origin on the ground (`obstacle_avoidance.ground_height` defaults to 0.0). The sim defines base = trunk position projected to z = 0 with the trunk's yaw. Head pose, detections, and depth all use this frame, so `brain`'s `camToRobot = headToBase · camToHead` reproduces the simulated camera exactly.
- **Out of scope:** opponents/teammates/multi-robot (shots come from the launcher, not a robot), occlusion in the ground-truth detections, team UDP communication (`enable_com=false`), real get-up motion (GetUp teleports the robot upright), damping collapse (damping/prepare modes just zero the velocity), YOLO for goalposts/markings, the field-line segmentation model.

## Global Constraints

- Platform: macOS Apple Silicon; ROS 2 **Kilted** from RoboStack channel `robostack-kilted` via pixi; Python `3.12.*`.
- `booster_deploy` pinned to commit `7bb1462e2ac742eba5c01659b815e2e99fbe617f`; task config `tasks.locomotion.robots.t2.T2WalkTaskCfg`; checkpoint `tasks/locomotion/robots/t2/models/t2_walk.pt`.
- T2 joint order is `booster_deploy.robots.t2.T2_31DOF_CFG.joint_names` (31 joints; index 0 = `aa_head_yaw_joint`, index 1 = `head_pitch_joint`). MJCF joint order and actuator order must equal it.
- Robotedge repo (`/Users/johnnytan5/Robotedge-5v5-RCAP2026`, env var `ROBOTEDGE_ROOT`) is **read-only**: no edits to any file in it. It is consumed through a symlink `ros_ws/src/robotedge -> $ROBOTEDGE_ROOT/src`.
- Booster SDK headers come from `~/Downloads/sdk_release.zip` (env var `SDK_ZIP`), unpacked to `vendor/booster_sdk/include` (git-ignored). Only headers are used; nothing links `libbooster_robotics_sdk.a`.
- Velocity limits (from `T2WalkControllerCfg.vel_command`): `vx ∈ [-0.7, 3.0]` m/s, `vy ∈ [-0.8, 0.8]` m/s, `vyaw ∈ [-2.0, 2.0]` rad/s. Requests outside are clamped.
- Loco RPC contract: request `header` JSON `{"api_id": N}`, `body` JSON params; response `header` `{"status": S}` (0 = ok), `uuid` echoed. IDs: 2000 ChangeMode `{"mode"}`, 2001 Move `{"vx","vy","vyaw"}`, 2004 RotateHead `{"pitch","yaw"}`, 2008 GetUp, 2017 GetMode → body `{"mode": M}`, 2038 VisualKick `{"start": bool, "version": int}` (version 0 = kV1, 1 = kV2). Any other id → status 0, no-op. Modes: 0 damping, 1 prepare, 2 walking, 3 custom, 4 soccer; initial mode 4; walking and kicking only in modes 2 and 4; any ChangeMode ends an active visual kick.
- Kick reference: topic `/kick_ball`, type `brain/msg/Kick` (`x`, `y` = ball position in robot frame; `dir` = desired kick direction relative to robot, rad; `power` = desired kick range, m). A reference older than 0.5 s is ignored.
- Camera (from `vision.yaml`): 544×448 px; `fx=208.712936, fy=208.883377, cx=272, cy=224`; `extrin` is camera(optical: x right, y down, z forward)→head, applied relative to MuJoCo body `head_pitch_link`. Topics `/boostercamera/head/rgb` (`rgb8`), `/boostercamera/head/depth` (`16UC1`, mm, 0 = invalid), `/boostercamera/head/rgb/camera_info`, `/head_pose_stamped` (frame `base`).
- Field: robo_league `FieldDimensions{22.003, 14.126, 3.635, 2.6, 1.99, 5.221, 8.121, 2.307, 5.083}`; own goal at −x; field markings must match `BrainConfig::calcMapMarkings()` (26 markings).
- Ball: radius 0.105 m (size 5, `vision.yaml` `ball_pose_estimator.ball_size: 5`), black/white checker texture (required for the YOLO model to see it); turf has a random-mark grass texture.
- YOLO ball model: `vendor/models/sim_data_det_0126.onnx` (git-ignored), fetched from `https://media.githubusercontent.com/media/BoosterRobotics/robocup_demo/main/src/vision/model/sim_data_det_0126.onnx`, sha256 `4410f04175bb3a7bc79ed7cfce2a5ab0e9c695be8cb488fbfd9d5619dc492005`. Input `images` `[1,3,320,320]` float RGB /255, letterbox pad 114; output `[1,13,2100]` = `cx, cy, w, h` + 9 class scores, class 0 = `Ball`. Ball confidence ≥ 0.4, NMS IoU 0.4 (Robotedge `vision.yaml` `post_process.confidence_thresholds.Ball` / `nms_threshold`). Range = bbox-centre ray ∩ plane z = 0.105 m. Inference on CPU via onnxruntime.
- Timing: real-time paced on wall clock, `policy_dt = 0.02` s (50 Hz), physics `dt = 0.002` s (decimation 10). Camera frames (render → YOLO → detections → publish) at 25 Hz (every 2nd tick). Never set `use_sim_time`.
- macOS: the MuJoCo viewer (`mujoco.viewer.launch_passive`) only works under `mjpython`; when the viewer is on, ROS spinning runs on a background thread. All MuJoCo stepping and rendering happens on the main thread.
- Policy safety fallback is disabled (`enable_safety_fallback = False`) because the sim must keep running after a fall.
- Gamepad: `/remote_controller_state` (`booster_interface/RemoteControllerState`) published at 20 Hz; axes clamped to [−1, 1]; manual drive when any axis |v| > 0.1: `vx = −ly·1.0`, `vy = −lx·0.5`, `vyaw = −rx·1.0` (m/s, rad/s); stale after 0.5 s → neutral.
- Referee: `/robocup/game_controller` at 2 Hz; `teams[0]` = us (`team_id` 131, `goalkeeper` = `goalkeeper_id` 1), `teams[1]` = opponent (`opp_team_id` 1); states INITIAL 0, READY 1, SET 2, PLAYING 3, FINISHED 4; set plays NONE 0, DIRECT_FREE_KICK 1, INDIRECT_FREE_KICK 2, PENALTY_KICK 3, THROW_IN 4, GOAL_KICK 5, CORNER_KICK 6; set-play countdown 30 s.
- Shot sliders (name, min, max, default): `shot_origin_x` −11..11 (−6), `shot_origin_y` −7..7 (0), `shot_speed` 0..10 m/s (4), `shot_direction_deg` −180..180 (180 = toward our goal at −x), `shot_elevation_deg` 0..45 (0), `shot_fire` 0..1 (0, fires on crossing 0.5). Scripted shots: `/sim/shoot` `Float64MultiArray` `[x, y, speed, direction_deg, elevation_deg]`; speed ≤ 15 m/s, elevation 0..80°.
- Boot sequence: bridge parameter `boot_sequence` (default false for tests; `run_sim.sh` passes true unless `--quick`) starts the robot in mode 1 (prepare); the LT+Up press edge (LT held, RT not held, hat up) switches mode 1/0 → 4. Match preset: `referee_schedule = "0:0"` (hold INITIAL).
- Controller tab: `http://127.0.0.1:8765/` (parameter `web_port`, 0 disables), bound to 127.0.0.1 only; POST bodies must be `application/json` (blocks cross-site form posts) and ≤ 4 KB.
- Commit messages: plain, no AI co-author/attribution lines.

## Review Focus

1. **Velocity commands beyond the policy's range** (goalie BT sends `vy_limit="1.2"`, chase sends `vx_limit="3.0"` plus factors) → clamped, robot keeps walking, never NaN. Test: Task 4 `test_move_is_clamped_to_policy_limits`.
2. **Malformed external input** — RPC (bad JSON, missing fields, NaN, non-bool kick `start`, unknown api_id), controller-tab HTTP (bad JSON, wrong content type, oversized body, unknown referee names or fields, NaN axes), `/sim/shoot` (wrong length, NaN, absurd speed) → rejected with an error status / 4xx / log line, state untouched, bridge keeps running. Tests: Task 4 `test_malformed_requests_are_rejected_without_state_change`; Task 7 `test_shot_from_dict_rejects_bad_input`, `test_invalid_override_changes_nothing`; Task 9 `test_bridge_survives_garbage_rpc`, `test_rejects_bad_json_wrong_type_and_oversized_bodies`, `test_joystick_from_dict_rejects_non_bool_buttons`.
3. **Objects behind the camera, at the image border, out of range, depth pixels hitting the sky, or a YOLO box whose ray points at/above the horizon** → omitted / encoded as 0; no NaN/inf ever reaches a published message; an empty frame yields no ball (not a stale or invented one). Tests: Task 3 `test_ball_behind_camera_is_not_detected`, `test_point_exactly_on_right_border_is_outside`, `test_all_outputs_finite`; Task 6 `test_ray_at_or_above_horizon_returns_none`, `test_real_model_sees_no_ball_in_empty_turf`; Task 9 `test_depth_image_is_millimetres_with_invalid_as_zero`.
4. **Robot falls / visual kick on a ball that is not really there** → sim keeps stepping, IMU reports the fall so `brain`'s recovery triggers, GetUp restores the robot; a kick reference that claims an aligned ball never launches a ball that is out of reach. Tests: Task 8 `test_fall_is_reported_and_getup_restores_upright`, Task 5 `test_no_strike_when_real_ball_out_of_reach`.
5. **Controller tab closed (or browser frozen) while a stick is held** → the gamepad reverts to neutral within 0.5 s, manual drive ends, and `brain` resumes control; the robot does not keep walking on a stale stick. Test: Task 9 `test_stale_joystick_releases_manual_drive`. (Odometer wraparound/continuity is also covered: Task 3 `test_odometry_theta_wraps`, Task 8 `test_odometry_continuous_after_getup`.)

---

## File Structure

```
soccerSim/
├── .gitignore
├── .gitmodules
├── README.md                         # usage, architecture, limitations (Task 10)
├── pixi.toml                         # ROS 2 Kilted + Python deps, env vars, tasks
├── pyproject.toml                    # soccer_sim package + pytest config
├── vendor/
│   ├── booster_deploy/               # git submodule @ 7bb1462
│   ├── booster_assets/               # git submodule (or unpacked zip, see Task 1)
│   ├── booster_sdk/include/          # unpacked SDK headers (git-ignored)
│   └── models/sim_data_det_0126.onnx # Booster's sim-trained YOLOv8 (git-ignored, fetched)
├── soccer_sim/                       # pure Python, no ROS imports
│   ├── __init__.py
│   ├── config.py                     # FieldDims, CameraCfg (team calibration), SimConfig
│   ├── field.py                      # markings, lines, goalposts, MuJoCo world builder
│   ├── geometry.py                   # angle wrap, quat→rpy/yaw, world→robot frame
│   ├── odometry.py                   # OdometryTracker (spawn-relative odometer)
│   ├── perception.py                 # optical Camera model + ground-truth detections/lines
│   ├── loco_rpc.py                   # LocoState + RPC request handling
│   ├── kick_model.py                 # VisualKickModel (stand-in for Booster's RL visual kick)
│   ├── yolo_ball.py                  # YoloBallDetector (real YOLOv8 ball detection + ranging)
│   ├── shots.py                      # Shot, slider panel reader, goal/no-goal tracker
│   ├── joystick.py                   # gamepad sticks -> manual velocity (firmware-style override)
│   ├── referee.py                    # schedule + overridable referee with set plays
│   └── sim.py                        # SoccerController (booster_deploy BaseController)
├── tests/                            # pytest, no ROS needed
│   ├── test_environment.py
│   ├── test_field.py
│   ├── test_perception.py
│   ├── test_loco_rpc.py
│   ├── test_kick_model.py
│   ├── test_yolo_ball.py
│   ├── test_controls.py              # shots, joystick, referee
│   └── test_sim.py                   # @slow, runs physics, rendering, YOLO
├── ros_ws/
│   └── src/
│       ├── robotedge -> $ROBOTEDGE_ROOT/src   # symlink, git-ignored
│       └── soccer_sim_bridge/        # ament_python package
│           ├── package.xml
│           ├── setup.py
│           ├── setup.cfg
│           ├── resource/soccer_sim_bridge
│           ├── launch/brain.launch.py
│           ├── soccer_sim_bridge/__init__.py
│           ├── soccer_sim_bridge/msg_conv.py
│           ├── soccer_sim_bridge/web.py          # controller-tab HTTP server (stdlib)
│           ├── soccer_sim_bridge/controller.html # virtual gamepad + referee + shoot + status
│           ├── soccer_sim_bridge/bridge_node.py
│           └── test/
│               ├── test_msg_conv.py
│               ├── test_web.py
│               ├── test_bridge.py
│               └── test_brain_in_sim.py   # end-to-end (Task 10)
└── scripts/
    ├── fetch_models.sh               # download + sha256-verify the YOLO model
    ├── setup_ws.sh                   # symlink + SDK headers + colcon build
    └── run_sim.sh                    # bridge (optionally with viewer) + brain
```

---

### Task 1: Repo bootstrap, environment, and the macOS build gate

**Files:**
- Create: `.gitignore`, `pixi.toml`, `pyproject.toml`, `soccer_sim/__init__.py`, `scripts/setup_ws.sh`, `tests/test_environment.py`
- Submodules: `vendor/booster_deploy`, `vendor/booster_assets`

**Interfaces:**
- Consumes: nothing.
- Produces: a pixi environment where `import booster_deploy`, `import tasks.locomotion.robots.t2`, `import booster_assets`, `import mujoco`, `import torch`, `import yaml` all work; env vars `ROBOTEDGE_ROOT`, `SDK_ZIP`; `pixi run build-ws` builds `brain` and its message packages into `ros_ws/install`; `vendor/models/sim_data_det_0126.onnx` present and checksum-verified.

This task is a **gate**: if `brain` cannot be built on macOS at Step 9, stop and report the exact compiler/solver error to the human partner before starting Task 2. The fallback (running `brain_node` inside an Ubuntu 24.04 arm64 VM with ROS 2 Kilted from apt, talking DDS to the Mac) changes Tasks 9–10 and needs re-planning.

- [ ] **Step 1: Initialise the repo and add the Booster submodules**

```bash
cd /Users/johnnytan5/soccerSim
git init
git submodule add https://github.com/BoosterRobotics/booster_deploy.git vendor/booster_deploy
git -C vendor/booster_deploy checkout 7bb1462e2ac742eba5c01659b815e2e99fbe617f
git submodule add https://github.com/BoosterRobotics/booster_assets.git vendor/booster_assets
ls vendor/booster_assets/robots/T2/T2_31dof/T2_31dof.xml
```

Expected: the last command prints the path. If it reports "No such file", the public repo lacks T2; replace the submodule with the zip the user provided:

```bash
git submodule deinit -f vendor/booster_assets && git rm -f vendor/booster_assets && rm -rf .git/modules/vendor/booster_assets
tmp=$(mktemp -d) && unzip -q ~/Downloads/booster_assets.zip -d "$tmp" && mv "$tmp/booster_assets" vendor/booster_assets && rm -rf "$tmp"
ls vendor/booster_assets/robots/T2/T2_31dof/T2_31dof.xml
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.pixi/
__pycache__/
*.egg-info/
.pytest_cache/
vendor/booster_sdk/
vendor/models/
ros_ws/build/
ros_ws/install/
ros_ws/log/
ros_ws/src/robotedge
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "soccer_sim"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools]
packages = ["soccer_sim"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: steps MuJoCo physics for several simulated seconds",
    "ros: needs a built and sourced ros_ws",
]
```

- [ ] **Step 4: Write `soccer_sim/__init__.py`**

```python
"""MuJoCo soccer world + Booster T2 walk policy, driven through brain's ROS interfaces."""
```

- [ ] **Step 5: Write `pixi.toml`**

```toml
[workspace]
name = "soccer-sim"
channels = ["robostack-kilted", "conda-forge"]
platforms = ["osx-arm64"]

[dependencies]
python = "3.12.*"
ros-kilted-desktop = "*"
ros-kilted-behaviortree-cpp = "*"
ros-kilted-backward-ros = "*"
ros-kilted-tf2-geometry-msgs = "*"
colcon-common-extensions = "*"
cmake = "*"
ninja = "*"
compilers = "*"
pkg-config = "*"
eigen = "*"
yaml-cpp = "*"
opencv = "*"
librerun-sdk = "*"
numpy = "*"
scipy = "*"
pyyaml = "*"
pytest = "*"

[pypi-dependencies]
mujoco = ">=3.2.6"
torch = ">=2.3"
onnxruntime = "*"
soccer_sim = { path = ".", editable = true }
booster_assets = { path = "vendor/booster_assets", editable = true }

[activation.env]
PYTHONPATH = "$PIXI_PROJECT_ROOT/vendor/booster_deploy"
ROBOTEDGE_ROOT = "/Users/johnnytan5/Robotedge-5v5-RCAP2026"
SDK_ZIP = "$HOME/Downloads/sdk_release.zip"

[tasks]
fetch-models = "bash scripts/fetch_models.sh"
build-ws = "bash scripts/setup_ws.sh"
test = "pytest -m 'not slow and not ros'"
test-slow = "pytest -m slow"
```

`opencv` from conda-forge also provides the Python `cv2` module used for letterboxing in Task 6.

`booster_deploy` is not pip-packaged (its own `scripts/deploy.py` does `sys.path.append(".")`), so it is put on `PYTHONPATH`; this makes both `booster_deploy` and its top-level `tasks` package importable.

- [ ] **Step 6: Install the environment**

Run: `cd /Users/johnnytan5/soccerSim && pixi install`
Expected: solves and installs without error. If the solver rejects a package (most likely `ros-kilted-backward-ros` or `librerun-sdk`), run `pixi search <name>` for the closest available build and pin that version in `pixi.toml`; if `librerun-sdk` has no build matching the C++ API brain uses, build the copy in `$ROBOTEDGE_ROOT/third_party/rerun_cpp_sdk-0.20.3-multiplatform.zip` per `$ROBOTEDGE_ROOT/scripts/install_rerun_cpp.sh` into `.pixi/envs/default` and drop `librerun-sdk` from `pixi.toml`.

- [ ] **Step 7: Write the failing environment test `tests/test_environment.py`**

```python
from pathlib import Path


def test_t2_walk_task_imports_without_linux_only_deps():
    from tasks.locomotion.robots.t2 import T2WalkTaskCfg

    cfg = T2WalkTaskCfg()
    assert cfg.policy.checkpoint_path == "robots/t2/models/t2_walk.pt"
    assert len(cfg.policy.policy_joint_names) == 23


def test_t2_policy_checkpoint_runs_on_cpu_with_expected_shapes():
    import torch
    import tasks.locomotion as loco
    from booster_deploy.utils.policy_runner import create_policy_runner

    path = Path(loco.__file__).parent / "robots/t2/models/t2_walk.pt"
    runner = create_policy_runner(str(path), torch.device("cpu"), use_actor_module=False)
    # ang_vel 3 + gravity 3 + command 3 + dof_pos 23 + dof_vel 23 + last_action 23 + heading_err 1 + standing 1
    obs_dim = 80
    out = runner(torch.zeros(10 * obs_dim)).reshape(-1)
    assert out.numel() == 23
    assert torch.isfinite(out).all()


def test_t2_mjcf_joint_and_actuator_order_matches_robot_cfg():
    import mujoco
    from booster_assets import BOOSTER_ASSETS_DIR
    from booster_deploy.robots.t2 import T2_31DOF_CFG

    path = T2_31DOF_CFG.mjcf_path.replace("{BOOSTER_ASSETS_DIR}", str(BOOSTER_ASSETS_DIR))
    model = mujoco.MjModel.from_xml_path(path)
    joints = [model.joint(i).name for i in range(1, model.njnt)]  # joint 0 is the free joint
    actuators = [model.actuator(i).name for i in range(model.nu)]
    assert joints == list(T2_31DOF_CFG.joint_names)
    assert actuators == list(T2_31DOF_CFG.joint_names)
```

- [ ] **Step 8: Run the environment tests**

Run: `pixi run pytest tests/test_environment.py -v`
Expected: 3 passed. A failure here means the vendored Booster code or assets differ from what this plan was written against (2026-10-10); stop and report which assertion failed.

- [ ] **Step 9: Write `scripts/setup_ws.sh` and build `brain` (the gate)**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROBOTEDGE_ROOT="${ROBOTEDGE_ROOT:-/Users/johnnytan5/Robotedge-5v5-RCAP2026}"
SDK_ZIP="${SDK_ZIP:-$HOME/Downloads/sdk_release.zip}"
PACKAGES="${PACKAGES:-brain}"

mkdir -p "$ROOT/ros_ws/src"
ln -sfn "$ROBOTEDGE_ROOT/src" "$ROOT/ros_ws/src/robotedge"

if [ ! -d "$ROOT/vendor/booster_sdk/include/booster" ]; then
  tmp="$(mktemp -d)"
  unzip -q "$SDK_ZIP" -d "$tmp"
  mkdir -p "$ROOT/vendor/booster_sdk"
  cp -R "$tmp/sdk_release/include" "$ROOT/vendor/booster_sdk/include"
  rm -rf "$tmp"
fi

cd "$ROOT/ros_ws"
# shellcheck disable=SC2086
colcon build --symlink-install \
  --packages-up-to $PACKAGES \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  "-DCMAKE_CXX_FLAGS=-I$ROOT/vendor/booster_sdk/include"
```

Run:
```bash
chmod +x scripts/setup_ws.sh
pixi run build-ws
pixi run bash -c 'source ros_ws/install/setup.bash && ros2 pkg executables brain && ros2 interface show brain/msg/Kick'
```
Expected: colcon summary lists `brain`, `booster_interface`, `booster_msgs`, `vision_interface`, `game_controller_interface` as finished (the `vision` package is not built — `brain` does not depend on it); the commands print `brain brain_node` and the `Kick` fields (`x`, `y`, `dir`, `goal_x`, `goal_y`, `robot_theta_to_field`, `power`).

**Gate:** if `brain` fails to compile, record the first error and stop (see the note at the top of this task). Do not edit Robotedge sources to make it compile.

- [ ] **Step 10: Write `scripts/fetch_models.sh` and fetch the YOLO ball model**

```bash
#!/usr/bin/env bash
# Booster's sim-trained YOLOv8 detector from the public robocup_demo repo (Git LFS object).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/vendor/models"
FILE="$DEST/sim_data_det_0126.onnx"
URL="https://media.githubusercontent.com/media/BoosterRobotics/robocup_demo/main/src/vision/model/sim_data_det_0126.onnx"
SHA256="4410f04175bb3a7bc79ed7cfce2a5ab0e9c695be8cb488fbfd9d5619dc492005"

mkdir -p "$DEST"
if [ -f "$FILE" ] && echo "$SHA256  $FILE" | shasum -a 256 -c --status; then
  echo "model present: $FILE"
  exit 0
fi
curl -fsSL -o "$FILE.tmp" "$URL"
echo "$SHA256  $FILE.tmp" | shasum -a 256 -c -
mv "$FILE.tmp" "$FILE"
echo "fetched: $FILE"
```

Run:
```bash
chmod +x scripts/fetch_models.sh
pixi run fetch-models
```
Expected: `...sim_data_det_0126.onnx.tmp: OK` then `fetched: .../vendor/models/sim_data_det_0126.onnx`. A checksum failure means Booster changed the file on `main`; stop and report rather than accepting a different model.

- [ ] **Step 11: Commit**

```bash
git add .gitignore .gitmodules pixi.toml pixi.lock pyproject.toml soccer_sim/__init__.py scripts/setup_ws.sh scripts/fetch_models.sh tests/test_environment.py vendor/booster_deploy vendor/booster_assets
git commit -m "Bootstrap soccerSim: pixi ROS 2 Kilted env, Booster submodules, brain build, YOLO model fetch"
```

---

### Task 2: Field model, calibrated camera, and MuJoCo world builder

**Files:**
- Create: `soccer_sim/config.py`, `soccer_sim/field.py`
- Test: `tests/test_field.py`

**Interfaces:**
- Consumes: `booster_assets.BOOSTER_ASSETS_DIR`, `booster_deploy.robots.t2.T2_31DOF_CFG` (Task 1 environment).
- Produces:
  - `config.FieldDims` (frozen dataclass: `length, width, penalty_dist, goal_width, circle_radius, penalty_area_length, penalty_area_width, goal_area_length, goal_area_width`)
  - `config.CameraCfg` (frozen: `width:int, height:int, fx, fy, cx, cy, cam_to_head: tuple of 4 row tuples`) with method `cam_to_head_matrix() -> np.ndarray` (4×4)
  - `config.SimConfig` (fields listed in Step 3)
  - `field.Segment = tuple[tuple[float, float], tuple[float, float]]`
  - `field.map_markings(fd) -> list[tuple[str, float, float]]`, `field.field_lines(fd, circle_segments=24) -> list[Segment]`, `field.goalposts(fd) -> list[tuple[float, float]]`
  - `field.OPTICAL_FROM_MUJOCO_CAM = np.diag([1.0, -1.0, -1.0])` — multiply a MuJoCo camera rotation by this to get the optical-frame rotation
  - `field.build_world(cfg: SimConfig) -> mujoco.MjModel` — model contains the T2 (free joint first, then the 31 joints, 31 motors), a free body `ball` with joint `ball_joint` and checker-textured geom `ball_geom`, a grass-textured `floor`, camera `head_rgb` on `head_pitch_link`, site `imu` on `trunk`, sensors `imu_gyro` and `imu_acc`.
  - `SimConfig.ball_detector` (`"yolo"` | `"ground_truth"`), `SimConfig.yolo_model_path`, `SimConfig.yolo_ball_conf`, `SimConfig.yolo_nms_iou` — consumed by Task 8.
  - `config.SHOT_SLIDERS: tuple[tuple[str, float, float, float], ...]` — `(name, min, max, default)` for the six shot sliders; `field.ROBOT_ACTUATOR_GROUP = 3`. `build_world` adds one zero-force actuator per slider after the 31 robot motors (actuator indices 31..36, group 0) and moves the robot motors to group 3.

- [ ] **Step 1: Write the failing tests `tests/test_field.py`**

```python
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
```

`test_camera_defaults_match_team_vision_yaml` fails loudly if the team recalibrates the camera, so the sim never silently drifts from the robot.

- [ ] **Step 2: Run the tests to see them fail**

Run: `pixi run pytest tests/test_field.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'soccer_sim.config'`.

- [ ] **Step 3: Write `soccer_sim/config.py`**

```python
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_YOLO_MODEL = str(Path(__file__).resolve().parents[1] / "vendor" / "models" / "sim_data_det_0126.onnx")

# (name, min, max, default) — drawn as sliders in the MuJoCo viewer's top-right Control panel.
SHOT_SLIDERS: tuple[tuple[str, float, float, float], ...] = (
    ("shot_origin_x", -11.0, 11.0, -6.0),
    ("shot_origin_y", -7.0, 7.0, 0.0),
    ("shot_speed", 0.0, 10.0, 4.0),  # m/s
    ("shot_direction_deg", -180.0, 180.0, 180.0),  # field frame; 180 = toward our goal (-x)
    ("shot_elevation_deg", 0.0, 45.0, 0.0),
    ("shot_fire", 0.0, 1.0, 0.0),  # drag above 0.5 to shoot, back below 0.5 to re-arm
)


@dataclass(frozen=True)
class FieldDims:
    """Mirror of Robotedge FD_ROBOLEAGUE (src/brain/include/types.h)."""

    length: float = 22.003
    width: float = 14.126
    penalty_dist: float = 3.635
    goal_width: float = 2.6
    circle_radius: float = 1.99
    penalty_area_length: float = 5.221
    penalty_area_width: float = 8.121
    goal_area_length: float = 2.307
    goal_area_width: float = 5.083


@dataclass(frozen=True)
class CameraCfg:
    """Mirror of $ROBOTEDGE_ROOT/src/vision/config/vision.yaml camera.intrin / camera.extrin.

    cam_to_head maps the optical frame (x right, y down, z forward) into the head frame,
    which the sim takes to be MuJoCo body head_pitch_link.
    """

    width: int = 544
    height: int = 448
    fx: float = 208.712936
    fy: float = 208.883377
    cx: float = 272.0
    cy: float = 224.0
    cam_to_head: tuple[tuple[float, float, float, float], ...] = (
        (0.0, -0.5716, 0.8206, 0.08684),
        (-1.0, 0.0, 0.0, 0.0349),
        (0.0, -0.8206, -0.5716, 0.08245),
        (0.0, 0.0, 0.0, 1.0),
    )

    def cam_to_head_matrix(self) -> np.ndarray:
        return np.array(self.cam_to_head, dtype=float)


@dataclass(frozen=True)
class SimConfig:
    field: FieldDims = field(default_factory=FieldDims)
    camera: CameraCfg = field(default_factory=CameraCfg)
    ball_radius: float = 0.105  # size 5, same as vision.yaml ball_pose_estimator.ball_size
    ball_mass: float = 0.45
    goalpost_radius: float = 0.05
    goal_height: float = 1.2
    line_width: float = 0.05
    # Default spawn = brain's "enter_field" prior for player_start_pos "right":
    # own half, just outside the right touchline, facing into the field.
    spawn_xy_yaw: tuple[float, float, float] = (-6.0, -7.5, 1.5707963267948966)
    ball_xy: tuple[float, float] = (-6.0, -4.0)
    max_ball_range: float = 8.0
    max_goalpost_range: float = 12.0
    max_marking_range: float = 5.0
    max_line_range: float = 5.0
    max_depth_m: float = 10.0
    # ponytail: ground-truth goalposts/markings/lines are perfect; raise to emulate vision noise.
    detection_noise_std: float = 0.0
    ball_detector: str = "yolo"  # "yolo" (real model on rendered frames) | "ground_truth"
    yolo_model_path: str = DEFAULT_YOLO_MODEL
    yolo_ball_conf: float = 0.4  # vision.yaml post_process.confidence_thresholds.Ball
    yolo_nms_iou: float = 0.4  # vision.yaml detection_model.nms_threshold
    initial_mode: int = 4  # RobotMode.kSoccer
```

- [ ] **Step 4: Write `soccer_sim/field.py`**

```python
import math

import mujoco
import numpy as np
from booster_assets import BOOSTER_ASSETS_DIR
from booster_deploy.robots.t2 import T2_31DOF_CFG

from .config import SHOT_SLIDERS, FieldDims, SimConfig

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


def robot_mjcf_path() -> str:
    return T2_31DOF_CFG.mjcf_path.replace("{BOOSTER_ASSETS_DIR}", str(BOOSTER_ASSETS_DIR))


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def build_world(cfg: SimConfig) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(robot_mjcf_path())
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
    spec.body("head_pitch_link").add_camera(
        name="head_rgb",
        pos=cam_to_head[:3, 3].tolist(),
        quat=cam_quat.tolist(),
        fovy=math.degrees(2.0 * math.atan(cfg.camera.height / 2 / cfg.camera.fy)),
    )

    spec.body("trunk").add_site(name="imu", pos=[0.0, 0.0, 0.0], size=[0.01, 0.0, 0.0])
    spec.add_sensor(name="imu_gyro", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(
        name="imu_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu"
    )

    for act in spec.actuators:  # only the 31 robot motors exist at this point
        act.group = ROBOT_ACTUATOR_GROUP
    for name, lo, hi, _default in SHOT_SLIDERS:
        # ponytail: zero-gain actuators used purely as viewer sliders; they never apply force.
        slider = spec.add_actuator(
            name=name,
            target="aa_head_yaw_joint",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE,
            ctrlrange=[lo, hi],
            group=0,
        )
        slider.gainprm = [0.0] * len(slider.gainprm)
        slider.biasprm = [0.0] * len(slider.biasprm)
    return spec.compile()
```

MuJoCo renders with square pixels, so the rendered image uses `fy` for both axes; `fx` differs from `fy` by 0.08 %, which is far below a pixel at this resolution.

- [ ] **Step 5: Run the tests**

Run: `pixi run pytest tests/test_field.py -v`
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add soccer_sim/config.py soccer_sim/field.py tests/test_field.py
git commit -m "Add robo_league field, textured ball/turf, calibrated head camera, shot sliders, and world builder"
```

---

### Task 3: Frame math, odometry, and ground-truth perception

**Files:**
- Create: `soccer_sim/geometry.py`, `soccer_sim/odometry.py`, `soccer_sim/perception.py`
- Test: `tests/test_perception.py`

**Interfaces:**
- Consumes: `config.CameraCfg`, `field.Segment` (Task 2).
- Produces:
  - `geometry.wrap_pi(a) -> float` (result in (−π, π]), `geometry.yaw_from_quat_wxyz(q) -> float`, `geometry.rpy_from_quat_wxyz(q) -> tuple[float, float, float]`, `geometry.world_to_robot_2d(px, py, rx, ry, ryaw) -> tuple[float, float]`
  - `perception.Pose2D(x, y, yaw)` (frozen dataclass)
  - `perception.Detection(label: str, confidence: float, bbox: tuple[int,int,int,int], pos_robot: tuple[float,float], target_uv: tuple[float,float] | None = None)`
  - `perception.LineObs(robot: tuple[float,float,float,float], uv: tuple[float,float,float,float])`
  - `perception.Camera(cfg: CameraCfg, pos_w, rot_w_optical)` — `rot_w_optical` is the 3×3 world-from-optical rotation (columns = optical x right, y down, z forward, in world); `.project(p_w) -> tuple[u, v, depth] | None`; attributes `fx, fy, cx, cy`
  - `perception.detect_ball(cam, robot, ball_w, radius, max_range, noise_std=0.0, rng=None) -> Detection | None`
  - `perception.detect_goalposts(cam, robot, posts, height, radius, max_range, noise_std=0.0, rng=None) -> list[Detection]`
  - `perception.detect_markings(cam, robot, markings, max_range, noise_std=0.0, rng=None) -> list[Detection]`
  - `perception.visible_lines(cam, robot, segments, max_range, step=0.05) -> list[LineObs]`
  - `odometry.OdometryTracker(origin: Pose2D)` with `.update(pose_w: Pose2D) -> tuple[float, float, float]`

- [ ] **Step 1: Write the failing tests `tests/test_perception.py`**

```python
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
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `pixi run pytest tests/test_perception.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'soccer_sim.geometry'`.

- [ ] **Step 3: Write `soccer_sim/geometry.py`**

```python
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
```

- [ ] **Step 4: Write `soccer_sim/perception.py`**

```python
import math
from dataclasses import dataclass

import numpy as np

from .config import CameraCfg
from .field import Segment
from .geometry import world_to_robot_2d

MARKING_HALF_SIZE_M = 0.15
CONFIDENCE = 0.9
MIN_DEPTH_M = 0.05


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # xmin, ymin, xmax, ymax in pixels
    pos_robot: tuple[float, float]  # ground projection, robot (base) frame, metres
    target_uv: tuple[float, float] | None = None


@dataclass(frozen=True)
class LineObs:
    robot: tuple[float, float, float, float]  # x0, y0, x1, y1 robot frame
    uv: tuple[float, float, float, float]  # u0, v0, u1, v1 pixels


class Camera:
    """Pinhole camera in the optical convention: x right, y down, z forward."""

    def __init__(self, cfg: CameraCfg, pos_w, rot_w_optical):
        self.cfg = cfg
        self.pos = np.asarray(pos_w, dtype=float).reshape(3)
        self.rot = np.asarray(rot_w_optical, dtype=float).reshape(3, 3)
        self.fx, self.fy, self.cx, self.cy = cfg.fx, cfg.fy, cfg.cx, cfg.cy

    def to_cam(self, p_w) -> np.ndarray:
        return self.rot.T @ (np.asarray(p_w, dtype=float).reshape(3) - self.pos)

    def project(self, p_w) -> tuple[float, float, float] | None:
        d = self.to_cam(p_w)
        if d[2] <= MIN_DEPTH_M:
            return None
        u = self.fx * d[0] / d[2] + self.cx
        v = self.fy * d[1] / d[2] + self.cy
        if not (0.0 <= u < self.cfg.width and 0.0 <= v < self.cfg.height):
            return None
        return float(u), float(v), float(d[2])


def _bbox(cam: Camera, u: float, v_top: float, v_bottom: float, half_w: float) -> tuple[int, int, int, int]:
    w, h = cam.cfg.width - 1, cam.cfg.height - 1
    return (
        int(max(0.0, min(w, u - half_w))),
        int(max(0.0, min(h, v_top))),
        int(max(0.0, min(w, u + half_w))),
        int(max(0.0, min(h, v_bottom))),
    )


def _robot_xy(robot: Pose2D, x: float, y: float, noise_std: float, rng) -> tuple[float, float]:
    rx, ry = world_to_robot_2d(x, y, robot.x, robot.y, robot.yaw)
    if noise_std > 0.0 and rng is not None:
        rx += float(rng.normal(0.0, noise_std))
        ry += float(rng.normal(0.0, noise_std))
    return rx, ry


def _in_range(robot: Pose2D, x: float, y: float, max_range: float) -> bool:
    return math.hypot(x - robot.x, y - robot.y) <= max_range


def detect_ball(cam, robot, ball_w, radius, max_range, noise_std=0.0, rng=None) -> Detection | None:
    bx, by, bz = (float(v) for v in ball_w)
    if not _in_range(robot, bx, by, max_range):
        return None
    p = cam.project((bx, by, bz))
    if p is None:
        return None
    u, v, depth = p
    half_u, half_v = cam.fx * radius / depth, cam.fy * radius / depth
    return Detection(
        "Ball", CONFIDENCE, _bbox(cam, u, v - half_v, v + half_v, half_u), _robot_xy(robot, bx, by, noise_std, rng)
    )


def detect_goalposts(cam, robot, posts, height, radius, max_range, noise_std=0.0, rng=None) -> list[Detection]:
    out = []
    for gx, gy in posts:
        if not _in_range(robot, gx, gy, max_range):
            continue
        bottom = cam.project((gx, gy, 0.0))
        if bottom is None:
            continue
        u, v_bottom, depth = bottom
        top = cam.to_cam((gx, gy, height))
        v_top = cam.fy * top[1] / top[2] + cam.cy if top[2] > MIN_DEPTH_M else 0.0
        out.append(
            Detection(
                "Goalpost",
                CONFIDENCE,
                _bbox(cam, u, v_top, v_bottom, cam.fx * radius / depth),
                _robot_xy(robot, gx, gy, noise_std, rng),
            )
        )
    return out


def detect_markings(cam, robot, markings, max_range, noise_std=0.0, rng=None) -> list[Detection]:
    out = []
    for label, mx, my in markings:
        if not _in_range(robot, mx, my, max_range):
            continue
        p = cam.project((mx, my, 0.0))
        if p is None:
            continue
        u, v, depth = p
        half_u, half_v = cam.fx * MARKING_HALF_SIZE_M / depth, cam.fy * MARKING_HALF_SIZE_M / depth
        out.append(
            Detection(
                label,
                CONFIDENCE,
                _bbox(cam, u, v - half_v, v + half_v, half_u),
                _robot_xy(robot, mx, my, noise_std, rng),
                target_uv=(u, v),
            )
        )
    return out


def visible_lines(cam, robot, segments: list[Segment], max_range: float, step: float = 0.05) -> list[LineObs]:
    """Sample each segment; every contiguous visible run becomes one observed segment."""
    out = []
    for (x0, y0), (x1, y1) in segments:
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step) + 1)
        run: list[tuple[float, float, float, float]] = []

        def flush():
            if len(run) >= 2:
                (ax, ay, au, av), (bx, by, bu, bv) = run[0], run[-1]
                pa = world_to_robot_2d(ax, ay, robot.x, robot.y, robot.yaw)
                pb = world_to_robot_2d(bx, by, robot.x, robot.y, robot.yaw)
                out.append(LineObs((pa[0], pa[1], pb[0], pb[1]), (au, av, bu, bv)))
            run.clear()

        for i in range(n):
            t = i / (n - 1)
            px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            p = cam.project((px, py, 0.0)) if _in_range(robot, px, py, max_range) else None
            if p is None:
                flush()
            else:
                run.append((px, py, p[0], p[1]))
        flush()
    return out
```

- [ ] **Step 5: Write `soccer_sim/odometry.py`**

```python
from .geometry import world_to_robot_2d, wrap_pi
from .perception import Pose2D


class OdometryTracker:
    """Ground-truth odometer zeroed at the spawn pose, like the robot's /odometer_state after boot."""

    def __init__(self, origin: Pose2D):
        self.origin = origin

    def update(self, pose_w: Pose2D) -> tuple[float, float, float]:
        o = self.origin
        x, y = world_to_robot_2d(pose_w.x, pose_w.y, o.x, o.y, o.yaw)
        return x, y, wrap_pi(pose_w.yaw - o.yaw)
```

`odometry.py` imports `Pose2D` from `perception.py`; `perception.py` must not import `odometry.py` (no cycle).

- [ ] **Step 6: Run the tests**

Run: `pixi run pytest tests/test_perception.py -v`
Expected: 15 passed.

- [ ] **Step 7: Commit**

```bash
git add soccer_sim/geometry.py soccer_sim/odometry.py soccer_sim/perception.py tests/test_perception.py
git commit -m "Add frame math, spawn-relative odometry, and optical-camera ground-truth perception"
```

---

### Task 4: Loco RPC protocol (what the robot SDK answers)

**Files:**
- Create: `soccer_sim/loco_rpc.py`
- Test: `tests/test_loco_rpc.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - constants `API_CHANGE_MODE=2000, API_MOVE=2001, API_ROTATE_HEAD=2004, API_GET_UP=2008, API_GET_MODE=2017, API_VISUAL_KICK=2038`; `MODE_DAMPING=0, MODE_PREPARE=1, MODE_WALKING=2, MODE_CUSTOM=3, MODE_SOCCER=4`; `MOVING_MODES=(2, 4)`; `KICK_V1=0, KICK_V2=1`; `STATUS_OK=0, STATUS_BAD_REQUEST=1`
  - `VelocityLimits(vx_min=-0.7, vx_max=3.0, vy_max=0.8, vyaw_max=2.0)`
  - `LocoState(mode=4, vx=0.0, vy=0.0, vyaw=0.0, head_pitch=0.0, head_yaw=0.0, getup_requested=False, visual_kick_active=False, visual_kick_version=1)` with `.effective_velocity() -> tuple[float, float, float]` and `.kicking() -> bool`
  - `RpcReply(header: str, body: str)`
  - `handle_request(state, header: str, body: str, limits=VelocityLimits()) -> RpcReply`

- [ ] **Step 1: Write the failing tests `tests/test_loco_rpc.py`**

```python
import json

import pytest

from soccer_sim.loco_rpc import (
    API_CHANGE_MODE, API_GET_MODE, API_GET_UP, API_MOVE, API_ROTATE_HEAD, API_VISUAL_KICK, KICK_V1, KICK_V2,
    MODE_PREPARE, MODE_SOCCER, MODE_WALKING, STATUS_BAD_REQUEST, STATUS_OK, LocoState, handle_request,
)


def req(api_id, **params):
    return json.dumps({"api_id": api_id}), json.dumps(params)


def status(reply):
    return json.loads(reply.header)["status"]


def test_move_sets_velocity():
    s = LocoState()
    r = handle_request(s, *req(API_MOVE, vx=0.5, vy=-0.2, vyaw=0.3))
    assert status(r) == STATUS_OK
    assert s.effective_velocity() == pytest.approx((0.5, -0.2, 0.3))


def test_move_is_clamped_to_policy_limits():
    s = LocoState()
    handle_request(s, *req(API_MOVE, vx=5.0, vy=1.2, vyaw=-3.0))
    assert s.effective_velocity() == pytest.approx((3.0, 0.8, -2.0))
    handle_request(s, *req(API_MOVE, vx=-2.0, vy=-1.2, vyaw=3.0))
    assert s.effective_velocity() == pytest.approx((-0.7, -0.8, 2.0))


def test_non_walking_modes_zero_the_velocity():
    s = LocoState()
    handle_request(s, *req(API_MOVE, vx=0.5, vy=0.0, vyaw=0.0))
    handle_request(s, *req(API_CHANGE_MODE, mode=MODE_PREPARE))
    assert s.effective_velocity() == (0.0, 0.0, 0.0)
    handle_request(s, *req(API_CHANGE_MODE, mode=MODE_WALKING))
    assert s.effective_velocity() == (0.0, 0.0, 0.0)  # old command does not resume by itself


def test_get_mode_reports_current_mode():
    s = LocoState()
    r = handle_request(s, *req(API_GET_MODE))
    assert status(r) == STATUS_OK
    assert json.loads(r.body) == {"mode": MODE_SOCCER}


def test_rotate_head_sets_targets():
    s = LocoState()
    handle_request(s, *req(API_ROTATE_HEAD, pitch=0.3, yaw=-0.5))
    assert (s.head_pitch, s.head_yaw) == (0.3, -0.5)


def test_get_up_sets_request_flag():
    s = LocoState()
    handle_request(s, *req(API_GET_UP))
    assert s.getup_requested


def test_visual_kick_start_and_stop():
    s = LocoState()
    assert status(handle_request(s, *req(API_VISUAL_KICK, start=True, version=KICK_V1))) == STATUS_OK
    assert s.kicking() and s.visual_kick_version == KICK_V1
    handle_request(s, *req(API_VISUAL_KICK, start=False, version=KICK_V2))
    assert not s.kicking()


def test_change_mode_ends_visual_kick_and_prepare_blocks_it():
    s = LocoState()
    handle_request(s, *req(API_VISUAL_KICK, start=True, version=KICK_V2))
    handle_request(s, *req(API_CHANGE_MODE, mode=MODE_SOCCER))
    assert not s.kicking()
    handle_request(s, *req(API_CHANGE_MODE, mode=MODE_PREPARE))
    handle_request(s, *req(API_VISUAL_KICK, start=True, version=KICK_V2))
    assert s.visual_kick_active and not s.kicking()  # flag set, but no kicking outside walking/soccer


def test_unknown_ids_are_acknowledged_no_ops():
    s = LocoState()
    assert status(handle_request(s, *req(9999, anything=1))) == STATUS_OK
    assert s.effective_velocity() == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "header,body",
    [
        ("not json", "{}"),
        (json.dumps({"no_api_id": 1}), "{}"),
        (json.dumps({"api_id": API_MOVE}), json.dumps({"vx": 0.1})),
        (json.dumps({"api_id": API_MOVE}), '{"vx": NaN, "vy": 0, "vyaw": 0}'),
        (json.dumps({"api_id": API_CHANGE_MODE}), json.dumps({"mode": 42})),
        (json.dumps({"api_id": API_ROTATE_HEAD}), '{"pitch": Infinity, "yaw": 0}'),
        (json.dumps({"api_id": API_VISUAL_KICK}), json.dumps({"start": "yes", "version": 1})),
    ],
)
def test_malformed_requests_are_rejected_without_state_change(header, body):
    s = LocoState()
    before = (s.mode, s.vx, s.vy, s.vyaw, s.head_pitch, s.head_yaw, s.visual_kick_active)
    r = handle_request(s, header, body)
    assert status(r) == STATUS_BAD_REQUEST
    assert (s.mode, s.vx, s.vy, s.vyaw, s.head_pitch, s.head_yaw, s.visual_kick_active) == before
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `pixi run pytest tests/test_loco_rpc.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'soccer_sim.loco_rpc'`.

- [ ] **Step 3: Write `soccer_sim/loco_rpc.py`**

```python
"""Simulated Booster loco RPC service: the subset of b1_loco_api brain_node uses."""

import json
import math
from dataclasses import dataclass

API_CHANGE_MODE = 2000
API_MOVE = 2001
API_ROTATE_HEAD = 2004
API_GET_UP = 2008
API_GET_MODE = 2017
API_VISUAL_KICK = 2038

MODE_DAMPING, MODE_PREPARE, MODE_WALKING, MODE_CUSTOM, MODE_SOCCER = 0, 1, 2, 3, 4
VALID_MODES = (MODE_DAMPING, MODE_PREPARE, MODE_WALKING, MODE_CUSTOM, MODE_SOCCER)
MOVING_MODES = (MODE_WALKING, MODE_SOCCER)
KICK_V1, KICK_V2 = 0, 1  # booster::robot::b1::VisualKickVersion

STATUS_OK = 0
# ponytail: the real SDK's error codes are undocumented; brain only checks status == 0.
STATUS_BAD_REQUEST = 1


@dataclass(frozen=True)
class VelocityLimits:
    vx_min: float = -0.7
    vx_max: float = 3.0
    vy_max: float = 0.8
    vyaw_max: float = 2.0


@dataclass
class LocoState:
    mode: int = MODE_SOCCER
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    getup_requested: bool = False
    visual_kick_active: bool = False
    visual_kick_version: int = KICK_V2

    def effective_velocity(self) -> tuple[float, float, float]:
        if self.mode in MOVING_MODES:
            return self.vx, self.vy, self.vyaw
        return 0.0, 0.0, 0.0

    def kicking(self) -> bool:
        return self.visual_kick_active and self.mode in MOVING_MODES


@dataclass(frozen=True)
class RpcReply:
    header: str
    body: str


def _finite(*values) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in out):
        raise ValueError("non-finite parameter")
    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ok(body: str = "") -> RpcReply:
    return RpcReply(json.dumps({"status": STATUS_OK}), body)


def handle_request(state: LocoState, header: str, body: str, limits: VelocityLimits = VelocityLimits()) -> RpcReply:
    try:
        api_id = int(json.loads(header)["api_id"])
        params = json.loads(body) if body else {}
        if api_id == API_MOVE:
            vx, vy, vyaw = _finite(params["vx"], params["vy"], params["vyaw"])
            state.vx = _clamp(vx, limits.vx_min, limits.vx_max)
            state.vy = _clamp(vy, -limits.vy_max, limits.vy_max)
            state.vyaw = _clamp(vyaw, -limits.vyaw_max, limits.vyaw_max)
        elif api_id == API_ROTATE_HEAD:
            state.head_pitch, state.head_yaw = _finite(params["pitch"], params["yaw"])
        elif api_id == API_CHANGE_MODE:
            mode = int(params["mode"])
            if mode not in VALID_MODES:
                raise ValueError(f"unknown mode {mode}")
            state.mode = mode
            state.vx = state.vy = state.vyaw = 0.0
            state.visual_kick_active = False
        elif api_id == API_GET_MODE:
            return _ok(json.dumps({"mode": state.mode}))
        elif api_id == API_GET_UP:
            state.getup_requested = True
        elif api_id == API_VISUAL_KICK:
            start = params["start"]
            if not isinstance(start, bool):
                raise ValueError("start must be a bool")
            version = int(params["version"])
            state.visual_kick_active = start
            state.visual_kick_version = version
        # Every other id: acknowledged no-op.
    except (KeyError, ValueError, TypeError):
        return RpcReply(json.dumps({"status": STATUS_BAD_REQUEST}), "")
    return _ok()
```

`json.JSONDecodeError` is a subclass of `ValueError`, so bad JSON is covered. Any mode change zeroes the stored velocity and ends a visual kick, matching `RobotClient::changeRobocupMode()` (ChangeMode followed by VisualKick false).

- [ ] **Step 4: Run the tests**

Run: `pixi run pytest tests/test_loco_rpc.py -v`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add soccer_sim/loco_rpc.py tests/test_loco_rpc.py
git commit -m "Add simulated loco RPC service with visual-kick mode, clamping, and strict validation"
```

---

### Task 5: `VisualKickModel` — stand-in for Booster's RL visual kick

**Files:**
- Create: `soccer_sim/kick_model.py`
- Test: `tests/test_kick_model.py`

**Interfaces:**
- Consumes: `loco_rpc.KICK_V2` (Task 4).
- Produces:
  - `KickRef(ball_x: float, ball_y: float, dir: float, power: float, stamp: float)` (frozen) — from `brain/msg/Kick`; `stamp` in seconds on the caller's clock
  - `KickParams` (frozen; fields in Step 3)
  - `KickOutput(vx, vy, vyaw, strike_speed=0.0, strike_dir=0.0)` (frozen) — `strike_speed > 0` means "launch the ball now at this speed in direction `strike_dir` (robot frame, rad)"
  - `VisualKickModel(params=KickParams())` with `.reset()` and `.step(ref: KickRef | None, now: float, true_ball_robot: tuple[float, float], version: int) -> KickOutput`
  - `ball_speed_for_power(power: float, version: int, p: KickParams) -> float`

- [ ] **Step 1: Write the failing tests `tests/test_kick_model.py`**

```python
import math

import pytest

from soccer_sim.kick_model import KickParams, KickRef, VisualKickModel, ball_speed_for_power
from soccer_sim.loco_rpc import KICK_V1, KICK_V2

P = KickParams()


def ref(x, y, direction=0.0, power=3.0, stamp=0.0):
    return KickRef(ball_x=x, ball_y=y, dir=direction, power=power, stamp=stamp)


def test_strike_when_aligned_and_in_reach():
    out = VisualKickModel().step(ref(P.strike_ball_x, 0.0), 0.1, (P.strike_ball_x, 0.0), KICK_V2)
    assert out.strike_speed == pytest.approx(ball_speed_for_power(3.0, KICK_V2, P))
    assert (out.vx, out.vy, out.vyaw) == (0.0, 0.0, 0.0)


def test_approach_moves_toward_ball_and_turns_toward_kick_direction():
    out = VisualKickModel().step(ref(1.0, 0.3, direction=0.4), 0.1, (1.0, 0.3), KICK_V2)
    assert out.strike_speed == 0.0
    assert out.vx > 0.0 and out.vy > 0.0 and out.vyaw > 0.0


def test_approach_velocity_is_clamped():
    out = VisualKickModel().step(ref(10.0, -10.0, direction=-3.0), 0.1, (10.0, -10.0), KICK_V2)
    assert out.vx == pytest.approx(P.max_vx)
    assert out.vy == pytest.approx(-P.max_vy)
    assert out.vyaw == pytest.approx(-P.max_vyaw)


def test_no_strike_when_real_ball_out_of_reach():
    # brain believes the ball is at the foot, but the true ball is 2 m away
    out = VisualKickModel().step(ref(P.strike_ball_x, 0.0), 0.1, (2.0, 0.0), KICK_V2)
    assert out.strike_speed == 0.0


def test_stale_or_missing_reference_stops_the_robot():
    m = VisualKickModel()
    assert m.step(None, 1.0, (1.0, 0.0), KICK_V2) == m.step(ref(1.0, 0.0, stamp=0.0), 1.0, (1.0, 0.0), KICK_V2)
    out = m.step(ref(1.0, 0.0, stamp=0.0), 1.0, (1.0, 0.0), KICK_V2)
    assert (out.vx, out.vy, out.vyaw, out.strike_speed) == (0.0, 0.0, 0.0, 0.0)


def test_cooldown_after_strike_then_ready_again():
    m = VisualKickModel()
    aligned = (P.strike_ball_x, 0.0)
    assert m.step(ref(*aligned, stamp=0.0), 0.0, aligned, KICK_V2).strike_speed > 0.0
    assert m.step(ref(*aligned, stamp=0.1), 0.1, aligned, KICK_V2).strike_speed == 0.0
    later = P.cooldown + 0.01
    assert m.step(ref(*aligned, stamp=later), later, aligned, KICK_V2).strike_speed > 0.0


def test_power_maps_to_rollout_distance():
    v = ball_speed_for_power(4.0, KICK_V1, P)
    assert v == pytest.approx(math.sqrt(2.0 * P.ball_decel * 4.0))
    assert ball_speed_for_power(-1.0, KICK_V1, P) == 0.0
    assert ball_speed_for_power(1e6, KICK_V2, P) == pytest.approx(P.max_ball_speed)


def test_v2_kicks_harder_than_v1():
    assert ball_speed_for_power(3.0, KICK_V2, P) > ball_speed_for_power(3.0, KICK_V1, P)
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `pixi run pytest tests/test_kick_model.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'soccer_sim.kick_model'`.

- [ ] **Step 3: Write `soccer_sim/kick_model.py`**

```python
"""Stand-in for Booster's closed RL visual kick (SDK topic rt/kick_ball).

On the robot, brain sends kVisualKick start and then streams /kick_ball references
(ball position relative to the robot, desired kick direction, power); the firmware
approaches and kicks. This model approaches with the walk policy and, once aligned
with the real ball within reach, launches the ball. Replace it if Booster publishes
the T2 visual-kick policy.
"""

import math
from dataclasses import dataclass

from .loco_rpc import KICK_V2


@dataclass(frozen=True)
class KickRef:
    ball_x: float  # ball position in robot frame (brain::msg::Kick.x)
    ball_y: float  # brain::msg::Kick.y
    dir: float  # desired kick direction relative to robot, rad (Kick.dir)
    power: float  # desired kick range, m (Kick.power)
    stamp: float  # seconds, caller's clock


@dataclass(frozen=True)
class KickParams:
    # ponytail: hand-tuned approximation of the firmware kick; calibrate against real kick videos.
    strike_ball_x: float = 0.25  # ball distance in front of the trunk at contact
    approach_gain: float = 1.5
    yaw_gain: float = 1.5
    max_vx: float = 0.8
    max_vy: float = 0.4
    max_vyaw: float = 1.0
    strike_tol_x: float = 0.08
    strike_tol_y: float = 0.06
    strike_tol_dir: float = 0.25
    reach: float = 0.45  # the real ball must be this close for a strike
    ball_decel: float = 0.8  # m/s^2 rolling deceleration used to turn "range" into launch speed
    v2_power_scale: float = 1.3  # VisualKickVersion.kV2 is documented as "stronger kicking force"
    max_ball_speed: float = 8.0
    ref_timeout: float = 0.5
    cooldown: float = 0.8


@dataclass(frozen=True)
class KickOutput:
    vx: float
    vy: float
    vyaw: float
    strike_speed: float = 0.0
    strike_dir: float = 0.0


def ball_speed_for_power(power: float, version: int, p: KickParams) -> float:
    """Launch speed that rolls the ball `power` metres under constant deceleration."""
    speed = math.sqrt(2.0 * p.ball_decel * max(0.0, power))
    if version == KICK_V2:
        speed *= p.v2_power_scale
    return min(p.max_ball_speed, speed)


def _clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))


STOP = KickOutput(0.0, 0.0, 0.0)


class VisualKickModel:
    def __init__(self, params: KickParams = KickParams()):
        self.p = params
        self._cooldown_until = -math.inf

    def reset(self) -> None:
        self._cooldown_until = -math.inf

    def step(self, ref: KickRef | None, now: float, true_ball_robot: tuple[float, float], version: int) -> KickOutput:
        p = self.p
        if ref is None or now - ref.stamp > p.ref_timeout or now < self._cooldown_until:
            return STOP
        ex = ref.ball_x - p.strike_ball_x
        ey = ref.ball_y
        edir = math.atan2(math.sin(ref.dir), math.cos(ref.dir))
        aligned = abs(ex) < p.strike_tol_x and abs(ey) < p.strike_tol_y and abs(edir) < p.strike_tol_dir
        if aligned and math.hypot(*true_ball_robot) < p.reach:
            self._cooldown_until = now + p.cooldown
            return KickOutput(0.0, 0.0, 0.0, ball_speed_for_power(ref.power, version, p), ref.dir)
        return KickOutput(
            _clamp(p.approach_gain * ex, p.max_vx),
            _clamp(p.approach_gain * ey, p.max_vy),
            _clamp(p.yaw_gain * edir, p.max_vyaw),
        )
```

- [ ] **Step 4: Run the tests**

Run: `pixi run pytest tests/test_kick_model.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add soccer_sim/kick_model.py tests/test_kick_model.py
git commit -m "Add VisualKickModel: approach-and-strike stand-in for Booster's closed RL visual kick"
```

---

### Task 6: `YoloBallDetector` — real ball detection on rendered frames

**Files:**
- Create: `soccer_sim/yolo_ball.py`
- Test: `tests/test_yolo_ball.py`

**Interfaces:**
- Consumes: `perception.Camera`, `perception.Detection`, `perception.Pose2D` (Task 3); `geometry.world_to_robot_2d` (Task 3); model file from Task 1.
- Produces:
  - `Letterbox(scale: float, pad_x: int, pad_y: int)` (frozen)
  - `letterbox(rgb: np.ndarray, size: int = 320) -> tuple[np.ndarray, Letterbox]` — returns the `[1,3,size,size]` float32 tensor
  - `decode_balls(output: np.ndarray, lb: Letterbox, width: int, height: int, conf_thresh: float, nms_iou: float) -> list[tuple[float, tuple[float, float, float, float]]]` — `output` is the `(13, 2100)` head; returns `(confidence, (xmin, ymin, xmax, ymax))` in image pixels, best first
  - `ground_point_at_height(cam: Camera, u: float, v: float, z: float) -> np.ndarray | None` — world point where the pixel ray meets the plane at height `z`
  - `YoloBallDetector(model_path: str, conf_thresh: float = 0.4, nms_iou: float = 0.4)` with `.detect(rgb, cam, robot, ball_radius) -> list[Detection]` (label `"Ball"`, `pos_robot` in the ground base frame)

- [ ] **Step 1: Write the failing tests `tests/test_yolo_ball.py`**

```python
import math
from pathlib import Path

import numpy as np
import pytest

from soccer_sim.config import DEFAULT_YOLO_MODEL, CameraCfg
from soccer_sim.perception import Camera, Pose2D
from soccer_sim.yolo_ball import Letterbox, YoloBallDetector, decode_balls, ground_point_at_height, letterbox

CFG = CameraCfg()


def level_camera(z=1.2):
    return Camera(CFG, np.array([0.0, 0.0, z]), np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]))


def test_letterbox_shape_scale_and_padding():
    rgb = np.full((448, 544, 3), 200, np.uint8)
    x, lb = letterbox(rgb)
    assert x.shape == (1, 3, 320, 320) and x.dtype == np.float32
    assert lb.scale == pytest.approx(320 / 544)
    assert lb.pad_x == 0 and lb.pad_y == (320 - round(448 * 320 / 544)) // 2
    assert x[0, :, 0, 0] == pytest.approx([114 / 255] * 3)  # top padding row
    assert x[0, :, 160, 160] == pytest.approx([200 / 255] * 3, abs=1e-6)


def test_decode_maps_boxes_back_to_image_and_applies_threshold_and_nms():
    out = np.zeros((13, 4), np.float32)
    # columns: cx, cy, w, h, then 9 class scores (class 0 = Ball, 1 = Goalpost)
    out[:4, 0], out[4, 0] = [160, 160, 20, 20], 0.9  # ball, kept
    out[:4, 1], out[4, 1] = [162, 160, 20, 20], 0.8  # overlapping ball, suppressed by NMS
    out[:4, 2], out[4, 2] = [60, 60, 10, 10], 0.3  # ball below threshold
    out[:4, 3], out[4, 3], out[5, 3] = [250, 250, 10, 10], 0.5, 0.9  # top class Goalpost, ignored
    boxes = decode_balls(out, Letterbox(scale=0.5, pad_x=0, pad_y=0), 1000, 1000, 0.4, 0.4)
    assert len(boxes) == 1
    conf, box = boxes[0]
    assert conf == pytest.approx(0.9)
    assert box == pytest.approx((300.0, 300.0, 340.0, 340.0))


def test_ground_point_at_ball_height():
    cam = level_camera(z=1.2)
    radius, forward = 0.105, 2.0
    v = cam.cy + cam.fy * (1.2 - radius) / forward
    p = ground_point_at_height(cam, cam.cx, v, radius)
    assert p == pytest.approx([forward, 0.0, radius], abs=1e-6)


def test_ray_at_or_above_horizon_returns_none():
    cam = level_camera()
    assert ground_point_at_height(cam, cam.cx, cam.cy, 0.105) is None  # horizontal ray
    assert ground_point_at_height(cam, cam.cx, cam.cy - 50, 0.105) is None  # upward ray


@pytest.mark.skipif(not Path(DEFAULT_YOLO_MODEL).exists(), reason="run `pixi run fetch-models`")
def test_real_model_sees_no_ball_in_empty_turf():
    det = YoloBallDetector(DEFAULT_YOLO_MODEL)
    rng = np.random.default_rng(0)
    turf = np.clip(rng.normal([40, 140, 50], 12, (448, 544, 3)), 0, 255).astype(np.uint8)
    assert det.detect(turf, level_camera(), Pose2D(0.0, 0.0, 0.0), 0.105) == []
```

The end-to-end "model finds the textured sim ball at the right range" check lives in Task 8, where the full world is rendered.

- [ ] **Step 2: Run the tests to see them fail**

Run: `pixi run pytest tests/test_yolo_ball.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'soccer_sim.yolo_ball'`.

- [ ] **Step 3: Write `soccer_sim/yolo_ball.py`**

```python
"""Ball detection with Booster's sim-trained YOLOv8 (robocup_demo sim_data_det_0126.onnx).

Mirrors Robotedge vision for the ball: per-class confidence >= 0.4, NMS IoU 0.4, and
range from the bbox-centre ray intersected with the plane z = ball radius
(BallPoseEstimator::EstimateByColor; use_depth is false for the ball in vision.yaml).
"""

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from .geometry import world_to_robot_2d
from .perception import Camera, Detection, Pose2D

BALL_CLASS = 0
INPUT_SIZE = 320
PAD_VALUE = 114


@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: int
    pad_y: int


def letterbox(rgb: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, Letterbox]:
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    pad_y, pad_x = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), PAD_VALUE, np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return tensor, Letterbox(scale, pad_x, pad_y)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0.0 else 0.0


def decode_balls(output, lb: Letterbox, width: int, height: int, conf_thresh: float, nms_iou: float):
    preds = np.asarray(output).T  # (N, 4 + classes)
    scores = preds[:, 4:]
    conf = scores[:, BALL_CLASS]
    keep = np.where((scores.argmax(1) == BALL_CLASS) & (conf >= conf_thresh))[0]
    boxes: list[tuple[float, tuple[float, float, float, float]]] = []
    for i in keep[np.argsort(-conf[keep])]:
        cx, cy, w, h = (float(v) for v in preds[i, :4])
        box = (
            max(0.0, (cx - w / 2 - lb.pad_x) / lb.scale),
            max(0.0, (cy - h / 2 - lb.pad_y) / lb.scale),
            min(width - 1.0, (cx + w / 2 - lb.pad_x) / lb.scale),
            min(height - 1.0, (cy + h / 2 - lb.pad_y) / lb.scale),
        )
        if all(_iou(box, kept) <= nms_iou for _, kept in boxes):
            boxes.append((float(conf[i]), box))
    return boxes


def ground_point_at_height(cam: Camera, u: float, v: float, z: float) -> np.ndarray | None:
    ray = cam.rot @ np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0])
    if ray[2] >= -1e-6:  # at or above the horizon: never meets the plane in front of the camera
        return None
    t = (z - cam.pos[2]) / ray[2]
    return cam.pos + t * ray if t > 0.0 else None


class YoloBallDetector:
    def __init__(self, model_path: str, conf_thresh: float = 0.4, nms_iou: float = 0.4):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        names = self.session.get_modelmeta().custom_metadata_map.get("names", "")
        if "'Ball'" not in names.split(",")[0]:
            raise ValueError(f"{model_path}: class 0 is not 'Ball' (names={names})")
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thresh = conf_thresh
        self.nms_iou = nms_iou

    def detect(self, rgb: np.ndarray, cam: Camera, robot: Pose2D, ball_radius: float) -> list[Detection]:
        tensor, lb = letterbox(rgb)
        output = self.session.run(None, {self.input_name: tensor})[0][0]
        h, w = rgb.shape[:2]
        out = []
        for conf, (x0, y0, x1, y1) in decode_balls(output, lb, w, h, self.conf_thresh, self.nms_iou):
            p = ground_point_at_height(cam, (x0 + x1) / 2, (y0 + y1) / 2, ball_radius)
            if p is None:
                continue
            pos = world_to_robot_2d(float(p[0]), float(p[1]), robot.x, robot.y, robot.yaw)
            out.append(Detection("Ball", conf, (int(x0), int(y0), int(x1), int(y1)), pos))
        return out
```

- [ ] **Step 4: Run the tests**

Run: `pixi run pytest tests/test_yolo_ball.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add soccer_sim/yolo_ball.py tests/test_yolo_ball.py
git commit -m "Add YoloBallDetector: Booster's sim-trained YOLOv8 with vision-identical ball ranging"
```

---

### Task 7: Test controls — shots, gamepad override, and referee with set plays

**Files:**
- Create: `soccer_sim/shots.py`, `soccer_sim/joystick.py`, `soccer_sim/referee.py`
- Test: `tests/test_controls.py`

**Interfaces:**
- Consumes: `config.FieldDims`, `config.SHOT_SLIDERS` (Task 2).
- Produces:
  - `shots.SHOT_FIELDS = ("x", "y", "speed", "direction_deg", "elevation_deg")`, `shots.MAX_SHOT_SPEED = 15.0`, `shots.MAX_ELEVATION_DEG = 80.0`
  - `shots.Shot(x, y, speed, direction_deg, elevation_deg)` (frozen) with `Shot.from_panel(values: dict[str, float]) -> Shot` (keys = slider names)
  - `shots.shot_velocity(speed, direction_deg, elevation_deg) -> tuple[float, float, float]` (field frame, m/s)
  - `shots.shot_from_dict(d: dict) -> Shot` — validates; raises `KeyError` (missing field) or `ValueError`
  - `shots.ShotPanel()` with `.read(values: dict[str, float]) -> Shot | None` — fires once when `shot_fire` crosses above 0.5, re-arms below 0.5
  - `shots.ShotTracker(fd, goal_height, timeout_s=8.0, rest_speed=0.05, grace_s=0.5)` with `.start(now)`, `.update(now, ball_pos, ball_vel) -> "goal" | "no_goal" | None`, attribute `tally: dict[str, int]`
  - `joystick.DEADZONE = 0.1`, `joystick.ManualLimits(vx=1.0, vy=0.5, vyaw=1.0)`, `joystick.manual_velocity(lx, ly, rx, ry, limits=ManualLimits()) -> tuple[float, float, float] | None`
  - `joystick.FirmwareRemote()` with `.update(lt: bool, hat_u: bool) -> bool` — True once per LT+Up press (the firmware "stand up into soccer mode" combo used by the boot sequence)
  - `referee.STATES`, `referee.SET_PLAYS` (name → int), `referee.SET_PLAY_SECS = 30.0`, `referee.name_of(table, value) -> str`
  - `referee.parse_schedule(text) -> list[tuple[int, float]]`, `referee.state_at(schedule, elapsed) -> int`
  - `referee.RefereeSnapshot(state, set_play, kicking_team, stopped, secondary_time)` (frozen)
  - `referee.Referee(schedule, team_id, opp_team_id, t0)` with `.override(now, state=None, set_play=None, kicking=None, stopped=None)` (names case-insensitive; `kicking` is `"us"` or `"opponent"`; all-or-nothing validation) and `.snapshot(now) -> RefereeSnapshot`

- [ ] **Step 1: Write the failing tests `tests/test_controls.py`**

```python
import math

import pytest

from soccer_sim.config import SHOT_SLIDERS, FieldDims
from soccer_sim.joystick import FirmwareRemote, ManualLimits, manual_velocity
from soccer_sim.referee import Referee, parse_schedule
from soccer_sim.shots import Shot, ShotPanel, ShotTracker, shot_from_dict, shot_velocity

FD = FieldDims()
GOAL_LINE_X = -FD.length / 2
PANEL = {name: default for name, _lo, _hi, default in SHOT_SLIDERS}


# ---- shots ---------------------------------------------------------------
def test_shot_velocity_ground_and_lofted():
    assert shot_velocity(4.0, 180.0, 0.0) == pytest.approx((-4.0, 0.0, 0.0), abs=1e-9)
    assert shot_velocity(5.0, 90.0, 30.0) == pytest.approx((0.0, 5 * math.cos(math.radians(30)), 2.5), abs=1e-9)


def test_shot_from_dict_accepts_valid_input():
    shot = shot_from_dict({"x": -6, "y": 1, "speed": 5, "direction_deg": 170, "elevation_deg": 10})
    assert shot == Shot(-6.0, 1.0, 5.0, 170.0, 10.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"x": -6, "y": 1, "speed": 5, "direction_deg": 170},  # missing field
        {"x": float("nan"), "y": 1, "speed": 5, "direction_deg": 170, "elevation_deg": 0},
        {"x": -6, "y": 1, "speed": 99, "direction_deg": 170, "elevation_deg": 0},
        {"x": -6, "y": 1, "speed": 5, "direction_deg": 170, "elevation_deg": -5},
        {"x": -6, "y": 1, "speed": 5, "direction_deg": 170, "elevation_deg": 0, "spin": 3},
    ],
)
def test_shot_from_dict_rejects_bad_input(bad):
    with pytest.raises((KeyError, ValueError)):
        shot_from_dict(bad)


def test_panel_fires_once_per_drag_and_rearms():
    panel = ShotPanel()
    assert panel.read(PANEL) is None
    assert panel.read({**PANEL, "shot_fire": 0.9}) == Shot(-6.0, 0.0, 4.0, 180.0, 0.0)
    assert panel.read({**PANEL, "shot_fire": 1.0}) is None  # still held: no repeat
    assert panel.read({**PANEL, "shot_fire": 0.1}) is None  # released: re-armed
    assert panel.read({**PANEL, "shot_fire": 0.6}) is not None


def test_tracker_reports_goal_between_posts():
    t = ShotTracker(FD, goal_height=1.2)
    t.start(0.0)
    assert t.update(0.2, (-9.0, 0.0, 0.1), (-5.0, 0.0, 0.0)) is None
    assert t.update(0.6, (GOAL_LINE_X - 0.05, 0.5, 0.1), (-5.0, 0.0, 0.0)) == "goal"
    assert t.tally == {"goal": 1, "no_goal": 0}
    assert t.update(0.7, (GOAL_LINE_X - 0.2, 0.5, 0.1), (-5.0, 0.0, 0.0)) is None  # one result per shot


def test_tracker_wide_shot_and_stopped_ball_are_no_goal():
    t = ShotTracker(FD, goal_height=1.2)
    t.start(0.0)
    assert t.update(0.6, (GOAL_LINE_X - 0.05, 3.0, 0.1), (-5.0, 0.0, 0.0)) == "no_goal"  # wide of the post
    t.start(1.0)
    assert t.update(1.2, (-8.0, 0.0, 0.1), (0.0, 0.0, 0.0)) is None  # grace period right after launch
    assert t.update(1.6, (-8.0, 0.0, 0.1), (0.0, 0.0, 0.0)) == "no_goal"  # stopped: saved or died
    assert t.tally == {"goal": 0, "no_goal": 2}


def test_tracker_times_out():
    t = ShotTracker(FD, goal_height=1.2, timeout_s=8.0)
    t.start(0.0)
    assert t.update(8.1, (-8.0, 0.0, 0.1), (0.3, 0.0, 0.0)) == "no_goal"


def test_over_the_bar_is_no_goal():
    t = ShotTracker(FD, goal_height=1.2)
    t.start(0.0)
    assert t.update(0.6, (GOAL_LINE_X - 0.05, 0.0, 1.5), (-5.0, 0.0, 1.0)) == "no_goal"


# ---- gamepad -------------------------------------------------------------
def test_centered_sticks_leave_brain_in_control():
    assert manual_velocity(0.05, -0.08, 0.02, 0.0) is None


def test_stick_mapping_matches_booster_remote_axes():
    lim = ManualLimits()
    # RemoteControllerState: ly = -1 is "push front", lx/rx = -1 is "push left".
    assert manual_velocity(0.0, -1.0, 0.0, 0.0) == pytest.approx((lim.vx, 0.0, 0.0))
    assert manual_velocity(-1.0, 0.0, 0.0, 0.0) == pytest.approx((0.0, lim.vy, 0.0))
    assert manual_velocity(0.0, 0.0, -1.0, 0.0) == pytest.approx((0.0, 0.0, lim.vyaw))


def test_right_stick_vertical_alone_holds_the_robot_still():
    assert manual_velocity(0.0, 0.0, 0.0, 0.8) == (0.0, 0.0, 0.0)


def test_lt_up_fires_once_per_press():
    remote = FirmwareRemote()
    assert remote.update(lt=True, hat_u=True) is True
    assert remote.update(lt=True, hat_u=True) is False  # still held
    assert remote.update(lt=True, hat_u=False) is False  # released
    assert remote.update(lt=True, hat_u=True) is True


def test_up_without_lt_is_not_the_stand_up_combo():
    assert FirmwareRemote().update(lt=False, hat_u=True) is False


# ---- referee -------------------------------------------------------------
def test_schedule_runs_states_in_order_then_holds_last():
    ref = Referee(parse_schedule("0:3,1:20,2:3,3:0"), 131, 1, t0=100.0)
    states = [ref.snapshot(100.0 + t).state for t in (0, 2.9, 3.1, 22.9, 23.1, 26.1, 999)]
    assert states == [0, 0, 1, 1, 2, 3, 3]


def test_override_holds_state_over_the_schedule():
    ref = Referee(parse_schedule("0:3,1:20,2:3,3:0"), 131, 1, t0=0.0)
    ref.override(1.0, state="PLAYING")
    assert ref.snapshot(1.5).state == 3
    assert ref.snapshot(50.0).state == 3


def test_opponent_corner_kick_counts_down_then_clears():
    ref = Referee(parse_schedule("3:0"), 131, 1, t0=0.0)
    ref.override(10.0, set_play="CORNER_KICK", kicking="opponent", stopped=True)
    snap = ref.snapshot(10.0)
    assert (snap.set_play, snap.kicking_team, snap.stopped, snap.secondary_time) == (6, 1, True, 30)
    assert ref.snapshot(25.2).secondary_time == 15
    cleared = ref.snapshot(40.5)
    assert (cleared.set_play, cleared.stopped, cleared.secondary_time) == (0, False, 0)


def test_our_free_kick_uses_our_team_number():
    ref = Referee(parse_schedule("3:0"), 131, 1, t0=0.0)
    ref.override(0.0, set_play="direct_free_kick", kicking="us")
    assert ref.snapshot(1.0).kicking_team == 131


@pytest.mark.parametrize(
    "kwargs",
    [
        {"state": "HALFTIME"},
        {"set_play": "OFFSIDE"},
        {"kicking": "them"},
        {"stopped": "yes"},
        {"state": "PLAYING", "set_play": "NOPE"},
    ],
)
def test_invalid_override_changes_nothing(kwargs):
    ref = Referee(parse_schedule("0:3,1:20"), 131, 1, t0=0.0)
    before = ref.snapshot(1.0)
    with pytest.raises(ValueError):
        ref.override(1.0, **kwargs)
    assert ref.snapshot(1.0) == before
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `pixi run pytest tests/test_controls.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'soccer_sim.joystick'`.

- [ ] **Step 3: Write `soccer_sim/shots.py`**

```python
"""Shot launcher pieces: slider panel reader, validated scripted shots, and goal tracking."""

import math
from dataclasses import dataclass

from .config import FieldDims

SHOT_FIELDS = ("x", "y", "speed", "direction_deg", "elevation_deg")
MAX_SHOT_SPEED = 15.0
MAX_ELEVATION_DEG = 80.0


@dataclass(frozen=True)
class Shot:
    x: float
    y: float
    speed: float
    direction_deg: float  # field frame; 180 = toward our goal (-x)
    elevation_deg: float

    @staticmethod
    def from_panel(values: dict[str, float]) -> "Shot":
        return Shot(
            values["shot_origin_x"],
            values["shot_origin_y"],
            values["shot_speed"],
            values["shot_direction_deg"],
            values["shot_elevation_deg"],
        )


def shot_velocity(speed: float, direction_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    d, e = math.radians(direction_deg), math.radians(elevation_deg)
    return speed * math.cos(e) * math.cos(d), speed * math.cos(e) * math.sin(d), speed * math.sin(e)


def shot_from_dict(d: dict) -> Shot:
    unknown = set(d) - set(SHOT_FIELDS)
    if unknown:
        raise ValueError(f"unknown shot fields: {sorted(unknown)}")
    values = {}
    for key in SHOT_FIELDS:
        v = float(d[key])
        if not math.isfinite(v):
            raise ValueError(f"{key} must be finite")
        values[key] = v
    if not 0.0 <= values["speed"] <= MAX_SHOT_SPEED:
        raise ValueError(f"speed must be in [0, {MAX_SHOT_SPEED}] m/s")
    if not 0.0 <= values["elevation_deg"] <= MAX_ELEVATION_DEG:
        raise ValueError(f"elevation_deg must be in [0, {MAX_ELEVATION_DEG}]")
    return Shot(**values)


class ShotPanel:
    """Edge-triggered reader of the viewer sliders: one shot per drag of shot_fire past 0.5."""

    def __init__(self):
        self._armed = True

    def read(self, values: dict[str, float]) -> Shot | None:
        firing = values["shot_fire"] > 0.5
        if not firing:
            self._armed = True
            return None
        if self._armed:
            self._armed = False
            return Shot.from_panel(values)
        return None


class ShotTracker:
    """Classifies the latest shot against our goal (-x): 'goal' or 'no_goal'."""

    def __init__(self, fd: FieldDims, goal_height: float, timeout_s: float = 8.0, rest_speed: float = 0.05,
                 grace_s: float = 0.5):
        self.fd = fd
        self.goal_height = goal_height
        self.timeout_s = timeout_s
        self.rest_speed = rest_speed
        self.grace_s = grace_s
        self.tally = {"goal": 0, "no_goal": 0}
        self._started: float | None = None

    def start(self, now: float) -> None:
        self._started = now

    def update(self, now: float, ball_pos, ball_vel) -> str | None:
        if self._started is None:
            return None
        x, y, z = (float(v) for v in ball_pos)
        if x < -self.fd.length / 2:
            inside = abs(y) < self.fd.goal_width / 2 and z < self.goal_height
            return self._finish("goal" if inside else "no_goal")
        elapsed = now - self._started
        resting = math.hypot(float(ball_vel[0]), float(ball_vel[1])) < self.rest_speed
        if elapsed > self.timeout_s or (elapsed > self.grace_s and resting):
            return self._finish("no_goal")
        return None

    def _finish(self, outcome: str) -> str:
        self._started = None
        self.tally[outcome] += 1
        return outcome
```

- [ ] **Step 4: Write `soccer_sim/joystick.py`**

```python
"""Booster-remote sticks -> velocity, the way the robot's firmware lets the remote override brain."""

from dataclasses import dataclass

DEADZONE = 0.1  # brain's go_manual threshold in Brain::joystickCallback


@dataclass(frozen=True)
class ManualLimits:
    # ponytail: comfortable driving speeds; the real remote's scaling lives in Booster firmware.
    vx: float = 1.0
    vy: float = 0.5
    vyaw: float = 1.0


def manual_velocity(lx: float, ly: float, rx: float, ry: float,
                    limits: ManualLimits = ManualLimits()) -> tuple[float, float, float] | None:
    """None = sticks centred (brain keeps control). Axes follow RemoteControllerState:
    ly = -1 is push front, lx/rx = -1 is push left."""
    if max(abs(lx), abs(ly), abs(rx), abs(ry)) <= DEADZONE:
        return None
    return -ly * limits.vx, -lx * limits.vy, -rx * limits.vyaw


class FirmwareRemote:
    """Robot-side remote combo used by the boot sequence: LT+Up stands the T2 up into soccer mode.

    ponytail: assumed meaning of LT+Up from the team's match procedure; Booster's firmware remote
    map is not public. brain also receives the press (it nudges vxFactor), exactly as on the robot.
    """

    def __init__(self):
        self._was_pressed = False

    def update(self, lt: bool, hat_u: bool) -> bool:
        pressed = lt and hat_u
        edge = pressed and not self._was_pressed
        self._was_pressed = pressed
        return edge
```

- [ ] **Step 5: Write `soccer_sim/referee.py`**

```python
"""GameController stand-in: a timed schedule that the controller tab can override, incl. set plays."""

import math
from dataclasses import dataclass

STATES = {"INITIAL": 0, "READY": 1, "SET": 2, "PLAYING": 3, "FINISHED": 4}
SET_PLAYS = {
    "NONE": 0,
    "DIRECT_FREE_KICK": 1,
    "INDIRECT_FREE_KICK": 2,
    "PENALTY_KICK": 3,
    "THROW_IN": 4,
    "GOAL_KICK": 5,
    "CORNER_KICK": 6,
}
SET_PLAY_SECS = 30.0


def name_of(table: dict[str, int], value: int) -> str:
    return next(name for name, v in table.items() if v == value)


def parse_schedule(text: str) -> list[tuple[int, float]]:
    """"0:3,1:20,2:3,3:0" -> [(0, 3.0), (1, 20.0), (2, 3.0), (3, 0.0)]; the last entry is held."""
    out = []
    for item in text.split(","):
        state_text, secs_text = item.split(":")
        state, secs = int(state_text), float(secs_text)
        if state not in STATES.values() or secs < 0:
            raise ValueError(f"bad schedule entry {item!r}")
        out.append((state, secs))
    if not out:
        raise ValueError("empty schedule")
    return out


def state_at(schedule: list[tuple[int, float]], elapsed: float) -> int:
    t = 0.0
    for state, secs in schedule[:-1]:
        t += secs
        if elapsed < t:
            return state
    return schedule[-1][0]


@dataclass(frozen=True)
class RefereeSnapshot:
    state: int
    set_play: int
    kicking_team: int
    stopped: bool
    secondary_time: int


def _lookup(table: dict[str, int], name, what: str) -> int:
    if not isinstance(name, str) or name.upper() not in table:
        raise ValueError(f"unknown {what} {name!r}; expected one of {list(table)}")
    return table[name.upper()]


class Referee:
    def __init__(self, schedule: list[tuple[int, float]], team_id: int, opp_team_id: int, t0: float):
        self.schedule = schedule
        self.team_id = team_id
        self.opp_team_id = opp_team_id
        self.t0 = t0
        self._manual_state: int | None = None
        self.set_play = 0
        self.kicking_team = team_id
        self.stopped = False
        self._set_play_started: float | None = None

    def override(self, now: float, state=None, set_play=None, kicking=None, stopped=None) -> None:
        # Validate everything first so a bad request never half-applies.
        new_state = None if state is None else _lookup(STATES, state, "state")
        new_set_play = None if set_play is None else _lookup(SET_PLAYS, set_play, "set_play")
        if kicking is not None and kicking not in ("us", "opponent"):
            raise ValueError(f"kicking must be 'us' or 'opponent', got {kicking!r}")
        if stopped is not None and not isinstance(stopped, bool):
            raise ValueError("stopped must be a bool")

        if new_state is not None:
            self._manual_state = new_state
        if new_set_play is not None:
            self.set_play = new_set_play
            self._set_play_started = now if new_set_play != 0 else None
        if kicking is not None:
            self.kicking_team = self.team_id if kicking == "us" else self.opp_team_id
        if stopped is not None:
            self.stopped = stopped

    def snapshot(self, now: float) -> RefereeSnapshot:
        state = self._manual_state if self._manual_state is not None else state_at(self.schedule, now - self.t0)
        secondary = 0
        if self.set_play != 0 and self._set_play_started is not None:
            remaining = SET_PLAY_SECS - (now - self._set_play_started)
            if remaining <= 0.0:
                self.set_play, self._set_play_started, self.stopped = 0, None, False
            else:
                secondary = int(math.ceil(remaining))
        return RefereeSnapshot(state, self.set_play, self.kicking_team, self.stopped, secondary)
```

- [ ] **Step 6: Run the tests**

Run: `pixi run pytest tests/test_controls.py -v`
Expected: 26 passed.

- [ ] **Step 7: Commit**

```bash
git add soccer_sim/shots.py soccer_sim/joystick.py soccer_sim/referee.py tests/test_controls.py
git commit -m "Add test controls: shot launcher logic, firmware-style gamepad override, referee with set plays"
```

---

### Task 8: `SoccerController` — walk policy, kick, camera rendering, and YOLO ball in the soccer world

**Files:**
- Create: `soccer_sim/sim.py`
- Test: `tests/test_sim.py`

**Interfaces:**
- Consumes: `field.build_world/map_markings/field_lines/goalposts/OPTICAL_FROM_MUJOCO_CAM` (Task 2), `perception.*`, `odometry.OdometryTracker`, `geometry.*` (Task 3), `loco_rpc.LocoState` (Task 4), `kick_model.VisualKickModel/KickRef` (Task 5), `yolo_ball.YoloBallDetector` (Task 6), `shots.Shot/shot_velocity` (Task 7), `config.SHOT_SLIDERS` (Task 2), `booster_deploy.controllers.base_controller.BaseController`, `tasks.locomotion.robots.t2.T2WalkTaskCfg`.
- Produces `sim.SoccerController(sim_cfg: SimConfig)` with:
  - attributes `mj_model`, `mj_data`, `cfg` (booster_deploy `ControllerCfg`), `sim_cfg`, `odometry`, `kick_model`, `is_running`
  - `start()` (inherited), `step()` — one 20 ms policy tick
  - `apply(loco: LocoState, now: float, kick_ref: KickRef | None = None, manual: tuple[float, float, float] | None = None) -> None` — sets the policy command (priority: `manual` gamepad velocity, then the kick model while `loco.kicking()`, then `loco`'s Move), head targets, launches the ball on a strike, services `getup_requested` (clears it)
  - `shot_panel_values() -> dict[str, float]` (current slider values, keyed by `SHOT_SLIDERS` names; initialised to their defaults) and `shoot(shot: Shot) -> None` (teleport the ball to the shot origin and launch it)
  - `reset_robot(x, y, yaw)`, `reset_ball(x, y)`, `fallen() -> bool`
  - `robot_pose() -> Pose2D`, `ball_position() -> np.ndarray` (3,), `ball_velocity() -> np.ndarray` (3,)
  - `odometer() -> tuple[float, float, float]`
  - `imu() -> tuple[rpy, gyro, acc]` (each a 3-tuple of floats)
  - `joint_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]` (q, dq, tau), each (31,) in `T2_31DOF_CFG.joint_names` order
  - `head_pose_in_base() -> tuple[np.ndarray, np.ndarray]` (position (3,), quaternion wxyz (4,)) — `head_pitch_link` in the ground base frame
  - `camera() -> Camera`, `perceive(rng=None, rgb=None) -> tuple[list[Detection], list[LineObs]]` — the ball comes from YOLO on `rgb` when `sim_cfg.ball_detector == "yolo"` and a frame is given, otherwise from ground truth; goalposts, markings and lines are always ground truth
  - attribute `yolo: YoloBallDetector | None` (constructed when `ball_detector == "yolo"`; a missing model file raises `FileNotFoundError` naming `pixi run fetch-models`)
  - `render() -> tuple[np.ndarray, np.ndarray]` — RGB `(448, 544, 3) uint8`, depth `(448, 544) float32` metres along the optical axis; must be called from the thread that steps the sim

- [ ] **Step 1: Write the failing tests `tests/test_sim.py`**

```python
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
```

The walking thresholds are deliberately loose (half the commanded distance): they prove the policy walks in the commanded direction in our world, not that it tracks speed precisely.

- [ ] **Step 2: Run the tests to see them fail**

Run: `pixi run pytest tests/test_sim.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'soccer_sim.sim'`.

- [ ] **Step 3: Write `soccer_sim/sim.py`**

```python
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from booster_deploy.controllers.base_controller import BaseController
from tasks.locomotion.robots.t2 import T2WalkTaskCfg

from .config import SHOT_SLIDERS, SimConfig
from .field import OPTICAL_FROM_MUJOCO_CAM, build_world, field_lines, goalposts, map_markings
from .geometry import rpy_from_quat_wxyz, world_to_robot_2d, yaw_from_quat_wxyz
from .kick_model import KickRef, VisualKickModel
from .loco_rpc import LocoState
from .odometry import OdometryTracker
from .perception import (
    Camera, Detection, LineObs, Pose2D, detect_ball, detect_goalposts, detect_markings, visible_lines,
)
from .shots import Shot, shot_velocity
from .yolo_ball import YoloBallDetector

HEAD_YAW_IDX, HEAD_PITCH_IDX = 0, 1  # T2_31DOF_CFG.joint_names order
UPRIGHT_MIN_ZZ = 0.5  # trunk z-axis . world z; same threshold as booster_deploy's safety fallback


def _make_ball_detector(cfg: SimConfig) -> YoloBallDetector | None:
    if cfg.ball_detector == "ground_truth":
        return None
    if cfg.ball_detector != "yolo":
        raise ValueError(f"ball_detector must be 'yolo' or 'ground_truth', got {cfg.ball_detector!r}")
    if not Path(cfg.yolo_model_path).exists():
        raise FileNotFoundError(f"{cfg.yolo_model_path} missing; run `pixi run fetch-models`")
    return YoloBallDetector(cfg.yolo_model_path, cfg.yolo_ball_conf, cfg.yolo_nms_iou)


class SoccerController(BaseController):
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
        self.sim_cfg = sim_cfg

        self.mj_model = build_world(sim_cfg)
        self.mj_model.opt.timestep = cfg.mujoco.physics_dt
        self.mj_data = mujoco.MjData(self.mj_model)
        self.decimation = cfg.mujoco.decimation

        nj = self.robot.num_joints
        self._nu_robot = nj  # robot motors are actuators 0..30; the shot sliders follow them
        self._q = slice(7, 7 + nj)
        self._dq = slice(6, 6 + nj)
        self._sliders = {name: self.mj_model.actuator(name).id for name, *_ in SHOT_SLIDERS}
        for name, _lo, _hi, default in SHOT_SLIDERS:
            self.mj_data.ctrl[self._sliders[name]] = default
        ball_joint = self.mj_model.joint("ball_joint")
        self._ball_q = int(ball_joint.qposadr[0])
        self._ball_dq = int(ball_joint.dofadr[0])
        self._cam = self.mj_model.camera("head_rgb").id
        self._head_body = self.mj_model.body("head_pitch_link").id
        self._trunk_body = self.mj_model.body("trunk").id
        self._gyro = int(self.mj_model.sensor("imu_gyro").adr[0])
        self._acc = int(self.mj_model.sensor("imu_acc").adr[0])

        self._kp = self.robot.joint_stiffness.numpy()
        self._kd = self.robot.joint_damping.numpy()
        self._effort = self.robot.effort_limit.numpy()
        self._head_target = (0.0, 0.0)  # (pitch, yaw)

        self._markings = map_markings(sim_cfg.field)
        self._lines = field_lines(sim_cfg.field)
        self._posts = goalposts(sim_cfg.field)

        self.kick_model = VisualKickModel()
        self._was_kicking = False
        self._renderer: mujoco.Renderer | None = None
        self.yolo = _make_ball_detector(sim_cfg)

        sx, sy, syaw = sim_cfg.spawn_xy_yaw
        self.odometry = OdometryTracker(Pose2D(sx, sy, syaw))
        self.reset_robot(sx, sy, syaw)
        self.reset_ball(*sim_cfg.ball_xy)

    # ---- resets -------------------------------------------------------------
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

    def reset_ball(self, x: float, y: float) -> None:
        d = self.mj_data
        d.qpos[self._ball_q:self._ball_q + 7] = [x, y, self.sim_cfg.ball_radius, 1.0, 0.0, 0.0, 0.0]
        d.qvel[self._ball_dq:self._ball_dq + 6] = 0.0
        mujoco.mj_forward(self.mj_model, d)

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

    # ---- inputs from brain -------------------------------------------------
    def apply(self, loco: LocoState, now: float, kick_ref: KickRef | None = None,
              manual: tuple[float, float, float] | None = None) -> None:
        if manual is not None:  # held gamepad stick: firmware-level remote beats brain and the kick
            vx, vy, vyaw = manual
            self.kick_model.reset()
        elif loco.kicking():
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

        self.vel_command.lin_vel_x = vx
        self.vel_command.lin_vel_y = vy
        self.vel_command.ang_vel_yaw = vyaw
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
        r.disable_depth_rendering()
        r.update_scene(self.mj_data, camera=self._cam)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(self.mj_data, camera=self._cam)
        depth = r.render().astype(np.float32)
        r.disable_depth_rendering()
        return rgb, depth
```

- [ ] **Step 4: Run the tests**

Run: `pixi run pytest tests/test_sim.py -v`
Expected: 16 passed (takes a minute or two). If `test_stands_still_without_falling` fails, the most likely cause is a mismatch between this controller's PD loop and `MujocoController.ctrl_step` in `vendor/booster_deploy/booster_deploy/controllers/mujoco_controller.py` — diff the two before changing anything else; first confirm the policy stands in Booster's own harness with `cd vendor/booster_deploy && mjpython scripts/deploy.py --task t2_walk --mujoco`. If only `test_visual_kick_walks_onto_ball_and_launches_it` fails, tune `KickParams` (start with `strike_tol_x`, `strike_tol_y`, `approach_gain`) and record the values that work. If `test_yolo_detects_the_rendered_ball_at_the_right_range` finds no ball, save the frame (`PIL.Image.fromarray(rgb).save("frame.png")`) and check the ball shows its checker pattern and is not cut off at the image edge; if the range is off by more than 0.3 m, check `test_camera_is_placed_with_team_calibration` first.

- [ ] **Step 5: Commit**

```bash
git add soccer_sim/sim.py tests/test_sim.py
git commit -m "Add SoccerController: T2 walk policy, kick stand-in, camera, YOLO ball, shots, gamepad override"
```

---

### Task 9: `soccer_sim_bridge` ROS 2 node and controller tab

**Files:**
- Create: `ros_ws/src/soccer_sim_bridge/package.xml`, `setup.py`, `setup.cfg`, `resource/soccer_sim_bridge`, `soccer_sim_bridge/__init__.py`, `soccer_sim_bridge/msg_conv.py`, `soccer_sim_bridge/web.py`, `soccer_sim_bridge/controller.html`, `soccer_sim_bridge/bridge_node.py`, `launch/brain.launch.py`
- Create: `scripts/run_sim.sh`
- Modify: `pixi.toml` (`[tasks]` — add `build-sim` and `test-ros`)
- Test: `ros_ws/src/soccer_sim_bridge/test/test_msg_conv.py`, `test/test_web.py`, `test/test_bridge.py`

**Interfaces:**
- Consumes: `soccer_sim.sim.SoccerController` (Task 8); `soccer_sim.config.SimConfig/CameraCfg`, `soccer_sim.field.ROBOT_ACTUATOR_GROUP` (Task 2); `soccer_sim.loco_rpc.LocoState/handle_request` (Task 4); `soccer_sim.kick_model.KickRef` (Task 5); `soccer_sim.perception.Detection/LineObs` (Task 3); `soccer_sim.shots.*`, `soccer_sim.joystick.manual_velocity/FirmwareRemote`, `soccer_sim.referee.*` (Task 7); ROS message packages built in Task 1 (`booster_interface`, `booster_msgs`, `vision_interface`, `game_controller_interface`, `brain` for `brain/msg/Kick`).
- Produces:
  - executable `ros2 run soccer_sim_bridge sim_bridge` and module entry `python -m soccer_sim_bridge.bridge_node [--viewer]`
  - class `bridge_node.SimBridge(Node)` (`SimBridge(**node_kwargs)` forwards to `Node`, e.g. `parameter_overrides`) with `.tick()`, `.status() -> dict`, `.on_web_joystick(dict)`, `.on_web_referee(dict)`, `.on_web_shoot(dict)`, and attributes `sim`, `loco`, `lock`, `kick_ref`, `referee`, `shot_tracker`, `web`
  - ROS parameters: `spawn_x, spawn_y, spawn_yaw, ball_x, ball_y` (floats), `ball_detector` (`"yolo"` | `"ground_truth"`), `team_id` (131), `opp_team_id` (1), `goalkeeper_id` (1), `referee_schedule` (`"0:3,1:20,2:3,3:0"`), `camera_every_n_ticks` (2 = 25 Hz; each camera tick renders RGB + depth, runs YOLO, and publishes images + detections + lines with one stamp), `web_port` (8765; 0 disables the controller tab), `joystick_timeout_s` (0.5), `boot_sequence` (false; true = robot boots in prepare mode and needs LT+Up on the gamepad to enter soccer mode, like powering on the real T2)
  - topics in: `LocoApiTopicReq`, `/kick_ball`, `/sim/shoot`; topics out: `LocoApiTopicResp`, `/odometer_state`, `/low_state`, `/head_pose_stamped`, `/booster_vision/detection`, `/booster_vision/line_segments`, `/boostercamera/head/{rgb,depth,rgb/camera_info}`, `/robocup/game_controller`, `/remote_controller_state`, `/sim/shot_result`
  - `web.ControlServer(port, on_joystick, on_referee, on_shoot, status, host="127.0.0.1")` with `.start()`, `.stop()`, attribute `port` (actual port; pass 0 for an ephemeral one); routes `GET /`, `GET /api/status`, `POST /api/joystick`, `POST /api/referee`, `POST /api/shoot`
  - launch file `brain.launch.py` (args `tree`, default `game.xml`; `role`, default empty = config value)

- [ ] **Step 1: Write the package metadata**

`ros_ws/src/soccer_sim_bridge/package.xml`:

```xml
<?xml version="1.0"?>
<package format="3">
  <name>soccer_sim_bridge</name>
  <version>0.1.0</version>
  <description>Impersonates the Booster T2 SDK, head camera, and vision topics for brain_node using a MuJoCo simulation.</description>
  <maintainer email="haowentan5@gmail.com">Johnny</maintainer>
  <license>Apache-2.0</license>
  <exec_depend>rclpy</exec_depend>
  <exec_depend>launch</exec_depend>
  <exec_depend>launch_ros</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>booster_interface</exec_depend>
  <exec_depend>booster_msgs</exec_depend>
  <exec_depend>vision_interface</exec_depend>
  <exec_depend>game_controller_interface</exec_depend>
  <exec_depend>brain</exec_depend>
  <test_depend>python3-pytest</test_depend>
  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

`ros_ws/src/soccer_sim_bridge/setup.py`:

```python
from glob import glob

from setuptools import setup

setup(
    name="soccer_sim_bridge",
    version="0.1.0",
    packages=["soccer_sim_bridge"],
    package_data={"soccer_sim_bridge": ["controller.html"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/soccer_sim_bridge"]),
        ("share/soccer_sim_bridge", ["package.xml"]),
        ("share/soccer_sim_bridge/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    entry_points={"console_scripts": ["sim_bridge = soccer_sim_bridge.bridge_node:main"]},
)
```

`ros_ws/src/soccer_sim_bridge/setup.cfg`:

```ini
[develop]
script_dir=$base/lib/soccer_sim_bridge
[install]
install_scripts=$base/lib/soccer_sim_bridge
```

Create empty files `ros_ws/src/soccer_sim_bridge/resource/soccer_sim_bridge` and `ros_ws/src/soccer_sim_bridge/soccer_sim_bridge/__init__.py`.

- [ ] **Step 2: Add pixi tasks for the bridge build and ROS tests**

In `pixi.toml` `[tasks]`, add:

```toml
build-sim = { cmd = "bash scripts/setup_ws.sh", env = { PACKAGES = "brain soccer_sim_bridge" } }
test-ros = "bash -c 'source ros_ws/install/setup.bash && pytest -m \"ros and not slow\" ros_ws/src/soccer_sim_bridge/test -p no:cacheprovider'"
```

- [ ] **Step 3: Write the failing conversion tests `test/test_msg_conv.py`**

```python
import numpy as np
import pytest
from brain.msg import Kick

from soccer_sim.config import CameraCfg
from soccer_sim.perception import Detection, LineObs
from soccer_sim.referee import RefereeSnapshot
from soccer_sim_bridge import msg_conv

pytestmark = pytest.mark.ros
STAMP = msg_conv.stamp_from_ns(1_500_000_000)


def test_detections_fill_the_fields_brain_reads():
    det = Detection("Ball", 0.9, (10, 20, 30, 40), (2.0, -0.5))
    mark = Detection("PenaltyPoint", 0.9, (1, 2, 3, 4), (3.0, 0.0), target_uv=(272.0, 300.0))
    msg = msg_conv.detections(STAMP, [det, mark])
    ball, pen = msg.detected_objects
    assert ball.label == "Ball" and ball.confidence == pytest.approx(0.9)
    assert (ball.xmin, ball.ymin, ball.xmax, ball.ymax) == (10, 20, 30, 40)
    assert list(ball.position_projection) == pytest.approx([2.0, -0.5])
    assert list(ball.target_uv) == []
    assert list(pen.target_uv) == pytest.approx([272.0, 300.0])
    assert msg.header.stamp.sec == 1 and msg.header.stamp.nanosec == 500_000_000


def test_line_segments_are_flat_and_paired():
    lines = [LineObs((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)), LineObs((0.0, 0.0, 1.0, 0.0), (1.0, 1.0, 2.0, 2.0))]
    msg = msg_conv.line_segments(STAMP, lines)
    assert list(msg.coordinates) == pytest.approx([1, 2, 3, 4, 0, 0, 1, 0])
    assert list(msg.coordinates_uv) == pytest.approx([5, 6, 7, 8, 1, 1, 2, 2])


def test_low_state_puts_head_yaw_and_pitch_first():
    q = np.arange(31, dtype=float) * 0.01
    msg = msg_conv.low_state((0.1, 0.2, 0.3), (0.0, 0.0, 0.5), (0.0, 0.0, 9.81), q, np.zeros(31), np.zeros(31))
    assert len(msg.motor_state_serial) == 31
    assert msg.motor_state_serial[0].q == pytest.approx(0.0)
    assert msg.motor_state_serial[1].q == pytest.approx(0.01)
    assert list(msg.imu_state.rpy) == pytest.approx([0.1, 0.2, 0.3])


def test_game_control_carries_state_set_play_and_teams():
    ref = RefereeSnapshot(state=3, set_play=6, kicking_team=1, stopped=True, secondary_time=25)
    msg = msg_conv.game_control(ref, team_id=131, opp_team_id=1, goalkeeper_id=1)
    assert (msg.state, msg.set_play, msg.kicking_team, msg.stopped, msg.secondary_time) == (3, 6, 1, True, 25)
    assert (msg.teams[0].team_number, msg.teams[0].goalkeeper) == (131, 1)
    assert msg.teams[1].team_number == 1
    assert len(msg.teams[0].players) == 20


def test_camera_info_uses_team_intrinsics():
    cfg = CameraCfg()
    msg = msg_conv.camera_info(STAMP, cfg)
    assert (msg.width, msg.height) == (544, 448)
    assert list(msg.k) == pytest.approx([cfg.fx, 0.0, cfg.cx, 0.0, cfg.fy, cfg.cy, 0.0, 0.0, 1.0])


def test_depth_image_is_millimetres_with_invalid_as_zero():
    depth = np.array([[2.5, np.inf], [70.0, np.nan]], dtype=np.float32)
    msg = msg_conv.depth_image(STAMP, depth, max_depth_m=10.0)
    assert msg.encoding == "16UC1" and (msg.height, msg.width) == (2, 2)
    assert msg.step == 4 and msg.is_bigendian == 0
    values = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(2, 2)
    assert values.tolist() == [[2500, 0], [0, 0]]


def test_rgb_image_layout():
    rgb = np.zeros((448, 544, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)
    msg = msg_conv.rgb_image(STAMP, rgb)
    assert msg.encoding == "rgb8" and msg.step == 544 * 3
    assert bytes(msg.data[:3]) == bytes([255, 0, 0])


def test_kick_ref_from_brain_kick_msg():
    msg = Kick(x=0.4, y=-0.1, dir=0.2, power=3.5)
    ref = msg_conv.kick_ref(msg, now=12.0)
    assert (ref.ball_x, ref.ball_y, ref.dir, ref.power, ref.stamp) == pytest.approx((0.4, -0.1, 0.2, 3.5, 12.0))


def test_joystick_from_dict_clamps_axes_and_sets_hat_centre():
    msg = msg_conv.joystick_from_dict({"lx": 3.0, "ly": -0.4, "lt": True, "b": True, "ignored_key": 1})
    assert (msg.lx, msg.ly, msg.rx, msg.ry) == pytest.approx((1.0, -0.4, 0.0, 0.0))
    assert msg.lt and msg.b and not msg.x
    assert msg.hat_c
    assert not msg_conv.joystick_from_dict({"hat_u": True}).hat_c


@pytest.mark.parametrize("bad", [{"b": 1}, {"lt": "true"}, {"lx": float("nan")}, {"lx": "left"}])
def test_joystick_from_dict_rejects_non_bool_buttons(bad):
    with pytest.raises(ValueError):
        msg_conv.joystick_from_dict(bad)
```

- [ ] **Step 4: Write `soccer_sim_bridge/msg_conv.py`**

```python
"""Pure builders between soccer_sim values and the ROS messages brain_node uses."""

import math

import numpy as np
from booster_interface.msg import ImuState, LowState, MotorState, Odometer, RemoteControllerState
from builtin_interfaces.msg import Time
from game_controller_interface.msg import GameControlData
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from vision_interface.msg import DetectedObject, Detections, LineSegments

from soccer_sim.config import CameraCfg
from soccer_sim.kick_model import KickRef
from soccer_sim.perception import Detection, LineObs
from soccer_sim.referee import RefereeSnapshot

CAMERA_FRAME = "head_camera"
BASE_FRAME = "base"
JOY_AXES = ("lx", "ly", "rx", "ry")
JOY_BUTTONS = ("a", "b", "x", "y", "lb", "rb", "lt", "rt", "ls", "rs", "back", "start",
               "hat_u", "hat_d", "hat_l", "hat_r")


def stamp_from_ns(ns: int) -> Time:
    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


def _header(stamp: Time, frame_id: str) -> Header:
    return Header(stamp=stamp, frame_id=frame_id)


def odometer(x: float, y: float, theta: float) -> Odometer:
    return Odometer(x=float(x), y=float(y), theta=float(theta))


def low_state(rpy, gyro, acc, q, dq, tau) -> LowState:
    msg = LowState()
    msg.imu_state = ImuState(rpy=[float(v) for v in rpy], gyro=[float(v) for v in gyro], acc=[float(v) for v in acc])
    msg.motor_state_serial = [
        MotorState(q=float(q[i]), dq=float(dq[i]), tau_est=float(tau[i])) for i in range(len(q))
    ]
    msg.motor_state_parallel = []
    return msg


def head_pose(stamp: Time, pos, quat_wxyz) -> PoseStamped:
    msg = PoseStamped()
    msg.header = _header(stamp, BASE_FRAME)
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in pos)
    w, x, y, z = (float(v) for v in quat_wxyz)
    msg.pose.orientation.w, msg.pose.orientation.x = w, x
    msg.pose.orientation.y, msg.pose.orientation.z = y, z
    return msg


def detections(stamp: Time, dets: list[Detection]) -> Detections:
    msg = Detections(header=_header(stamp, CAMERA_FRAME))
    for d in dets:
        obj = DetectedObject()
        obj.label = d.label
        obj.confidence = float(d.confidence)
        obj.xmin, obj.ymin, obj.xmax, obj.ymax = (int(v) for v in d.bbox)
        obj.position_projection = [float(d.pos_robot[0]), float(d.pos_robot[1])]
        obj.position = [float(d.pos_robot[0]), float(d.pos_robot[1]), 0.0]
        obj.target_uv = [float(v) for v in d.target_uv] if d.target_uv is not None else []
        msg.detected_objects.append(obj)
    return msg


def line_segments(stamp: Time, lines: list[LineObs]) -> LineSegments:
    msg = LineSegments(header=_header(stamp, CAMERA_FRAME))
    msg.coordinates = [float(v) for line in lines for v in line.robot]
    msg.coordinates_uv = [float(v) for line in lines for v in line.uv]
    return msg


def camera_info(stamp: Time, cfg: CameraCfg) -> CameraInfo:
    msg = CameraInfo(header=_header(stamp, CAMERA_FRAME), width=cfg.width, height=cfg.height)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5
    msg.k = [cfg.fx, 0.0, cfg.cx, 0.0, cfg.fy, cfg.cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [cfg.fx, 0.0, cfg.cx, 0.0, 0.0, cfg.fy, cfg.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def rgb_image(stamp: Time, rgb: np.ndarray) -> Image:
    h, w, _ = rgb.shape
    return Image(
        header=_header(stamp, CAMERA_FRAME), height=h, width=w, encoding="rgb8",
        is_bigendian=0, step=w * 3, data=np.ascontiguousarray(rgb, dtype=np.uint8).tobytes(),
    )


def depth_image(stamp: Time, depth_m: np.ndarray, max_depth_m: float) -> Image:
    """16UC1 millimetres (brain's vision.depth_scale_16u = 0.001); 0 marks invalid pixels."""
    d = np.where(np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= max_depth_m), depth_m, 0.0)
    mm = np.round(d * 1000.0).astype("<u2")
    h, w = mm.shape
    return Image(
        header=_header(stamp, CAMERA_FRAME), height=h, width=w, encoding="16UC1",
        is_bigendian=0, step=w * 2, data=mm.tobytes(),
    )


def game_control(ref: RefereeSnapshot, team_id: int, opp_team_id: int, goalkeeper_id: int) -> GameControlData:
    # header/version are validated by Robotedge's UDP game_controller node, not by brain; left default.
    msg = GameControlData()
    msg.players_per_team = 5
    msg.state = ref.state
    msg.set_play = ref.set_play
    msg.kicking_team = ref.kicking_team
    msg.stopped = ref.stopped
    msg.secondary_time = ref.secondary_time
    msg.first_half = True
    msg.secs_remaining = 600
    msg.teams[0].team_number = team_id
    msg.teams[0].goalkeeper = goalkeeper_id
    msg.teams[1].team_number = opp_team_id
    msg.teams[1].goalkeeper = 1
    return msg


def joystick_from_dict(d: dict) -> RemoteControllerState:
    """Controller-tab JSON -> RemoteControllerState. Axes clamped to [-1, 1]; unknown keys ignored."""
    msg = RemoteControllerState()
    for key in JOY_AXES:
        value = float(d.get(key, 0.0))
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        setattr(msg, key, max(-1.0, min(1.0, value)))
    for key in JOY_BUTTONS:
        value = d.get(key, False)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true/false")
        setattr(msg, key, value)
    msg.hat_c = not (msg.hat_u or msg.hat_d or msg.hat_l or msg.hat_r)
    return msg


def kick_ref(msg, now: float) -> KickRef:
    """brain/msg/Kick -> KickRef stamped on the bridge's clock at receipt."""
    return KickRef(ball_x=float(msg.x), ball_y=float(msg.y), dir=float(msg.dir), power=float(msg.power), stamp=now)
```

- [ ] **Step 5: Write the failing controller-tab server tests `test/test_web.py`**

```python
import json
import urllib.error
import urllib.request

import pytest

from soccer_sim_bridge.web import ControlServer

pytestmark = pytest.mark.ros


@pytest.fixture
def server():
    calls = {"joystick": [], "referee": [], "shoot": []}

    def on_referee(payload):
        if payload.get("state") == "BOGUS":
            raise ValueError("unknown state 'BOGUS'")
        calls["referee"].append(payload)

    srv = ControlServer(0, calls["joystick"].append, on_referee, calls["shoot"].append, lambda: {"state": "PLAYING"})
    srv.start()
    yield f"http://127.0.0.1:{srv.port}", calls
    srv.stop()


def post(url, body: bytes, ctype="application/json"):
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_serves_the_controller_page(server):
    base, _ = server
    with urllib.request.urlopen(base + "/", timeout=2) as r:
        assert r.status == 200 and b"Soccer Sim Controller" in r.read()


def test_status_endpoint(server):
    base, _ = server
    with urllib.request.urlopen(base + "/api/status", timeout=2) as r:
        assert json.loads(r.read()) == {"state": "PLAYING"}


def test_posts_reach_their_callbacks(server):
    base, calls = server
    assert post(base + "/api/joystick", b'{"lx": 0.5}')[0] == 200
    assert post(base + "/api/referee", b'{"state": "READY"}')[0] == 200
    assert post(base + "/api/shoot", b"{}")[0] == 200
    assert calls == {"joystick": [{"lx": 0.5}], "referee": [{"state": "READY"}], "shoot": [{}]}


def test_rejects_bad_json_wrong_type_and_oversized_bodies(server):
    base, calls = server
    assert post(base + "/api/joystick", b"{not json")[0] == 400
    assert post(base + "/api/joystick", b"[1, 2]")[0] == 400
    assert post(base + "/api/joystick", b'{"lx": 0.5}', ctype="text/plain")[0] == 415
    assert post(base + "/api/joystick", b"{" + b" " * 5000 + b"}")[0] == 413
    assert calls["joystick"] == []


def test_callback_validation_errors_become_400(server):
    base, _ = server
    status, body = post(base + "/api/referee", b'{"state": "BOGUS"}')
    assert status == 400 and "unknown state" in body["error"]


def test_unknown_paths_are_404(server):
    base, _ = server
    assert post(base + "/api/nope", b"{}")[0] == 404
```

- [ ] **Step 6: Write `soccer_sim_bridge/web.py`**

```python
"""Controller tab HTTP server: page + JSON API, stdlib only, bound to 127.0.0.1."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

MAX_BODY_BYTES = 4096
PAGE_PATH = Path(__file__).with_name("controller.html")


class ControlServer:
    def __init__(self, port: int, on_joystick: Callable[[dict], None], on_referee: Callable[[dict], None],
                 on_shoot: Callable[[dict], None], status: Callable[[], dict], host: str = "127.0.0.1"):
        routes = {"/api/joystick": on_joystick, "/api/referee": on_referee, "/api/shoot": on_shoot}
        page = PAGE_PATH.read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):  # keep the ROS console clean
                pass

            def _reply(self, code: int, body: bytes, ctype: str = "application/json") -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _error(self, code: int, message: str) -> None:
                self._reply(code, json.dumps({"error": message}).encode())

            def do_GET(self):
                if self.path == "/":
                    self._reply(200, page, "text/html; charset=utf-8")
                elif self.path == "/api/status":
                    self._reply(200, json.dumps(status()).encode())
                else:
                    self._error(404, "not found")

            def do_POST(self):
                handler = routes.get(self.path)
                if handler is None:
                    return self._error(404, "not found")
                # Requiring a JSON content type forces a CORS preflight for any cross-site request,
                # which this server never answers, so other web pages cannot drive the sim.
                if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                    return self._error(415, "Content-Type must be application/json")
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    return self._error(400, "bad Content-Length")
                if length > MAX_BODY_BYTES:
                    return self._error(413, "body too large")
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(payload, dict):
                        raise ValueError("expected a JSON object")
                    handler(payload)
                except (ValueError, TypeError, KeyError) as exc:
                    return self._error(400, str(exc))
                self._reply(200, b'{"ok": true}')

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
```

- [ ] **Step 7: Write `soccer_sim_bridge/controller.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Soccer Sim Controller</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; background: #111; color: #eee; }
  section { border: 1px solid #333; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
  h1 { font-size: 20px; } h2 { font-size: 16px; margin: 0 0 8px; }
  .sticks { display: flex; gap: 24px; flex-wrap: wrap; }
  canvas { background: #222; border-radius: 50%; touch-action: none; }
  button { margin: 3px; padding: 8px 12px; border-radius: 6px; border: 1px solid #555; background: #2a2a2a; color: #eee; }
  button.on { background: #2f6f3f; }
  label { margin-right: 12px; }
  #status { font-family: ui-monospace, monospace; white-space: pre; }
</style>
</head>
<body>
<h1>Soccer Sim Controller</h1>

<section>
  <h2>Gamepad</h2>
  <p id="pad">Drag the sticks. A physical gamepad plugged into this computer is used automatically.</p>
  <div class="sticks">
    <div><div>Left stick: walk</div><canvas id="left" width="180" height="180"></canvas></div>
    <div><div>Right stick: turn</div><canvas id="right" width="180" height="180"></canvas></div>
  </div>
  <div id="buttons"></div>
  <div>
    <button data-combo="hat_u">Stand up (LT+Up)</button>
    <button data-combo="b">Auto (LT+B)</button>
    <button data-combo="x">Cancel / manual (LT+X)</button>
    <button data-combo="a">Recalibrate (LT+A)</button>
  </div>
</section>

<section>
  <h2>Referee</h2>
  <div id="states"></div>
  <label>Set play
    <select id="setplay">
      <option>NONE</option><option>DIRECT_FREE_KICK</option><option>INDIRECT_FREE_KICK</option>
      <option>PENALTY_KICK</option><option>THROW_IN</option><option>GOAL_KICK</option><option>CORNER_KICK</option>
    </select>
  </label>
  <label><input type="radio" name="kicking" value="opponent" checked> opponent kicks</label>
  <label><input type="radio" name="kicking" value="us"> we kick</label>
  <label><input type="checkbox" id="stopped"> stopped (ball being placed)</label>
  <button id="apply">Apply set play</button>
</section>

<section>
  <h2>Shot</h2>
  <p>Set origin, speed, direction and elevation with the sliders in the MuJoCo window (top right).</p>
  <button id="shoot">Shoot with the MuJoCo slider values</button>
</section>

<section><h2>Status</h2><div id="status">connecting...</div></section>

<script>
const joy = {lx: 0, ly: 0, rx: 0, ry: 0};
const held = {};
const BUTTONS = ["a", "b", "x", "y", "lb", "rb", "lt", "rt", "back", "start", "hat_u", "hat_d", "hat_l", "hat_r"];
const LATCHED = new Set(["lt", "rt"]);  // click to hold, click again to release (a mouse can only press one button)
const PAD_BUTTONS = {0: "a", 1: "b", 2: "x", 3: "y", 4: "lb", 5: "rb", 6: "lt", 7: "rt", 8: "back", 9: "start",
                     12: "hat_u", 13: "hat_d", 14: "hat_l", 15: "hat_r"};

async function post(path, body) {
  const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  if (!r.ok) console.warn(path, r.status, await r.text());
}

function stick(id, kx, ky) {
  const c = document.getElementById(id), g = c.getContext("2d"), R = c.width / 2;
  function draw() {
    g.clearRect(0, 0, c.width, c.height);
    g.beginPath();
    g.arc(R + joy[kx] * R * 0.8, R + joy[ky] * R * 0.8, 18, 0, 2 * Math.PI);
    g.fillStyle = "#6cf";
    g.fill();
  }
  function move(e) {
    const b = c.getBoundingClientRect();
    let x = (e.clientX - b.left - R) / (R * 0.8), y = (e.clientY - b.top - R) / (R * 0.8);
    const n = Math.hypot(x, y);
    if (n > 1) { x /= n; y /= n; }
    joy[kx] = x; joy[ky] = y;  // screen-down is +y, matching RemoteControllerState (push front = -1)
    draw();
  }
  c.addEventListener("pointerdown", e => { c.setPointerCapture(e.pointerId); move(e); });
  c.addEventListener("pointermove", e => { if (c.hasPointerCapture(e.pointerId)) move(e); });
  c.addEventListener("pointerup", () => { joy[kx] = 0; joy[ky] = 0; draw(); });
  draw();
}
stick("left", "lx", "ly");
stick("right", "rx", "ry");

const box = document.getElementById("buttons");
for (const name of BUTTONS) {
  const btn = document.createElement("button");
  btn.textContent = name.toUpperCase();
  if (LATCHED.has(name)) {
    btn.onclick = () => { held[name] = !held[name]; btn.classList.toggle("on", held[name]); };
  } else {
    btn.onpointerdown = () => { held[name] = true; btn.classList.add("on"); };
    btn.onpointerup = btn.onpointerleave = () => { held[name] = false; btn.classList.remove("on"); };
  }
  box.appendChild(btn);
}
document.querySelectorAll("[data-combo]").forEach(btn => btn.onclick = () => {
  const k = btn.dataset.combo;
  held.lt = true; held[k] = true;
  setTimeout(() => { held.lt = false; held[k] = false; }, 300);
});

function gamepadState() {
  const pads = navigator.getGamepads ? [...navigator.getGamepads()] : [];
  const pad = pads.find(p => p && p.connected);
  if (!pad) return null;
  const s = {lx: pad.axes[0] || 0, ly: pad.axes[1] || 0, rx: pad.axes[2] || 0, ry: pad.axes[3] || 0};
  for (const [i, name] of Object.entries(PAD_BUTTONS)) s[name] = !!(pad.buttons[i] && pad.buttons[i].pressed);
  return s;
}

setInterval(() => {
  const state = {...joy};
  for (const name of BUTTONS) state[name] = !!held[name];
  const pad = gamepadState();
  if (pad) {
    for (const k of ["lx", "ly", "rx", "ry"]) if (Math.abs(pad[k]) > Math.abs(state[k])) state[k] = pad[k];
    for (const name of BUTTONS) state[name] = state[name] || pad[name];
    document.getElementById("pad").textContent = "Physical gamepad connected.";
  }
  post("/api/joystick", state);
}, 50);

for (const s of ["INITIAL", "READY", "SET", "PLAYING", "FINISHED"]) {
  const b = document.createElement("button");
  b.textContent = s;
  b.onclick = () => post("/api/referee", {state: s});
  document.getElementById("states").appendChild(b);
}
document.getElementById("apply").onclick = () => post("/api/referee", {
  set_play: document.getElementById("setplay").value,
  kicking: document.querySelector("input[name=kicking]:checked").value,
  stopped: document.getElementById("stopped").checked,
});
document.getElementById("shoot").onclick = () => post("/api/shoot", {});

setInterval(async () => {
  try {
    const s = await (await fetch("/api/status")).json();
    document.getElementById("status").textContent = JSON.stringify(s, null, 2);
  } catch (e) {
    document.getElementById("status").textContent = "bridge not reachable";
  }
}, 500);
</script>
</body>
</html>
```

- [ ] **Step 8: Write the failing bridge tests `test/test_bridge.py`**

```python
import json
import time
import uuid
from types import SimpleNamespace

import pytest
import rclpy
from booster_interface.msg import Odometer, RemoteControllerState
from booster_msgs.msg import RpcReqMsg, RpcRespMsg
from brain.msg import Kick
from game_controller_interface.msg import GameControlData
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String
from vision_interface.msg import Detections

from soccer_sim_bridge.bridge_node import SimBridge

pytestmark = pytest.mark.ros


def make_harness(extra_params=()):
    rclpy.init()
    bridge = SimBridge(parameter_overrides=[
        Parameter("web_port", value=0),
        Parameter("joystick_timeout_s", value=0.2),
        *extra_params,
    ])
    probe = rclpy.create_node("probe")
    got = {k: [] for k in ("resp", "odom", "det", "rgb", "depth", "joy", "gc", "result")}
    probe.create_subscription(RpcRespMsg, "LocoApiTopicResp", got["resp"].append, 10)
    probe.create_subscription(Odometer, "/odometer_state", got["odom"].append, 10)
    probe.create_subscription(Detections, "/booster_vision/detection", got["det"].append, 10)
    probe.create_subscription(Image, "/boostercamera/head/rgb", got["rgb"].append, 10)
    probe.create_subscription(Image, "/boostercamera/head/depth", got["depth"].append, 10)
    probe.create_subscription(RemoteControllerState, "/remote_controller_state", got["joy"].append, 10)
    probe.create_subscription(GameControlData, "/robocup/game_controller", got["gc"].append, 10)
    probe.create_subscription(String, "/sim/shot_result", got["result"].append, 10)
    rpc_pub = probe.create_publisher(RpcReqMsg, "LocoApiTopicReq", 10)
    ex = SingleThreadedExecutor()
    ex.add_node(bridge)
    ex.add_node(probe)
    bridge.sim.start()

    def send(api_id, body="", header=None):
        msg = RpcReqMsg(uuid=str(uuid.uuid4()), header=header or json.dumps({"api_id": api_id}), body=body)
        rpc_pub.publish(msg)
        return msg.uuid

    def ticks(n):
        for _ in range(n):
            ex.spin_once(timeout_sec=0.0)
            bridge.tick()
        ex.spin_once(timeout_sec=0.06)

    def close():
        ex.shutdown()
        bridge.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()

    ns = SimpleNamespace(
        bridge=bridge, got=got, send=send, ticks=ticks, close=close,
        kick_pub=probe.create_publisher(Kick, "/kick_ball", 10),
        shoot_pub=probe.create_publisher(Float64MultiArray, "/sim/shoot", 10),
    )
    ticks(5)  # let discovery settle
    return ns


@pytest.fixture
def h():
    ns = make_harness()
    yield ns
    ns.close()


@pytest.fixture
def booting():
    ns = make_harness([Parameter("boot_sequence", value=True)])
    yield ns
    ns.close()


def test_get_mode_response_echoes_uuid(h):
    req_id = h.send(2017)
    h.ticks(5)
    match = [r for r in h.got["resp"] if r.uuid == req_id]
    assert match and json.loads(match[0].header)["status"] == 0
    assert json.loads(match[0].body) == {"mode": 4}


def test_move_request_moves_the_odometer(h):
    for _ in range(3):
        h.send(2001, json.dumps({"vx": 0.5, "vy": 0.0, "vyaw": 0.0}))
        h.ticks(2)
    h.ticks(150)  # 3 s of sim
    assert h.got["odom"][-1].x > 0.6


def test_bridge_survives_garbage_rpc(h):
    bad = h.send(0, body="{", header="not json")
    h.ticks(5)
    match = [r for r in h.got["resp"] if r.uuid == bad]
    assert match and json.loads(match[0].header)["status"] != 0
    h.ticks(5)
    assert h.got["odom"]  # still publishing


def test_publishes_detections_and_camera_images(h):
    h.ticks(10)
    assert h.got["det"]
    assert h.got["rgb"] and h.got["rgb"][-1].encoding == "rgb8"
    assert h.got["depth"] and h.got["depth"][-1].encoding == "16UC1"


def test_visual_kick_rpc_and_kick_ball_reach_the_sim(h):
    h.send(2038, json.dumps({"start": True, "version": 1}))
    h.kick_pub.publish(Kick(x=1.0, y=0.0, dir=0.0, power=3.0))
    h.ticks(5)
    assert h.bridge.loco.kicking()
    assert h.bridge.kick_ref is not None and h.bridge.kick_ref.ball_x == pytest.approx(1.0)


def test_stale_joystick_releases_manual_drive(h):
    h.bridge.on_web_joystick({"ly": -1.0})
    h.ticks(5)
    assert h.bridge.status()["manual_drive"] is True
    assert h.got["joy"][-1].ly == pytest.approx(-1.0)
    time.sleep(0.3)  # longer than joystick_timeout_s = 0.2 set in the fixture
    h.ticks(5)
    assert h.bridge.status()["manual_drive"] is False
    assert h.got["joy"][-1].ly == 0.0


def test_controller_tab_set_play_reaches_game_controller_topic(h):
    h.bridge.on_web_referee({"state": "PLAYING", "set_play": "CORNER_KICK", "kicking": "opponent", "stopped": False})
    time.sleep(0.6)  # referee publishes at 2 Hz
    h.ticks(5)
    last = h.got["gc"][-1]
    assert (last.state, last.set_play, last.kicking_team) == (3, 6, 1)
    with pytest.raises(ValueError):
        h.bridge.on_web_referee({"state": "PLAYING", "colour": "red"})


def test_boot_sequence_waits_for_lt_up_before_walking(booting):
    h = booting
    req = h.send(2017)
    h.ticks(5)
    assert json.loads([r for r in h.got["resp"] if r.uuid == req][0].body) == {"mode": 1}  # prepare: standing
    h.send(2001, json.dumps({"vx": 0.5, "vy": 0.0, "vyaw": 0.0}))
    h.ticks(50)
    assert abs(h.got["odom"][-1].x) < 0.2  # brain's Move is ignored until the robot is stood up
    h.bridge.on_web_joystick({"lt": True, "hat_u": True})
    h.ticks(3)
    h.bridge.on_web_joystick({})
    req = h.send(2017)
    h.ticks(5)
    assert json.loads([r for r in h.got["resp"] if r.uuid == req][0].body) == {"mode": 4}  # soccer


def test_scripted_shot_scores_and_reports_goal(h):
    h.shoot_pub.publish(Float64MultiArray(data=[-9.0, 0.0, 6.0, 180.0, 0.0]))
    h.ticks(60)  # 1.2 s of sim: 2 m to the line at 6 m/s
    assert any(m.data == "goal" for m in h.got["result"])
    assert h.bridge.status()["shots"]["goal"] == 1
```

- [ ] **Step 9: Write `soccer_sim_bridge/bridge_node.py`**

```python
"""ROS 2 stand-in for the Booster T2 SDK, head camera, and vision stack, backed by soccer_sim."""

import sys
import threading
import time

import rclpy
from booster_interface.msg import LowState, Odometer, RemoteControllerState
from booster_msgs.msg import RpcReqMsg, RpcRespMsg
from brain.msg import Kick
from game_controller_interface.msg import GameControlData
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64MultiArray, String
from vision_interface.msg import Detections, LineSegments

from soccer_sim.config import SimConfig
from soccer_sim.field import ROBOT_ACTUATOR_GROUP
from soccer_sim.joystick import FirmwareRemote, manual_velocity
from soccer_sim.kick_model import KickRef
from soccer_sim.loco_rpc import MODE_DAMPING, MODE_PREPARE, MODE_SOCCER, LocoState, handle_request
from soccer_sim.referee import SET_PLAYS, STATES, Referee, name_of, parse_schedule
from soccer_sim.shots import SHOT_FIELDS, Shot, ShotPanel, ShotTracker, shot_from_dict
from soccer_sim.sim import SoccerController

from . import msg_conv
from .web import ControlServer

REFEREE_KEYS = {"state", "set_play", "kicking", "stopped"}


class SimBridge(Node):
    def __init__(self, **node_kwargs):
        super().__init__("soccer_sim_bridge", **node_kwargs)
        d = SimConfig()
        sx, sy, syaw = d.spawn_xy_yaw
        defaults = {
            "spawn_x": sx, "spawn_y": sy, "spawn_yaw": syaw,
            "ball_x": d.ball_xy[0], "ball_y": d.ball_xy[1],
            "ball_detector": d.ball_detector,
            "team_id": 131, "opp_team_id": 1, "goalkeeper_id": 1,
            "referee_schedule": "0:3,1:20,2:3,3:0",
            "camera_every_n_ticks": 2,
            "web_port": 8765,
            "joystick_timeout_s": 0.5,
            "boot_sequence": False,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

        def g(name):
            return self.get_parameter(name).value

        cfg = SimConfig(
            spawn_xy_yaw=(float(g("spawn_x")), float(g("spawn_y")), float(g("spawn_yaw"))),
            ball_xy=(float(g("ball_x")), float(g("ball_y"))),
            ball_detector=str(g("ball_detector")),
        )
        self.sim = SoccerController(cfg)
        # boot_sequence: like a freshly powered T2, stand in prepare mode until LT+Up on the remote.
        self.loco = LocoState(mode=MODE_PREPARE if bool(g("boot_sequence")) else cfg.initial_mode)
        self.firmware_remote = FirmwareRemote()
        self.kick_ref: KickRef | None = None
        self.lock = threading.Lock()
        self.team_id, self.opp_team_id = int(g("team_id")), int(g("opp_team_id"))
        self.goalkeeper_id = int(g("goalkeeper_id"))
        self.camera_every = max(1, int(g("camera_every_n_ticks")))
        self.joystick_timeout = float(g("joystick_timeout_s"))
        self.referee = Referee(
            parse_schedule(str(g("referee_schedule"))), self.team_id, self.opp_team_id, time.monotonic()
        )
        self.shot_panel = ShotPanel()
        self.shot_tracker = ShotTracker(cfg.field, cfg.goal_height)
        self.last_shot = ""
        self.tick_count = 0
        self._joy: tuple[RemoteControllerState, float] | None = None
        self._pending_shots: list[Shot] = []
        self._fire_panel = False
        self._status: dict = {}

        self.create_subscription(RpcReqMsg, "LocoApiTopicReq", self.on_rpc, 10)
        self.create_subscription(Kick, "/kick_ball", self.on_kick_ball, 10)
        self.create_subscription(Float64MultiArray, "/sim/shoot", self.on_sim_shoot, 10)
        self.resp_pub = self.create_publisher(RpcRespMsg, "LocoApiTopicResp", 10)
        self.odom_pub = self.create_publisher(Odometer, "/odometer_state", 10)
        self.low_pub = self.create_publisher(LowState, "/low_state", 10)
        self.head_pub = self.create_publisher(PoseStamped, "/head_pose_stamped", 10)
        self.det_pub = self.create_publisher(Detections, "/booster_vision/detection", 10)
        self.line_pub = self.create_publisher(LineSegments, "/booster_vision/line_segments", 10)
        self.rgb_pub = self.create_publisher(Image, "/boostercamera/head/rgb", 5)
        self.depth_pub = self.create_publisher(Image, "/boostercamera/head/depth", 5)
        self.caminfo_pub = self.create_publisher(CameraInfo, "/boostercamera/head/rgb/camera_info", 10)
        self.gc_pub = self.create_publisher(GameControlData, "/robocup/game_controller", 10)
        self.joy_pub = self.create_publisher(RemoteControllerState, "/remote_controller_state", 10)
        self.shot_result_pub = self.create_publisher(String, "/sim/shot_result", 10)
        self.create_timer(0.5, self.publish_referee)
        self.create_timer(1.0, self.publish_camera_info)
        self.create_timer(0.05, self.publish_joystick)

        self.web: ControlServer | None = None
        port = int(g("web_port"))
        if port > 0:
            self.web = ControlServer(port, self.on_web_joystick, self.on_web_referee, self.on_web_shoot, self.status)
            self.web.start()
            self.get_logger().info(f"controller tab: http://127.0.0.1:{self.web.port}/")

    # ---- inputs (executor / HTTP threads; only touch state under self.lock) ----
    def _stamp(self):
        return self.get_clock().now().to_msg()

    def _fresh_joystick(self, now: float) -> RemoteControllerState | None:
        """Caller holds self.lock."""
        if self._joy is None or now - self._joy[1] > self.joystick_timeout:
            return None
        return self._joy[0]

    def on_rpc(self, msg: RpcReqMsg) -> None:
        with self.lock:
            reply = handle_request(self.loco, msg.header, msg.body)
        self.resp_pub.publish(RpcRespMsg(uuid=msg.uuid, header=reply.header, body=reply.body))

    def on_kick_ball(self, msg: Kick) -> None:
        ref = msg_conv.kick_ref(msg, time.monotonic())
        with self.lock:
            self.kick_ref = ref

    def on_sim_shoot(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != len(SHOT_FIELDS):
            self.get_logger().warning(f"/sim/shoot needs {len(SHOT_FIELDS)} values {SHOT_FIELDS}")
            return
        try:
            shot = shot_from_dict(dict(zip(SHOT_FIELDS, msg.data)))
        except ValueError as exc:
            self.get_logger().warning(f"/sim/shoot rejected: {exc}")
            return
        with self.lock:
            self._pending_shots.append(shot)

    def on_web_joystick(self, payload: dict) -> None:
        msg = msg_conv.joystick_from_dict(payload)
        with self.lock:
            self._joy = (msg, time.monotonic())

    def on_web_referee(self, payload: dict) -> None:
        unknown = set(payload) - REFEREE_KEYS
        if unknown:
            raise ValueError(f"unknown referee fields: {sorted(unknown)}")
        with self.lock:
            self.referee.override(time.monotonic(), **payload)

    def on_web_shoot(self, payload: dict) -> None:
        shot = shot_from_dict(payload) if payload else None  # empty body = use the MuJoCo slider values
        with self.lock:
            if shot is None:
                self._fire_panel = True
            else:
                self._pending_shots.append(shot)

    def status(self) -> dict:
        with self.lock:
            return dict(self._status)

    # ---- main-thread tick: the only code that touches mj_data or the renderer ----
    def tick(self) -> None:
        now = time.monotonic()
        with self.lock:
            joy = self._fresh_joystick(now)
            stand_up = self.firmware_remote.update(joy.lt and not joy.rt, joy.hat_u) if joy is not None else \
                self.firmware_remote.update(False, False)
            if stand_up and self.loco.mode in (MODE_DAMPING, MODE_PREPARE):
                self.loco.mode = MODE_SOCCER
                self.get_logger().info("remote LT+Up: robot stood up into soccer mode")
            manual = manual_velocity(joy.lx, joy.ly, joy.rx, joy.ry) if joy is not None else None
            self.sim.apply(self.loco, now, self.kick_ref, manual)
            shots, self._pending_shots = self._pending_shots, []
            fire_panel, self._fire_panel = self._fire_panel, False

        panel = self.sim.shot_panel_values()
        panel_shot = self.shot_panel.read(panel)
        if panel_shot is not None:
            shots.append(panel_shot)
        if fire_panel:
            shots.append(Shot.from_panel(panel))
        for shot in shots:
            self.sim.shoot(shot)
            self.shot_tracker.start(now)
            self.get_logger().info(f"shot: {shot}")

        self.sim.step()

        outcome = self.shot_tracker.update(now, self.sim.ball_position(), self.sim.ball_velocity())
        if outcome is not None:
            self.last_shot = outcome
            self.shot_result_pub.publish(String(data=outcome))
            self.get_logger().info(f"shot result: {outcome}  tally={self.shot_tracker.tally}")

        stamp = self._stamp()
        self.odom_pub.publish(msg_conv.odometer(*self.sim.odometer()))
        rpy, gyro, acc = self.sim.imu()
        self.low_pub.publish(msg_conv.low_state(rpy, gyro, acc, *self.sim.joint_state()))
        self.head_pub.publish(msg_conv.head_pose(stamp, *self.sim.head_pose_in_base()))
        if self.tick_count % self.camera_every == 0:
            rgb, depth = self.sim.render()
            dets, lines = self.sim.perceive(rgb=rgb)  # ball via YOLO on this frame, the rest ground truth
            self.rgb_pub.publish(msg_conv.rgb_image(stamp, rgb))
            self.depth_pub.publish(msg_conv.depth_image(stamp, depth, self.sim.sim_cfg.max_depth_m))
            self.det_pub.publish(msg_conv.detections(stamp, dets))
            self.line_pub.publish(msg_conv.line_segments(stamp, lines))

        with self.lock:
            ref = self.referee.snapshot(now)
            self._status = {
                "state": name_of(STATES, ref.state),
                "set_play": name_of(SET_PLAYS, ref.set_play),
                "kicking": "us" if ref.kicking_team == self.team_id else "opponent",
                "stopped": ref.stopped,
                "set_play_secs_left": ref.secondary_time,
                "robot_mode": self.loco.mode,
                "visual_kick": self.loco.kicking(),
                "manual_drive": manual is not None,
                "shots": dict(self.shot_tracker.tally),
                "last_shot": self.last_shot,
            }
        self.tick_count += 1

    # ---- timers ----
    def publish_referee(self) -> None:
        with self.lock:
            ref = self.referee.snapshot(time.monotonic())
        self.gc_pub.publish(msg_conv.game_control(ref, self.team_id, self.opp_team_id, self.goalkeeper_id))

    def publish_camera_info(self) -> None:
        self.caminfo_pub.publish(msg_conv.camera_info(self._stamp(), self.sim.sim_cfg.camera))

    def publish_joystick(self) -> None:
        with self.lock:
            joy = self._fresh_joystick(time.monotonic())
        self.joy_pub.publish(joy if joy is not None else msg_conv.joystick_from_dict({}))

    def destroy_node(self):
        if self.web is not None:
            self.web.stop()
            self.web = None
        super().destroy_node()


def _run_realtime(node: SimBridge, viewer) -> None:
    period = node.sim.cfg.policy_dt
    next_t = time.monotonic()
    while rclpy.ok() and (viewer is None or viewer.is_running()):
        node.tick()
        if viewer is not None:
            viewer.cam.lookat[:] = node.sim.mj_data.qpos[0:3]
            viewer.sync()
        next_t += period
        delay = next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        elif delay < -5 * period:
            node.get_logger().warning(f"sim running {-delay:.2f}s behind real time; resyncing", throttle_duration_sec=5.0)
            next_t = time.monotonic()


def main(argv=None) -> None:
    argv = sys.argv if argv is None else argv
    use_viewer = "--viewer" in argv
    rclpy.init(args=[a for a in argv if a != "--viewer"])
    node = SimBridge()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    node.sim.start()
    try:
        if use_viewer:
            import mujoco.viewer  # needs mjpython on macOS

            with mujoco.viewer.launch_passive(
                node.sim.mj_model, node.sim.mj_data, show_left_ui=False, show_right_ui=True
            ) as viewer:
                with viewer.lock():
                    # Hide the robot motors so the top-right Control panel shows only the shot sliders.
                    viewer.opt.actuatorgroup[ROBOT_ACTUATOR_GROUP] = 0
                _run_realtime(node, viewer)
        else:
            _run_realtime(node, None)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
```

Threading model: ROS callbacks and timers run on the executor thread, HTTP handlers on the server's threads; all of them only read/write `loco`, `kick_ref`, `referee`, `_joy`, `_pending_shots`, `_fire_panel`, `_status` under `self.lock`, and they never touch `mj_data`. `tick()` runs on the main thread and is the only code that touches `mj_data`, the renderer, the shot panel and the shot tracker. Head pose and camera images in the same tick share one stamp, which satisfies `brain`'s `vision.depth_pose_max_sync_diff_ms` pairing.

- [ ] **Step 10: Write `launch/brain.launch.py`**

```python
"""Start the unmodified Robotedge brain_node against the simulator (vision package not required)."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _brain(context):
    root = os.environ.get("ROBOTEDGE_ROOT", "/Users/johnnytan5/Robotedge-5v5-RCAP2026")
    brain_dir = os.path.join(root, "src", "brain")
    overrides = {
        "tree_file_path": os.path.join(brain_dir, "behavior_trees", LaunchConfiguration("tree").perform(context)),
        "vision_config_path": os.path.join(root, "src", "vision", "config", "vision.yaml"),
        "vision_config_local_path": os.path.join(root, "src", "vision", "config", "vision_local.yaml"),
        "rerunLog.enable_file": False,
        "rerunLog.enable_tcp": False,
        "odomLog.enable": False,
        "enable_com": False,
    }
    role = LaunchConfiguration("role").perform(context)
    if role:
        overrides["game.player_role"] = role
    return [
        Node(
            package="brain",
            executable="brain_node",
            output="screen",
            parameters=[os.path.join(brain_dir, "config", "config.yaml"), overrides],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("tree", default_value="game.xml"),
            DeclareLaunchArgument("role", default_value=""),
            OpaqueFunction(function=_brain),
        ]
    )
```

These overrides mirror `$ROBOTEDGE_ROOT/src/brain/launch/launch.py` (`disable_log`, `disable_com`), minus its `get_package_share_directory('vision')` lookup, which fails because the vision package is not built. `vision_local.yaml` may not exist; brain's own launch passes a possibly-missing path the same way. Brain reads camera intrinsics and `camToHead` from the same `vision.yaml` the sim's `CameraCfg` mirrors (Task 2 test).

- [ ] **Step 11: Write `scripts/run_sim.sh`**

```bash
#!/usr/bin/env bash
# Usage: pixi run bash scripts/run_sim.sh [--viewer] [--quick] [tree:=game.xml] [role:=striker]
#   default : real match procedure — robot boots in prepare mode (press "Stand up (LT+Up)" in the
#             controller tab), stands at the sideline, referee holds INITIAL until you press READY.
#   --quick : robot already in soccer mode, scripted referee INITIAL 3 s -> READY 20 s -> SET 3 s -> PLAYING.
# Extra bridge parameters: BRIDGE_ARGS='-p ball_detector:=ground_truth' pixi run bash scripts/run_sim.sh --viewer
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/ros_ws/install/setup.bash"

PY=python
VIEWER=""
MATCH_ARGS="-p boot_sequence:=true -p referee_schedule:=0:0"
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --viewer) PY=mjpython; VIEWER="--viewer" ;;
    --quick) MATCH_ARGS="" ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

# shellcheck disable=SC2086
"$PY" -m soccer_sim_bridge.bridge_node $VIEWER --ros-args $MATCH_ARGS ${BRIDGE_ARGS:-} &
BRIDGE_PID=$!
trap 'kill $BRIDGE_PID 2>/dev/null || true' EXIT
echo "controller tab: http://127.0.0.1:8765/"
ros2 launch soccer_sim_bridge brain.launch.py "$@"
```

- [ ] **Step 12: Build and run the ROS tests**

Run:
```bash
chmod +x scripts/run_sim.sh
pixi run build-sim
pixi run test-ros
```
Expected: colcon finishes `soccer_sim_bridge`; `test_msg_conv.py` 13 passed, `test_web.py` 6 passed, `test_bridge.py` 9 passed.

- [ ] **Step 13: Commit**

```bash
git add ros_ws/src/soccer_sim_bridge scripts/run_sim.sh pixi.toml
git commit -m "Add soccer_sim_bridge: T2 SDK/camera/vision stand-in, controller tab, referee set plays, shot launcher"
```

---

### Task 10: End-to-end — `brain_node` playing in the simulator

**Files:**
- Create: `ros_ws/src/soccer_sim_bridge/test/test_brain_in_sim.py`
- Create: `README.md`

**Interfaces:**
- Consumes: everything above; `brain_node` from Task 1; `scripts/run_sim.sh`, `brain.launch.py`, controller tab (Task 9).
- Produces: a documented one-command way to run the full pipeline and an automated check that `brain` actually drives the simulated robot.

- [ ] **Step 1: Write the end-to-end test**

```python
import json
import math
import os
import signal
import subprocess
import time

import pytest
import rclpy
from booster_interface.msg import Odometer
from booster_msgs.msg import RpcReqMsg

pytestmark = [pytest.mark.ros, pytest.mark.slow]
TIMEOUT_S = 90.0


def test_brain_drives_the_simulated_robot():
    env = dict(os.environ)
    bridge = subprocess.Popen(
        ["python", "-m", "soccer_sim_bridge.bridge_node", "--ros-args",
         "-p", "spawn_x:=-6.0", "-p", "spawn_y:=-7.5", "-p", "ball_x:=-6.0", "-p", "ball_y:=-4.0",
         "-p", "web_port:=0"],  # scripted referee, already in soccer mode, no controller tab
        env=env, start_new_session=True,
    )
    brain = subprocess.Popen(
        ["ros2", "launch", "soccer_sim_bridge", "brain.launch.py", "role:=striker"],
        env=env, start_new_session=True,
    )
    rclpy.init()
    probe = rclpy.create_node("e2e_probe")
    seen = {"api_ids": set(), "nonzero_move": False, "odom": []}

    def on_req(msg):
        api_id = json.loads(msg.header).get("api_id")
        seen["api_ids"].add(api_id)
        if api_id == 2001:
            body = json.loads(msg.body)
            if abs(body["vx"]) + abs(body["vy"]) + abs(body["vyaw"]) > 1e-3:
                seen["nonzero_move"] = True

    probe.create_subscription(RpcReqMsg, "LocoApiTopicReq", on_req, 10)
    probe.create_subscription(Odometer, "/odometer_state", seen["odom"].append, 10)
    try:
        deadline = time.monotonic() + TIMEOUT_S
        moved = 0.0
        while time.monotonic() < deadline:
            rclpy.spin_once(probe, timeout_sec=0.1)
            if seen["odom"]:
                last = seen["odom"][-1]
                moved = math.hypot(last.x, last.y)
            if seen["nonzero_move"] and moved > 0.5:
                break
        assert bridge.poll() is None, "bridge exited"
        assert brain.poll() is None, "brain_node exited"
        assert 2017 in seen["api_ids"], "brain never queried the robot mode"
        assert seen["nonzero_move"], "brain never commanded a non-zero velocity"
        assert moved > 0.5, f"robot only moved {moved:.2f} m"
        print(f"visual kick requests seen: {2038 in seen['api_ids']}")
    finally:
        probe.destroy_node()
        rclpy.shutdown()
        for proc in (brain, bridge):
            os.killpg(proc.pid, signal.SIGINT)
        for proc in (brain, bridge):
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
```

- [ ] **Step 2: Run it**

Run: `pixi run bash -c 'source ros_ws/install/setup.bash && pytest -m "ros and slow" ros_ws/src/soccer_sim_bridge/test/test_brain_in_sim.py -v -s'`
Expected: 1 passed within ~90 s. While it runs, `brain_node` output should show the game-state progression (INITIAL → READY → SET → PLAYING); `game.xml` starts `brain` in `control_state=3`, so no button press is needed.

If it fails, diagnose in this order (each is a one-command check in a second terminal after `source ros_ws/install/setup.bash`):
1. `ros2 topic echo /LocoApiTopicReq --once` — no output means brain is not commanding; check brain's log for parameter or behavior-tree load errors.
2. `ros2 topic echo /booster_vision/detection --once` — empty `detected_objects` means the camera sees nothing; check the head pitch and that `test_camera_is_placed_with_team_calibration` passes.
3. `ros2 topic hz /odometer_state` — should be ~50 Hz; lower means the sim is slower than real time (the bridge logs a warning). Try `-p camera_every_n_ticks:=4` on the bridge (12.5 Hz camera).
4. Brain logs `Ignoring depth image: no synchronized T2 head pose` → head pose and depth stamps are not pairing; confirm both come from the same `tick()`.
5. `ros2 topic echo /booster_vision/detection --field detected_objects` shows goalposts/markings but never `Ball` while the ball is in view → YOLO is missing it; view `/boostercamera/head/rgb` and check the ball's checker pattern is visible. Rerun the bridge with `-p ball_detector:=ground_truth` to confirm the rest of the pipeline works without YOLO.

- [ ] **Step 3: Rehearse the real match procedure (manual check)**

Run: `pixi run bash scripts/run_sim.sh --viewer role:=goal_keeper` and open `http://127.0.0.1:8765/` in a browser tab.
Expected, in order:
1. The T2 stands at the right sideline in prepare mode; the status panel shows `robot_mode: 1` and `state: INITIAL`. `brain`'s Move commands do nothing yet.
2. Click **Stand up (LT+Up)** → bridge logs `remote LT+Up: robot stood up into soccer mode`, status `robot_mode: 4`.
3. Click **Recalibrate (LT+A)**, then **Auto (LT+B)** → `brain` logs `State => 2: RECALIBRATE`, then `State => 3: ACTION`.
4. Click **READY** in the referee panel → the robot walks onto the field to its ready position; **SET** → it holds; **PLAYING** → the keeper takes its blocking position.
5. In the MuJoCo window's top-right Control panel (only the six `shot_*` sliders are listed), set e.g. origin x −7, y 1, speed 5, direction 190, then drag `shot_fire` to the right → the ball launches at the goal and the bridge logs `shot result: goal|no_goal` with the tally (also shown in the controller tab).
6. In the referee panel choose **CORNER_KICK**, **opponent kicks**, tick **stopped**, Apply → `brain` sees an opponent set play (the free-kick behaviour from issue #1 can be observed here); untick **stopped** and Apply to move to execution; it clears itself after 30 s.
7. Drag the left virtual stick → the robot walks manually and `brain` pauses strategy (`go_manual`); release → `brain` resumes.

Then `pixi run bash scripts/run_sim.sh --viewer --quick role:=striker` for the fast path: scripted referee, robot already in soccer mode. In PLAYING the striker approaches the ball, and when `brain` chooses `auto_visual_kick` the bridge log shows the visual kick starting, the robot lines up, and the ball is launched toward the commanded direction. In a second terminal, `ros2 run rqt_image_view rqt_image_view /boostercamera/head/rgb` shows the head camera.

This is the loop for testing `goalie-intercept-lane` changes: rebuild brain (`pixi run build-sim`), rerun, and compare the shot tally over the same set of shots.

- [ ] **Step 4: Write `README.md`**

````markdown
# soccerSim

Run the unmodified Robotedge `brain_node` against a simulated Booster T2 on a RoboCup field (macOS, MuJoCo).

- Locomotion: Booster's trained T2 walking policy (`vendor/booster_deploy`, task `t2_walk`) on Booster's real T2 model (`vendor/booster_assets`), with Booster's PD gains.
- Camera: the head camera is placed with the team's calibration (`vision.yaml`); RGB and 16-bit depth are rendered on brain's camera topics.
- Perception: the ball is detected by a real YOLOv8 model on the rendered camera frames (ranged like the robot's vision node); goalposts, markings and field lines are ground truth projected through the same camera. `-p ball_detector:=ground_truth` on the bridge switches the ball to ground truth.
- Visual kick: a stand-in for Booster's closed RL visual kick, driven by brain's `kVisualKick` RPC and `/kick_ball` references.
- Interface: `soccer_sim_bridge` answers brain's loco RPC (`LocoApiTopicReq/Resp`) and publishes `/odometer_state`, `/low_state`, `/head_pose_stamped`, `/boostercamera/head/*`, `/booster_vision/*`, `/robocup/game_controller`, `/remote_controller_state`.
- Test controls: shot launcher sliders in the MuJoCo window (top right), and a browser controller tab at `http://127.0.0.1:8765/` with a virtual gamepad (any physical gamepad works too), a referee panel with set plays, a Shoot button, and live status incl. the goal/no-goal tally. Scripted shots: `ros2 topic pub --once /sim/shoot std_msgs/Float64MultiArray "{data: [-7.0, 1.0, 5.0, 190.0, 0.0]}"` (x, y, speed m/s, direction deg, elevation deg); results on `/sim/shot_result`.

## Setup

```bash
git clone --recurse-submodules <this repo> && cd soccerSim
pixi install
pixi run fetch-models       # Booster's sim-trained YOLOv8 ball detector (sha256-checked)
pixi run build-sim          # needs ~/Downloads/sdk_release.zip and $ROBOTEDGE_ROOT
pixi run test               # fast tests
pixi run test-slow          # physics + rendering tests
pixi run test-ros           # bridge tests
```

## Run

```bash
pixi run bash scripts/run_sim.sh --viewer role:=goal_keeper          # real match procedure (default)
pixi run bash scripts/run_sim.sh --viewer --quick role:=striker      # scripted fast start
```

Real match procedure (default), driven from the controller tab: the robot boots in prepare mode at the sideline → **Stand up (LT+Up)** → **Recalibrate (LT+A)** → **Auto (LT+B)** → referee **READY** → **SET** → **PLAYING**. Set plays: choose the set play, who kicks, and whether play is stopped, then Apply.

`--quick`: robot starts in soccer mode and the referee runs INITIAL 3 s → READY 20 s → SET 3 s → PLAYING by itself. Other bridge parameters go in `BRIDGE_ARGS`, e.g. `BRIDGE_ARGS='-p ball_detector:=ground_truth'`.

In the MuJoCo window's Control panel (top right) only the shot sliders are shown: `shot_origin_x/y` (ball start), `shot_speed`, `shot_direction_deg` (180 = straight at our goal), `shot_elevation_deg`; drag `shot_fire` right to shoot, back left to re-arm.

After changing brain code in the Robotedge repo: `pixi run build-sim`, then run again. If the team recalibrates the camera, `tests/test_field.py::test_camera_defaults_match_team_vision_yaml` fails until `CameraCfg` is updated.

## Known limitations

- The visual kick is an approximation (`soccer_sim/kick_model.py`, parameters in `KickParams`), not Booster's policy; swap it out if Booster publishes the T2 visual kick.
- The ball is detected by Booster's public sim-trained YOLOv8, not the team's own model (Jetson TensorRT only); its misses and range errors are realistic in kind but not identical to the robot's. Ultralytics weights are AGPL-3.0 — keep them out of anything distributed.
- Goalposts, field markings and lines are ground truth: no occlusion, no misdetections, no noise unless `detection_noise_std` is raised.
- Single robot; no opponents or teammates (shots come from the launcher); team UDP communication disabled.
- GetUp teleports the robot upright; damping/prepare modes stop the robot instead of collapsing it.
- Gamepad sticks drive the robot directly (like the firmware remote) at fixed speeds (`ManualLimits`); LT+Up is assumed to be the firmware "stand up into soccer mode" combo.
- Ball rolling friction and the kick's power→speed mapping are uncalibrated guesses.
````

- [ ] **Step 5: Commit**

```bash
git add ros_ws/src/soccer_sim_bridge/test/test_brain_in_sim.py README.md
git commit -m "Add end-to-end brain-in-sim test, match-procedure walkthrough, and README"
```
