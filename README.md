# soccerSim

Run the unmodified Robotedge `brain_node` against a simulated Booster T2 on a RoboCup field (macOS, MuJoCo).

- Locomotion: Booster's trained T2 walking policy (`vendor/booster_deploy`, task `t2_walk`) on Booster's real T2 model (`vendor/booster_assets`), with Booster's PD gains.
- Camera: the head camera is placed with the team's calibration (`vision.yaml`); RGB and 16-bit depth are rendered on brain's camera topics.
- Perception: the ball is detected by a real YOLOv8 model on the rendered camera frames (ranged like the robot's vision node); goalposts, markings and field lines are ground truth projected through the same camera. `BRIDGE_ARGS='-p ball_detector:=ground_truth'` switches the ball to ground truth.
- Visual kick: a stand-in for Booster's closed RL visual kick, driven by brain's `kVisualKick` RPC and `/kick_ball` references.
- Interface: `soccer_sim_bridge` answers brain's loco RPC (`LocoApiTopicReq/Resp`) and publishes `/odometer_state`, `/low_state`, `/head_pose_stamped`, `/boostercamera/head/*`, `/booster_vision/*`, `/robocup/game_controller`, `/remote_controller_state`.
- Test controls: shot launcher sliders in the MuJoCo window (top right), and a browser controller tab at `http://127.0.0.1:8765/` with a virtual gamepad (any physical gamepad works too), a referee panel with set plays, a Shoot button, and live status incl. the goal/no-goal tally. Scripted shots: `ros2 topic pub --once /sim/shoot std_msgs/Float64MultiArray "{data: [-7.0, 1.0, 5.0, 190.0, 0.0]}"` (x, y, speed m/s, direction deg, elevation deg); results on `/sim/shot_result`.

## Setup

```bash
git clone --recurse-submodules <this repo> && cd soccerSim
pixi install
pixi run fetch-models       # Booster's sim-trained YOLOv8 ball detector (sha256-checked)
pixi run build-sim          # needs $SDK_ZIP (default ~/Downloads/sdk_release.zip) and $ROBOTEDGE_ROOT (default ~/Robotedge-5v5-RCAP2026); export either to override
pixi run test               # fast tests
pixi run test-slow          # physics + rendering tests
pixi run test-ros           # bridge tests
```

`build-sim` copies `$ROBOTEDGE_ROOT/src/brain` into `ros_ws/src/brain` and applies `patches/brain-clang.patch` (8 behaviour-identical lines that macOS clang needs and the robot's GCC does not); the Robotedge repo itself is never modified. If you edit one of the patched lines, the build stops and asks for the patch to be refreshed. It also builds the rerun C++ SDK 0.20.3 that brain expects into `vendor/rerun_sdk` on first run.

## Run

```bash
pixi run bash scripts/run_sim.sh --viewer role:=goal_keeper                    # real match procedure (default)
pixi run bash scripts/run_sim.sh --viewer --quick role:=striker player_id:=2   # scripted fast start
```

Real match procedure (default), driven from the controller tab: the robot boots in prepare mode at the sideline → **Stand up (LT+Up)** → **Recalibrate (LT+A)** → **Auto (LT+B)** → referee **READY** → **SET** → **PLAYING**. Set plays: choose the set play, who kicks, and whether play is stopped, then Apply.

`--quick`: robot starts in soccer mode and the referee runs INITIAL 3 s → READY 20 s → SET 3 s → PLAYING by itself. Other bridge parameters go in `BRIDGE_ARGS`, e.g. `BRIDGE_ARGS='-p ball_detector:=ground_truth'`.

**Striker needs `player_id:=2`.** brain makes player `game.initial_goalkeeper_id` (1 in config.yaml) the goalkeeper whatever `role:=` says, and the robot is player 1 by default.

In the MuJoCo window's Control panel (top right) only the shot sliders are shown: `shot_origin_x/y` (ball start), `shot_speed`, `shot_direction_deg` (180 = straight at our goal), `shot_elevation_deg`; drag `shot_fire` right to shoot, back left to re-arm.

After changing brain code in the Robotedge repo: `pixi run build-sim`, then run again. If the team recalibrates the camera, `tests/test_field.py::test_camera_defaults_match_team_vision_yaml` fails until `CameraCfg` is updated.

## Known limitations

- The visual kick is an approximation (`soccer_sim/kick_model.py`, parameters in `KickParams`), not Booster's policy; swap it out if Booster publishes the T2 visual kick.
- The ball is detected by Booster's public sim-trained YOLOv8, not the team's own model (Jetson TensorRT only); its misses and range errors are realistic in kind but not identical to the robot's. Ultralytics weights are AGPL-3.0 — keep them out of anything distributed.
- Goalposts, field markings and lines are ground truth: no occlusion, no misdetections, no noise unless `detection_noise_std` is raised.
- Single robot; no opponents or teammates (shots come from the launcher); team UDP communication disabled.
- GetUp teleports the robot upright; damping/prepare modes stop the robot instead of collapsing it.
- Gamepad sticks drive the robot directly (like the firmware remote) at fixed speeds (`ManualLimits`); LT+Up is assumed to be the firmware "stand up into soccer mode" combo.
- Camera frames render without shadows/reflections (`SimConfig.render_shadows`) to stay real time on a Mac.
- Ball rolling friction and the kick's power→speed mapping are uncalibrated guesses.
