"""Watch the Zoomlion Z01 walk on the field in the MuJoCo viewer. Plain Python, no ROS (works on Windows).

    python scripts/view_z01.py --vx 0.6 --vyaw 0.3
    python scripts/view_z01.py --head-pitch 0.5 --seconds 10

The command is clamped to the policy's trained range, and the policy barely steps below |v| ~0.33 m/s.
The viewer's shot sliders (Control panel, top right) work as in the full sim.
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soccer_sim.field import ROBOT_ACTUATOR_GROUP  # noqa: E402
from soccer_sim.loco_rpc import LocoState  # noqa: E402
from soccer_sim.shots import ShotPanel  # noqa: E402
from soccer_sim.z01 import Z01SoccerController, z01_sim_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--vyaw", type=float, default=0.0)
    ap.add_argument("--head-pitch", type=float, default=0.0)
    ap.add_argument("--head-yaw", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="close after this long (0 = until the window closes)")
    args = ap.parse_args()

    sim = Z01SoccerController(z01_sim_config(ball_detector="ground_truth", spawn_xy_yaw=(-3.0, 0.0, 0.0),
                                             ball_xy=(0.0, 0.0)))
    sim.start()
    loco = LocoState(vx=args.vx, vy=args.vy, vyaw=args.vyaw, head_pitch=args.head_pitch, head_yaw=args.head_yaw)
    panel = ShotPanel()
    with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data, show_left_ui=False) as viewer:
        viewer.opt.actuatorgroup[ROBOT_ACTUATOR_GROUP] = 0  # the Control panel then shows only the shot sliders
        t0 = time.perf_counter()
        sim_t = 0.0
        while viewer.is_running() and (args.seconds <= 0 or sim_t < args.seconds):
            sim.apply(loco, sim_t)
            shot = panel.read(sim.shot_panel_values())
            if shot is not None:
                sim.shoot(shot)
            sim.step()
            sim_t += sim.policy_dt
            viewer.cam.lookat[:] = sim.mj_data.qpos[0:3]
            viewer.sync()
            ahead = sim_t - (time.perf_counter() - t0)
            if ahead > 0:
                time.sleep(ahead)


if __name__ == "__main__":
    main()
