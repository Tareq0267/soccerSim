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
        ["ros2", "launch", "soccer_sim_bridge", "brain.launch.py", "role:=striker", "player_id:=2"],
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
