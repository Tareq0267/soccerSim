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
