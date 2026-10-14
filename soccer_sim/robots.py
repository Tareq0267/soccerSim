"""What the soccer world needs to know about a robot body, independent of how it walks."""

from collections.abc import Callable
from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class RobotSpec:
    name: str
    mjcf_path: str
    root_body: str  # the free-floating trunk; its z-axis decides "fallen"
    head_body: str  # last link of the neck chain; the head camera hangs off it
    slider_joint: str  # any robot joint: the zero-gain viewer sliders need a transmission target
    # Called on the robot's MjSpec before the field is added, e.g. to create motors the MJCF lacks.
    prepare: Callable[[mujoco.MjSpec], None] | None = None


def t2_spec() -> RobotSpec:
    # Imported here so the Z01 path runs without Booster's packages on the import path.
    from booster_assets import BOOSTER_ASSETS_DIR
    from booster_deploy.robots.t2 import T2_31DOF_CFG

    return RobotSpec(
        name="t2",
        mjcf_path=T2_31DOF_CFG.mjcf_path.replace("{BOOSTER_ASSETS_DIR}", str(BOOSTER_ASSETS_DIR)),
        root_body="trunk",
        head_body="head_pitch_link",
        slider_joint="aa_head_yaw_joint",
    )
