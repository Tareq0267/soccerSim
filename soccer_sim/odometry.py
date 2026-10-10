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
