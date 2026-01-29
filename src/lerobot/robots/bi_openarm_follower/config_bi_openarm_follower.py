from ..config import RobotConfig
from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig


@RobotConfig.register_subclass("bi_openarm_follower")
@dataclass
class BiOpenarmFollowerConfig(RobotConfig):
    """Configuration class for Bi Openarm Follower robots."""

    left_port: str
    right_port: str
    # 摄像头
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
