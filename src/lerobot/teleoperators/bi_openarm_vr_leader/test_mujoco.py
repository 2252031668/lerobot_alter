import logging
import time
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from visualizer import ArmVisualizer
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_vr_leader,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.teleoperators.bi_openarm_vr_leader import BiOpenarmVRLeaderConfig, BiOpenarmVRLeader
import sys




if __name__ == "__main__":
    fps = 30

    logging.basicConfig(
        level=logging.INFO,  # 设置为INFO级别，不显示DEBUG信息
        format='%(levelname)s:%(name)s:%(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # 强制重新配置日志
    )
    # 硬编码配置参数
    config = BiOpenarmVRLeaderConfig(
        # 如果有特定的配置参数，在此处添加
    )
    # 创建VR遥操作器实例
    teleop = BiOpenarmVRLeader(config)
    visualizer = ArmVisualizer()

    teleop.connect()

    time.sleep(3)
    try:
        # time.sleep(3)
        while True:
            loop_start = time.perf_counter()

            raw_action = teleop.get_action()
            visualizer.update_joint_angles_from_action_dict(raw_action)
            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt_s, 0.0))

    except KeyboardInterrupt:
        teleop.disconnect()

    finally:
        teleop.disconnect()