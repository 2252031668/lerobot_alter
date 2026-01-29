# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
简单的脚本来通过遥操作控制机器人

示例:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

使用双臂SO100进行遥操作的示例:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr  # Rerun库，用于数据可视化和分析，不弹出窗口，而是启动一个可视化服务器

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    reachy2,
    so_follower,
    bi_openarm_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    reachy2_teleoperator,
    so_leader,
    bi_openarm_vr_leader,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data  # Rerun可视化工具函数


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig  # 遥操作设备配置
    robot: RobotConfig  # 机器人配置
    # 限制最大帧率
    fps: int = 60
    teleop_time_s: float | None = None  # 遥操作持续时间（秒），None表示无限
    # 在屏幕上显示所有摄像头画面
    display_data: bool = False
    # 在远程Rerun服务器上显示数据
    display_ip: str | None = None
    # 远程Rerun服务器的端口
    display_port: int | None = None
    # 是否在Rerun中显示压缩图像
    display_compressed_images: bool = False


def teleop_loop(
    teleop: Teleoperator,  # 遥操作设备实例
    robot: Robot,  # 机器人实例
    fps: int,  # 控制循环的目标帧率
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],  # 遥操作动作处理器
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],  # 机器人动作处理器
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],  # 机器人观测处理器
    display_data: bool = False,  # 是否显示数据
    duration: float | None = None,  # 最大持续时间
    display_compressed_images: bool = False,  # 是否显示压缩图像
):
    """
    这个函数连续从遥操作设备读取动作，通过可选的管道处理它们，然后发送到机器人，
    并可选择性地显示机器人的状态。循环以指定的频率运行，直到达到设定的持续时间或被手动中断。

    参数:
        teleop: 提供控制动作的遥操作设备实例。
        robot: 正在被控制的机器人实例。
        fps: 控制循环的目标帧率（每秒帧数）。
        display_data: 如果为True，则获取机器人观测并在控制台和Rerun中显示。
        display_compressed_images: 如果为True，则在发送到Rerun显示前压缩图像。
        duration: 遥操作循环的最大持续时间（秒）。如果为None，则循环无限运行。
        teleop_action_processor: 用于处理来自遥操作的原始动作的管道。
        robot_action_processor: 在发送到机器人之前处理动作的管道。
        robot_observation_processor: 用于处理来自机器人的原始观测的管道。
    """

    display_len = max(len(key) for key in robot.action_features)  # 计算用于显示的长度
    start = time.perf_counter()  # 记录开始时间

    while True:  # 主循环
        loop_start = time.perf_counter()  # 记录循环开始时间

        # 获取机器人观测
        # 目前主要用于可视化
        # teleop_action_processor 可以将 None 作为观测接收
        # 因为默认情况下它是身份处理器
        obs = robot.get_observation()  # 获取机器人当前状态（关节角度、摄像头图像等）

        # 获取遥操作动作
        raw_action = teleop.get_action()  # 从遥操作设备获取原始动作数据

        # 通过管道处理遥操作动作
        teleop_action = teleop_action_processor((raw_action, obs))

        # 通过管道处理动作以供机器人使用
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # 发送处理后的动作到机器人（robot_action_processor.to_output 应返回RobotAction）
        _ = robot.send_action(robot_action_to_send)  # 将动作发送给机器人执行

        if display_data:  # 如果需要显示数据
            # 通过管道处理机器人观测
            obs_transition = robot_observation_processor(obs)

            # 记录用于Rerun可视化的数据
            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # 显示发送到机器人的最终动作
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start  # 计算循环耗时
        precise_sleep(max(1 / fps - dt_s, 0.0))  # 精确睡眠以维持目标帧率
        loop_s = time.perf_counter() - loop_start  # 计算实际循环时间
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")  # 打印循环时间和频率
        move_cursor_up(1)  # 移动光标

        if duration is not None and time.perf_counter() - start >= duration:  # 如果达到最大持续时间
            return


@parser.wrap()  # 配置解析装饰器
def teleoperate(cfg: TeleoperateConfig):  # 主遥操作函数
    init_logging()  # 初始化日志
    logging.info(pformat(asdict(cfg)))  # 记录配置信息
    if cfg.display_data:  # 如果需要显示数据
        # 初始化Rerun可视化服务器，这会启动一个后台gRPC服务器
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)

    # 确定是否显示压缩图像
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    # 根据配置创建遥操作设备
    teleop = make_teleoperator_from_config(cfg.teleop)
    # 根据配置创建机器人
    robot = make_robot_from_config(cfg.robot)
    # 创建默认处理器（用于处理动作和观测数据）
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # 连接遥操作设备和机器人
    teleop.connect()
    robot.connect()

    try:
        # 开始遥操作主循环
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:  # 捕获Ctrl+C中断
        pass
    finally:
        if cfg.display_data:  # 如果启用了数据可视化
            rr.rerun_shutdown()  # 关闭Rerun服务器
        # 断开遥操作设备和机器人连接
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()  # 注册第三方插件
    teleoperate()  # 执行遥操作


if __name__ == "__main__":
    main()
