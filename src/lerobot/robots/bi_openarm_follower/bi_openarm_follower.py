import json
import logging
import signal
from functools import cached_property
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.robots.bi_openarm_follower.config_bi_openarm_follower import BiOpenarmFollowerConfig
from lerobot.motors.DM import MotorControl, Motor, DM_Motor_Type
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
import serial
import numpy as np
import time
from multiprocessing import Process, Queue, Pipe, Event
from multiprocessing.queues import Full  # 修复Queue.Full未解析问题
from typing import List
import atexit


logger = logging.getLogger(__name__)

# --------------------------- 轨迹执行进程（仅硬件控制） ---------------------------
class ArmExecutionProcess:
    """单个手臂的轨迹执行进程（仅硬件交互，无计算）"""

    def __init__(self, arm_type: str, port: str, kp: List[float], kd: List[float]):
        self.arm_type = arm_type
        self.port = port
        self.kp = kp
        self.kd = kd
        self.control_dt = 0.001

        # IPC组件
        self.action_queue = Queue(maxsize=10)  # 接收生成进程的轨迹
        self.state_pipe = (None, None)  # (parent_conn, child_conn) 状态查询管道
        self.stop_event = Event()

        # 进程对象
        self.process = None

        # 硬件资源（子进程内初始化）
        self.serial_device = None
        self.controller = None
        self.motors = None

    def _init_hardware(self):
        """子进程内初始化硬件"""
        try:
            self.serial_device = serial.Serial(self.port, 921600, timeout=0.1)
            logger.info(f"{self.arm_type} arm serial port connected: {self.port}")

            # 初始化电机
            self.motors = [
                Motor(DM_Motor_Type.DM8009, 0x01, 0x11),
                Motor(DM_Motor_Type.DM8009, 0x02, 0x12),
                Motor(DM_Motor_Type.DM4340, 0x03, 0x13),
                Motor(DM_Motor_Type.DM4340, 0x04, 0x14),
                Motor(DM_Motor_Type.DM4310, 0x05, 0x15),
                Motor(DM_Motor_Type.DM4310, 0x06, 0x16),
                Motor(DM_Motor_Type.DM4310, 0x07, 0x17),
                Motor(DM_Motor_Type.DM4310, 0x08, 0x18),
            ]

            self.controller = MotorControl(self.serial_device)
            for m in self.motors:
                self.controller.addMotor(m)
                self.controller.enable(m)
                time.sleep(0.05)

            # 移动到零位
            self._move_to_zero()
            logger.info(f"{self.arm_type} arm hardware initialized")
        except Exception as e:
            logger.error(f"{self.arm_type} arm hardware init failed: {e}")
            raise

    def _move_to_zero(self):
        """移动到零位（初始化用）"""
        init_angles = [0.0] * 8
        num_steps = int(3 / 0.003) + 1
        during_time = 3

        target_positions = np.zeros((num_steps, len(self.motors)))
        current_angles = self._get_joint_angles()
        for i in range(len(self.motors)):
            target_positions[:, i] = self.quintic_trajectory(
                current_angles[i], init_angles[i], during_time, num_steps
            )

        for q_targets in target_positions:
            # if self.stop_event.is_set():
            #     break
            for i, motor in enumerate(self.motors):
                self.controller.controlMIT(
                    motor, kp=self.kp[i], kd=self.kd[i], q=float(q_targets[i]), dq=0, tau=0
                )
            time.sleep(0.001)

    def _get_joint_angles(self) -> List[float]:
        """获取当前关节角度"""
        angles = []
        for motor in self.motors:
            self.controller.refresh_motor_status(motor)
            angles.append(motor.getPosition())
        return angles

    def quintic_trajectory(self, q0, qT, T, num_steps):
        """仅初始化零位用的插值函数"""
        t = np.linspace(0, T, num_steps)
        a0 = q0
        a1 = 0
        a2 = 0
        a3 = 10 * (qT - q0) / T ** 3
        a4 = -15 * (qT - q0) / T ** 4
        a5 = 6 * (qT - q0) / T ** 5
        return a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5

    def _execute_loop(self):
        """轨迹执行主循环（纯硬件控制）"""
        self._init_hardware()
        parent_conn, child_conn = self.state_pipe

        while not self.stop_event.is_set():
            try:
                # 1. 非阻塞处理轨迹执行（最高优先级）
                if not self.action_queue.empty():
                    # 直接获取关节角度列表
                    angles = self.action_queue.get(timeout=0.001)
                    # 发送电机控制指令
                    for i, motor in enumerate(self.motors):
                        self.controller.controlMIT(
                            motor, kp=self.kp[i], kd=self.kd[i],
                            q=angles[i], dq=0, tau=0
                        )

                # 2. 低优先级处理状态查询（非阻塞）
                if child_conn.poll(0.0005):
                    msg = child_conn.recv()
                    if msg == "get_state":
                        angles = self._get_joint_angles()
                        child_conn.send(angles)
            except Exception as e:
                logger.error(f"{self.arm_type} execution process error: {e}", exc_info=True)
                break
                #continue
        self._cleanup()

    def _cleanup(self):
        """清理硬件资源"""
        for m in self.motors:
            self.controller.disable(m)
            logger.info(f"{self.arm_type}失能")
        self.serial_device.close()
        logger.info(f"{self.arm_type} execution process cleaned up")

    def start(self):
        """启动执行进程"""
        if self.process is None or not self.process.is_alive():
            self.stop_event.clear()
            self.state_pipe = Pipe()  # 重新创建管道（避免进程复用问题）
            self.process = Process(target=self._execute_loop, daemon=True)
            self.process.start()
            logger.info(f"{self.arm_type} execution process started (PID: {self.process.pid})")

    def stop(self):
        """停止执行进程"""
        # 首先设置停止事件，让执行循环结束并执行_cleanup()
        self.stop_event.set()
        time.sleep(1)
        # 等待进程自然结束（给_cleanup() 时间执行）
        self.process.terminate()
        logger.info(f"{self.arm_type} execution process stopped")

    def get_joint_angles(self) -> List[float]:
        """查询关节角度"""
        if self.process:
            # 安全地检查进程状态
            try:
                alive_check = self.process.is_alive()
            except AssertionError:
                # 如果不能检查进程状态，则假定进程仍在运行
                alive_check = True
                
            if alive_check:
                parent_conn, _ = self.state_pipe
                parent_conn.send("get_state")
                if parent_conn.poll(1.0):
                    return parent_conn.recv()

        return [0.0] * 8


# --------------------------- 主机器人类 ---------------------------
class BiOpenarmFollower(Robot):
    """双机械臂主类（进程并行版）"""
    config_class = BiOpenarmFollowerConfig
    name = "bi_openarm_follower"

    def __init__(self, config: BiOpenarmFollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)

        # 控制参数
        self.kp = [200, 200, 150, 100, 25, 25, 25, 5]
        self.kd = [25, 15, 8, 5, 1.5, 1.5, 1.5, 0.3]

        # 关节限制
        self.left_joint_limits = [
            [-3.0, 1.2], [-1.5, 0.01], [-1.52, 1.56], [-0.01, 2.3],
            [-1.55, 1.55], [-0.5, 0.35], [-0.9, 0.9], [0.0, 1.0]
        ]
        self.right_joint_limits = [
            [-1.2, 3.0], [-0.01, 1.5], [-1.51, 1.59], [-2.3, 0.01],
            [-1.6, 1.6], [-0.4, 0.4], [-1.0, 1.0], [-1.0, 0.0]
        ]

        # 关节名称
        self.joint_names = [
            "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4",
            "left_joint_5", "left_joint_6", "left_joint_7", "left_trigger",
            "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4",
            "right_joint_5", "right_joint_6", "right_joint_7", "right_trigger"
        ]

        # 初始化2个并行进程
        # 左臂执行进程（关联生成进程的轨迹队列）
        self.left_executor = ArmExecutionProcess(
            arm_type="left",
            port=self.config.left_port,
            kp=self.kp,
            kd=self.kd
        )

        # 右臂执行进程（关联生成进程的轨迹队列）
        self.right_executor = ArmExecutionProcess(
            arm_type="right",
            port=self.config.right_port,
            kp=self.kp,
            kd=self.kd
        )
        # 注册退出清理
        atexit.register(self.disconnect)
        
        # 用于跟踪是否已连接
        self._is_connected_flag = False
        # 信号处理标志

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{joint_name}.pos": float for joint_name in self.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """检查所有进程和相机的连接状态"""
        return  self._is_connected_flag

        # cam_connected = all(cam.is_connected for cam in self.cameras.values())
        # executor_alive = (self.left_executor.process and self.left_executor.process.is_alive() and
        #                   self.right_executor.process and self.right_executor.process.is_alive())
        # return cam_connected and executor_alive

    def connect(self, calibrate: bool = True) -> None:
        """启动所有进程+连接相机+注册信号处理"""
        if self._is_connected_flag:
            raise DeviceAlreadyConnectedError("BiOpenarm already connected")
        
        # 启动执行进程
        self.left_executor.start()
        self.right_executor.start()

        # 连接相机
        for cam in self.cameras.values():
            cam.connect()

        self._is_connected_flag = True
        
        # 在连接后自动注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("BiOpenarm all processes started successfully")

    def _signal_handler(self, signum, frame):
        """内部信号处理函数"""
        logger.info(f"Received signal {signum}, shutting down robot gracefully...")
        self.disconnect()

    def disconnect(self) -> None:
        """停止所有进程+清理资源"""
        if not self._is_connected_flag:
            return  # 避免重复触发

        # 停执行进程
        self.left_executor.stop()
        self.right_executor.stop()

        # 断开相机
        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected_flag = False
        logger.info("BiOpenarm all processes stopped successfully")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> RobotObservation:
        """获取观测（关节角度+相机图像）"""
        obs_dict = {}
        start = time.perf_counter()

        # 获取左右臂关节角度（并行查询，减少耗时）
        left_angles = self.left_executor.get_joint_angles()
        right_angles = self.right_executor.get_joint_angles()

        # 填充角度数据
        for i in range(8):
            obs_dict[f"{self.joint_names[i]}.pos"] = left_angles[i]
            obs_dict[f"{self.joint_names[i + 8]}.pos"] = right_angles[i]

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"Read joint positions cost: {dt_ms:.1f}ms")

        # 读取相机图像
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"Read camera {cam_key} cost: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: RobotAction) -> RobotAction:
        """下发整体动作 16 个关节的字典 """
        # 拆分左右臂动作
        left_action = list(action.values())[:8]
        right_action = list(action.values())[8:]

        #判断是不是紧急停止信号
        if left_action[7] ==2 or right_action[7] ==2:
            self.disconnect()

        # 应用关节限制进行裁剪
        left_action = self._clip_to_joint_limits(left_action, self.left_joint_limits)
        right_action = self._clip_to_joint_limits(right_action, self.right_joint_limits)

        # 发送到对应的执行进程
        self._send_action_to_executor(self.left_executor, left_action)
        self._send_action_to_executor(self.right_executor, right_action)

        return action

    def _clip_to_joint_limits(self, action: List[float], limits: List[List[float]]) -> List[float]:
        """将动作限制在关节范围内"""
        clipped_action = []
        for i, (angle, (min_limit, max_limit)) in enumerate(zip(action, limits)):
            clipped_angle = max(min_limit, min(max_limit, angle))
            clipped_action.append(clipped_angle)
        return clipped_action

    def _send_action_to_executor(self, executor: ArmExecutionProcess, action: List[float]) -> None:
        """向执行器发送动作"""
        try:
            if not executor.action_queue.full():
                executor.action_queue.put_nowait(action)
            else:
                logger.warning(f"{executor.arm_type} arm action queue is full, dropping action")
        except Full:
            logger.warning(f"{executor.arm_type} arm action queue is full, dropping action")


    # def send_action_left(self, action: list) -> None:
    #     """下发左臂原始动作list """
    #     # 应用关节限制进行裁剪
    #     clipped_action = self._clip_to_joint_limits(action, self.left_joint_limits)
    #     self._send_action_to_executor(self.left_executor, clipped_action)
    #
    # def send_action_right(self, action: list) -> None:
    #     """下发右臂原始动作list """
    #     # 应用关节限制进行裁剪
    #     clipped_action = self._clip_to_joint_limits(action, self.right_joint_limits)
    #     self._send_action_to_executor(self.right_executor, clipped_action)


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 初始化配置
    config = BiOpenarmFollowerConfig(
        left_port=r'/dev/ttyACM0',  # 替换为实际串口
        right_port=r'/dev/ttyACM1',
        cameras={}  # 无相机时置空
    )

    # 创建机器人实例
    robot = BiOpenarmFollower(config)
    try:
        # 连接机器人（启动四个进程）
        robot.connect()
        # 获取观测
        obs = robot.get_observation()
        print(obs)
        # 添加一些测试指令
        # 构造左臂关节动作字典
        left_joint_names = [f'left_joint_{i}' for i in range(1, 9)]
        left_action_dict = {f'{name}.pos': angle for name, angle in
                            zip(left_joint_names, [0, 0, 0.2, 0, 0.2, 0.1, 0, 0.2])}

        # 构造右臂关节动作字典
        right_joint_names = [f'right_joint_{i}' for i in range(1, 9)]
        right_action_dict = {f'{name}.pos': angle for name, angle in
                             zip(right_joint_names, [0, 0, 0.2, 0, 0.2, 0.1, 0, -0.2])}

        # 合并左右臂动作
        action_dict = {**left_action_dict, **right_action_dict}
        robot.send_action(action_dict)

        time.sleep(0.4)
        obs = robot.get_observation()
        print(obs)

        # 回到零位
        left_zero_dict = {f'{name}.pos': 0.0 for name in left_joint_names}
        right_zero_dict = {f'{name}.pos': 0.0 for name in right_joint_names}
        zero_action_dict = {**left_zero_dict, **right_zero_dict}
        robot.send_action(zero_action_dict)

        time.sleep(0.6)
        obs = robot.get_observation()
        print(obs)

        # 其他测试动作
        left_test_dict = {f'{name}.pos': angle for name, angle in
                          zip(left_joint_names, [0, 0, -0.1, 0, -0.2, -0.1, 0, 0.5])}
        right_test_dict = {f'{name}.pos': angle for name, angle in
                           zip(right_joint_names, [0, 0, -0.1, 0, -0.2, -0.1, 0, -0.5])}
        test_action_dict = {**left_test_dict, **right_test_dict}
        robot.send_action(test_action_dict)

        time.sleep(0.4)

        # 回到零位
        robot.send_action(zero_action_dict)
        obs = robot.get_observation()
        print(obs)

        time.sleep(0.4)

        # 最后一个测试动作
        left_final_dict = {f'{name}.pos': angle for name, angle in
                           zip(left_joint_names, [0, 0, 0.2, 0, 0.2, 0, 0, 0.2])}
        right_final_dict = {f'{name}.pos': angle for name, angle in
                            zip(right_joint_names, [0, 0, 0.2, 0, 0.2, 0, 0, -0.2])}
        final_action_dict = {**left_final_dict, **right_final_dict}
        robot.send_action(final_action_dict)
        print("最后的动作发送完毕，请等待机器人执行")
        time.sleep(3)


    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down robot gracefully...")
    except Exception as e:
        logger.error(f"Test error: {e}", exc_info=True)
    finally:
        # 断开连接（停止所有进程）
        robot.disconnect()
        logger.info("Test completed")