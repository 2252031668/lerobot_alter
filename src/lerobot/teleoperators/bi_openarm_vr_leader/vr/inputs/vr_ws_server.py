"""
VR WebSocket server for receiving controller data from web browsers.
Adapted from the original vr_robot_teleop.py script.
"""

import asyncio
import json
import ssl
import time

import websockets
import numpy as np
import math
import logging
from typing import Dict, Optional, Set
from scipy.spatial.transform import Rotation as R

from .base import BaseInputProvider, ControlGoal, ControlMode
from ..config import VRConfig

logger = logging.getLogger(__name__)


class VRControllerState:
    """用于VR控制器的状态跟踪。"""

    def __init__(self, hand: str):
        self.hand = hand
        self.grip_active = False  # squeeze/grip键激活状态
        self.trigger_active = False

        # 相对移动的位置跟踪
        self.origin_position = None  # grip键首次按下时的起始位置
        self.origin_rotation = None

        # 基于四元数的旋转跟踪（比欧拉角更稳定）
        self.origin_quaternion = None  # grip键首次按下时的起始姿态（绝对姿态）
        self.accumulated_rotation_quat = None  # 当前控制器姿态（绝对姿态）

        # 手腕控制的旋转跟踪（相对于起始姿态的变化量）
        self.z_axis_rotation = 0.0  # 用于wrist_roll（相对于起始姿态的变化量）
        self.x_axis_rotation = 0.0  # 用于wrist_flex (pitch)（相对于起始姿态的变化量）

        # 位置跟踪
        self.current_position = None

        # 旋转跟踪
        self.origin_wrist_angle = 0.0

    def reset_grip(self):
        """重置握持状态但保留扳机状态。"""
        self.grip_active = False
        self.origin_position = None
        self.origin_rotation = None
        self.origin_quaternion = None
        self.accumulated_rotation_quat = None
        self.z_axis_rotation = 0.0
        self.x_axis_rotation = 0.0

    # def reset_origin(self):
    #     """重置原点位置和旋转以用于自动控制模式。"""
    #     self.origin_position = None
    #     self.origin_rotation = None
    #     self.origin_quaternion = None
    #     self.accumulated_rotation_quat = None
    #     self.z_axis_rotation = 0.0
    #     self.x_axis_rotation = 0.0


class VRWebSocketServer(BaseInputProvider):
    """用于VR控制器输入的WebSocket服务器。"""

    def __init__(self, command_queue: asyncio.Queue, config: VRConfig, print_only: bool = False):
        super().__init__(command_queue)
        self.config = config
        self.clients: Set = set()
        self.server = None
        self.print_only = print_only  # 仅打印模式的新标志

        # 控制器状态
        self.left_controller = VRControllerState("left")
        self.right_controller = VRControllerState("right")

        # 机器人状态跟踪（用于相对位置计算）
        self.left_arm_origin_position = None
        self.right_arm_origin_position = None

    def setup_ssl(self) -> Optional[ssl.SSLContext]:
        """设置WebSocket服务器的SSL上下文。"""
        # 如果不存在则自动生成SSL证书
        if not self.config.ssl_files_exist:
            logger.info("未找到WebSocket服务器的SSL证书，尝试生成...")
            if not self.config.ensure_ssl_certificates():
                logger.error("无法生成WebSocket服务器的SSL证书")
                return None

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            ssl_context.load_cert_chain(certfile=self.config.certfile, keyfile=self.config.keyfile)
            logger.info("SSL证书和密钥已成功加载到WebSocket服务器")
            return ssl_context
        except ssl.SSLError as e:
            logger.error(f"加载SSL证书/密钥时出错: {e}")
            return None

    async def start(self):
        """启动WebSocket服务器。"""
        if not self.config.enable_vr:
            logger.info("VR WebSocket服务器在配置中被禁用")
            return

        ssl_context = self.setup_ssl()
        if ssl_context is None:
            logger.error("无法为WebSocket服务器设置SSL")
            return

        host = self.config.host_ip
        port = self.config.websocket_port

        try:
            self.server = await websockets.serve(
                self.websocket_handler,
                host,
                port,
                ssl=ssl_context
            )
            self.is_running = True
            logger.info(f"VR WebSocket服务器运行在 wss://{host}:{port}")
        except Exception as e:
            logger.error(f"无法启动WebSocket服务器: {e}")

    async def stop(self):
        """停止WebSocket服务器。"""
        self.is_running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("VR WebSocket服务器已停止")

    async def websocket_handler(self, websocket, path=None):
        """处理来自VR控制器的WebSocket连接。"""
        client_address = websocket.remote_address
        logger.info(f"VR客户端已连接: {client_address}")
        self.clients.add(websocket)

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_controller_data(data)
                except json.JSONDecodeError:
                    logger.warning(f"收到非JSON消息: {message}")
                except Exception as e:
                    logger.error(f"处理VR数据时出错: {e}")
                    # 添加更多调试上下文
                    logger.error(f"导致错误的数据: {data}")
                    import traceback
                    logger.error(f"跟踪: {traceback.format_exc()}")

        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"VR客户端 {client_address} 正常断开连接")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"VR客户端 {client_address} 错误断开连接: {e}")
        except Exception as e:
            logger.error(f"VR客户端 {client_address} 出现意外错误: {e}")
        finally:
            self.clients.discard(websocket)
            # 客户端断开连接时处理握持释放
            await self.handle_grip_release('left')
            await self.handle_grip_release('right')
            logger.info(f"VR客户端 {client_address} 清理完成")

    async def process_controller_data(self, data: Dict):
        """处理传入的VR控制器数据。"""
        # 检查是否有摇杆或按钮操作，只在有操作时打印
        has_thumbstick_or_button_activity = False
        thumbstick_info = []
        button_info = []

        if "timestamp" in data:
            timestamp = data['timestamp']
        else:
            timestamp = time.time()


        # 检查左右手柄的摇杆和按钮状态
        for hand in ['leftController', 'rightController']:
            if hand in data:
                controller_data = data[hand]
                hand_name = hand.replace('Controller', '').upper()

                # # 检查摇杆
                # if 'thumbstick' in controller_data:
                #     thumbstick = controller_data['thumbstick']
                #     x = thumbstick.get('x', 0)
                #     y = thumbstick.get('y', 0)
                #     # 只在摇杆有实际输入时打印（阈值0.1）
                #     if abs(x) > 0.1 or abs(y) > 0.1:
                #         has_thumbstick_or_button_activity = True
                #         thumbstick_info.append(f"[{hand_name}] 摇杆: x={x:.2f}, y={y:.2f}")

                # 检查按钮
                if 'buttons' in controller_data:
                    buttons = controller_data['buttons']
                    pressed_buttons = []
                    stop_contorl = False

                    for button_name, is_pressed in buttons.items():
                        if is_pressed:
                            #has_thumbstick_or_button_activity = True
                            pressed_buttons.append(button_name)
                            # if button_name == 'a':
                            #     stop_contorl = True #设置按钮a 为紧急制动
                    if pressed_buttons:
                        #print(pressed_buttons)
                        button_stop = ControlGoal(
                            arm=hand_name,
                            mode=ControlMode.POSITION_CONTROL,  # 保持在位置控制
                            target_position=None,  # 特殊信号
                            metadata={
                                #"source": f"button_{hand_name}",
                                #"stop_contorl": stop_contorl,  # 紧急停止控制
                                "timestamp": timestamp,
                                "buttons_list": pressed_buttons #按钮信息
                            }
                        )
                        await self.send_goal(button_stop)




                    #if pressed_buttons:
                        #button_info.append(f"[{hand_name}] 按钮: {', '.join(pressed_buttons)}")
                        #print(pressed_buttons) #

        # 只在有操作时打印
        # if has_thumbstick_or_button_activity:
        #     print(f"[VR_WS] 检测到活动:")
        #     for info in thumbstick_info:
        #         print(f"  {info}")
        #     for info in button_info:
        #         print(f"  {info}")

        # Process headset data if available
        # if 'headset' in data:
        #     headset_data = data['headset']
        #     if headset_data and headset_data.get('position'):
        #         pos = headset_data['position']
        #         rot = headset_data.get('rotation', {})
        #         quat = headset_data.get('quaternion', {})
        #
        #         # print(f"[VR_WS] 头戴设备 - 位置: [{pos.get('x', 0):.3f}, {pos.get('y', 0):.3f}, {pos.get('z', 0):.3f}], "
        #         #       f"旋转: [{rot.get('x', 0):.1f}, {rot.get('y', 0):.1f}, {rot.get('z', 0):.1f}]")
        #
        #         # 创建头戴设备控制目标
        #         headset_position = np.array([pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)])
        #         # 如果有四元数信息，使用它来设置目标姿态
        #         headset_orientation_quat = None
        #         if quat and all(k in quat for k in ['x', 'y', 'z', 'w']):
        #             headset_orientation_quat = np.array([quat['x'], quat['y'], quat['z'], quat['w']])
        #
        #         # 对于头戴设备，我们仍然使用绝对姿态
        #         headset_goal = ControlGoal(
        #             arm="headset",
        #             mode=ControlMode.POSITION_CONTROL,
        #             target_position=headset_position,
        #             wrist_roll_deg=rot.get('y', 0),  # 偏航旋转
        #             wrist_flex_deg=rot.get('x', 0),   # 俯仰旋转
        #             target_orientation_quat=headset_orientation_quat,
        #             metadata={
        #                 "source": "vr_headset",
        #                 "relative_position": False,
        #                 "vr_position": headset_position.tolist(),
        #                 "rotation": rot,
        #                 "quaternion": quat
        #             }
        #         )
        #         await self.send_goal(headset_goal)

        # Process controller data
        if 'leftController' in data:
            await self.process_single_controller('left', data['leftController'], timestamp)

        if 'rightController' in data:
            await self.process_single_controller('right', data['rightController'], timestamp)

    async def process_single_controller(self, hand: str, data: Dict, timestamp):
        """处理单个控制器的数据。"""
        position = data.get('position', {})
        rotation = data.get('rotation', {})
        quaternion = data.get('quaternion', {})  # 直接获取四元数数据
        grip_active = data.get('gripActive', False)  # squeeze/grip键状态
        trigger = data.get('trigger', 0.0)  #0-1 0表示没有下按，之后是不同的下按程度
        thumbstick = data.get('thumbstick', {})

        controller = self.left_controller if hand == 'left' else self.right_controller

        # # 处理扳机键用于夹爪控制
        # trigger_active = trigger > 0.5
        # if trigger_active != controller.trigger_active:
        #     controller.trigger_active = trigger_active
        #
        #     # 发送夹爪控制目标 - 不指定模式以避免干扰位置控制
        #     # 反向行为：默认夹爪打开，按下扳机时关闭
        #     gripper_goal = ControlGoal(
        #         arm=hand,
        #         gripper_closed=not trigger_active,  # 反向：扳机未激活时关闭
        #         metadata={
        #             "source": "vr_trigger",
        #             "trigger": trigger,
        #             "trigger_active": trigger_active,
        #             "thumbstick": thumbstick
        #         }
        #     )
        #     await self.send_goal(gripper_goal)
        #
        #     logger.info(f"🤏 {hand.upper()} 夹爪 {'已打开' if trigger_active else '已关闭'}")

        # 使用squeeze/grip键控制臂移动（原始逻辑）
        if grip_active:
            if not controller.grip_active:
                # Grip刚刚激活 - 设置原点并重置目标位置
                controller.grip_active = True
                # 将位置字典转换为numpy数组以便稍后进行正确的减法运算
                controller.origin_position = np.array([position.get('x', 0), position.get('y', 0), position.get('z', 0)])
                
                # 如果可用则直接使用四元数数据，否则回退到欧拉角转换
                if quaternion and all(k in quaternion for k in ['x', 'y', 'z', 'w']):
                    controller.origin_quaternion = np.array([quaternion['x'], quaternion['y'], quaternion['z'], quaternion['w']])
                    controller.origin_rotation = controller.origin_quaternion  # 存储以兼容
                else:
                    # 回退到欧拉角转换
                    controller.origin_quaternion = self.euler_to_quaternion(rotation) if rotation else None
                    controller.origin_rotation = controller.origin_quaternion
                
                # 初始化当前姿态为初始姿态
                controller.accumulated_rotation_quat = controller.origin_quaternion
                controller.z_axis_rotation = 0.0  # 相对于起始姿态的滚动角变化量
                controller.x_axis_rotation = 0.0  # 相对于起始姿态的俯仰角变化量
                
                # 发送重置信号到控制循环，将目标位置重置为当前机器人位置
                # 在重置时，相对旋转为单位四元数（表示没有旋转变化）
                identity_quat = np.array([0, 0, 0, 1])  # 单位四元数，表示无旋转变化

                reset_goal = ControlGoal(
                    arm=hand,
                    mode=ControlMode.POSITION_CONTROL,  # 保持在位置控制
                    target_position=None,  # 特殊信号
                    target_orientation_quat=identity_quat,  # 重置时的相对旋转为单位四元数
                    metadata={
                        #"source": f"vr_grip_reset_{hand}",
                        "reset_target_to_current": True,  # 重置目标到当前位置的信号
                        "trigger": trigger,
                        #"thumbstick": thumbstick,
                        "timestamp": timestamp,
                    }
                )

                await self.send_goal(reset_goal)
                
                #logger.info(f"🔒 {hand.upper()} 握持激活 - 控制 {hand} 臂 (目标重置到当前位置)")
            
            # 计算目标位置
            if controller.origin_position is not None:
                # 将位置字典转换为numpy数组以便进行正确的减法运算
                position_array = np.array([position.get('x', 0), position.get('y', 0), position.get('z', 0)])
                
                # 确保origin_position是numpy数组
                if isinstance(controller.origin_position, dict):
                    # 如果origin_position仍然是字典，将其转换为numpy数组
                    logger.warning(f"origin_position是字典，正在转换为numpy数组用于{hand}控制器")
                    controller.origin_position = np.array([controller.origin_position.get('x', 0), controller.origin_position.get('y', 0), controller.origin_position.get('z', 0)])
                elif not isinstance(controller.origin_position, np.ndarray):
                    # 如果origin_position既不是字典也不是numpy数组，记录警告并跳过
                    logger.warning(f"origin_position是{type(controller.origin_position)}，跳过{hand}控制器的位置计算")
                    return
                
                relative_delta = (position_array - controller.origin_position) * self.config.vr_to_robot_scale
                
                # 计算Z轴旋转用于wrist_roll控制
                # 计算X轴旋转用于wrist_flex控制
                if controller.origin_quaternion is not None:
                    # 更新基于四元数的旋转跟踪
                    if quaternion and all(k in quaternion for k in ['x', 'y', 'z', 'w']):
                        # 直接使用四元数数据（当前绝对姿态）
                        current_quat = np.array([quaternion['x'], quaternion['y'], quaternion['z'], quaternion['w']])
                        self.update_quaternion_rotation_direct(controller, current_quat)
                    else:
                        # 回退到欧拉角转换
                        self.update_quaternion_rotation(controller, rotation)
                    
                    # 从四元数获取相对于起始姿态的旋转变化量
                    controller.z_axis_rotation = self.extract_roll_from_quaternion(controller.accumulated_rotation_quat, controller.origin_quaternion)
                    controller.x_axis_rotation = self.extract_pitch_from_quaternion(controller.accumulated_rotation_quat, controller.origin_quaternion)
                
                # 创建位置控制目标
                # 注意：我们在这里发送相对位置，控制循环将处理
                # 添加到机器人当前位置
                # 计算相对旋转四元数（从初始姿态到当前姿态的变化）
                relative_rotation_quat = None
                if controller.origin_quaternion is not None and controller.accumulated_rotation_quat is not None:
                    # 计算相对旋转：从初始姿态到当前姿态的旋转变化
                    origin_rotation = R.from_quat(controller.origin_quaternion)
                    current_rotation = R.from_quat(controller.accumulated_rotation_quat)
                    relative_rotation = current_rotation * origin_rotation.inv()
                    relative_rotation_quat = relative_rotation.as_quat()  # [x, y, z, w]
                
                goal = ControlGoal(
                    arm=hand,
                    mode=ControlMode.POSITION_CONTROL,
                    target_position=relative_delta,  # 相对位置增量
                    # wrist_roll_deg=-controller.z_axis_rotation,  # 相对于起始姿态的旋转变化量
                    # wrist_flex_deg=-controller.x_axis_rotation,  # 相对于起始姿态的旋转变化量
                    # 添加相对姿态四元数（从初始姿态到当前姿态的旋转变化）
                    target_orientation_quat=relative_rotation_quat,
                    metadata={
                        #"source": "vr_grip",
                        #"relative_position": True,
                        #"origin_position": controller.origin_position.tolist(),
                        "trigger": trigger,
                        #"thumbstick": thumbstick,
                        "timestamp": timestamp
                    }
                )
                await self.send_goal(goal)
        else:controller.grip_active = False



    async def handle_grip_release(self, hand: str):
        """处理控制器的握持释放。"""
        if hand == 'left':
            controller = self.left_controller
        elif hand == 'right':
            controller = self.right_controller
        else:
            return

        if controller.grip_active:
            controller.reset_grip()
            # # 发送空闲目标以停止臂控制
            # goal = ControlGoal(
            #     arm=hand,
            #     mode=ControlMode.IDLE,
            #     metadata={
            #         "source": "vr_grip_release",
            #         "trigger": 0.0,
            #         "trigger_active": False,
            #         "thumbstick": {}
            #     }
            # )
            # await self.send_goal(goal)

            #logger.info(f"🔓 {hand.upper()} 握持释放 - 臂控制停止")

    # async def handle_trigger_release(self, hand: str):
    #     """处理控制器的扳机释放。"""
    #     controller = self.left_controller if hand == 'left' else self.right_controller
    #
    #     if controller.trigger_active:
    #         controller.trigger_active = False
    #
    #         # 发送夹爪关闭目标 - 反向行为：释放扳机时夹爪关闭
    #         goal = ControlGoal(
    #             arm=hand,
    #             gripper_closed=True,  # 释放扳机时关闭夹爪
    #             metadata={
    #                 "source": "vr_trigger_release",
    #                 "trigger": 0.0,
    #                 "trigger_active": False,
    #                 "thumbstick": {}
    #             }
    #         )
    #         await self.send_goal(goal)
    #
    #         logger.info(f"🤏 {hand.upper()} 夹爪已关闭 (扳机释放)")

    def euler_to_quaternion(self, euler_deg: Dict[str, float]) -> np.ndarray:
        """将欧拉角（度）转换为四元数 [x, y, z, w]。"""
        euler_rad = [math.radians(euler_deg['x']), math.radians(euler_deg['y']), math.radians(euler_deg['z'])]
        rotation = R.from_euler('xyz', euler_rad)
        return rotation.as_quat()

    def update_quaternion_rotation(self, controller: VRControllerState, current_euler: dict):
        """更新基于四元数的旋转跟踪。"""
        if not current_euler:
            return

        # 将当前欧拉角转换为四元数（当前绝对姿态）
        current_quat = self.euler_to_quaternion(current_euler)

        # 存储当前四元数用于累积旋转计算（当前绝对姿态）
        controller.accumulated_rotation_quat = current_quat

    def update_quaternion_rotation_direct(self, controller: VRControllerState, current_quat: np.ndarray):
        """直接使用四元数数据更新基于四元数的旋转跟踪。"""
        if current_quat is None:
            return

        # 存储当前四元数用于累积旋转计算（当前绝对姿态）
        controller.accumulated_rotation_quat = current_quat

    def extract_roll_from_quaternion(self, current_quat: np.ndarray, origin_quat: np.ndarray) -> float:
        """从相对四元数旋转中提取Z轴（滚动）旋转变化量（相对于起始姿态）。
        
        Args:
            current_quat: 当前控制器姿态（绝对四元数）
            origin_quat: 起始控制器姿态（绝对四元数）
        
        Returns:
            相对于起始姿态的滚动角变化量（度）
        """
        if current_quat is None or origin_quat is None:
            return 0.0

        try:
            # 计算相对旋转四元数（从起始姿态到当前姿态）
            origin_rotation = R.from_quat(origin_quat)
            current_rotation = R.from_quat(current_quat)
            relative_rotation = current_rotation * origin_rotation.inv()

            # 将相对旋转投影到Z轴（滚动）
            # 获取旋转矢量（轴角表示）
            rotvec = relative_rotation.as_rotvec()

            # 旋转矢量的Z分量表示Z轴（滚动）旋转
            z_rotation_rad = rotvec[2]
            z_rotation_deg = -np.degrees(z_rotation_rad)

            return z_rotation_deg
        except Exception as e:
            logger.warning(f"从四元数提取滚动时出错: {e}")
            return 0.0

    def extract_pitch_from_quaternion(self, current_quat: np.ndarray, origin_quat: np.ndarray) -> float:
        """从相对四元数旋转中提取X轴（俯仰）旋转变化量（相对于起始姿态）。
        
        Args:
            current_quat: 当前控制器姿态（绝对四元数）
            origin_quat: 起始控制器姿态（绝对四元数）
        
        Returns:
            相对于起始姿态的俯仰角变化量（度）
        """
        if current_quat is None or origin_quat is None:
            return 0.0

        try:
            # 计算相对旋转四元数（从起始姿态到当前姿态）
            origin_rotation = R.from_quat(origin_quat)
            current_rotation = R.from_quat(current_quat)
            relative_rotation = current_rotation * origin_rotation.inv()

            # 将相对旋转投影到X轴（俯仰）
            # 获取旋转矢量（轴角表示）
            rotvec = relative_rotation.as_rotvec()

            # 旋转矢量的X分量表示X轴（俯仰）旋转
            x_rotation_rad = rotvec[0]
            x_rotation_deg = np.degrees(x_rotation_rad)

            return x_rotation_deg
        except Exception as e:
            logger.warning(f"从四元数提取俯仰时出错: {e}")
            return 0.0

    async def send_goal(self, goal: ControlGoal):
        """发送控制目标到命令队列，如果在仅打印模式下则打印。"""
        if self.print_only:
            # 以格式化方式打印ControlGoal
            print(f"\n🎮 控制目标:")
            print(f"   臂: {goal.arm}")
            print(f"   模式: {goal.mode}")
            if goal.target_position is not None:
                print(f"   目标位置: [{goal.target_position[0]:.3f}, {goal.target_position[1]:.3f}, {goal.target_position[2]:.3f}]")
            if goal.wrist_roll_deg is not None:
                print(f"   手腕滚动: {goal.wrist_roll_deg:.1f}°")
            if goal.wrist_flex_deg is not None:
                print(f"   手腕弯曲: {goal.wrist_flex_deg:.1f}°")
            if goal.target_orientation_quat is not None:
                print(f"   目标姿态四元数 (相对): [{goal.target_orientation_quat[0]:.3f}, {goal.target_orientation_quat[1]:.3f}, {goal.target_orientation_quat[2]:.3f}, {goal.target_orientation_quat[3]:.3f}]")
            if goal.gripper_closed is not None:
                print(f"   夹爪: {'已关闭' if goal.gripper_closed else '已打开'}")
            if goal.metadata:
                print(f"   元数据: {goal.metadata}")
            print()
        else:
            # 使用父类方法发送到队列
            await super().send_goal(goal) 