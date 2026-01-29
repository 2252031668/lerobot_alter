import asyncio
import logging
from functools import cached_property

from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.bi_openarm_vr_leader.config_bi_openarm_vr_leader import BiOpenarmVRLeaderConfig
from lerobot.teleoperators.bi_openarm_vr_leader.op_ik import PinocchioIK


# VR相关
from lerobot.teleoperators.bi_openarm_vr_leader.vr.config import VRConfig
from lerobot.teleoperators.bi_openarm_vr_leader.vr.inputs.vr_ws_server import VRWebSocketServer
import threading
import http.server
import ssl
import os
import socket
import time
from multiprocessing import Process, Array, Event, Lock
import ctypes

import numpy as np
from scipy.spatial.transform import Rotation as R


logger = logging.getLogger(__name__)

# 获取当前文件所在的目录作为根路径

# 常量定义 - 替换原来的共享变量阈值
POSITION_ERROR_THRESHOLD = 0.1  # meters
ANGLE_CHANGE_THRESHOLD = 0.5  # radians
MOVE_DIS_THRESHOLD = 0.1  # meters


def get_local_ip():
    """获取本机IP地址。"""
    try:
        # 连接到远程地址以确定本地IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            # 备选方案：获取主机名IP
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            # 最终备选方案
            return "localhost"


def get_move_trajectory(cur_angles, tar_angles, during_time=3, dt=0.01):
    """点到点的插值轨迹"""
    num_steps = int(during_time / dt) + 1
    tar_angles_list = np.zeros((num_steps, 8))
    t = np.linspace(0, during_time, num_steps)
    for i in range(8):
        # 内联五次多项式轨迹计算
        q0 = cur_angles[i]
        qT = tar_angles[i]
        a0 = q0
        a1 = 0
        a2 = 0
        a3 = 10 * (qT - q0) / during_time ** 3
        a4 = -15 * (qT - q0) / during_time ** 4
        a5 = 6 * (qT - q0) / during_time ** 5
        tar_angles_list[:, i] = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5
    return tar_angles_list


def left_arm_process_func(stop_event, reset_event, new_event,B_event,input_shared_array, input_lock,
                          output_shared_array, output_lock, root_path):
    """
    左臂处理进程函数
    优化点：
    1. 移除了重复的is_running_value，只使用stop_event控制停止
    2. 移除了重复的重置信号判断，只使用reset_event控制重置
    3. 阈值改为常量，不再使用共享变量
    """

    left_joint_limits = [
        [-3.0, 1.2], [-1.5, 0.01], [-1.52, 1.56], [-0.01, 2.3],
        [-1.55, 1.55], [-0.5, 0.35], [-0.9, 0.9]
    ]

    # 初始化IK求解器
    xml_path = os.path.join(root_path, 'v1', 'openarm_bimanual.xml')
    ik_solver = PinocchioIK(xml_path, 'left', left_joint_limits)

    # 初始化关节角度
    joint_angles = [0.0] * 7
    trigger_val = 0.0
    fk_result = ik_solver.fk(joint_angles.copy())
    initial_pos = fk_result.translation.copy()
    initial_rotmat = fk_result.rotation.copy()
    cur_position = np.array(initial_pos)

    logger.debug("🚀 启动左臂处理进程")

    while not stop_event.is_set():
        try:
            if B_event.is_set():
                logger.info("🔄 执行左臂B指令")

                tar_angle = [0.8, -0.4, 0.0, 1.85, -0.45, 0.1, -0.65,0] #设定的关节位置
                cur_angle = joint_angles + [trigger_val]
                trajectory = get_move_trajectory(cur_angle,tar_angle,during_time=3, dt=0.01)
                for angle in trajectory:
                    with output_lock:
                        output_shared_array[1:] = angle  # 关节角度
                    time.sleep(0.01)
                    if stop_event.is_set():
                        break
                joint_angles = tar_angle[:7]
                fk_result = ik_solver.fk(joint_angles.copy())
                cur_position = fk_result.translation.copy()
                logger.info("🚀 左臂已到达B指令位置")
                B_event.clear()  # 清除B指令事件
                continue
            # 检查是否需要重置
            if reset_event.is_set():
                logger.info("🔄 执行左臂重置")
                fk_result = ik_solver.fk(joint_angles.copy())
                initial_pos = fk_result.translation.copy()
                initial_rotmat = fk_result.rotation.copy()
                reset_event.clear()  # 清除重置事件
                continue
            if new_event.is_set():
                with input_lock:
                    # 无嵌套，切片拷贝即可
                    change_pos = input_shared_array[:3]
                    quat_data = input_shared_array[3:7]
                    trigger_val = input_shared_array[7]
                    timestamp = input_shared_array[8]
                new_event.clear()  # 清除新指令事件
            else:
                time.sleep(0.005)
                continue

            # 更新目标位置 - 基于初始位置 + VR空间中的位移增量
            target_pos = np.array(initial_pos) + np.array(change_pos)

            # 检查位移是否超过移动限制（使用常量）
            change_distance = np.linalg.norm(target_pos - cur_position)
            if change_distance > MOVE_DIS_THRESHOLD:
                logger.warning(
                    f"⚠️ 左臂change_pos位移 {change_distance:.4f} 超过阈值 {MOVE_DIS_THRESHOLD:.4f}，跳过更新"
                )
                time.sleep(0.005)
                continue

            # VR四元数转旋转矩阵
            vr_quat = quat_data
            vr_rot = R.from_quat(vr_quat)

            # 坐标系变换
            coord_transform = np.array([
                [0, 0, -1],
                [-1, 0, 0],
                [0, 1, 0]
            ])
            coord_transform_rot = R.from_matrix(coord_transform)
            relative_rot = (coord_transform_rot * vr_rot * coord_transform_rot.inv()).as_matrix()
            # 重新构建初始旋转矩阵
            target_rot = relative_rot @ initial_rotmat

            # 执行IK求解
            ik_angles = ik_solver.ik(
                target_position=target_pos,
                target_orientation=target_rot,
                now_joints=joint_angles.copy(),
            )
            if ik_angles is None:
                logger.warning("⚠️ 左臂IK求解失败，跳过更新")
                continue

            # 更新关节角度和位置
            joint_angles = ik_angles
            cur_position = ik_solver.fk(ik_angles.copy()).translation.copy()

            # 将结果写入输出共享数组 (时间戳 + 7个关节角度 + 夹爪值)
            with output_lock:
                output_shared_array[0] = timestamp  # 时间戳
                output_shared_array[1:8] = ik_angles  # 关节角度
                output_shared_array[8] = trigger_val  # 夹爪值

        except Exception as e:
            logger.error(f"❌ 左臂处理进程出错: {e}")
    #发送特殊电机失能信号,接受端会自己解析
    with output_lock:
        output_shared_array[8] = 2  #加爪的值不可能是2 ，所以这里作为特殊信号


def right_arm_process_func(stop_event, reset_event, new_event,B_event, input_shared_array, input_lock,
                           output_shared_array, output_lock, root_path):
    """
    右臂处理进程函数
    优化点：
    1. 移除了重复的is_running_value，只使用stop_event控制停止
    2. 移除了重复的重置信号判断，只使用reset_event控制重置
    3. 阈值改为常量，不再使用共享变量
    4. 与左臂一样使用PinocchioIK求解器
    """

    
    # 右臂关节限制
    right_joint_limits = [
        [-1.2, 3.0], [-0.01, 1.5], [-1.51, 1.59], [-2.3, 0.01],
        [-1.6, 1.6], [-0.4, 0.4], [-1.0, 1.0]
    ]

    # 初始化IK求解器
    xml_path = os.path.join(root_path, 'v1', 'openarm_bimanual.xml')
    ik_solver = PinocchioIK(xml_path, 'right', right_joint_limits)

    # 初始化关节角度
    joint_angles = [0.0] * 7
    trigger_val = 0.0
    fk_result = ik_solver.fk(joint_angles.copy())
    initial_pos = fk_result.translation.copy()
    initial_rotmat = fk_result.rotation.copy()
    cur_position = np.array(initial_pos)
    
    logger.debug("🚀 启动右臂处理进程")

    while not stop_event.is_set():
        try:
            if B_event.is_set():
                logger.info("🔄 执行右臂B指令")

                tar_angle = [-0.8, 0.4, 0.0, -1.9, 0.3, -0.1, 0.65,0] #设定的关节位置
                cur_angle = joint_angles + [trigger_val]
                trajectory = get_move_trajectory(cur_angle,tar_angle,during_time=3, dt=0.01)
                for angle in trajectory:
                    with output_lock:
                        output_shared_array[1:] = angle.copy()  # 关节角度
                    time.sleep(0.01)
                    if stop_event.is_set():
                        break
                joint_angles = tar_angle[:7]
                fk_result = ik_solver.fk(joint_angles.copy())
                cur_position = fk_result.translation.copy()
                logger.info("🚀 右臂已到达B指令位置")
                B_event.clear()  # 清除B指令事件
                continue
            # 检查是否需要重置
            if reset_event.is_set():
                logger.info("🔄 执行右臂重置")
                fk_result = ik_solver.fk(joint_angles.copy())
                initial_pos = fk_result.translation.copy()
                initial_rotmat = fk_result.rotation.copy()
                reset_event.clear()  # 清除重置事件
                continue
            if new_event.is_set():
                with input_lock:
                    # 无嵌套，切片拷贝即可
                    change_pos = input_shared_array[:3]
                    quat_data = input_shared_array[3:7]
                    trigger_val = input_shared_array[7]
                    timestamp = input_shared_array[8]
                new_event.clear()  # 清除新指令事件
            else:
                time.sleep(0.005)
                continue

            # 更新目标位置 - 基于初始位置 + VR空间中的位移增量
            target_pos = np.array(initial_pos) + np.array(change_pos)

            # 检查位移是否超过移动限制（使用常量）
            change_distance = np.linalg.norm(target_pos - cur_position)
            if change_distance > MOVE_DIS_THRESHOLD:
                logger.warning(
                    f"⚠️ 右臂change_pos位移 {change_distance:.4f} 超过阈值 {MOVE_DIS_THRESHOLD:.4f}，跳过更新"
                )
                time.sleep(0.005)
                continue

            # VR四元数转旋转矩阵
            vr_quat = quat_data
            vr_rot = R.from_quat(vr_quat)

            # 坐标系变换
            coord_transform = np.array([
                [0, 0, -1],
                [-1, 0, 0],
                [0, 1, 0]
            ])
            coord_transform_rot = R.from_matrix(coord_transform)
            relative_rot = (coord_transform_rot * vr_rot * coord_transform_rot.inv()).as_matrix()
            # 重新构建初始旋转矩阵
            target_rot = relative_rot @ initial_rotmat

            # 执行IK求解
            ik_angles = ik_solver.ik(
                target_position=target_pos,
                target_orientation=target_rot,
                now_joints=joint_angles.copy(),
            )
            if ik_angles is None:
                logger.warning("⚠️ 右臂IK求解失败，跳过更新")
                continue

            # 更新关节角度和位置
            joint_angles = ik_angles
            cur_position = ik_solver.fk(ik_angles.copy()).translation.copy()

            # 将结果写入输出共享数组 (时间戳 + 7个关节角度 + 夹爪值)
            with output_lock:
                output_shared_array[0] = timestamp  # 时间戳
                output_shared_array[1:8] = ik_angles  # 关节角度
                output_shared_array[8] = trigger_val  # 夹爪值

        except Exception as e:
            logger.error(f"❌ 右臂处理进程出错: {e}")
    #发送特殊电机失能信号,接受端会自己解析
    with output_lock:
        output_shared_array[8] = 2  #加爪的值不可能是2 ，所以这里作为特殊信号





class SimpleAPIHandler(http.server.BaseHTTPRequestHandler):
    """简化的HTTP请求处理器，只提供基本web服务"""
    # 类变量用于存储ROOT_PATH
    root_path = None

    def end_headers(self):
        """为所有响应添加CORS头。"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        try:
            super().end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, ssl.SSLError):
            pass

    def do_OPTIONS(self):
        """处理预检CORS请求。"""
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        """重写以减少HTTP请求日志噪音。"""
        pass  # 禁用默认HTTP日志

    def do_GET(self):
        """处理GET请求。"""
        if self.path == '/' or self.path == '/index.html':
            # 从web-ui目录提供主页面
            self.serve_file('web-ui/index.html', 'text/html')
        elif self.path.endswith('.css'):
            # 从web-ui目录提供CSS文件
            self.serve_file(f'web-ui{self.path}', 'text/css')
        elif self.path.endswith('.js'):
            # 从web-ui目录提供JS文件
            self.serve_file(f'web-ui{self.path}', 'application/javascript')
        elif self.path.endswith('.ico'):
            self.serve_file(self.path[1:], 'image/x-icon')
        elif self.path.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            # 从web-ui目录提供图像文件
            content_type = 'image/jpeg' if self.path.endswith(('.jpg', '.jpeg')) else 'image/png' if self.path.endswith(
                '.png') else 'image/gif'
            self.serve_file(f'web-ui{self.path}', content_type)
        else:
            self.send_error(404, "Not Found")

    def serve_file(self, filename, content_type):
        """使用给定的内容类型提供文件。"""
        try:
            # 使用类变量ROOT_PATH
            file_path = os.path.join(SimpleAPIHandler.root_path, filename)

            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    content = f.read()

                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "File not found")
        except Exception as e:
            logger.error(f"提供文件时出错 {filename}: {e}")
            self.send_error(500, "Internal server error")


class SimpleHTTPSServer:
    """封装了 HTTPS 服务器的启动 / 停止逻辑"""

    def __init__(self, config, root_path):
        self.config = config
        self.root_path = root_path
        self.httpd = None
        self.server_thread = None

    async def start(self):
        """启动HTTPS服务器。"""
        try:
            # 创建服务器
            logger.info(f"🌐 启动HTTP服务器在 {self.config.host_ip}:{self.config.https_port}")
            
            # 设置类变量root_path供SimpleAPIHandler使用
            SimpleAPIHandler.root_path = self.root_path
            # 创建HTTP服务器
            self.httpd = http.server.HTTPServer((self.config.host_ip, self.config.https_port), SimpleAPIHandler)

            # 设置SSL
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

            context.load_cert_chain(self.config.certfile, self.config.keyfile)
            self.httpd.socket = context.wrap_socket(self.httpd.socket, server_side=True)

            # 在单独的线程中启动服务器
            self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.server_thread.start()

        except Exception as e:
            logger.error(f"❌ 启动HTTPS服务器失败: {e}")
            raise

    async def stop(self):
        """停止HTTPS服务器。"""
        if self.httpd:
            self.httpd.shutdown()
            if self.server_thread:
                self.server_thread.join(timeout=5)
            logger.info("🌐 HTTPS服务器已停止")


class BiOpenarmVRLeader(Teleoperator):
    config_class = BiOpenarmVRLeaderConfig
    name = "bi_openarm_vr_leader"

    def __init__(self, config: BiOpenarmVRLeaderConfig):
        super().__init__(config)
        self.ROOT_PATH = config.ROOT_PATH
        self.joint_names = [
            "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4",
            "left_joint_5", "left_joint_6", "left_joint_7", "left_trigger",
            "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4",
            "right_joint_5", "right_joint_6", "right_joint_7", "right_trigger"
        ]
        self.vr_config = VRConfig()
        self.vr_config.https_port = config.https_port
        self.vr_config.websocket_port= config.websocket_port  # 如果修改需要同时修改vr_app.js
        self.vr_config.host_ip = config.host_ip
        self.vr_config.certfile = os.path.join(self.ROOT_PATH, 'cert.pem')
        self.vr_config.keyfile = os.path.join(self.ROOT_PATH, 'key.pem')



        # 创建命令队列
        self.command_queue = asyncio.Queue()  # 先进先出的队列
        # 创建VR服务器
        self.vr_server = VRWebSocketServer(
            command_queue=self.command_queue,
            config=self.vr_config,
            print_only=False  # 更改为False以发送数据到队列
        )
        # 创建HTTP服务器
        self.https_server = SimpleHTTPSServer(self.vr_config, self.ROOT_PATH)

        self.is_running = True

        # ========== 优化后的共享变量和锁 ==========
        # 输入共享数组：位置(3) + 姿态(4) + 触发值(1) + 时间戳(1) = 9个float
        self.left_input_shared_array = Array(ctypes.c_double, [0.0] * 9)
        self.right_input_shared_array = Array(ctypes.c_double, [0.0] * 9)

        # 输出共享数组：时间戳(1) + 7个关节角度 + 夹爪值(1) = 9个float
        self.left_output_shared_array = Array(ctypes.c_double, [0.0] * 9)
        self.right_output_shared_array = Array(ctypes.c_double, [0.0] * 9)

        # 输入锁和输出锁
        self.left_input_lock = Lock()
        self.left_output_lock = Lock()
        self.right_input_lock = Lock()
        self.right_output_lock = Lock()

        # 事件对象（只保留必要的）
        self.left_reset_event = Event()
        self.left_stop_event = Event()
        self.left_new_event = Event()
        self.left_B_event = Event()
        self.right_reset_event = Event()
        self.right_stop_event = Event()
        self.right_new_event = Event()
        self.right_B_event = Event()


        # 进程对象
        self.left_arm_process = None
        self.right_arm_process = None

        self.vr_thread = None

    async def start_monitoring(self):
        """启动并监控VR控制信息"""
        try:
            # 启动HTTPS服务器
            await self.https_server.start()
            # 启动VR服务器
            await self.vr_server.start()
            logger.info("✅ VR监控器正在运行")
            # 显示连接信息
            host_display = get_local_ip() if self.vr_config.host_ip == "0.0.0.0" else self.vr_config.host_ip
            logger.info(f"📱 在您的VR头戴设备浏览器中打开:")
            logger.info(f"   https://{host_display}:{self.vr_config.https_port}")
            print(f"https://{host_display}:{self.vr_config.https_port}")

            # 启动双臂处理进程（优化：移除了is_running_value和阈值共享变量），传递ROOT_PATH
            self.left_arm_process = Process(
                target=left_arm_process_func,
                args=(
                    self.left_stop_event,
                    self.left_reset_event,
                    self.left_new_event,
                    self.left_B_event,
                    self.left_input_shared_array,
                    self.left_input_lock,
                    self.left_output_shared_array,
                    self.left_output_lock,
                    self.ROOT_PATH  # 传递ROOT_PATH参数
                )
            )
            self.right_arm_process = Process(
                target=right_arm_process_func,
                args=(
                    self.right_stop_event,
                    self.right_reset_event,
                    self.right_new_event,
                    self.right_B_event,
                    self.right_input_shared_array,
                    self.right_input_lock,
                    self.right_output_shared_array,
                    self.right_output_lock,
                    self.ROOT_PATH  # 传递ROOT_PATH参数
                )
            )

            self.left_arm_process.start()
            self.right_arm_process.start()

            # 监控命令队列
            await self.monitor_commands()

        except KeyboardInterrupt:
            logger.warning("\n⏹️  正在停止VR监控器...")
        except Exception as e:
            logger.error(f"❌ VR监控器出错: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            await self.stop_monitoring()

    async def monitor_commands(self):
        """监控来自VR控制器的命令（优化版）"""
        logger.debug("📊 监控VR控制命令...")

        while self.is_running:
            try:
                # 等待命令，超时1秒
                goal = await asyncio.wait_for(self.command_queue.get(), timeout=0.5)

                buttons_list = goal.metadata.get("buttons_list", [])
                if "a" in buttons_list:
                    logger.warning("🚨 紧急停止已经捕获")
                    self.is_running = False
                    # 设置停止事件
                    self.left_stop_event.set()
                    self.right_stop_event.set()
                    await self.stop_monitoring()
                    continue
                elif "b" in buttons_list: #启动预设动作B
                    self.left_B_event.set()
                    self.right_B_event.set()
                    continue
                # 还有 x，y 空余

                if goal.arm == "left":
                    # 处理重置信号
                    if goal.metadata.get("reset_target_to_current", False):
                        logger.info("🔄 左臂重置信号已经捕获")
                        # 设置重置事件
                        self.left_reset_event.set()
                    else:
                        # 更新左臂最新指令
                        with self.left_input_lock:
                            # 位置数据
                            self.left_input_shared_array[:3] = [-goal.target_position[2], -goal.target_position[0], goal.target_position[1]]
                            # 姿态数据 (四元数)
                            self.left_input_shared_array[3:7] = goal.target_orientation_quat
                            # 触发值
                            self.left_input_shared_array[7] = goal.metadata.get("trigger", 0.0)
                            # 时间戳
                            self.left_input_shared_array[8] = goal.metadata.get("timestamp", 0.0)
                        self.left_new_event.set()

                elif goal.arm == "right":
                    # 处理重置信号
                    if goal.metadata.get("reset_target_to_current", False):
                        logger.info("🔄 右臂重置信号已经捕获")
                        # 设置重置事件
                        self.right_reset_event.set()
                    else:
                        # 更新右臂最新指令
                        with self.right_input_lock:
                            # 位置数据
                            self.right_input_shared_array[:3] = [-goal.target_position[2], -goal.target_position[0], goal.target_position[1]]
                            # 姿态数据 (四元数)
                            self.right_input_shared_array[3:7] = goal.target_orientation_quat
                            # 触发值
                            self.right_input_shared_array[7] = - goal.metadata.get("trigger", 0.0) #右臂映射是 反的-1 到0
                            # 时间戳
                            self.right_input_shared_array[8] = goal.metadata.get("timestamp", 0.0)
                        self.right_new_event.set()

            except asyncio.TimeoutError:
                time.sleep(0.05)
                continue
            except Exception as e:
                logger.error(f"❌ 处理命令时出错: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")


    async def stop_monitoring(self):
        """停止监控（优化版）"""
        self.is_running = False

        # 设置停止事件
        self.left_stop_event.set()
        self.right_stop_event.set()

        # 等待双臂处理进程结束
        if self.left_arm_process and self.left_arm_process.is_alive():
            self.left_arm_process.join(timeout=2.0)
        if self.right_arm_process and self.right_arm_process.is_alive():
            self.right_arm_process.join(timeout=2.0)

        # 停止服务器
        if self.vr_server:
            await self.vr_server.stop()
        if self.https_server:
            await self.https_server.stop()

        logger.info("✅ VR监控器已停止")

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{joint_name}.pos": float for joint_name in self.joint_names}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.is_running

    def connect(self, calibrate: bool = True) -> None:
        # 启动VR监听线程
        self.vr_thread = threading.Thread(target=lambda: asyncio.run(self.start_monitoring()), daemon=True)
        self.vr_thread.start()

        logger.info("✅ VR遥操作器已连接，进程已启动")

    def disconnect(self) -> None:
        # 设置停止标志
        self.is_running = False
        self.vr_thread.join(timeout=2.0)  # 2秒超时

        # 等待进程结束
        if self.left_arm_process and self.left_arm_process.is_alive():
            self.left_stop_event.set()
            self.left_arm_process.join(timeout=2.0)
        if self.right_arm_process and self.right_arm_process.is_alive():
            self.right_stop_event.set()
            self.right_arm_process.join(timeout=2.0)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        '''包含夹爪 长度16'''
        action_dict = {}

        # 读取左臂输出
        with self.left_output_lock:
            left_action= self.left_output_shared_array[1:9]
        # 读取右臂输出
        with self.right_output_lock:
            right_action = self.right_output_shared_array[1:9]
        action = left_action + right_action  # 列表拼接
        for i, joint_name in enumerate(self.joint_names):
            action_dict[f"{joint_name}.pos"] = float(action[i])
        return action_dict

    def get_action_timestamp(self) -> list:
        '''包含时间戳 长度18'''
        # 读取左臂输出
        with self.left_output_lock:
            left_action = self.left_output_shared_array[0:9] #加入第一个时间戳
        # 读取右臂输出
        with self.right_output_lock:
            right_action = self.right_output_shared_array[0:9]
        return left_action + right_action  # 列表拼接


    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO: Implement force feedback
        raise NotImplementedError


if __name__ == '__main__':
    from lerobot.teleoperators.bi_openarm_vr_leader import BiOpenarmVRLeaderConfig
    from lerobot.robots.bi_openarm_follower.bi_openarm_follower import BiOpenarmFollowerConfig, BiOpenarmFollower
    #from visualizer import ArmVisualizer  # 更新导入路径

    #初始化配置
    config = BiOpenarmFollowerConfig(
        left_port=r'/dev/ttyACM0',  # 替换为实际串口
        right_port=r'/dev/ttyACM1',
        cameras={}  # 无相机时置空
    )

    # 创建机器人实例
    robot = BiOpenarmFollower(config)

    # 创建可视化器实例
    #visualizer = ArmVisualizer()

    # 硬编码配置参数
    config = BiOpenarmVRLeaderConfig(
        # 如果有特定的配置参数，在此处添加
    )
    # 创建VR遥操作器实例
    vr_leader = BiOpenarmVRLeader(config)
    try:
        # 连接遥操作器

        #robot.connect()
        vr_leader.connect()
        time.sleep(5)
        # 持续获取动作并更新可视化
        time_last_l =0
        time_last_r =0

        while vr_leader.is_running:
            action = vr_leader.get_action()
            #robot.send_action(action)
            #obs = robot.get_observation()
            time.sleep(0.02)

        ## 带时间戳
        # while vr_leader.is_running:
        #     action = vr_leader.get_action_timestamp()
        #     if action[0] != time_last_l:
        #         robot.send_action_left(action[1:9])
        #         time_last_l = action[0]
        #
        #     if action[9] != time_last_r:
        #         robot.send_action_right(action[10:18])
        #         time_last_r = action[9]
        #     time.sleep(0.02)

        # while vr_leader.is_running:
        #     action = vr_leader.get_action_timestamp()
        #     if action[0] != time_last_l or action[9] != time_last_r:
        #         #print(action)
        #         visualizer.update_joint_angles_from_action_list(action[1:9]+action[10:18])
        #         time_last_l = action[0]
        #         time_last_r = action[9]
        #     time.sleep(0.05)

    except KeyboardInterrupt:
        robot.disconnect()
        vr_leader.is_running = False
        logger.warning("\n测试被用户中断")

    finally:
        # 设置停止标志
        vr_leader.is_running = False
        # 设置进程停止事件
        vr_leader.left_stop_event.set()
        vr_leader.right_stop_event.set()
        # 断开连接
        vr_leader.disconnect()