# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
import serial
import serial.tools.list_ports
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QGridLayout, QDoubleSpinBox,
    QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

# =============================
# DM_CAN 电机控制模块
# =============================
from lerobot.motors.DM import MotorControl, Motor, DM_Motor_Type

# 比例增益：控制响应速度
kp1_default = [200, 200, 150, 100, 25, 25, 25, 5]  # 8个电机的P参数，最后一个是夹爪
# 微分增益：抑制振荡，提高稳定性
kd1_default = [25, 15, 8, 5, 1.5, 1.5, 1.5, 0.3]  # 8个电机的D参数，最后一个是夹爪

# 当前的 kp 和 kd 值，初始化为默认值
kp1 = kp1_default.copy()
kd1 = kd1_default.copy()


# ==========================================================
# 电机实时位置刷新线程
# ==========================================================
class PositionThread(QThread):
    pos_signal = pyqtSignal(int, float)  # joint_index, pos

    def __init__(self, controller, motors, dt=0.005):
        super().__init__()
        self.controller = controller
        self.motors = motors
        self.dt = dt
        self.running = True

    def run(self):
        while self.running:
            try:
                self.controller.recv()  # 更新所有电机数据
                for i, m in enumerate(self.motors):
                    pos = m.getPosition()
                    self.pos_signal.emit(i, pos)
            except Exception as e:
                print("PositionThread error:", e)

            time.sleep(self.dt)

    def stop(self):
        self.running = False


# ==========================================================
# 电机点到点控制线程（五次多项式）
# ==========================================================
class P2PThread(QThread):
    # 定义信号
    update_signal = pyqtSignal(int, float)  # (joint_idx, current_angle)
    finished_signal = pyqtSignal()  # 运动完成信号

    def __init__(self, controller, motors, target_angles, duration, dt=0.002):
        super().__init__()
        self.controller = controller  # 电机控制器
        self.motors = motors  # 电机列表
        self.target_angles = np.array(target_angles)  # 目标角度数组
        self.duration = duration  # 运动持续时间
        self.dt = dt  # 控制周期
        self.running = True  # 线程运行标志

    def quintic_trajectory(self, q0, qT, T, num_steps):
        """
        计算五次多项式轨迹
        Args:
            q0: 初始角度
            qT: 目标角度
            T: 运动时间
            num_steps: 步数
        Returns:
            q: 轨迹点数组
        """
        t = np.linspace(0, T, num_steps)  # 时间序列
        # 五次多项式系数
        a0 = q0
        a1 = 0
        a2 = 0
        a3 = 10 * (qT - q0) / T ** 3
        a4 = -15 * (qT - q0) / T ** 4
        a5 = 6 * (qT - q0) / T ** 5
        # 计算轨迹
        q = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5
        return q

    def run(self):
        """线程主循环：执行点到点运动"""
        # 获取初始位置
        init_positions = np.array([m.getPosition() for m in self.motors])
        # 生成时间序列
        times = np.linspace(0, self.duration, int(self.duration / self.dt) + 1)
        # 初始化目标位置数组
        target_positions = np.zeros((len(times), len(self.motors)))

        print("Init positions:", init_positions)

        # 为每个电机计算轨迹
        for i in range(len(self.motors)):
            target_positions[:, i] = self.quintic_trajectory(
                init_positions[i], self.target_angles[i],
                self.duration, len(times)
            )

        # 控制循环
        start = time.time()
        for step, q_targets in enumerate(target_positions):
            if not self.running:  # 检查停止标志
                break
            # 控制每个电机
            for i, motor in enumerate(self.motors):
                q = float(q_targets[i])  # 当前目标角度
                # 发送控制命令
                self.controller.controlMIT(
                    motor, kp=kp1[i], kd=kd1[i],
                    q=q, dq=0, tau=0
                )
                # 发送更新信号
                self.update_signal.emit(i, q)

            time.sleep(self.dt)  # 我发现dt设置为0.002的时候已经满足一直，因为本身指令就要时间

        end = time.time()
        print("Time:", end - start)
        self.finished_signal.emit()  # 发送完成信号

    def stop(self):
        """停止线程运行"""
        self.running = False


# ==========================================================
# 主界面类
# ==========================================================
class ArmP2P(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("点到点控制面板")  # 设置窗口标题
        self.resize(700, 400)  # 设置窗口大小，增加以适应第八个关节

        # 初始化成员变量
        self.serial_device = None  # 串口设备
        self.controller = None  # 电机控制器
        self.motors = []  # 电机列表
        self.control_thread = None  # 控制线程
        self.position_thread = None  # 位置更新线程

        # 初始化 kp 和 kd 控制器
        self.kp_boxes = []  # kp 调节框
        self.kd_boxes = []  # kd 调节框

        self._build_ui()  # 构建用户界面

    def _build_ui(self):
        """构建用户界面"""
        layout = QVBoxLayout()  # 主布局

        # 串口选择区域
        port_layout = QHBoxLayout()
        self.port_combo = QComboBox()  # 串口下拉框
        self.btn_refresh = QPushButton("刷新")  # 刷新按钮
        self.btn_connect = QPushButton("连接")  # 连接按钮
        self.btn_disconnect = QPushButton("断开连接")  # 断开连接按钮
        port_layout.addWidget(QLabel("串口:"))
        port_layout.addWidget(self.port_combo)
        port_layout.addWidget(self.btn_refresh)
        port_layout.addWidget(self.btn_connect)
        port_layout.addWidget(self.btn_disconnect)
        layout.addLayout(port_layout)
        # 绑定按钮事件
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_connect.clicked.connect(self.connect_serial)
        self.btn_disconnect.clicked.connect(self.disconnect_serial)
        self.refresh_ports()  # 初始刷新串口列表

        # 关节角度控制和参数调节区域
        grid = QGridLayout()
        self.joint_boxes = []  # 角度输入框列表
        self.joint_labels = []  # 位置显示标签列表
        self.kp_boxes = []  # kp 调节框列表
        self.kd_boxes = []  # kd 调节框列表

        # 表头
        grid.addWidget(QLabel("关节"), 0, 0)
        grid.addWidget(QLabel("活动范围  左臂/右臂"), 0, 1)
        grid.addWidget(QLabel("目标角度 (rad)"), 0, 2)
        grid.addWidget(QLabel("当前角度"), 0, 3)
        grid.addWidget(QLabel("Kp"), 0, 4)
        grid.addWidget(QLabel("Kd"), 0, 5)

        # 为每个关节创建控件
        for i in range(8):  # 修改为8个关节（包括夹爪）
            # 关节标签
            grid.addWidget(QLabel(f"J{i + 1}:"), i + 1, 0)

            # 活动范围标签
            if i == 0:  # 关节1 (对应J2)
                range_label = QLabel("左: -1.0 ~ 0.7      右: -0.7 ~ 1.0")  # 左臂和右臂
            elif i == 1:  # 关节2 (对应J3)
                range_label = QLabel("左: -0.8 ~ 0.01    右: -0.01 ~ 0.8")  # 左臂和右臂
            elif i == 2:  # 关节3 (对应J4)
                range_label = QLabel("左: -1.52 ~ 1.56  右: -1.51 ~ 1.59")  # 左臂和右臂
            elif i == 3:  # 关节4 (对应J5)
                range_label = QLabel("左: -0.01 ~ 2.3    右: -2.3 ~ 0.01")  # 左臂和右臂
            elif i == 4:  # 关节5 (对应J6)
                range_label = QLabel("左: -1.55 ~ 1.55    右: -1.6 ~ 1.6")  # 左臂和右臂
            elif i == 5:  # 关节6 (对应J7)
                range_label = QLabel("左: -0.35 ~ 0.35  右: -0.35 ~ 0.35")  # 左臂和右臂
            elif i == 6:  # 关节7 (对应J8)
                range_label = QLabel("左: -1.5 ~ 1.0      右: -1.0 ~ 1.5")  # 左臂和右臂
            elif i == 7:  # 夹爪 (对应J9)
                range_label = QLabel("左爪: 0 ~ 1           右爪: -1 ~ 0")  # 夹爪的活动范围

            range_label.setStyleSheet("QLabel { background-color: #f0f8ff; padding: 5px; border: 1px solid gray; }")
            grid.addWidget(range_label, i + 1, 1)

            # 角度输入框
            box = QDoubleSpinBox()
            if i == 7:  # 如果是夹爪，调整范围
                box.setRange(-1, 1.)  # 夹爪范围通常是0到1
            else:
                box.setRange(-3.14, 3.14)  # 设置范围
            box.setDecimals(2)  # 设置小数位数
            box.setValue(0.00)  # 设置默认值
            box.setSingleStep(0.05)  # 设置步长
            grid.addWidget(box, i + 1, 2)
            self.joint_boxes.append(box)

            # 位置显示标签
            label = QLabel("当前: 0.0000")
            label.setStyleSheet("color: #007acc; font-weight: bold;")
            grid.addWidget(label, i + 1, 3)
            self.joint_labels.append(label)

            # Kp 调节框
            kp_box = QDoubleSpinBox()
            kp_box.setRange(0, 1000)  # 设置范围
            kp_box.setDecimals(2)  # 设置小数位数
            kp_box.setValue(kp1_default[i])  # 设置默认值
            kp_box.setSingleStep(1)  # 设置步长
            grid.addWidget(kp_box, i + 1, 4)
            self.kp_boxes.append(kp_box)

            # Kd 调节框
            kd_box = QDoubleSpinBox()
            kd_box.setRange(0, 100)  # 设置范围
            kd_box.setDecimals(2)  # 设置小数位数
            kd_box.setValue(kd1_default[i])  # 设置默认值
            kd_box.setSingleStep(0.1)  # 设置步长
            grid.addWidget(kd_box, i + 1, 5)
            self.kd_boxes.append(kd_box)

        layout.addLayout(grid)

        # 时间设置区域
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("过渡时间 (s):"))
        self.time_box = QDoubleSpinBox()
        self.time_box.setRange(1, 10.0)  # 时间范围
        self.time_box.setValue(1.0)  # 默认值
        time_layout.addWidget(self.time_box)
        layout.addLayout(time_layout)

        # 按钮区域
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("执行点到点")  # 执行按钮
        self.btn_stop = QPushButton("停止")  # 停止按钮
        self.btn_exit = QPushButton("退出")  # 退出按钮
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_exit)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # 绑定按钮事件
        self.btn_run.clicked.connect(self.start_p2p)
        self.btn_stop.clicked.connect(self.stop_p2p)
        self.btn_exit.clicked.connect(self.close)

    # ==========================================================
    # 串口管理
    # ==========================================================
    def refresh_ports(self):
        """刷新可用串口列表"""
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self.port_combo.addItem(p.device)
        if not ports:
            self.port_combo.addItem("未找到串口")

    def connect_serial(self):
        """连接串口并初始化电机"""
        port = self.port_combo.currentText()

        try:
            # 打开串口
            self.serial_device = serial.Serial(port, 921600, timeout=0.1)
            self.controller = MotorControl(self.serial_device)
            self._init_motors()  # 初始化电机

            # 启动位置更新线程
            self.position_thread = PositionThread(self.controller, self.motors)
            self.position_thread.pos_signal.connect(self._on_pos_update)
            self.position_thread.start()

            QMessageBox.information(self, "成功", f"已连接 {port}")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def _init_motors(self):
        """初始化所有电机"""
        self.motors = [
            Motor(DM_Motor_Type.DM8009, 0x01, 0x11),  # 关节1
            Motor(DM_Motor_Type.DM8009, 0x02, 0x12),  # 关节2
            Motor(DM_Motor_Type.DM4340, 0x03, 0x13),  # 关节3
            Motor(DM_Motor_Type.DM4340, 0x04, 0x14),  # 关节4
            Motor(DM_Motor_Type.DM4310, 0x05, 0x15),  # 关节5
            Motor(DM_Motor_Type.DM4310, 0x06, 0x16),  # 关节6
            Motor(DM_Motor_Type.DM4310, 0x07, 0x17),  # 关节7
            Motor(DM_Motor_Type.DM4310, 0x08, 0x18)   # 夹爪
        ]
        # 添加并启用所有电机
        for m in self.motors:
            self.controller.addMotor(m)
            self.controller.enable(m)
            time.sleep(0.05)  # 等待电机启动

    # ==========================================================
    # 实时位置刷新
    # ==========================================================
    def _on_pos_update(self, idx, pos):
        """更新关节位置显示"""
        self.joint_labels[idx].setText(f"当前: {pos:+.4f}")

    # ==========================================================
    # 点到点控制
    # ==========================================================
    def start_p2p(self):
        """启动点到点运动"""
        if not self.controller:
            QMessageBox.warning(self, "警告", "请先连接串口！")
            return

        # 获取目标角度和运动时间
        target_angles = [b.value() for b in self.joint_boxes]
        duration = self.time_box.value()

        # 获取当前界面设置的 kp 和 kd 值
        global kp1, kd1
        kp1 = [box.value() for box in self.kp_boxes]
        kd1 = [box.value() for box in self.kd_boxes]

        # 停止当前运动（如果有）
        if self.control_thread and self.control_thread.isRunning():
            self.control_thread.stop()
            self.control_thread.wait()

        # 创建并启动新的控制线程
        self.control_thread = P2PThread(self.controller, self.motors, target_angles, duration)
        self.control_thread.update_signal.connect(self._on_p2p_update)
        self.control_thread.finished_signal.connect(lambda: QMessageBox.information(self, "完成", "运动完成"))
        self.control_thread.start()

    def _on_p2p_update(self, idx, angle):
        """更新运动过程中的角度显示"""
        self.joint_boxes[idx].setValue(round(angle, 3))

    def disconnect_serial(self):
        """断开串口连接，取消电机使能，关闭串口"""
        if self.controller and self.motors:
            # 禁用所有电机
            for m in self.motors:
                self.controller.disable(m)
            QMessageBox.information(self, "断开连接", "已取消电机使能")

        # 关闭串口
        if self.serial_device:
            self.serial_device.close()
            self.serial_device = None

        # 停止所有线程
        if self.control_thread and self.control_thread.isRunning():
            self.control_thread.stop()
            self.control_thread.wait()

        if self.position_thread and self.position_thread.isRunning():
            self.position_thread.stop()
            self.position_thread.wait()

        # 重置控制器
        self.controller = None
        self.motors = []

        QMessageBox.information(self, "断开连接", "已断开串口连接")

    def stop_p2p(self):
        """停止当前运动"""
        if self.control_thread:
            self.control_thread.stop()
            self.control_thread.wait()
            QMessageBox.information(self, "停止", "运动已停止")

    # ==========================================================
    # 退出处理
    # ==========================================================
    def closeEvent(self, event):
        """程序退出时的清理工作"""
        # 停止所有线程
        if self.control_thread and self.control_thread.isRunning():
            self.control_thread.stop()
            self.control_thread.wait()

        if self.position_thread and self.position_thread.isRunning():
            self.position_thread.stop()
            self.position_thread.wait()

        # 禁用所有电机
        if self.controller and self.motors:
            for m in self.motors:
                self.controller.disable(m)

        # 关闭串口
        if self.serial_device:
            self.serial_device.close()
        event.accept()


# ==========================================================
# 主程序入口
# ==========================================================
if __name__ == "__main__":
    # 启用高DPI支持
    from PyQt5 import QtCore

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    # 创建应用程序
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 设置界面风格

    # 创建并显示主窗口
    gui = ArmP2P()
    gui.show()

    # 启动事件循环
    sys.exit(app.exec_())