#!/usr/bin/env python3
"""
Arm Visualizer 
直接接收关节角度数据并更新MuJoCo仿真可视化
"""

import os
import time
import mujoco
from mujoco import viewer


class ArmVisualizer:
    """机械臂可视化类 - 去除ROS2依赖，简化实现"""

    def __init__(self):
        mjcf_path = os.path.join(os.path.dirname(__file__), 'v1', 'openarm_bimanual.xml')

        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        # 启动可视化查看器
        self.viewer = viewer.launch_passive(self.model, self.data)
        
        print("=== Arm Visualizer Initialized ===")

    def update_joint_angles_from_action_dict(self, action_dict):
        """从动作字典更新关节角度并立即刷新视图"""
        joint_angles = list(action_dict.values())

        for i in range(7):
            self.data.qpos[i] = joint_angles[i]
        # 更新左臂爪子角度 (索引 7-8)，将0-1范围映射到0-0.044米(0-44毫米)
        left_gripper_pos = joint_angles[7] * 0.044  # 将0-1映射到0-0.044米
        self.data.qpos[7] = left_gripper_pos
        self.data.qpos[8] = left_gripper_pos
            # 更新右臂关节角度 (索引 9-15)
        for i in range(7):
            self.data.qpos[i + 9] = joint_angles[i+8]
        # 更新右臂爪子角度 (索引 16-17)，将-1 到0 范围映射到0-0.044米(0-44毫米)
        right_gripper_pos = joint_angles[15] * -0.044
        self.data.qpos[16] = right_gripper_pos
        self.data.qpos[17] = right_gripper_pos

        # 执行前向动力学计算
        mujoco.mj_forward(self.model, self.data)
        
        # 同步viewer
        with self.viewer.lock():
            self.viewer.sync()

    def update_joint_angles_from_action_list(self, joint_angles):
        """从动作字典更新关节角度并立即刷新视图"""

        for i in range(7):
            self.data.qpos[i] = joint_angles[i]
        # 更新左臂爪子角度 (索引 7-8)，将0-1范围映射到0-0.044米(0-44毫米)
        left_gripper_pos = joint_angles[7] * 0.044  # 将0-1映射到0-0.044米
        self.data.qpos[7] = left_gripper_pos
        self.data.qpos[8] = left_gripper_pos
        # 更新右臂关节角度 (索引 9-15)
        for i in range(7):
            self.data.qpos[i + 9] = joint_angles[i + 8]
        # 更新右臂爪子角度 (索引 16-17)，将-1 到0 范围映射到0-0.044米(0-44毫米)
        right_gripper_pos = joint_angles[15] * -0.044
        self.data.qpos[16] = right_gripper_pos
        self.data.qpos[17] = right_gripper_pos

        # 执行前向动力学计算
        mujoco.mj_forward(self.model, self.data)

        # 同步viewer
        with self.viewer.lock():
            self.viewer.sync()

if __name__ == '__main__':
    visualizer = ArmVisualizer()
    # 模拟一些关节角度数据
    import math

    try:
        for t in range(100):
            # 模拟动作字典
            action_dict = {}
            for i in range(16):  # 现在有16个关节：左臂7个关节+1个夹爪，右臂7个关节+1个夹爪
                joint_name = f"joint_{i + 1}"
                action_dict[f"{joint_name}.pos"] = math.sin(t * 0.1 + i)
            # 特别设置夹爪关节，使其在0到1之间变化
            action_dict["joint_8.pos"] = abs(math.sin(t * 0.1 + 7))  # 左臂夹爪
            action_dict["joint_16.pos"] = abs(math.sin(t * 0.1 + 15))  # 右臂夹爪
            visualizer.update_joint_angles_from_action_dict(action_dict)
            time.sleep(0.1)  # 模拟控制循环
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        visualizer.viewer.close()
