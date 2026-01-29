# ------------------------------------------------------------------------------
# 版权所有 2025 2toINF 和 HuggingFace Inc. (https://github.com/2toINF)
#
# 根据Apache许可证2.0版本授权；除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，根据许可证分发的软件是基于"按原样"分发的，
# 不附带任何形式的明示或暗示保证或条件。请参阅许可证以了解特定语言的权限和限制。
# ------------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

# =============================================================================
# 注册表
# =============================================================================
ACTION_REGISTRY: dict[str, type[BaseActionSpace]] = {}


def register_action(name: str):
    """用于注册新动作空间的装饰器。"""

    def _wrap(cls):
        key = name.lower()
        if key in ACTION_REGISTRY:
            raise KeyError(f"动作空间 '{key}' 已经注册 -> {ACTION_REGISTRY[key]}")
        ACTION_REGISTRY[key] = cls
        cls.name = key
        return cls

    return _wrap


def build_action_space(name: str, **kwargs) -> BaseActionSpace:
    """通过名称实例化一个已注册的动作空间。"""
    key = name.lower()
    if key not in ACTION_REGISTRY:
        raise KeyError(f"未知的动作空间 '{name}'。可用的有: {list(ACTION_REGISTRY.keys())}")
    return ACTION_REGISTRY[key](**kwargs)


# =============================================================================
# 基类
# =============================================================================
class BaseActionSpace(nn.Module):
    """
    所有动作空间定义的抽象基类。

    每个子类定义:
      - `dim_action`: 动作向量的维度。
      - `gripper_idx`: 夹爪通道的索引。
      - `compute_loss(pred, target)`: 此空间的监督损失。
      - `preprocess(proprio, action, mode)`: 预处理修改。
      - `postprocess(action)`: 后处理校正（例如应用sigmoid）。
    """

    name: str = "base"
    dim_action: int = 0
    gripper_idx: tuple[int, ...] = ()

    def __init__(self):
        super().__init__()

    # ---------------------------------------------------------------------
    # 核心监督损失
    # ---------------------------------------------------------------------
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """compute_loss的别名。"""
        return self.compute_loss(pred, target)

    # ---------------------------------------------------------------------
    # 空间级钩子
    # ---------------------------------------------------------------------
    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
        mode: str = "train",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """默认: 返回不变值。"""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """默认: 返回不变值。"""
        return action


# =============================================================================
# 工具函数
# =============================================================================
def _ensure_indices_valid(dim_action: int, idx: Iterable[int], name: str) -> None:
    bad = [i for i in idx if i < 0 or i >= dim_action]
    if bad:
        raise IndexError(f"{name} 包含超出范围的索引 {bad}，对于动作维度 dim_action={dim_action}")


# =============================================================================
# 实现
# =============================================================================
@register_action("bi_openarm")
class BimanualOpenarmActionSpace(BaseActionSpace):
    """
    双臂Openarm机器人：每个臂有7个关节+触发器。

    布局（真实机器人）:
    [左臂 (7个关节+触发器), 右臂 (7个关节+触发器)]
    - 左臂:  joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, trigger
    - 右臂: joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, trigger

    真实动作维度: 16
    模型面向维度: 20 (末尾额外4个虚拟维度)
    """

    # 模型输出/训练维度（与预训练策略匹配）
    dim_action = 20

    # 真实机器人动作维度
    REAL_DIM = 16

    # 真实vs虚拟通道的索引
    REAL_IDXS = tuple(range(REAL_DIM))  # 0..15
    DUMMY_IDXS = tuple(range(REAL_DIM, dim_action))  # 16..19

    # 触发器位于真实部分
    gripper_idx = (7, 15)  # 左触发器在索引7，右触发器在索引15
    GRIPPER_SCALE = 1.0
    JOINTS_SCALE = 1.0

    # 左右臂关节的索引（不包括触发器）
    LEFT_ARM_JOINTS = (0, 1, 2, 3, 4, 5, 6)
    RIGHT_ARM_JOINTS = (8, 9, 10, 11, 12, 13, 14)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    # ---------- 辅助函数 ----------

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """如果最后一个维度是REAL_DIM (16)，则填充零以达到dim_action (20)。"""
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.REAL_DIM:
            raise ValueError(
                f"期望最后一个维度为{self.REAL_DIM}或{self.dim_action}，得到{x.size(-1)}"
            )
        pad_shape = list(x.shape[:-1]) + [self.dim_action - self.REAL_DIM]
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        """只为真实机器人保留前REAL_DIM (16)个维度。"""
        return x[..., : self.REAL_DIM]

    # ---------- 损失 ----------

    def compute_loss(self, pred, target):
        """
        pred:  [B, T, 20] 来自模型
        target: [B, T, 16] 或 [B, T, 20]
        我们将目标→20填充并仅在真实维度上计算损失。
        """
        # 确保两者都是[B, T, 20]
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        assert pred.shape == target.shape

        # ---- 所有真实维度(0–15)的MSE ----
        real_dims = 16

        joints_loss = (
            self.mse(
                pred[:, :, :real_dims],
                target[:, :, :real_dims],
            )
            * self.JOINTS_SCALE
        )

        left_arm_loss = self.mse(pred[:, :, :8], target[:, :, :8])
        right_arm_loss = self.mse(pred[:, :, 8:16], target[:, :, 8:16])

        gripper_loss = (
            self.mse(
                pred[:, :, [7, 15]],
                target[:, :, [7, 15]],
            )
            * self.GRIPPER_SCALE
        )

        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
            "left_arm_loss": left_arm_loss,
            "right_arm_loss": right_arm_loss,
        }

    # ---------- 预处理/后处理 ----------

    def preprocess(self, proprio, action, mode="train"):
        """
        - 如果proprio/action是16维，则将它们填充到20供模型使用。
        - 将proprio/action中的触发器通道置零，以专注于关节学习。
        """
        proprio_m = self._pad_to_model_dim(proprio.clone())
        action_m = self._pad_to_model_dim(action.clone()) if action is not None else None

        proprio_m[..., self.gripper_idx] = 0.0
        if action_m is not None:
            action_m[..., self.gripper_idx] = 0.0

        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """
        - 模型输出[*, 20]
        - 对触发器logits应用sigmoid
        - 仅为真实机器人返回前16个维度:
          ["left_joint_1.pos",
           "left_joint_2.pos",
           "left_joint_3.pos",
           "left_joint_4.pos",
           "left_joint_5.pos",
           "left_joint_6.pos",
           "left_joint_7.pos",
           "left_trigger.pos",
           "right_joint_1.pos",
           "right_joint_2.pos",
           "right_joint_3.pos",
           "right_joint_4.pos",
           "right_joint_5.pos",
           "right_joint_6.pos",
           "right_joint_7.pos",
           "right_trigger.pos"]
        """
        # 确保我们至少有真实维度+触发器
        if action.size(-1) < self.REAL_DIM:
            raise ValueError(f"期望动作中至少有{self.REAL_DIM}个维度，得到{action.size(-1)}")

        # 在模型空间中的触发器通道上应用sigmoid（索引7和15）
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])

        # 仅为环境返回真实的16维控制向量
        return self._trim_to_real_dim(action)


@register_action("ee6d")
class EE6DActionSpace(BaseActionSpace):
    """末端执行器布局，包含xyz位置、6D旋转和夹爪通道。"""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0

    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape, "预测值/目标值形状必须匹配"
        batch_size, seq_len, action_dim = pred.shape
        _ensure_indices_valid(action_dim, self.gripper_idx, "gripper_idx")

        # 夹爪BCE
        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        # XYZ位置
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1])
            + self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE

        # 6D旋转
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1])
            + self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """将proprio/action中的夹爪通道置零。"""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """对夹爪logits应用sigmoid。"""
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("joint")
class JointActionSpace(BaseActionSpace):
    """关节空间布局，仅包含关节+夹爪。"""

    dim_action = 14
    gripper_idx = (6, 13)
    GRIPPER_SCALE = 0.1
    JOINTS_SCALE = 1.0

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape
        batch_size, seq_len, action_dim = pred.shape
        _ensure_indices_valid(action_dim, self.gripper_idx, "gripper_idx")

        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        joints_idx = tuple(i for i in range(action_dim) if i not in set(self.gripper_idx))
        joints_loss = self.mse(pred[:, :, joints_idx], target[:, :, joints_idx]) * self.JOINTS_SCALE

        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """将proprio/action中的夹爪通道置零。"""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """对夹爪logits应用sigmoid。"""
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("agibot_ee6d")
class AGIBOTEE6DActionSpace(BaseActionSpace):
    """AGI-bot变体的EE6DActionSpace，对所有组件使用MSE。"""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 10.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape
        batch_size, seq_len, action_dim = pred.shape
        _ensure_indices_valid(action_dim, self.gripper_idx, "gripper_idx")

        gripper_loss = (
            self.mse(pred[:, :, self.gripper_idx], target[:, :, self.gripper_idx]) * self.GRIPPER_SCALE
        )
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1])
            + self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1])
            + self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """在AGIBOT变体中不应用预处理。"""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """AGIBOT不进行后处理。"""
        return action


@register_action("franka_joint7")
class FrankaJoint7ActionSpace(BaseActionSpace):
    """
    Franka Panda关节空间：7个关节，带夹爪。

    - 真实机器人动作维度：7
    - 模型面向维度：20（用零填充）
      与预期20维的预训练VLA模型兼容。
    """

    dim_action = 20  # 模型维度
    REAL_DIM = 7  # 实际Franka关节

    JOINTS_SCALE = 1.0

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """将7 → 20维度填充（虚拟通道用零）。"""
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.REAL_DIM:
            raise ValueError(
                f"期望最后一个维度为{self.REAL_DIM}或{self.dim_action}，得到{x.size(-1)}"
            )

        pad_shape = list(x.shape[:-1]) + [self.dim_action - self.REAL_DIM]  # 13个零
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        """修剪模型输出20 → 7维度。"""
        return x[..., : self.REAL_DIM]

    def compute_loss(self, pred, target):
        """
        pred :  [B, T, 20]
        target : [B, T, 7] 或 [B, T, 20]

        只计算前7个维度的MSE。
        """
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)

        assert pred.shape == target.shape

        joints_loss = (
            self.mse(
                pred[:, :, : self.REAL_DIM],  # 只使用前7个关节
                target[:, :, : self.REAL_DIM],
            )
            * self.JOINTS_SCALE
        )

        return {"joints_loss": joints_loss}

    def preprocess(self, proprio, action, mode="train"):
        """
        训练期间:
        - 填充 [7] → [20]
        """
        return proprio, self._pad_to_model_dim(action)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """
        模型预测后:
        - 修剪 [20] → [7] 用于真实机器人控制。
        """
        return self._trim_to_real_dim(action)


@register_action("auto")
class AutoActionSpace(BaseActionSpace):
    """
    自动检测动作空间，适应任何动作维度。

    - 自动从策略特征中检测真实动作维度
    - 模型输出max_dim以与预训练模型兼容
    - 损失仅在前real_dim维度上计算
    - 后处理将输出修剪回real_dim

    参数:
        real_dim: 来自数据集/策略特征的实际动作维度
        max_dim: 模型的输出维度，用于与预训练VLA兼容
    """

    JOINTS_SCALE = 1.0

    def __init__(self, real_dim: int, max_dim: int):
        super().__init__()
        self.real_dim = real_dim
        self.dim_action = max_dim  # 模型面向维度
        self.mse = nn.MSELoss()

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """将real_dim → max_dim填充（虚拟通道用零）。"""
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.real_dim:
            # 如果维度都不匹配，则先填充/修剪到real_dim
            if x.size(-1) < self.real_dim:
                pad_shape = list(x.shape[:-1]) + [self.real_dim - x.size(-1)]
                pad = x.new_zeros(pad_shape)
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[..., : self.real_dim]

        pad_shape = list(x.shape[:-1]) + [self.dim_action - self.real_dim]
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        """修剪模型输出max_dim → real_dim。"""
        return x[..., : self.real_dim]

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        仅在前real_dim维度上计算损失。

        pred:   [B, T, max_dim] 来自模型
        target: [B, T, real_dim] 或 [B, T, max_dim]

        损失 = MSE(pred[:,:,:real_dim], target[:,:,:real_dim])
        """
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        assert pred.shape == target.shape, f"形状不匹配: pred {pred.shape} vs target {target.shape}"

        # 仅在真实维度上计算损失
        joints_loss = (
            self.mse(
                pred[:, :, : self.real_dim],
                target[:, :, : self.real_dim],
            )
            * self.JOINTS_SCALE
        )

        return {"joints_loss": joints_loss}

    def preprocess(self, proprio: torch.Tensor, action: torch.Tensor, mode: str = "train"):
        """
        将动作从real_dim填充到max_dim供模型使用。
        """
        return proprio, self._pad_to_model_dim(action)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """
        将模型输出从max_dim修剪到real_dim用于真实机器人控制。
        """
        return self._trim_to_real_dim(action)


@register_action("so101_bimanual")
class BimanualSO101ActionSpace(BaseActionSpace):
    """
    双臂SO101机器人：每个臂有5个关节+夹爪。

    布局（真实机器人）:
    [左臂 (5个关节+夹爪), 右臂 (5个关节+夹爪)]
    - 左臂:  肩部旋转, 肩部提升, 肘部弯曲, 腕部弯曲, 腕部旋转, 夹爪
    - 右臂: 肩部旋转, 肩部提升, 肘部弯曲, 腕部弯曲, 腕部旋转, 夹爪

    真实动作维度: 12
    模型面向维度: 20 (末尾额外8个虚拟维度)
    """

    # 模型输出/训练维度（与预训练策略匹配）
    dim_action = 20

    # 真实机器人动作维度
    REAL_DIM = 12

    # 真实vs虚拟通道的索引
    REAL_IDXS = tuple(range(REAL_DIM))  # 0..11
    DUMMY_IDXS = tuple(range(REAL_DIM, dim_action))  # 12..19

    # 夹爪位于真实部分
    gripper_idx = (5, 11)  # 左夹爪在索引5，右夹爪在索引11
    GRIPPER_SCALE = 1.0
    JOINTS_SCALE = 1.0

    # 左右臂关节的索引（不包括夹爪）
    LEFT_ARM_JOINTS = (0, 1, 2, 3, 4)
    RIGHT_ARM_JOINTS = (6, 7, 8, 9, 10)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    # ---------- 辅助函数 ----------

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """如果最后一个维度是REAL_DIM (12)，则填充零以达到dim_action (20)。"""
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.REAL_DIM:
            raise ValueError(
                f"期望最后一个维度为{self.REAL_DIM}或{self.dim_action}，得到{x.size(-1)}"
            )
        pad_shape = list(x.shape[:-1]) + [self.dim_action - self.REAL_DIM]
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        """只为真实机器人保留前REAL_DIM (12)个维度。"""
        return x[..., : self.REAL_DIM]

    # ---------- 损失 ----------

    def compute_loss(self, pred, target):
        """
        pred:  [B, T, 20] 来自模型
        target: [B, T, 12] 或 [B, T, 20]
        我们将目标→20填充并仅在真实维度上计算损失。
        """
        # 确保两者都是[B, T, 20]
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        assert pred.shape == target.shape

        # ---- 所有真实维度(0–11)的MSE ----
        real_dims = 12

        joints_loss = (
            self.mse(
                pred[:, :, :real_dims],
                target[:, :, :real_dims],
            )
            * self.JOINTS_SCALE
        )

        left_arm_loss = self.mse(pred[:, :, :6], target[:, :, :6])
        right_arm_loss = self.mse(pred[:, :, 6:12], target[:, :, 6:12])

        gripper_loss = (
            self.mse(
                pred[:, :, [5, 11]],
                target[:, :, [5, 11]],
            )
            * self.GRIPPER_SCALE
        )

        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
            "left_arm_loss": left_arm_loss,
            "right_arm_loss": right_arm_loss,
        }

    # ---------- 预处理/后处理 ----------

    def preprocess(self, proprio, action, mode="train"):
        """
        - 如果proprio/action是12维，则将它们填充到20供模型使用。
        - 将proprio/action中的夹爪通道置零，以专注于关节学习。
        """
        proprio_m = self._pad_to_model_dim(proprio.clone())
        action_m = self._pad_to_model_dim(action.clone()) if action is not None else None

        proprio_m[..., self.gripper_idx] = 0.0
        if action_m is not None:
            action_m[..., self.gripper_idx] = 0.0

        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """
        - 模型输出[*, 20]
        - 对夹爪logits应用sigmoid
        - 仅为真实机器人返回前12个维度:
          ["left_shoulder_pan.pos",
           "left_shoulder_lift.pos",
           "left_elbow_flex.pos",
           "left_wrist_flex.pos",
           "left_wrist_roll.pos",
           "left_gripper.pos",
           "right_shoulder_pan.pos",
           "right_shoulder_lift.pos",
           "right_elbow_flex.pos",
           "right_wrist_flex.pos",
           "right_wrist_roll.pos",
           "right_gripper.pos"]
        """
        # 确保我们至少有真实维度+夹爪
        if action.size(-1) < self.REAL_DIM:
            raise ValueError(f"期望动作中至少有{self.REAL_DIM}个维度，得到{action.size(-1)}")

        # 在模型空间中的夹爪通道上应用sigmoid（索引5和11）
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])

        # 仅为环境返回真实的12维控制向量
        return self._trim_to_real_dim(action)


# =============================================================================
# 导出
# =============================================================================
__all__ = [
    "BaseActionSpace",
    "build_action_space",
    "register_action",
    "EE6DActionSpace",
    "JointActionSpace",
    "AGIBOTEE6DActionSpace",
    "FrankaJoint7ActionSpace",
    "AutoActionSpace",
    "BimanualSO101ActionSpace",
    "BimanualOpenarmActionSpace",
    "ACTION_REGISTRY",
]
