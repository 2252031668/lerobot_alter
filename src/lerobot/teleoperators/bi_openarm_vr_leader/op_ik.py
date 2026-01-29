"""Utilities for differential IK."""
import numpy as np
import pinocchio


def check_within_limits(model, q):
    return np.all(q >= model.lowerPositionLimit) and np.all(
        q <= model.upperPositionLimit
    )
def get_random_state(model, padding=0.0):
    return np.random.uniform(
        model.lowerPositionLimit + padding, model.upperPositionLimit - padding
    )
def joint_limit_nullspace_component(model, q, gain=1.0, padding=0.0):
    """
    Returns a joint limits avoidance nullspace component.
    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        q : array-like
            The joint configuration for the model.
        gain : float, optional
            A gain to modify the relative weight of this term.
        padding : float, optional
            Optional padding around the joint limits.

    Returns
    -------
        array-like
            An array containing the joint space avoidance nullspace terms.
    """
    upper_limits = model.upperPositionLimit - padding
    lower_limits = model.lowerPositionLimit + padding

    grad = np.zeros_like(model.lowerPositionLimit)
    for idx in range(len(grad)):
        if q[idx] > upper_limits[idx]:
            grad[idx] = -gain * (q[idx] - upper_limits[idx])
        elif q[idx] < lower_limits[idx]:
            grad[idx] = -gain * (q[idx] - lower_limits[idx])
    return grad

class DifferentialIkOptions:
    """Options for differential IK."""

    def __init__(
            self,
            max_iters=200,
            max_retries=10,
            max_translation_error=1e-3,
            max_rotation_error=1e-3,
            damping=1e-3,
            min_step_size=0.1,
            max_step_size=0.5,
            ignore_joint_indices=[],
            joint_weights=None,
            rng_seed=None,
            translation_error_threshold=None,
            rotation_error_threshold=None,
    ):
        """
        Initializes a set of differential IK options.

        Parameters
        ----------
            max_iters : int
                Maximum number of iterations per try.
            max_retries : int
                Maximum number of retries with random restarts.
                If set to 0, only the initial state provided will be used.
            max_translation_error : float
                Maximum translation error, in meters, to consider IK solved.
            max_rotation_error : float
                Maximum rotation error, in radians, to consider IK solved.
            damping : float
                Damping value, between 0 and 1, for the Jacobian pseudoinverse.
                Setting this to a nonzero value is using Levenberg-Marquardt.
            min_step_size : float
                Minimum gradient step size, between 0 and 1, based on ratio of current distance to target to initial distance to target.
                To use a fixed step size, set both minimum and maximum values to be equal.
            max_step_size : float
                Maximum gradient step size, between 0 and 1, based on ratio of current distance to target to initial distance to target.
                To use a fixed step size, set both minimum and maximum values to be equal.
            joint_weights : list[float], optional
                A list of relative weights for different joints, used in computing the Jacobian pseudoinverse.
                If your robot has redundant joints, assigning a higher weight to some joints will cause them to move less than lower weight joints.
                If not specified, all joints are weighted equally with unit weight.
            ignore_joint_indices : list[int], optional
                A list of joints to ignore changing when solving IK.
            rng_seed : int, optional
                Sets the seed for random number generation. Use to generate deterministic results.
            translation_error_threshold : float, optional
                Translation error threshold for accepting a suboptimal solution when max iterations reached.
            rotation_error_threshold : float, optional
                Rotation error threshold for accepting a suboptimal solution when max iterations reached.
        """
        self.max_iters = max_iters
        self.max_retries = max_retries
        self.max_translation_error = max_translation_error
        self.max_rotation_error = max_rotation_error
        self.damping = damping
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.joint_weights = joint_weights
        self.ignore_joint_indices = ignore_joint_indices
        self.rng_seed = rng_seed
        self.translation_error_threshold = translation_error_threshold
        self.rotation_error_threshold = rotation_error_threshold


class DifferentialIk:
    """
    Differential IK solver.

    This is a numerical IK solver that uses the manipulator's Jacobian to take first-order steps towards a solution.
    It contains several of the common options such as damped least squares (Levenberg-Marquardt), random restarts, and nullspace projection.

    Some good resources:
      * https://motion.cs.illinois.edu/RoboticSystems/InverseKinematics.html
      * https://homes.cs.washington.edu/~todorov/courses/cseP590/06_JacobianMethods.pdf
      * https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
      * http://www.diag.uniroma1.it/deluca/rob2_en/02_KinematicRedundancy_1.pdf
    """

    def __init__(
            self,
            model,
            collision_model=None,
            data=None,
            options=DifferentialIkOptions(),
    ):
        """
        Creates an instance of a DifferentialIk solver.

        Parameters
        ----------
            model : `pinocchio.Model`
                The model to use for this solver.
            collision_model : `pinocchio.Model`, optional
                The model to use for collision checking. If None, no collision checking takes place.
            data : `pinocchio.Data`, optional
                The model data to use for this solver. If None, data is created automatically.
            collision_data : `pinocchio.GeometryData`, optional
                The collision_model data to use for this solver. If None, data is created automatically.
            visualizer : `pinocchio.visualize.meshcat_visualizer.MeshcatVisualizer`, optional
                The visualizer to use for this solver.
            options : `DifferentialIkOptions`, optional
                The options to use for solving IK. If not specified, default options are used.
        """
        self.model = model
        self.collision_model = collision_model

        if not data:
            data = model.createData()
        self.data = data
        self.options = options

    def solve(
            self,
            target_frame,
            target_tform,
            init_state,
            nullspace_components=[],
            verbose=False,
    ):
        """
        Solves an IK query.

        Parameters
        ----------
            target_frame : str
                The name of the target frame in the model.
            target_tform : `pinocchio.SE3`
                The desired transformation of the target frame in the model.
            init_state : array-like,
                The initial state to solve from. If not specified, a random initial state will be selected.
            nullspace_components : list[function], optional
                An optional list of nullspace components to use when solving.
                These components must take the form `lambda model, q: component(model, q, <other_args>)`.
            verbose : bool, optional
                If True, prints additional information to the console.

        Returns
        -------
            array-like or None
                A list of joint configuration values with the solution, if one was found. Otherwise, returns None.
        """
        np.random.seed(self.options.rng_seed)
        target_frame_id = self.model.getFrameId(target_frame)

        # Get the active joint indices.
        active_joint_indices = [
            idx
            for idx in range(self.model.nq)
            if idx not in self.options.ignore_joint_indices
        ]
        num_active_joints = len(active_joint_indices)

        # Create the joint weights.
        if self.options.joint_weights is None:
            # Use identity weights if they are not specified.
            W = np.eye(num_active_joints)
        elif len(self.options.joint_weights) != num_active_joints:
            raise ValueError(
                f"Joint weights, if specified, must have {num_active_joints} elements."
            )
        elif np.any(np.array(self.options.joint_weights) <= 0.0):
            raise ValueError(f"All joint weights must be strictly positive.")
        else:
            # Invert the weights so that higher weight means less joint motion.
            W = np.linalg.inv(np.diag(self.options.joint_weights))

        # Initialize IK
        solved = False
        n_tries = 0
        q_cur = init_state
        initial_error_norm = None

        # Track the best solution found so far
        best_solution = None
        best_translation_error = float('inf')
        best_rotation_error = float('inf')

        while n_tries <= self.options.max_retries:
            # Reset best solution for this try
            best_solution = q_cur.copy()
            best_translation_error = float('inf')
            best_rotation_error = float('inf')

            n_iters = 0
            while n_iters < self.options.max_iters:
                # Compute forward kinematics at the current state
                pinocchio.framesForwardKinematics(self.model, self.data, q_cur)
                cur_tform = self.data.oMf[target_frame_id]

                # Check the error using actInv
                error = target_tform.actInv(cur_tform)
                error = -pinocchio.log(error).vector

                # Extract translation and rotation errors
                current_translation_error = np.linalg.norm(error[:3])
                current_rotation_error = np.linalg.norm(error[3:])

                # Update best solution if current is better
                if (current_translation_error < best_translation_error and
                    current_rotation_error < best_rotation_error):
                    best_solution = q_cur.copy()
                    best_translation_error = current_translation_error
                    best_rotation_error = current_rotation_error

                if (
                        current_translation_error < self.options.max_translation_error
                        and current_rotation_error < self.options.max_rotation_error
                ):
                    # Wrap to the range -/+ pi, and then check joint limits and collision.
                    q_cur = (q_cur + np.pi) % (2 * np.pi) - np.pi
                    if check_within_limits(self.model, q_cur):
                        solved = True
                        if verbose:
                            print("Solved within joint limits!")

                    else:
                        if verbose:
                            print("Solved, but outside joint limits.")
                    break

                # Calculate the Jacobian for the active joints.
                J = pinocchio.computeFrameJacobian(
                    self.model,
                    self.data,
                    q_cur,
                    target_frame_id,
                    pinocchio.ReferenceFrame.LOCAL,
                )[:, active_joint_indices]

                # Compute the (optionally damped and weighted) Jacobian pseudoinverse.
                jjt = (J @ W @ J.T) + self.options.damping ** 2 * np.eye(6)

                # Compute the gradient descent step size.
                error_norm = np.linalg.norm(error)
                if initial_error_norm is None:
                    initial_error_norm = error_norm
                alpha = self.options.min_step_size + (
                        1.0 - error_norm / initial_error_norm
                ) * (self.options.max_step_size - self.options.min_step_size)

                # Gradient descent step
                if not nullspace_components:
                    q_step = alpha * W @ J.T @ np.linalg.solve(jjt, error)
                else:
                    nullspace_term = sum(
                        [
                            comp(self.model, q_cur)[active_joint_indices]
                            for comp in nullspace_components
                        ]
                    )
                    q_step = alpha * (
                            W @ J.T @ (np.linalg.solve(jjt, error - J @ (nullspace_term)))
                            + nullspace_term
                    )

                # Zero out the values for the ignored indices before returning.
                for q, idx in zip(q_step, active_joint_indices):
                    q_cur[idx] += q

                # Clip joint values to respect position limits after each gradient step
                q_cur = np.clip(q_cur, self.model.lowerPositionLimit, self.model.upperPositionLimit)

                n_iters += 1

                # Protect against numerical instability.
                if np.any(np.isinf(q_cur)):
                    print(f"Terminating due to numerical instability.")
                    break

            # Check results at the end of this try
            if solved:
                if verbose:
                    print(f"Solved in {n_tries + 1} tries.")
                break
            else:
                # Check if the best solution of this try meets the threshold criteria
                meets_translation_threshold = (
                    self.options.translation_error_threshold is not None and
                    best_translation_error <= self.options.translation_error_threshold
                )
                meets_rotation_threshold = (
                    self.options.rotation_error_threshold is not None and
                    best_rotation_error <= self.options.rotation_error_threshold
                )

                # If both thresholds are specified and met, return the best solution
                if (self.options.translation_error_threshold is not None and
                    self.options.rotation_error_threshold is not None and
                    meets_translation_threshold and meets_rotation_threshold):
                    if verbose:
                        print(f"Returning suboptimal solution with translation error {best_translation_error:.6f} "
                              f"(threshold {self.options.translation_error_threshold:.6f}) and "
                              f"rotation error {best_rotation_error:.6f} "
                              f"(threshold {self.options.rotation_error_threshold:.6f})")
                    return best_solution

                # If only translation threshold is specified and met, return the best solution
                elif (self.options.translation_error_threshold is not None and
                      self.options.rotation_error_threshold is None and
                      meets_translation_threshold):
                    if verbose:
                        print(f"Returning suboptimal solution with translation error {best_translation_error:.6f} "
                              f"(threshold {self.options.translation_error_threshold:.6f})")
                    return best_solution

                # If only rotation threshold is specified and met, return the best solution
                elif (self.options.rotation_error_threshold is not None and
                      self.options.translation_error_threshold is None and
                      meets_rotation_threshold):
                    if verbose:
                        print(f"Returning suboptimal solution with rotation error {best_rotation_error:.6f} "
                              f"(threshold {self.options.rotation_error_threshold:.6f})")
                    return best_solution

                # If thresholds are specified but not met, continue with retry
                elif (self.options.translation_error_threshold is not None or
                      self.options.rotation_error_threshold is not None):
                    if verbose:
                        print(f"Best solution did not meet threshold criteria. Translation error: {best_translation_error:.6f}, "
                              f"Rotation error: {best_rotation_error:.6f}")

                # Generate new random state for next try
                q_cur = get_random_state(self.model)
                # Ensure the new initial state respects joint limits
                q_cur = np.clip(q_cur, self.model.lowerPositionLimit, self.model.upperPositionLimit)
                n_tries += 1
                if verbose:
                    print(f"Retry {n_tries}")

        # Check final results
        if solved:
            return q_cur
        else:
            # If max retries reached and thresholds are enabled, return the best solution found
            if (self.options.translation_error_threshold is not None or
                self.options.rotation_error_threshold is not None):
                meets_translation_threshold = (
                    self.options.translation_error_threshold is not None and
                    best_translation_error <= self.options.translation_error_threshold
                )
                meets_rotation_threshold = (
                    self.options.rotation_error_threshold is not None and
                    best_rotation_error <= self.options.rotation_error_threshold
                )

                # Check if both thresholds are met (if both are specified)
                if (self.options.translation_error_threshold is not None and
                    self.options.rotation_error_threshold is not None and
                    meets_translation_threshold and meets_rotation_threshold):
                    if verbose:
                        print(f"Max retries reached, returning best solution with translation error {best_translation_error:.6f} "
                              f"(threshold {self.options.translation_error_threshold:.6f}) and "
                              f"rotation error {best_rotation_error:.6f} "
                              f"(threshold {self.options.rotation_error_threshold:.6f})")
                    return best_solution
                # Check if only translation threshold is met (if only translation threshold is specified)
                elif (self.options.translation_error_threshold is not None and
                      self.options.rotation_error_threshold is None and
                      meets_translation_threshold):
                    if verbose:
                        print(f"Max retries reached, returning best solution with translation error {best_translation_error:.6f} "
                              f"(threshold {self.options.translation_error_threshold:.6f})")
                    return best_solution
                # Check if only rotation threshold is met (if only rotation threshold is specified)
                elif (self.options.rotation_error_threshold is not None and
                      self.options.translation_error_threshold is None and
                      meets_rotation_threshold):
                    if verbose:
                        print(f"Max retries reached, returning best solution with rotation error {best_rotation_error:.6f} "
                              f"(threshold {self.options.rotation_error_threshold:.6f})")
                    return best_solution
                else:
                    if verbose:
                        print("Max retries reached and best solution does not meet threshold criteria.")
                    return None
            else:
                return None




class PinocchioIK:
    """
    基于Pinocchio库的单臂逆运动学类
    用于加载双臂机械臂模型，但选择一手臂提供IK/FK功能
    """

    def __init__(self, mjcf_path="v1/openarm_bimanual.xml", arm_type="left", joint_limits=None):
        """
        初始化单臂IK类，xml文件是双臂，但是选择一侧手臂初始化
        :param mjcf_path: MJCF文件路径
        :param arm_type: 选择哪只手臂 ('left' 或 'right')
        :param joint_limits: 关节限制列表
        """
        self.model, _, _ = pinocchio.buildModelsFromMJCF(mjcf_path)
        self.data = pinocchio.Data(self.model)  # 添加数据结构

        self.joint_indices_in_nq = []  # 存储单个臂关节ID  共7个关节
        self.ignore_joint_indices_in_nq = []  # 存储忽略的关节ID
        self.end_effector_frame = None  # 单臂末端执行器名字
        self.end_effector_id = None  # 单臂末端执行器序号

        # 加载机器人模型
        self._load_arm(arm_type)

        # 初始化关节角度向量
        self.q = pinocchio.neutral(self.model)

        # 设置关节限位置
        self.set_joint_limits(joint_limits)
        options = DifferentialIkOptions(
            max_iters=800,  # 最大迭代次数（迭代求解的终止条件）
            max_retries=0,  # 求解失败时重试次数  #不要重新尝试，没有意义
            max_translation_error=0.001, #精准解范围
            max_rotation_error=0.02,
            translation_error_threshold=0.05,  #次优解范围
            rotation_error_threshold=0.5,
            damping=0.0001,  # 阻尼因子（避免雅可比矩阵奇异）
            min_step_size=0.01,  # 关节角度最小步长
            max_step_size=0.2,  # 关节角度最大步长
            ignore_joint_indices=self.ignore_joint_indices_in_nq,  # 忽略的关节
            joint_weights=[1.5, 1.5, 1.5, 1.0, 0.7, 0.7, 0.7] #权重最低（移动最多）
        )
        self.differential_ik = DifferentialIk(
            self.model,  # mujoco/pyroboplan机器人模型
            data=self.data,  # 模型数据（存储关节/末端状态）
            options=options,  # IK求解参数配置
        )
        self.nullspace_components = [#当逆运动学求解器在迭代过程中接近关节极限时，这个零空间分量会在不影响末端执行器轨迹的前提下，调整关节角度使其远离极限位置
            lambda model, q: joint_limit_nullspace_component(
                model, q, gain=1.0, padding=0.05
            ),
        ]

    def print_model_info(self):
        """
        打印模型的详细信息，包括关节、连杆、帧等
        """
        print("=" * 60)
        print("ROBOT MODEL INFORMATION")
        print("=" * 60)

        # 基本信息
        print(f"Number of joints: {self.model.njoints}")
        print(f"Number of degrees of freedom (nq): {self.model.nq}")
        print(f"Number of velocities (nv): {self.model.nv}")
        print()

        # 关节信息
        print("JOINT INFORMATION:")
        print("-" * 30)
        for i in range(len(self.model.names)):
            joint_name = self.model.names[i]
            joint_id = i
            # 获取关节类型
            joint_type = self.model.joints[joint_id].shortname()

            # 获取关节限位信息
            if joint_id < len(self.model.lowerPositionLimit) and joint_id < len(self.model.upperPositionLimit):
                lower_limit = self.model.lowerPositionLimit[joint_id]
                upper_limit = self.model.upperPositionLimit[joint_id]
                joint_range_info = f"Range: [{lower_limit:.3f}, {upper_limit:.3f}]"
            else:
                joint_range_info = "Range: N/A"

            print(f"ID {joint_id:2d}: {joint_name:<30} Type: {joint_type:<15} {joint_range_info})")
        print()
        # 连杆(Frame)信息
        print("FRAME INFORMATION:")
        print("-" * 30)
        print(f"Total frames: {len(self.model.frames)}")
        for i, frame in enumerate(self.model.frames):
            print(f"Frame ID {i:3d}: {frame.name:<30} Type: {frame.type.name:<10} Parent Joint: {frame.parentJoint}")

    def _load_arm(self, arm_type):

        if arm_type == "left":
            # 左臂关节ID (不包括手指关节) - 根据打印的信息确定
            # 根据实际的关节名称查找在nq空间中的索引
            left_joint_names = ['openarm_left_joint1', 'openarm_left_joint2', 'openarm_left_joint3',
                                'openarm_left_joint4', 'openarm_left_joint5', 'openarm_left_joint6',
                                'openarm_left_joint7']
            for joint_name in left_joint_names:
                joint_id = self.model.getJointId(joint_name)
                # 需要找到这个关节在q向量中的位置
                if joint_id < self.model.njoints:
                    # 获取关节在q向量中的起始索引
                    joint_idx_in_q = self.model.joints[joint_id].idx_q
                    if joint_idx_in_q < self.model.nq:  # 确保不超过q向量的大小
                        self.joint_indices_in_nq.append(joint_idx_in_q)

            # 左臂末端执行器TCP frame
            self.end_effector_frame = "openarm_left_hand_tcp"  # openarm_left_hand_tcp
            self.end_effector_id = self.model.getFrameId(self.end_effector_frame)

        elif arm_type == "right":
            # 右臂关节ID (不包括手指关节) - 根据打印的信息确定
            # 根据实际的关节名称查找在nq空间中的索引
            right_joint_names = ['openarm_right_joint1', 'openarm_right_joint2', 'openarm_right_joint3',
                                 'openarm_right_joint4', 'openarm_right_joint5', 'openarm_right_joint6',
                                 'openarm_right_joint7']
            for joint_name in right_joint_names:
                joint_id = self.model.getJointId(joint_name)
                # 需要找到这个关节在q向量中的位置
                if joint_id < self.model.njoints:
                    # 获取关节在q向量中的起始索引
                    joint_idx_in_q = self.model.joints[joint_id].idx_q
                    if joint_idx_in_q < self.model.nq:  # 确保不超过q向量的大小
                        self.joint_indices_in_nq.append(joint_idx_in_q)

            # 右臂末端执行器TCP frame
            self.end_effector_frame = "openarm_right_hand_tcp"  # openarm_right_hand_tcp
            self.end_effector_id = self.model.getFrameId(self.end_effector_frame)

        # 确保末端执行器ID存在
        if self.end_effector_id >= len(self.model.frames):
            raise ValueError(f"End effector frame '{self.end_effector_frame}' not found in model")

        # 初始化忽略的关节索引列表，包含除了当前臂之外的所有其他关节
        for i in range(self.model.nq):
            if i not in self.joint_indices_in_nq:
                self.ignore_joint_indices_in_nq.append(i)

    def set_joint_limits(self, joint_limits):
        """
        设置关节限位
        传的格式类似
        joint_limits = [
            [-3.0, 1.2], [-1.5, 0.01], [-1.52, 1.56], [-0.01, 2.3],
            [-1.55, 1.55], [-0.5, 0.35], [-0.9, 0.9]
        ]
        """
        if joint_limits is None:
            # 如果没有提供关节限制，则使用默认的模型限制
            return

        if len(joint_limits) != 7:
            raise ValueError(f"Expected 7 joint limits for 7-DOF arm, got {len(joint_limits)}")

        # 获取当前的关节限位
        lower_limits = self.model.lowerPositionLimit.copy()
        upper_limits = self.model.upperPositionLimit.copy()

        # 根据手臂类型设置相应的关节限位
        for i, (lower, upper) in enumerate(joint_limits):
            joint_idx_in_q = self.joint_indices_in_nq[i]  # 使用nq空间中的索引
            lower_limits[joint_idx_in_q] = lower
            upper_limits[joint_idx_in_q] = upper

        # 更新模型的关节限位
        self.model.lowerPositionLimit = lower_limits
        self.model.upperPositionLimit = upper_limits

    def fk(self, joint_angles):
        """
        计算单臂的FK，返回末端执行器的位置和旋转矩阵
        :param joint_angles: 单个手臂关节角度列表，7个关节角度
        :return: 4x4 齐次变换矩阵
        """

        # 将输入的关节角度更新到对应的nq空间中的关节索引
        for i, joint_idx_in_q in enumerate(self.joint_indices_in_nq):
            if i < len(joint_angles):
                self.q[joint_idx_in_q] = joint_angles[i]

        # 计算正运动学
        pinocchio.forwardKinematics(self.model, self.data, self.q)
        pinocchio.updateFramePlacements(self.model, self.data)

        # 获取末端执行器的姿态
        end_effector_pose = self.data.oMf[self.end_effector_id]  # oMf是相对于世界坐标系的位姿
        # target_pos = end_effector_pose.translation
        # target_rot = end_effector_pose.rotation
        # 返回
        return end_effector_pose

    def ik(self, target_position, target_orientation, now_joints, max_iter=1000, tolerance=1e-4):
        """
        逆运动学计算 - 参考官方示例改进版本
        :param target_position: 末端的目标位置 [x, y, z]
        :param target_orientation: 末端的目标方向矩阵 (3x3)
        :param now_joints: 当前的关节角度
        :param max_iter: 最大迭代次数
        :param tolerance: 收敛容差
        :return: 关节角度列表 7个关节
        """

        # 将当前的关节角度更新到对应的nq空间中的关节索引
        for i, joint_idx_in_q in enumerate(self.joint_indices_in_nq):
            if i < len(now_joints):
                self.q[joint_idx_in_q] = now_joints[i]
        # 构建目标变换矩阵
        target_transform = pinocchio.SE3(target_orientation, target_position)

        q_sol = self.differential_ik.solve(
            self.end_effector_frame,  # 目标帧：hand（末端执行器）
            target_transform,  # 目标位姿（pinocchio.SE3格式）
            init_state=self.q,  # 迭代初始值
            nullspace_components=self.nullspace_components,
            verbose=False,  # 打印求解过程
        )
        if q_sol is not None:
            # 提取手臂关节的角度
            result_joints = []
            for joint_idx_in_q in self.joint_indices_in_nq:
                result_joints.append(q_sol[joint_idx_in_q])
            return result_joints
        else:
            return None




if __name__ == '__main__':
    # 测试代码

    # 定义关节限位 (示例值)
    # 关节限制
    left_joint_limits = [
        [-3.0, 1.2], [-1.5, 0.01], [-1.52, 1.56], [-0.01, 2.3],
        [-1.55, 1.55], [-0.5, 0.35], [-0.9, 0.9]
    ]
    right_joint_limits = [
        [-1.2, 3.0], [-0.01, 1.5], [-1.51, 1.59], [-2.3, 0.01],
        [-1.6, 1.6], [-0.4, 0.4], [-1.0, 1.0]
    ]

    # 创建IK对象 - 测试左臂
    print("Testing Left Arm:")
    ik_solver_left = PinocchioIK("v1/openarm_bimanual.xml", "left", left_joint_limits)

    # 打印模型信息
    #ik_solver_left.print_model_info()

    # 输出手臂关节在nq空间中的索引
    print(f"\nLeft arm joint indices in q vector: {ik_solver_left.joint_indices_in_nq}")

    # 测试FK功能
    print("\nTesting Forward Kinematics (FK):")
    init_joints = [0, 0, 0, 0, 0, 0, 0]  # 关节角度
    init_pose = ik_solver_left.fk(init_joints)
    print(f"Initial end-effector pose:\n{init_pose.translation}R:\n{init_pose.rotation}")
    test_joints = [-1, -1, 0, 0.4, 0.5, 0.1, 0.1]  # 关节角度
    pose = ik_solver_left.fk(test_joints)
    print(f"Test end-effector pose:{pose.translation}R:\n{pose.rotation}")
    a = ik_solver_left.ik(pose.translation, pose.rotation, init_joints)
    if a is None:
        print("IK Failed")
        exit(1)
    print("IK Result:", a)
    pose = ik_solver_left.fk(a)
    print(f"IK end-effector pose:{pose.translation}R:\n{pose.rotation}")
