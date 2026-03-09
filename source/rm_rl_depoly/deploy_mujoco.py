import time
import mujoco.viewer
import mujoco
import numpy as np

import torch
import yaml
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Action indices (根据 ActionsCfg 定义)
# hip_pos: [0, 1], knee_pos: [2, 3], wheel_vel: [4, 5]
hip_indices = np.array([0, 1])
knee_indices = np.array([2, 3])
wheel_indices = np.array([4, 5])

# DOF indices in qpos/qvel (根据 URDF 关节顺序)
# hip_left, knee_left, wheel_left, hip_right, knee_right, wheel_right
hip_dof_indices = np.array([0, 3])
knee_dof_indices = np.array([1, 4])
wheel_dof_indices = np.array([2, 5])

def forward_kinematics(theta_1, theta_2):
    """正运动学：从关节角度计算虚拟腿状态"""
    L0 = np.sqrt(l1**2 + l2**2 + 2 * l1 * l2 * np.cos(theta_2))
    theta_0 = theta_1 + np.arccos((L0**2 + l1**2 - l2**2) / (2 * L0 * l1))
    return theta_0, L0

def differential_kinematics(hip_pos, hip_vel, knee_pos, knee_vel, dt=0.005):
    """微分运动学：计算虚拟腿的角度、长度及其导数
    
    Args:
        hip_pos: 髋关节位置 [left, right]
        hip_vel: 髋关节速度 [left, right]
        knee_pos: 膝关节位置 [left, right]
        knee_vel: 膝关节速度 [left, right]
        dt: 微分时间步长
    
    Returns:
        theta0: 虚拟腿角度
        l0: 虚拟腿长度
        theta0_dot: 虚拟腿角速度
        l0_dot: 虚拟腿长度变化率
    """
    theta0, l0 = forward_kinematics(hip_pos, knee_pos)
    theta0_next, l0_next = forward_kinematics(hip_pos + hip_vel * dt, knee_pos + knee_vel * dt)
    theta0_dot = (theta0_next - theta0) / dt
    l0_dot = (l0_next - l0) / dt
    return theta0, l0, theta0_dot, l0_dot

def VMC(F, T, theta1, theta2, theta0, l0):
    L0_theta1 = 0
    L0_theta2 = -l1 * l2 * np.sin(theta2) / l0
    theta0_theta1 = 1
    theta0_theta2 = (l0 - l1 * np.cos(theta1 - theta0)) / l0

    T1 = L0_theta1 * F + theta0_theta1 * T
    T2 = L0_theta2 * F + theta0_theta2 * T

    return T1, T2

def compute_torques(action_tmp, hip_pos_tmp, hip_vel_tmp, knee_pos_tmp, knee_vel_tmp, wheel_vel_tmp):
    """
    根据 ActionsCfg 定义计算关节扭矩:
    - action[0:2]: hip position (scale=3.14, use_default_offset=True)
    - action[2:4]: knee position (scale=1.2, use_default_offset=True)
    - action[4:6]: wheel velocity (scale=60.0)
    
    注意: default_angles 按 URDF 顺序 [hip_l, knee_l, wheel_l, hip_r, knee_r, wheel_r]
          action 按 ActionsCfg 顺序 [hip_l, hip_r, knee_l, knee_r, wheel_l, wheel_r]
    """
    # 计算目标位置/速度 (使用 dof 索引获取 default_angles)
    hip_target = default_angles[hip_dof_indices] + action_tmp[hip_indices] * action_scale_hip
    knee_target = default_angles[knee_dof_indices] + action_tmp[knee_indices] * action_scale_knee
    wheel_vel_target = action_tmp[wheel_indices] * action_scale_wheel

    # PD 控制
    torque_hip = kp_joint * (hip_target - hip_pos_tmp) - kd_joint * hip_vel_tmp
    torque_knee = kp_joint * (knee_target - knee_pos_tmp) - kd_joint * knee_vel_tmp
    torque_wheel = kd_wheel * (wheel_vel_target - wheel_vel_tmp)

    return torque_hip, torque_knee, torque_wheel

def get_gravity_orientation(quaternion):
    gravity_vec = [0, 0, -1]
    return quat_rotate_inv(quaternion, gravity_vec)

def quat_rotate_pos(quat, vec):
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]
    r = R.from_quat([qx, qy, qz, qw])
    return r.apply(vec)

def quat_rotate_inv(quat, vec):
    quat_conj = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
    return quat_rotate_pos(quat_conj, vec)

def quat_to_euler(quat):
    # quat 的格式是 [w, x, y, z]
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return r.as_euler('xyz')  # 返回 [roll, pitch, yaw]

def euler_to_quat(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw])
    quat = r.as_quat()  # 返回 [x, y, z, w]
    return [quat[3], quat[0], quat[1], quat[2]]  # 转换为 [w, x, y, z] 格式

def wrap_to_pi(angles):
    """将角度wrap到[-pi, pi]"""
    return (angles + np.pi) % (2 * np.pi) - np.pi

def get_cmd_vector(base_quat_tmp, cmd_init):
    cmd_tmp = cmd_init.copy()
    forward_vec = np.array([1.0, 0.0, 0.0])
    forward = quat_rotate_pos(base_quat_tmp, forward_vec)
    heading = np.arctan2(forward[1], forward[0])
    cmd_tmp[1] = np.clip(1.5 * wrap_to_pi(cmd_init[3] - heading), -5, 5)

    return cmd_tmp

def normalize_wheel_pos(wheel_pos):
    """将轮子位置归一化到[-2π, 2π]范围内"""
    return (wheel_pos + 2 * np.pi) % (4 * np.pi) - 2 * np.pi

if __name__ == "__main__":
    # get config file name from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, default="Infantry_25_headless.yaml", help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file

    # initialize history
    theta0_history = []
    l0_history = []
    wheel_vel_history = []
    wheel_pos_history = []  # 新增：记录轮子位置
    torque_history = []
    ang_vel_history = []
    lin_vel_history = []
    est_lin_vel_history = []
    cmd_history = []
    base_pos_history = []

    base_lin_vel = []
    base_ang_vel = []

    last_base_pos = np.zeros(3, dtype=float)
    base_pos = np.zeros(3, dtype=float)

    with open(f"deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kp_joint = config["kp_joint"]
        kd_joint = config["kd_joint"]
        kd_wheel = config["kd_wheel"]

        default_angles = config["default_angles"]

        action_scale_hip = config["action_scale_hip"]
        action_scale_knee = config["action_scale_knee"]
        action_scale_wheel = config["action_scale_wheel"]

        num_actions = config["num_actions"]
        clip_actions = config["clip_actions"]

        cmd_lin_vel_scale = np.array(config["lin_vel_scale"], dtype=np.float32)
        cmd_ang_vel_scale = np.array(config["ang_vel_scale"], dtype=np.float32)
        cmd_height_scale = np.array(config["height_scale"], dtype=np.float32)
        command_init = np.array(config["command_init"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        l0_scale = config["l0_scale"]
        l0_dot_scale = config["l0_dot_scale"]
        
        clip_observations = config["clip_observations"]
        num_obs = config["num_obs"]

        l1 = config["l1"]
        l2 = config["l2"]

        
    action = np.zeros(num_actions, dtype=np.float32)
    latent = None
    obs = np.zeros(num_obs, dtype=np.float32)
    target_dof_pos = default_angles.copy()

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    # Get base body id
    base_body_name = "base_link"
    base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base_body_name)

    theta0_indices = [0, 3]
    l0_indices = [1, 4]

    # Load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds
        start_time = time.time()
        while viewer.is_running() and time.time() - start_time < simulation_duration:
            step_start = time.time()

            # Get joint state
            dof_pos = d.qpos[7:7+len(default_angles)].copy()
            dof_vel = d.qvel[6:6+len(default_angles)].copy()

            # 获取关节状态 (按 URDF 关节顺序)
            hip_pos = dof_pos[hip_dof_indices]
            hip_vel = dof_vel[hip_dof_indices]
            knee_pos = dof_pos[knee_dof_indices]
            knee_vel = dof_vel[knee_dof_indices]         
            wheel_pos = dof_pos[wheel_dof_indices]
            wheel_vel = dof_vel[wheel_dof_indices]

            # 计算虚拟腿状态 (用于 logging 和 obs)
            theta0, l0, theta0_dot, l0_dot = differential_kinematics(hip_pos, hip_vel, knee_pos, knee_vel)

            # 计算扭矩
            torque_hip, torque_knee, torque_wheel = compute_torques(
                action_tmp=action,
                hip_pos_tmp=hip_pos,
                hip_vel_tmp=hip_vel,
                knee_pos_tmp=knee_pos,
                knee_vel_tmp=knee_vel,
                wheel_vel_tmp=wheel_vel,
            )

            torques = np.array(
                [torque_hip[0], torque_knee[0], torque_wheel[0],
                torque_hip[1], torque_knee[1], torque_wheel[1]],
                dtype=np.float32
            )
            # TODO: 限制幅度
            torques[[0, 1, 3, 4]] = np.clip(torques[[0, 1, 3, 4]], -100.0, 100.0)
            torques[[2, 5]] = np.clip(torques[[2, 5]], -5, 5)

            # logging
            theta0_history.append(theta0.copy())
            l0_history.append(l0.copy())
            wheel_vel_history.append(wheel_vel.copy())
            wheel_pos_history.append(wheel_pos.copy())  # 记录轮子位置
            torque_history.append(torques.copy())

            # Step
            d.ctrl[:len(torques)] = torques
        
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Get Observations from mujoco
                dt = simulation_dt
                
                base_quat = d.xquat[base_body_id]
                base_pos = d.xpos[base_body_id]
                base_lin_vel = d.qvel[0:3]
                base_ang_vel = d.qvel[3:6]

                est_lin_vel = (base_pos - last_base_pos) / (control_decimation * simulation_dt)
                
                projected_gravity = get_gravity_orientation(base_quat)

                cmd_vec = np.array([2.0, 0.0, 0.12], dtype=np.float32)
                cmd_scale = np.array([cmd_lin_vel_scale, cmd_ang_vel_scale, cmd_height_scale], dtype=np.float32)

                # history
                cmd_history.append(cmd_vec.copy())
                lin_vel_history.append(base_lin_vel.copy())
                est_lin_vel_history.append(est_lin_vel[0].copy())
                ang_vel_history.append(base_ang_vel.copy())
                base_pos_history.append(base_pos.copy())
                
                obs[0:3] = base_ang_vel * ang_vel_scale
                obs[3:6] = projected_gravity
                obs[6:9] = cmd_vec[:3] * cmd_scale
                obs[9:11] = theta0 * dof_pos_scale
                obs[11:13] = theta0_dot * dof_vel_scale
                obs[13:15] = l0 * l0_scale
                obs[15:17] = l0_dot * l0_dot_scale
                obs[17:19] = wheel_pos * dof_pos_scale
                obs[19:21] = wheel_vel * dof_vel_scale
                obs[21:27] = action

                action = policy(obs).detach().numpy().squeeze().clip(-clip_actions, clip_actions)

                last_base_pos = base_pos.copy()

            # 同步 viewer
            viewer.sync()
            # time control
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    plt.figure(figsize=(25, 20))

    # 绘制theta0
    plt.subplot(6, 3, 1)
    theta0_array = np.array(theta0_history)
    plt.plot(theta0_array[:, 0], label='theta0_left')
    plt.plot(theta0_array[:, 1], label='theta0_right')
    plt.title('Theta0 History')
    plt.legend()
    plt.grid(True)

    # 绘制l0
    plt.subplot(6, 3, 2)
    l0_array = np.array(l0_history)
    plt.plot(l0_array[:, 0], label='l0_left')
    plt.plot(l0_array[:, 1], label='l0_right')
    plt.title('L0 History')
    plt.legend()
    plt.grid(True)

    # 绘制wheel_vel
    plt.subplot(6, 3, 3)
    wheel_vel_array = np.array(wheel_vel_history)
    plt.plot(wheel_vel_array[:, 0], label='wheel_vel_left')
    plt.plot(wheel_vel_array[:, 1], label='wheel_vel_right')
    plt.title('Wheel Velocity History')
    plt.legend()
    plt.grid(True)

    # 轮子位置历史
    plt.subplot(6, 3, 4)
    wheel_pos_array = np.array(wheel_pos_history)
    plt.plot(wheel_pos_array[:, 0], label='wheel_pos_left')
    plt.plot(wheel_pos_array[:, 1], label='wheel_pos_right')
    plt.title('Wheel Position History')
    plt.legend()
    plt.grid(True)
    plt.ylabel('Position [rad]')

    # 轮子速度对比（替换原来的位置差）
    plt.subplot(6, 3, 5)
    plt.plot(wheel_vel_array[:, 0], label='wheel_vel_left', color='blue', linewidth=2)
    plt.plot(wheel_vel_array[:, 1], label='wheel_vel_right', color='red', linewidth=2)
    plt.title('Wheel Velocity Comparison')
    plt.legend()
    plt.grid(True)
    plt.ylabel('Velocity [rad/s]')

    # 轮子累计转数
    plt.subplot(6, 3, 6)
    wheel_revolutions_left = wheel_pos_array[:, 0] / (2 * np.pi)
    wheel_revolutions_right = wheel_pos_array[:, 1] / (2 * np.pi)
    plt.plot(wheel_revolutions_left, label='wheel_revolutions_left')
    plt.plot(wheel_revolutions_right, label='wheel_revolutions_right')
    plt.title('Wheel Revolutions')
    plt.legend()
    plt.grid(True)
    plt.ylabel('Revolutions')

    # 髋关节扭矩对比 [0,3]
    plt.subplot(6, 3, 7)
    torque_array = np.array(torque_history)
    plt.plot(torque_array[:, 0], label='torque_hip_left', color='blue', linewidth=2)
    plt.plot(torque_array[:, 3], label='torque_hip_right', color='red', linewidth=2)
    plt.title('Hip Torque Comparison [0,3]')
    plt.legend()
    plt.grid(True)
    plt.ylabel('Torque [Nm]')

    # 膝关节扭矩对比 [1,4]
    plt.subplot(6, 3, 8)
    plt.plot(torque_array[:, 1], label='torque_knee_left', color='blue', linewidth=2)
    plt.plot(torque_array[:, 4], label='torque_knee_right', color='red', linewidth=2)
    plt.title('Knee Torque Comparison [1,4]')
    plt.legend()
    plt.grid(True)
    plt.ylabel('Torque [Nm]')

    # 轮子扭矩对比 [2,5]
    plt.subplot(6, 3, 9)
    plt.plot(torque_array[:, 2], label='torque_wheel_left', color='blue', linewidth=2)
    plt.plot(torque_array[:, 5], label='torque_wheel_right', color='red', linewidth=2)
    plt.title('Wheel Torque Comparison [2,5]')
    plt.legend()
    plt.grid(True)
    plt.ylabel('Torque [Nm]')

    # 髋关节扭矩统计
    plt.subplot(6, 3, 10)
    hip_torque_left = torque_array[:, 0]
    hip_torque_right = torque_array[:, 3]
    plt.hist(hip_torque_left, bins=50, alpha=0.7, label='Hip Left', color='blue')
    plt.hist(hip_torque_right, bins=50, alpha=0.7, label='Hip Right', color='red')
    plt.title('Hip Torque Distribution')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Torque [Nm]')
    plt.ylabel('Frequency')

    # 膝关节扭矩统计
    plt.subplot(6, 3, 11)
    knee_torque_left = torque_array[:, 1]
    knee_torque_right = torque_array[:, 4]
    plt.hist(knee_torque_left, bins=50, alpha=0.7, label='Knee Left', color='blue')
    plt.hist(knee_torque_right, bins=50, alpha=0.7, label='Knee Right', color='red')
    plt.title('Knee Torque Distribution')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Torque [Nm]')
    plt.ylabel('Frequency')

    # 轮子扭矩统计
    plt.subplot(6, 3, 12)
    wheel_torque_left = torque_array[:, 2]
    wheel_torque_right = torque_array[:, 5]
    plt.hist(wheel_torque_left, bins=50, alpha=0.7, label='Wheel Left', color='blue')
    plt.hist(wheel_torque_right, bins=50, alpha=0.7, label='Wheel Right', color='red')
    plt.title('Wheel Torque Distribution')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Torque [Nm]')
    plt.ylabel('Frequency')

    # 角速度历史
    plt.subplot(6, 3, 13)
    cmd_array = np.array(cmd_history)
    ang_vel_array = np.array(ang_vel_history)
    plt.plot(cmd_array[:, 1], label='cmd_ang_vel_z')
    plt.plot(ang_vel_array[:, 2], label='real_ang_vel_z')
    plt.title('Angular Velocity History')
    plt.legend()
    plt.grid(True)

    # 线速度历史
    plt.subplot(6, 3, 14)
    lin_vel_array = np.array(lin_vel_history)
    est_lin_vel_array = np.array(est_lin_vel_history)
    plt.plot(cmd_array[:, 0], label='cmd_lin_vel_x')
    plt.plot(lin_vel_array[:, 0], label='real_lin_vel_x')
    plt.title('Linear Velocity History')
    plt.legend()
    plt.grid(True)

    # 位置历史
    plt.subplot(6, 3, 15)
    base_pos_array = np.array(base_pos_history)
    plt.plot(base_pos_array[:, 2], label='base_pos_z')
    plt.plot(base_pos_array[:, 1], label='base_pos_y')
    plt.plot(base_pos_array[:, 0], label='base_pos_x')
    plt.title('Base Position History')
    plt.legend()
    plt.grid(True)

    # 扭矩绝对值对比
    plt.subplot(6, 3, 16)
    torque_abs_mean = [
        np.mean(np.abs(torque_array[:, 0])),  # 髋左
        np.mean(np.abs(torque_array[:, 3])),  # 髋右
        np.mean(np.abs(torque_array[:, 1])),  # 膝左
        np.mean(np.abs(torque_array[:, 4])),  # 膝右
        np.mean(np.abs(torque_array[:, 2])),  # 轮左
        np.mean(np.abs(torque_array[:, 5]))   # 轮右
    ]
    labels = ['Hip L', 'Hip R', 'Knee L', 'Knee R', 'Wheel L', 'Wheel R']
    colors = ['blue', 'red', 'blue', 'red', 'blue', 'red']
    bars = plt.bar(labels, torque_abs_mean, color=colors, alpha=0.7)
    plt.title('Average Absolute Torque')
    plt.ylabel('Torque [Nm]')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.show()