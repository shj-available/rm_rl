# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Virtual Leg Model for Wheel-Legged Infantry Robot.

Maps hip and knee joint angles to virtual leg parameters (angle and length).

Coordinate System:
    - Virtual leg angle: 0 when pointing straight down (vertical), 
      positive when swinging forward
    - Virtual leg length: distance from hip joint to wheel center

Kinematic Model:
    hip_joint ---(L1)--- knee_joint ---(L2)--- wheel_center
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============== Robot Parameters ==============
# Link lengths from URDF (Infantry_25)
L1 = 0.210  # hip link length (hip_joint to knee_joint)
L2 = 0.248  # knee link length (knee_joint to wheel_center)

# Virtual leg length limits
LEG_LENGTH_MIN = abs(L1 - L2) + 0.01  # ~0.048m
LEG_LENGTH_MAX = L1 + L2 - 0.01       # ~0.448m


# ============== Core Conversion Functions ==============

def _get_joint_positions(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get hip and knee joint positions for both legs from the environment.
    
    Returns:
        hip_l, hip_r, knee_l, knee_r: Joint angles [num_envs]
    """
    asset: Articulation = env.scene["robot"]
    joint_pos = asset.data.joint_pos
    
    # Get joint indices (cached after first call)
    hip_l_idx = asset.find_joints("hip_l_joint")[0][0]
    hip_r_idx = asset.find_joints("hip_r_joint")[0][0]
    knee_l_idx = asset.find_joints("knee_l_joint")[0][0]
    knee_r_idx = asset.find_joints("knee_r_joint")[0][0]
    
    return (
        joint_pos[:, hip_l_idx],
        joint_pos[:, hip_r_idx],
        joint_pos[:, knee_l_idx],
        joint_pos[:, knee_r_idx],
    )


def _get_joint_velocities(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get hip and knee joint velocities for both legs from the environment.
    
    Returns:
        hip_l_vel, hip_r_vel, knee_l_vel, knee_r_vel: Joint velocities [num_envs]
    """
    asset: Articulation = env.scene["robot"]
    joint_vel = asset.data.joint_vel
    
    hip_l_idx = asset.find_joints("hip_l_joint")[0][0]
    hip_r_idx = asset.find_joints("hip_r_joint")[0][0]
    knee_l_idx = asset.find_joints("knee_l_joint")[0][0]
    knee_r_idx = asset.find_joints("knee_r_joint")[0][0]
    
    return (
        joint_vel[:, hip_l_idx],
        joint_vel[:, hip_r_idx],
        joint_vel[:, knee_l_idx],
        joint_vel[:, knee_r_idx],
    )


def _joint_to_virtual_leg(
    hip_angle: torch.Tensor, 
    knee_angle: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert hip and knee joint angles to virtual leg angle and length.
    
    URDF Convention:
    - hip_joint has rpy="0 1.5708 0" (starts horizontal forward)
    - knee_joint has rpy="0 3.1416 0" (folds back)
    
    Args:
        hip_angle: Hip joint angles [num_envs]
        knee_angle: Knee joint angles [num_envs]
        
    Returns:
        leg_angle: Virtual leg angle from vertical (down=0, forward=positive) [num_envs]
        leg_length: Virtual leg length [num_envs]
    """
    # Upper leg angle from vertical (positive = forward swing)
    # In URDF: hip starts at pi/2 (horizontal), so angle from vertical = pi/2 - hip_angle
    alpha = -hip_angle
    
    # Lower leg angle from vertical
    # Due to knee joint convention (rpy pi rotation)
    gamma = alpha - knee_angle + torch.pi
    
    # Forward kinematics: wheel position relative to hip
    x1 = L1 * torch.sin(alpha)           # upper leg endpoint x
    z1 = -L1 * torch.cos(alpha)          # upper leg endpoint z (down is negative)
    
    x2 = x1 + L2 * torch.sin(gamma)      # wheel x
    z2 = z1 - L2 * torch.cos(gamma)      # wheel z
    
    # Virtual leg length
    leg_length = torch.sqrt(x2**2 + z2**2)
    
    # Virtual leg angle (from vertical down, positive forward)
    leg_angle = torch.atan2(x2, -z2)
    
    return leg_angle, leg_length




# ============== Observation Functions ==============

def virtual_leg_angle(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Get virtual leg angles for both legs.
    
    Returns:
        Tensor [num_envs, 2]: [left_leg_angle, right_leg_angle]
        Angle is 0 when vertical (down), positive when forward.
    """
    hip_l, hip_r, knee_l, knee_r = _get_joint_positions(env)
    
    angle_l, _ = _joint_to_virtual_leg(hip_l, knee_l)
    angle_r, _ = _joint_to_virtual_leg(hip_r, knee_r)
    
    return torch.stack([angle_l, angle_r], dim=1)


def virtual_leg_length(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Get virtual leg lengths for both legs.
    
    Returns:
        Tensor [num_envs, 2]: [left_leg_length, right_leg_length]
    """
    hip_l, hip_r, knee_l, knee_r = _get_joint_positions(env)
    
    _, length_l = _joint_to_virtual_leg(hip_l, knee_l)
    _, length_r = _joint_to_virtual_leg(hip_r, knee_r)
    
    return torch.stack([length_l, length_r], dim=1)


def virtual_leg_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Get full virtual leg state for both legs.
    
    Returns:
        Tensor [num_envs, 4]: [angle_l, angle_r, length_l, length_r]
    """
    hip_l, hip_r, knee_l, knee_r = _get_joint_positions(env)
    
    angle_l, length_l = _joint_to_virtual_leg(hip_l, knee_l)
    angle_r, length_r = _joint_to_virtual_leg(hip_r, knee_r)
    
    return torch.stack([angle_l, angle_r, length_l, length_r], dim=1)


def virtual_leg_angle_velocity(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Get virtual leg angle velocities using Jacobian.
    
    Returns:
        Tensor [num_envs, 2]: [angle_vel_l, angle_vel_r]
    """
    hip_l, hip_r, knee_l, knee_r = _get_joint_positions(env)
    hip_l_vel, hip_r_vel, knee_l_vel, knee_r_vel = _get_joint_velocities(env)
    
    # Numerical Jacobian (finite difference)
    eps = 1e-4
    
    # Left leg
    angle_l, _ = _joint_to_virtual_leg(hip_l, knee_l)
    angle_l_dh, _ = _joint_to_virtual_leg(hip_l + eps, knee_l)
    angle_l_dk, _ = _joint_to_virtual_leg(hip_l, knee_l + eps)
    
    d_angle_d_hip = (angle_l_dh - angle_l) / eps
    d_angle_d_knee = (angle_l_dk - angle_l) / eps
    angle_vel_l = d_angle_d_hip * hip_l_vel + d_angle_d_knee * knee_l_vel
    
    # Right leg
    angle_r, _ = _joint_to_virtual_leg(hip_r, knee_r)
    angle_r_dh, _ = _joint_to_virtual_leg(hip_r + eps, knee_r)
    angle_r_dk, _ = _joint_to_virtual_leg(hip_r, knee_r + eps)
    
    d_angle_d_hip = (angle_r_dh - angle_r) / eps
    d_angle_d_knee = (angle_r_dk - angle_r) / eps
    angle_vel_r = d_angle_d_hip * hip_r_vel + d_angle_d_knee * knee_r_vel
    
    return torch.stack([angle_vel_l, angle_vel_r], dim=1)


def virtual_leg_length_velocity(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Get virtual leg length velocities using Jacobian.
    
    Returns:
        Tensor [num_envs, 2]: [length_vel_l, length_vel_r]
    """
    hip_l, hip_r, knee_l, knee_r = _get_joint_positions(env)
    hip_l_vel, hip_r_vel, knee_l_vel, knee_r_vel = _get_joint_velocities(env)
    
    eps = 1e-4
    
    # Left leg
    _, length_l = _joint_to_virtual_leg(hip_l, knee_l)
    _, length_l_dh = _joint_to_virtual_leg(hip_l + eps, knee_l)
    _, length_l_dk = _joint_to_virtual_leg(hip_l, knee_l + eps)
    
    d_length_d_hip = (length_l_dh - length_l) / eps
    d_length_d_knee = (length_l_dk - length_l) / eps
    length_vel_l = d_length_d_hip * hip_l_vel + d_length_d_knee * knee_l_vel
    
    # Right leg
    _, length_r = _joint_to_virtual_leg(hip_r, knee_r)
    _, length_r_dh = _joint_to_virtual_leg(hip_r + eps, knee_r)
    _, length_r_dk = _joint_to_virtual_leg(hip_r, knee_r + eps)
    
    d_length_d_hip = (length_r_dh - length_r) / eps
    d_length_d_knee = (length_r_dk - length_r) / eps
    length_vel_r = d_length_d_hip * hip_r_vel + d_length_d_knee * knee_r_vel
    
    return torch.stack([length_vel_l, length_vel_r], dim=1)


# ============== Debug Functions ==============

# Debug configuration - set which variables to observe
DEBUG_CONFIG = {
    "velocity_tracking": True,  # Command vs actual velocity comparison + orientation
    "virtual_leg": False,       # Virtual leg angle and length
    "base_state": False,        # Base position, orientation, velocity
    "joint_state": False,       # Joint positions and velocities
    "wheel_state": False,       # Wheel velocities
    "joint_torque": False,      # Joint applied torques
    "link_wrench": False,       # Link incoming joint wrench (force/torque from parent)
    "command": False,           # Velocity commands (legacy, use velocity_tracking)
}


def _debug_velocity_tracking(env: "ManagerBasedRLEnv", env_id: int) -> str:
    """Compare commanded velocity vs actual velocity, and show orientation."""
    import math
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
    
    asset: Articulation = env.scene["robot"]
    
    # Get actual velocities in robot's yaw-aligned frame
    root_quat = asset.data.root_quat_w[env_id:env_id+1]
    root_lin_vel_w = asset.data.root_lin_vel_w[env_id:env_id+1]
    root_ang_vel_w = asset.data.root_ang_vel_w[env_id]
    
    # Transform linear velocity to yaw-aligned frame (gravity-aligned, yaw from robot)
    vel_yaw = quat_apply_inverse(yaw_quat(root_quat), root_lin_vel_w)[0]
    vx_actual = vel_yaw[0].item()
    vy_actual = vel_yaw[1].item()
    wz_actual = root_ang_vel_w[2].item()
    
    # Get commanded velocities
    try:
        cmd = env.command_manager.get_command("base_velocity")[env_id]
        vx_cmd = cmd[0].item()
        vy_cmd = cmd[1].item()
        wz_cmd = cmd[2].item()
    except Exception:
        vx_cmd, vy_cmd, wz_cmd = 0.0, 0.0, 0.0
    
    # Get orientation (roll, pitch) from projected gravity
    proj_grav = asset.data.projected_gravity_b[env_id]
    pitch = torch.atan2(proj_grav[0], -proj_grav[2]).item()
    roll = torch.atan2(proj_grav[1], -proj_grav[2]).item()
    
    lines = [
        "--- Velocity Tracking ---",
        f"  Vx: cmd={vx_cmd:+.2f}, actual={vx_actual:+.2f}, err={vx_actual - vx_cmd:+.2f} m/s",
        f"  Vy: cmd={vy_cmd:+.2f}, actual={vy_actual:+.2f}, err={vy_actual - vy_cmd:+.2f} m/s",
        f"  Wz: cmd={wz_cmd:+.2f}, actual={wz_actual:+.2f}, err={wz_actual - wz_cmd:+.2f} rad/s",
        f"--- Orientation ---",
        f"  Roll: {math.degrees(roll):+.1f}°, Pitch: {math.degrees(pitch):+.1f}°",
    ]
    return "\n".join(lines)


def _debug_virtual_leg(env: "ManagerBasedRLEnv", env_id: int) -> str:
    """Get virtual leg debug info."""
    import math
    hip_l, hip_r, knee_l, knee_r = _get_joint_positions(env)
    
    angle_l, length_l = _joint_to_virtual_leg(hip_l[env_id:env_id+1], knee_l[env_id:env_id+1])
    angle_r, length_r = _joint_to_virtual_leg(hip_r[env_id:env_id+1], knee_r[env_id:env_id+1])
    
    lines = [
        "--- Virtual Leg ---",
        f"  Angle L: {angle_l[0].item():+.3f} rad ({math.degrees(angle_l[0].item()):+.1f}°)",
        f"  Angle R: {angle_r[0].item():+.3f} rad ({math.degrees(angle_r[0].item()):+.1f}°)",
        f"  Length L: {length_l[0].item():.3f} m",
        f"  Length R: {length_r[0].item():.3f} m",
    ]
    return "\n".join(lines)


def _debug_base_state(env: "ManagerBasedRLEnv", env_id: int) -> str:
    """Get base state debug info."""
    import math
    asset: Articulation = env.scene["robot"]
    
    pos = asset.data.root_pos_w[env_id]
    quat = asset.data.root_quat_w[env_id]
    lin_vel = asset.data.root_lin_vel_w[env_id]
    ang_vel = asset.data.root_ang_vel_w[env_id]
    
    # Convert quaternion to euler (roll, pitch, yaw)
    # Simplified: extract pitch from projected gravity
    proj_grav = asset.data.projected_gravity_b[env_id]
    pitch = torch.atan2(proj_grav[0], -proj_grav[2]).item()
    roll = torch.atan2(proj_grav[1], -proj_grav[2]).item()
    
    lines = [
        "--- Base State ---",
        f"  Pos: x={pos[0].item():.2f}, y={pos[1].item():.2f}, z={pos[2].item():.2f}",
        f"  Roll: {math.degrees(roll):+.1f}°, Pitch: {math.degrees(pitch):+.1f}°",
        f"  Lin Vel: vx={lin_vel[0].item():+.2f}, vy={lin_vel[1].item():+.2f}, vz={lin_vel[2].item():+.2f}",
        f"  Ang Vel: wx={ang_vel[0].item():+.2f}, wy={ang_vel[1].item():+.2f}, wz={ang_vel[2].item():+.2f}",
    ]
    return "\n".join(lines)


def _debug_joint_state(env: "ManagerBasedRLEnv", env_id: int) -> str:
    """Get joint state debug info."""
    import math
    hip_l, hip_r, knee_l, knee_r = _get_joint_positions(env)
    hip_l_vel, hip_r_vel, knee_l_vel, knee_r_vel = _get_joint_velocities(env)
    
    lines = [
        "--- Joint State ---",
        f"  Hip L:  pos={hip_l[env_id].item():+.3f} rad ({math.degrees(hip_l[env_id].item()):+.1f}°), vel={hip_l_vel[env_id].item():+.2f}",
        f"  Hip R:  pos={hip_r[env_id].item():+.3f} rad ({math.degrees(hip_r[env_id].item()):+.1f}°), vel={hip_r_vel[env_id].item():+.2f}",
        f"  Knee L: pos={knee_l[env_id].item():+.3f} rad ({math.degrees(knee_l[env_id].item()):+.1f}°), vel={knee_l_vel[env_id].item():+.2f}",
        f"  Knee R: pos={knee_r[env_id].item():+.3f} rad ({math.degrees(knee_r[env_id].item()):+.1f}°), vel={knee_r_vel[env_id].item():+.2f}",
    ]
    return "\n".join(lines)


def _debug_wheel_state(env: "ManagerBasedRLEnv", env_id: int) -> str:
    """Get wheel state debug info."""
    asset: Articulation = env.scene["robot"]
    joint_vel = asset.data.joint_vel
    
    wheel_l_idx = asset.find_joints("wheel_l_joint")[0][0]
    wheel_r_idx = asset.find_joints("wheel_r_joint")[0][0]
    
    wheel_l_vel = joint_vel[env_id, wheel_l_idx].item()
    wheel_r_vel = joint_vel[env_id, wheel_r_idx].item()
    
    # Estimate linear velocity from wheel (assuming wheel radius ~0.06m)
    wheel_radius = 0.06
    lin_vel_l = wheel_l_vel * wheel_radius
    lin_vel_r = wheel_r_vel * wheel_radius
    
    lines = [
        "--- Wheel State ---",
        f"  Wheel L: vel={wheel_l_vel:+.2f} rad/s -> {lin_vel_l:+.2f} m/s",
        f"  Wheel R: vel={wheel_r_vel:+.2f} rad/s -> {lin_vel_r:+.2f} m/s",
        f"  Avg wheel vel: {(lin_vel_l + lin_vel_r) / 2:+.2f} m/s",
    ]
    return "\n".join(lines)


def _debug_joint_torque(env: "ManagerBasedRLEnv", env_id: int) -> str:
    """Get joint torque debug info for all joints.
    
    Shows:
    - applied_torque: torque actually applied to simulation
    - computed_torque: torque computed by actuator model (before saturation/clipping)
    - For wheels: additional analysis of torque sources
    """
    asset: Articulation = env.scene["robot"]
    applied_torque = asset.data.applied_torque
    computed_torque = asset.data.computed_torque
    joint_names = asset.joint_names
    joint_vel = asset.data.joint_vel
    
    lines = ["--- Joint Torques ---"]
    for i, name in enumerate(joint_names):
        applied = applied_torque[env_id, i].item()
        computed = computed_torque[env_id, i].item()
        vel = joint_vel[env_id, i].item()
        lines.append(f"  {name}: applied={applied:+.3f} Nm, computed={computed:+.3f} Nm, vel={vel:+.2f} rad/s")
    
    # Wheel-specific analysis
    wheel_l_idx = asset.find_joints("wheel_l_joint")[0][0]
    wheel_r_idx = asset.find_joints("wheel_r_joint")[0][0]
    
    lines.append("--- Wheel Torque Analysis ---")
    for idx, name in [(wheel_l_idx, "Wheel L"), (wheel_r_idx, "Wheel R")]:
        vel = joint_vel[env_id, idx].item()
        torque = applied_torque[env_id, idx].item()
        # Estimate: if damping=0.2, τ_damping = damping * Δvel
        # Power = τ * ω
        power = torque * vel
        lines.append(f"  {name}: τ={torque:+.3f} Nm, ω={vel:+.2f} rad/s, P={power:+.2f} W")
    
    # Total torque magnitude
    total_torque = applied_torque[env_id].abs().sum().item()
    lines.append(f"  Total |τ|: {total_torque:.3f} Nm")
    
    return "\n".join(lines)


def _debug_link_wrench(env: "ManagerBasedRLEnv", env_id: int) -> str:
    """Get link incoming joint wrench debug info.
    
    Shows the 6-D wrench (force and torque) applied to each link by the parent joint.
    Format: [Fx, Fy, Fz, Tx, Ty, Tz] in body frame.
    
    Particularly useful for:
    - Analyzing torque around Y axis (pitch direction) on wheel links
    - Understanding reaction forces through the kinematic chain
    """
    asset: Articulation = env.scene["robot"]
    
    # Get link incoming joint wrench: shape (num_envs, num_bodies, 6)
    # Format: [Fx, Fy, Fz, Tx, Ty, Tz] in parent body frame
    wrench = asset.data.body_incoming_joint_wrench_b[env_id]  # (num_bodies, 6)
    body_names = asset.body_names
    
    lines = ["--- Link Incoming Joint Wrench (body frame) ---"]
    lines.append("  Format: [Fx, Fy, Fz | Tx, Ty, Tz] in Nm")
    
    for i, name in enumerate(body_names):
        fx, fy, fz = wrench[i, 0].item(), wrench[i, 1].item(), wrench[i, 2].item()
        tx, ty, tz = wrench[i, 3].item(), wrench[i, 4].item(), wrench[i, 5].item()
        
        # Highlight Y-axis torque (pitch torque)
        lines.append(
            f"  {name:12s}: F=[{fx:+6.2f}, {fy:+6.2f}, {fz:+6.2f}] "
            f"T=[{tx:+6.2f}, {ty:+6.2f}★, {tz:+6.2f}]"
        )
    
    # Wheel-specific: focus on Ty (torque around Y, which is rolling direction)
    lines.append("--- Wheel Y-Torque (Rolling Resistance Analysis) ---")
    for name in body_names:
        if "wheel" in name.lower():
            idx = body_names.index(name)
            ty = wrench[idx, 4].item()  # Y-axis torque
            lines.append(f"  {name}: Ty = {ty:+.4f} Nm")
    
    return "\n".join(lines)


def _debug_command(env: "ManagerBasedRLEnv", env_id: int) -> str:
    """Get command debug info."""
    try:
        cmd = env.command_manager.get_command("base_velocity")[env_id]
        lines = [
            "--- Command ---",
            f"  Lin Vel X: {cmd[0].item():+.2f} m/s",
            f"  Lin Vel Y: {cmd[1].item():+.2f} m/s",
            f"  Ang Vel Z: {cmd[2].item():+.2f} rad/s",
        ]
    except Exception:
        lines = ["--- Command ---", "  (No command available)"]
    return "\n".join(lines)


def debug_print(
    env: "ManagerBasedRLEnv",
    env_id: int = 0,
    print_interval: int = 100,
    config: dict | None = None,
) -> None:
    """
    Main debug print function. Call this in play.py to observe variables.
    
    Args:
        env: Environment instance
        env_id: Which environment to debug (default 0)
        print_interval: How often to print (in simulation steps)
        config: Optional dict to override DEBUG_CONFIG. Keys:
            - "virtual_leg": bool
            - "base_state": bool
            - "joint_state": bool
            - "wheel_state": bool
            - "command": bool
    
    Usage in play.py:
        from rm_rl.tasks.manager_based.rm_rl.mdp.utils import debug_print
        debug_print(env.unwrapped, env_id=0, print_interval=100)
    """
    if env.common_step_counter % print_interval != 0:
        return
    
    cfg = config if config is not None else DEBUG_CONFIG
    
    print(f"\n{'='*50}")
    print(f"DEBUG (Step {env.common_step_counter}, Env {env_id})")
    print('='*50)
    
    if cfg.get("velocity_tracking", False):
        print(_debug_velocity_tracking(env, env_id))
    
    if cfg.get("command", False):
        print(_debug_command(env, env_id))
    
    if cfg.get("base_state", False):
        print(_debug_base_state(env, env_id))
    
    if cfg.get("virtual_leg", False):
        print(_debug_virtual_leg(env, env_id))
    
    if cfg.get("joint_state", False):
        print(_debug_joint_state(env, env_id))
    
    if cfg.get("wheel_state", False):
        print(_debug_wheel_state(env, env_id))
    
    if cfg.get("joint_torque", False):
        print(_debug_joint_torque(env, env_id))
    
    if cfg.get("link_wrench", False):
        print(_debug_link_wrench(env, env_id))
    
    print('='*50)


# Legacy function for backwards compatibility
def debug_virtual_leg_kinematics(env: "ManagerBasedRLEnv", env_id: int = 0, print_interval: int = 100) -> None:
    """
    Legacy debug function. Use debug_print() instead.
    """
    debug_print(env, env_id, print_interval, config={"virtual_leg": True, "joint_state": True})




