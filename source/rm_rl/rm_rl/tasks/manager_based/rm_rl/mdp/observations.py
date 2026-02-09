# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for Wheel-Legged Infantry Robot."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import (
    _get_joint_positions,
    _get_joint_velocities,
    _joint_to_virtual_leg,
)

# Gravitational acceleration magnitude
GRAVITY_MAGNITUDE = 9.81


# ============== IMU Observation Functions ==============

def imu_measured_gravity(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Simulated IMU gravity direction measurement.
    
    Real IMU accelerometers measure specific force (a_measured = a_body - g).
    When normalized, this gives an estimate of gravity direction that is 
    corrupted by body acceleration.
    
    This function simulates this by:
        imu_gravity = projected_gravity - body_lin_acc_b / |g|
    
    Returns:
        Tensor [num_envs, 3]: Simulated gravity direction in body frame (not normalized)
    """
    asset: Articulation = env.scene["robot"]
    
    # Get true projected gravity (pure geometry, from quaternion)
    projected_gravity = asset.data.projected_gravity_b  # [num_envs, 3]
    
    # Get root body linear acceleration in world frame
    # body_lin_acc_w shape: [num_envs, num_bodies, 3], index 0 is root
    root_lin_acc_w = asset.data.body_lin_acc_w[:, 0, :]  # [num_envs, 3]
    
    # Transform acceleration to body frame
    root_quat_w = asset.data.root_link_quat_w  # [num_envs, 4]
    root_lin_acc_b = quat_apply_inverse(root_quat_w, root_lin_acc_w)  # [num_envs, 3]
    
    # Simulate IMU measurement: gravity direction is corrupted by body acceleration
    # a_measured = a_body - g, so measured_gravity_direction ≈ -a_measured/|g| = g/|g| - a_body/|g|
    # Since projected_gravity is already g_direction (unit vector pointing in gravity direction),
    # the IMU would measure: imu_gravity = projected_gravity - body_acc / |g|
    imu_gravity = projected_gravity - root_lin_acc_b / GRAVITY_MAGNITUDE
    
    return imu_gravity


# ============== Virtual Leg Observation Functions ==============

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
