# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize joint position deviation from a target value."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # wrap the joint positions to (-pi, pi)
#     joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
#     # compute the reward
#     return torch.sum(torch.square(joint_pos - target), dim=1)

def forward_exp_alignment(env, asset_cfg: SceneEntityCfg):
    """
    reward = v_x_body * exp(dot(forward, command))
    """

    robot = env.scene[asset_cfg.name]

    # forward direction in world frame
    forwards = math_utils.quat_apply(
        robot.data.root_link_quat_w,
        robot.data.FORWARD_VEC_B,
    )

    # command direction (N, 3)
    commands = env.command_manager.get_term("base_velocity").command

    # alignment: dot(forward, command)
    alignment = torch.sum(forwards * commands, dim=-1)

    # forward velocity (body frame x)
    forward_vel = robot.data.root_com_lin_vel_b[:, 0]

    # final reward
    reward = forward_vel * torch.exp(alignment)

    return reward