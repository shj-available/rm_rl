# mdp/observations.py

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg



def command_alignment(env, asset_cfg: SceneEntityCfg):
    """dot(forward, command)"""

    robot = env.scene[asset_cfg.name]

    forwards = math_utils.quat_apply(
        robot.data.root_link_quat_w,
        robot.data.FORWARD_VEC_B,
    )

    commands = env.command_manager.get_term("base_velocity").command  # shape: (N, 3)

    dot = torch.sum(forwards * commands, dim=-1, keepdim=True)
    return dot

def command_cross_z(env, asset_cfg: SceneEntityCfg):
    """cross(forward, command).z"""

    robot = env.scene[asset_cfg.name]

    forwards = math_utils.quat_apply(
        robot.data.root_link_quat_w,
        robot.data.FORWARD_VEC_B,
    )

    commands = env.command_manager.get_term("base_velocity").command

    cross = torch.cross(forwards, commands, dim=-1)
    return cross[:, 2:3]   # keepdim=True

def forward_speed(env, asset_cfg: SceneEntityCfg):
    """forward speed in robot body x direction"""

    robot = env.scene[asset_cfg.name]

    lin_vel_b = math_utils.quat_apply_inverse(
        robot.data.root_link_quat_w,
        robot.data.root_lin_vel_w,
    )

    return lin_vel_b[:, 0:1]  # keepdim=True