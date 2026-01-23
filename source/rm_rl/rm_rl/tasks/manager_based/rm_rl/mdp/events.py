import torch
from isaaclab.managers import SceneEntityCfg


def reset_root_state(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """
    Reset root pose and velocity to default.
    Equivalent to DirectRLEnv _reset_idx root part.
    """

    robot = env.scene[asset_cfg.name]

    if env_ids.numel() == 0:
        return

    # default root state
    root_state = robot.data.default_root_state[env_ids].clone()

    # shift by environment origins
    root_state[:, :3] += env.scene.env_origins[env_ids]

    robot.write_root_state_to_sim(root_state, env_ids)

def reset_joints_to_default(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]

    if env_ids.numel() == 0:
        return

    joint_pos = robot.data.default_joint_pos[env_ids]
    joint_vel = robot.data.default_joint_vel[env_ids]

    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
