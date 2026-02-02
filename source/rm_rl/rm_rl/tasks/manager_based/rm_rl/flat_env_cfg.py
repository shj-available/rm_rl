# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import RoughEnvCfg


@configclass
class FlatEnvCfg(RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_lin_vel_xy_exp_enhance.weight = 5.0
        self.rewards.track_ang_vel_z_exp.weight = 5.0
        self.rewards.base_height_l2.weight = -200.0

        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -5.0e-4
        self.rewards.leg_acc_l2.weight = -5.0e-6
        self.rewards.wheel_acc_l2.weight = -1.0e-6
        # self.rewards.action_rate_l2.weight = -1.0
        self.rewards.leg_action_rate_l2.weight = -1.0
        self.rewards.wheel_action_rate_l2.weight = -1.0
  

        self.rewards.undesired_contacts.weight = -20.0
        self.rewards.desired_contacts.weight = 2.0
        self.rewards.flat_orientation_l2.weight = -25.0
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.joint_devitation.weight = -1.0

        self.rewards.termination_penalty.weight = -2000.0

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-2.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0., 0.)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


class FlatEnvCfg_PLAY(FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
