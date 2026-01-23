# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CommandTermCfg as CommandTerm
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

from rm_rl.robots.jetbot import JETBOT_CONFIG

##
# Scene definition
##


@configclass
class RmRlSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.1,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command_dot = ObsTerm(
            func=mdp.command_alignment,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        command_cross = ObsTerm(
            func=mdp.command_cross_z,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        forward_speed = ObsTerm(
            func=mdp.forward_speed,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:

    wheels = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint", "right_wheel_joint"],
        scale=50.0,
    )
@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_root = EventTerm(
        func=mdp.reset_root_state,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_to_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    #alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    #terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep forward corresponds to commanded velocity
    forward = RewTerm(
        func=mdp.forward_exp_alignment,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) out of bounds


##
# Environment configuration
##


@configclass
class RmRlEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: RmRlSceneCfg = RmRlSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()


    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation