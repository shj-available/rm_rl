# from isaaclab.managers import SceneEntityCfg
# from isaaclab.managers.action_manager import ActionTerm


# class DifferentialDriveActionCfg(ActionTerm):
#     asset_name: str
#     left_joint_names: list[str]
#     right_joint_names: list[str]
#     scale: float = 1.0

#     def __call__(self, env, actions):
#         robot = env.scene[self.asset_name]
#         # actions: (N, 2) -> left, right wheel
#         torques = actions * self.scale
#         robot.set_joint_effort_target(
#             torques,
#             joint_names=self.left_joint_names + self.right_joint_names,
#         )
