# # mdp/commands.py

# import torch
# from isaaclab.managers.command_manager import CommandTerm, CommandTermCfg
# from isaaclab.utils import configclass


# @configclass
# class RandomXYCommandCfg(CommandTermCfg):
#     resampling_time_range: tuple[float, float] = (5.0, 5.0)

# class RandomXYCommand(CommandTerm):

#     def __init__(self, cfg: RandomXYCommandCfg, env):
#         super().__init__(cfg, env)
#         self._command = torch.zeros((env.num_envs, 3), device=env.device)

#     # ====== 必须实现的抽象方法 ======

#     def _resample_command(self, env_ids: torch.Tensor):
#         cmd = torch.randn((len(env_ids), 3), device=self.device)
#         cmd[:, 2] = 0.0
#         cmd = cmd / torch.linalg.norm(cmd, dim=1, keepdim=True)
#         self._command[env_ids] = cmd

#     def _update_command(self):
#         pass  # command 在 resample 后保持不变

#     def _update_metrics(self):
#         pass  # 不记录额外指标

#     @property
#     def command(self):
#         return self._command

