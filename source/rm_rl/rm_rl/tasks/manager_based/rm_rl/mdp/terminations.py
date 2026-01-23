import torch


def time_out(env):
    return env.episode_length_buf >= env.max_episode_length


# def fallen_over(env):
#     # 示例：pitch / roll 过大
#     gravity = env.robot.data.projected_gravity_b
#     return gravity[:, 2] < 0.2
