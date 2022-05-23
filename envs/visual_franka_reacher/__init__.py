from gym.envs.registration import register

register(
    id='franka_agile_grasping-v0',
    entry_point='envs.visual_franka_reacher.reacher_env_visual:FrankaPanda_agile_grasping_V0',
)

# import envs.visual_ur5_reacher.reacher_env_visual