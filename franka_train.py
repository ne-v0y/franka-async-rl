import numpy as np
import torch
import argparse
import os
import time
import json
from algo.sac_rad_agent import SacRadAgent
import utils
from logger import Logger
import torch.multiprocessing as mp
from configs.franka_config import config
# from envs.franka_wrapper import FrankaWrapper

import gym
import envs.visual_franka_reacher.reacher_env_visual

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--setup', default='Franka')
    parser.add_argument('--ip', default='172.16.0.1', type=str)
    parser.add_argument('--camera_id', default=0, type=int)
    parser.add_argument('--image_width', default=160, type=int)
    parser.add_argument('--image_height', default=120, type=int)
    parser.add_argument('--target_type', default='reaching', type=str)
    parser.add_argument('--random_action_repeat', default=1, type=int)
    parser.add_argument('--agent_action_repeat', default=1, type=int)
    parser.add_argument('--image_history', default=3, type=int)
    parser.add_argument('--joint_history', default=1, type=int)
    parser.add_argument('--ignore_joint', default=False, action='store_true')
    parser.add_argument('--episode_length', default=4.0, type=float)
    parser.add_argument('--dt', default=0.04, type=float)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_step', default=100, type=int)
    parser.add_argument('--env_step', default=100000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--async_mode', default=False, action='store_true')
    parser.add_argument('--max_update_freq', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    # misc
    parser.add_argument('--seed', default=9, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--ignore_monitor_comm', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=10000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--device', default='', type=str)
    parser.add_argument('--lock', default=False, action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # env = FrankaWrapper(
    #     setup = args.setup,
    #     ip = args.ip,
    #     seed = args.seed,
    #     camera_id = args.camera_id,
    #     image_width = args.image_width,
    #     image_height = args.image_height,
    #     target_type = args.target_type,
    #     image_history = args.image_history,
    #     joint_history = args.joint_history,
    #     episode_length = args.episode_length,
    #     dt = args.dt,
    #     ignore_joint = args.ignore_joint,
    #     ignore_monitor_comm=args.ignore_monitor_comm,
    # )
    env = gym.make('franka_agile_grasping-v0')
    utils.set_seed_everywhere(args.seed)
    if not args.async_mode:
        version = 'SACv0'
    elif args.async_mode and args.lock:
        version = 'SACv1'
    elif args.async_mode:
        version = 'SACv2'
    else:
        raise NotImplementedError('Not supported mode!')
    args.work_dir += f'/results/{version}_{args.target_type}_' \
                     f'dt={args.dt}_bs={args.batch_size}_' \
                     f'dim={args.image_width}*{args.image_height}_{args.seed}/'
    utils.make_dir(args.work_dir)
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    # buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    if args.device is '':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    agent = SacRadAgent(
        image_shape=env.observation_space.shape,
        proprioception_shape=env.state_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        net_params=config,
        discount=args.discount,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        actor_lr=args.actor_lr,
        actor_update_freq=args.actor_update_freq,
        critic_lr=args.critic_lr,
        critic_tau=args.critic_tau,
        critic_target_update_freq=args.critic_target_update_freq,
        encoder_tau=args.encoder_tau,
        rad_offset=args.rad_offset,
        async_mode=args.async_mode,
        replay_buffer_capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        max_updates_per_step=args.max_update_freq,
        init_steps=args.init_step,
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, episode_step, done = 0, 0, 0, True
    start_time = time.time()
    obs, state = env.reset()

    for step in range(args.env_step + 1 + args.init_step):
        # sample action for data collection
        if step < args.init_step:
            if step % args.random_action_repeat == 0:
                action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                if step % args.agent_action_repeat == 0:
                    # observation = env.get_state()
                    # obs = observation["image"]
                    # state = observation["joints"]
                    # print("state now", observation["joints"])
                    action = agent.sample_action(obs, state)

        # step in the environment
        next_obs, reward, done, _ = env.step(action)
        next_state = next_obs["joints"]
        next_obs = next_obs["image"]
        episode_reward += reward

        agent.push_sample(obs, state, action, reward, next_obs, next_state, done)

        obs = next_obs
        state = next_state
        episode_step += 1

        if done and step > 0:
            L.log('train/duration', time.time() - start_time, step)
            L.log('train/episode_reward', episode_reward, step)
            start_time = time.time()
            L.dump(step)
            obs, state = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            L.log('train/episode', episode, step)
            if args.save_model and step > 0 and step % args.save_model_freq == 0:
                agent.save(model_dir, step)

        stat = agent.update_networks(step)
        for k, v in stat.items():
            L.log(k, v, step)

    agent.close()
    # Terminate environment processes
    env.terminate()

if __name__ == '__main__':
    main()
