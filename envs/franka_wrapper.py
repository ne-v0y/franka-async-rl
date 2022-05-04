import time
import copy
import numpy as np
import torch.multiprocessing as mp
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Process, Value, Manager
import gym
from envs.visual_franka_reacher.reacher_env import ReacherEnv
from senseact.utils import tf_set_seeds, NormalizedEnv
import argparse
import multiprocessing as mp


def make_env_franka(setup='Franka',
             ip='172.16.0.2',
             seed=9,
             camera_id=0,
             image_width=160,
             image_height=120,
             target_type='reaching',
             image_history=3,
             joint_history=1,
             episode_length=4.0,
             dt=0.1,  
             ignore_monitor_comm=False,):
    # state
    np.random.seed(seed)
    rand_state = np.random.get_state()
    # Create Visual UR5 Reacher environment
    env = ReacherEnv(
            setup=setup,
            host=ip,
            dof=7,
            camera_id=camera_id,
            image_width=image_width,
            image_height=image_height,
            channel_first=True,
            control_type="velocity",
            target_type=target_type,
            image_history=image_history,
            joint_history=joint_history,
            reset_type="zero",
            reward_type="dense",
            derivative_type="none",
            deriv_action_max=5,
            first_deriv_max=2,
            accel_max=1.4,
            speed_max=0.6,
            speedj_a=1.4,
            episode_length_time=episode_length,
            episode_length_step=None,
            actuation_sync_period=1,
            dt=dt,
            run_mode="multiprocess",
            rllab_box=False,
            movej_t=2.0,
            delay=0.0,
            random_state=rand_state,
            ignore_monitor_comm=ignore_monitor_comm
        )
    env = NormalizedEnv(env)
    env.start()
    return env


# class UR5Wrapper():
#     def __init__(self,
#                  setup='Franka',
#                  ip='129.128.159.210',
#                  seed=9,
#                  camera_id=0,
#                  image_width=160,
#                  image_height=120,
#                  target_type='stationary',
#                  image_history=3,
#                  joint_history=1,
#                  episode_length=4.0,
#                  dt=0.04,
#                  ignore_joint=False,
#                  ignore_monitor_comm=False,
#                  ):
#         self.env = make_env_franka(
#                         setup,
#                         ip,
#                         seed,
#                         camera_id,
#                         image_width,
#                         image_height,
#                         target_type,
#                         image_history,
#                         joint_history,
#                         episode_length,
#                         dt,
#                         ignore_monitor_comm,
#                         )

#         self.observation_space = self.env.observation_space['image']
#         self.ignore_joint = ignore_joint
#         if ignore_joint:
#             self.state_space = gym.spaces.Box(low=0, high=1., shape=(0, ), dtype=np.float32)
#             pass
#         else:
#             self.state_space = self.env.observation_space['joint']

#         self.action_space = self.env.action_space

#     def step(self, action):
#         obs_dict, reward, done, _ = self.env.step(action)
#         if self.ignore_joint:
#             return obs_dict['image'], None, reward, done, _
#         else:
#             return obs_dict['image'], obs_dict['joint'], reward, done, _

#     def reset(self):
#         obs_dict = self.env.reset()
#         if self.ignore_joint:
#             return obs_dict['image'], None
#         else:
#             return obs_dict['image'], obs_dict['joint']

#     def terminate(self):
#         self.env.terminate()

class FrankaWrapper():
    def __init__(self,
                 setup='Franka',
                 ip='172.16.0.2',
                 seed=9,
                 camera_id=0,
                 image_width=160,
                 image_height=120,
                 target_type='reaching',
                 image_history=3,
                 joint_history=1,
                 episode_length=4.0,
                 dt=0.01,
                 ignore_joint=False,
                 ignore_monitor_comm=False,
                 ):
        self.env = make_env_franka(
                        setup,
                        ip,
                        seed,
                        camera_id,
                        image_width,
                        image_height,
                        target_type,
                        image_history,
                        joint_history,
                        episode_length,
                        dt,
                        ignore_monitor_comm,
                        )

        self.observation_space = self.env.observation_space['image']
        self.ignore_joint = ignore_joint
        if ignore_joint:
            self.state_space = gym.spaces.Box(low=0, high=1., shape=(0, ), dtype=np.float32)
            pass
        else:
            self.state_space = self.env.observation_space['joint']

        self.action_space = self.env.action_space

    def step(self, action):
        obs_dict, reward, done, _ = self.env.step(action)
        if self.ignore_joint:
            return obs_dict['image'], None, reward, done, _
        else:
            return obs_dict['image'], obs_dict['joint'], reward, done, _

    def reset(self):
        obs_dict = self.env.reset()
        if self.ignore_joint:
            return obs_dict['image'], None
        else:
            return obs_dict['image'], obs_dict['joint']

    def terminate(self):
        self.env.terminate()

if __name__ == '__main__':
    pass

