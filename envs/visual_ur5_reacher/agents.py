#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from utils import *
import tf
import copy

MAX_JOINT_VELs = max_joint_vel = (np.ones(7)*0.0).tolist()
MAX_JOINT_VELs[6] = max_joint_vel[6] = 0.6
MAX_JOINT_VELs[5] = max_joint_vel[5] = 0.3
MAX_CART_VELs = [0.008, 0.008, 0.008, 0.02]


def clamp_action(action):
    for i in range(action.shape[0]):
        action[i] = min(
            max(action[i], -MAX_JOINT_VELs[i]), MAX_JOINT_VELs[i]
        )
    return action


class ExpertAgent:
    def __init__(self):
        self.sigma = 0.1

    def act(self, ee_states, ref_ee_states, jacobian, add_noise=True):
        print(ee_states[:3])
        ct = pbvs6(ee_states, ref_ee_states)
        joint_vel = np.matmul(np.linalg.pinv(jacobian), np.expand_dims(ct, axis=1))
        action = np.squeeze(joint_vel)
        if add_noise:
            action += np.random.normal(0, self.sigma, len(action))
        return clamp_action(action)


class RandomAgent:
    def __init__(self, ref_pos):
        self.random_target = self.initialize_random_target(copy.deepcopy(ref_pos))

    def initialize_random_target(self, ref_pos):
        random_xy = np.random.uniform(-0.10, 0.10, 2)
        random_z = np.random.uniform(0, 0.2, 1)[0]
        ref_pos[0:2] += random_xy
        ref_pos[2] += random_z
        rotation_angle = np.radians(10)
        random_eulers = np.random.uniform(-rotation_angle, rotation_angle, 3)
        eulers = tf.transformations.euler_from_quaternion((ref_pos[4], ref_pos[5], ref_pos[6], ref_pos[3]))
        eulers = np.array(eulers) + random_eulers
        quaternions = tf.transformations.quaternion_from_euler(eulers[0], eulers[1], eulers[2])  # tf output quaternion as x, y, z, w
        ref_pos[3:7] = np.array([quaternions[3], quaternions[0], quaternions[1], quaternions[2]])
        return ref_pos

    def act(self, ee_states, ref_ee_states=None, jacobian=None, add_noise=True):
        ct = pbvs6(ee_states, self.random_target)
        joint_vel = np.matmul(np.linalg.pinv(jacobian), np.expand_dims(ct, axis=1))
        action = np.squeeze(joint_vel)  # [random.uniform(-max_vel * 0.2, max_vel * 0.2) for max_vel in MAX_JOINT_VELs]
        sigma = 0.1
        if add_noise:
            action += np.random.normal(0, sigma, len(action))*0.5
        # add clamp to fix max vels
        return clamp_action(action)

class RandomAgent2:
    def __init__(self, ref_pos):
        self.random_target = self.initialize_random_target(copy.deepcopy(ref_pos))
        self.f = 1

    def initialize_random_target(self, ref_pos):
        random_xy = np.random.uniform(-0.10, 0.10, 2)
        random_z = np.random.uniform(0, 0.2, 1)[0]
        ref_pos[0:2] += random_xy
        ref_pos[2] += random_z
        rotation_angle = np.radians(10)
        random_eulers = np.random.uniform(-rotation_angle, rotation_angle, 3)
        eulers = tf.transformations.euler_from_quaternion((ref_pos[4], ref_pos[5], ref_pos[6], ref_pos[3]))
        eulers = np.array(eulers) + random_eulers
        quaternions = tf.transformations.quaternion_from_euler(eulers[0], eulers[1], eulers[2])  # tf output quaternion as x, y, z, w
        ref_pos[3:7] = np.array([quaternions[3], quaternions[0], quaternions[1], quaternions[2]])
        return ref_pos

    def act(self, ee_states, ref_ee_states=None, jacobian=None, add_noise=True):
        ct = pbvs6(ee_states, self.random_target)
        joint_vel = np.matmul(np.linalg.pinv(jacobian), np.expand_dims(ct, axis=1))
        action = 1*np.squeeze(joint_vel)  # [random.uniform(-max_vel * 0.2, max_vel * 0.2) for max_vel in MAX_JOINT_VELs]
        sigma = 0.1
        self.f = -self.f
        action += self.f * 0.6

        # if add_noise:
        #     action += np.random.normal(0, sigma, len(action))*0.5
        # add clamp to fix max vels
        return clamp_action(action)
