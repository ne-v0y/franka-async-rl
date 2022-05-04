from audioop import mul
import imp
from ntpath import join
from pickle import FALSE
from cv2 import ORB_HARRIS_SCORE
import gym
from gym.core import ActionWrapper
import numpy as np
from gym import spaces
import os

from numpy.core.defchararray import count
import rospy
from PIL import Image
import math
from collections import deque
from envs.visual_franka_reacher.utils import *

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
import time
import logging

from envs.visual_franka_reacher.agents import *
from franka_interface import ArmInterface, RobotEnable, GripperInterface
# ids camera lib for use of IDS ueye cameras.
# https://www.ids-imaging.us/files/downloads/ids-peak/readme/ids-peak-linux-readme-1.2_EN.html
#import ids
import time
import signal
import cv2
import multiprocessing
from gym.spaces import Box as GymBox


class FrankaPanda_agile_grasping_V0(gym.Env):
    """
    Gym env for the real franka robot. Set up to perform the placement of a peg that starts in the robots hand into a slot
    """
    def __init__(self, config_file =None):
        """
        Initializes a franka gym environment for one arm. Sets up parameters required to load the scene -
        reset() must be called to load the environment.

        Parameters
        ----------
        config_file : str
            Path to yaml file consisting of simulation configuration.
        """
        signal.signal(signal.SIGINT, self.exit_handler)
        config_file = os.path.join(os.path.dirname(__file__), os.pardir, 'visual_franka_reacher/reacher.yaml')
        self.configs = configure(config_file)
        self.conf_exp = self.configs['experiment']
        self.conf_env = self.configs['environment']
        rospy.init_node("franka_robot_gym")
        self.init_joints_bound = self.conf_env['reset-bound']
        #self.target_joints = self.conf_env['target-bound']
        self.safe_bound_box = np.array(self.conf_env['safe-bound-box'])
        self.target_box = np.array(self.conf_env['target-box'])
        self.joint_angle_bound = np.array(self.conf_env['joint-angle-bound'])
        self.return_point = self.conf_env['return-point']
        self.out_of_boundary_flag = False
        self.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']

        self.robot = ArmInterface(True)
        force = 1e-6
        self.robot.set_collision_threshold(cartesian_forces=[force,force,force,force,force,force])
        self.robot.exit_control_mode(0.1)
        self.robot_status = RobotEnable()
        self.control_frequency = 25
        self.rate = rospy.Rate(self.control_frequency)

        self.ct = 0.04
        self.tv = time.time()

        self._image_width = 640 // 4
        self._image_height = 480 // 4


        self.joint_states_history = deque(np.zeros((5, 21)), maxlen=5)
        self.torque_history = deque(np.zeros((5, 7)), maxlen=5)
        self.last_action_history = deque(np.zeros((5, 7)), maxlen=5)
        self.time_out_reward = False
        action_dim = 7
        self.prev_action = np.zeros(action_dim)
        self.obs_image = None
        self.max_time_steps = 200

        self.reward_functions = {'default': self.get_reward}

        self.camera = camera()

        self.previous_place_down = None

        self.joint_action_limit = 0.1

        self.action_space = GymBox(low=-self.joint_action_limit * np.ones(7), high=self.joint_action_limit*np.ones(7))
        self.joint_angle_low = [j[0] for j in self.joint_angle_bound]
        self.joint_angle_high = [j[1] for j in self.joint_angle_bound]

        self.state_space = GymBox(
            low=np.array(
                self.joint_angle_low  # q_actual
                + list(-np.ones(7)*self.joint_action_limit)  # qd_actual
                + list(-np.ones(7)*self.joint_action_limit)  # previous action in cont space
            ),
            high=np.array(
                self.joint_angle_high  # q_actual
                + list(np.ones(7)*self.joint_action_limit)  # qd_actual
                + list(np.ones(7)*self.joint_action_limit)    # previous action in cont space
            )
        )
        self.observation_space = GymBox(low=0., high=255.,
                                     shape=[self._image_height, self._image_width, 3],
                                     dtype=np.uint8)

    def reset(self):
        """
        reset robot to random pose
        Returns
        -------
        object
            Observation of the current state of this env in the format described by this envs observation_space.
        """
        self.time_steps = 0
        self.robot_status.enable()
        # stop the robot
        self.apply_joint_vel(np.zeros((7,)))

        # # random pose
        # random_reset_pose = [random_val_continous(joint_range) for joint_range in self.angle_safety_bound]
        # random_rest_joints = dict(zip(self.joint_names, random_reset_pose))
        # smoothly_move_to_position_vel(self.robot, self.robot_status, random_rest_joints ,MAX_JOINT_VELs=1.3)

        self.target_pose = [np.random.uniform(box_range[0], box_range[1]) for box_range in self.target_box]
        # # self.target_pose = [1.9, 0, -1.93, -1.52, 0.10, 1.52, 0.8]
        # self.target_pose[1] = 0  # set y to 0
        self.target_pose[2] = 0.5  # set z to string length
        self.reset_ee_quaternion = [0,-1.,0,0]
        
        # get latest observation
        #time.sleep(1)
        _ = self.render()  # skip one frame of the camera
        obs = self.get_state()
        #print(self.ee_position)
        # obs = np.concatenate((obs['joints'], self.target_pose))
        
        self.out_of_boundary_flag = False


        reset_pose = dict(zip(self.joint_names, [random.randint(-10, 10) / 100, 0, 0, -1.6, 0, 1.6, 0.8]))
        smoothly_move_to_position_vel(self.robot, self.robot_status, reset_pose ,MAX_JOINT_VELs=1.3)

        # reset end-effector pose
        # self.move_to_pose_ee(self.target_pose)#np.array([0.42, 0, 0.35]))
        #smoothly_move_to_position_vel(self.robot, self.robot_status, random_rest_joints,MAX_JOINT_VELs=1.3)
        
        # self.move_to_pose_ee(np.array([0.42, 0, 0.05]))

        reset_pose = dict(zip(self.joint_names,self.return_point))
        smoothly_move_to_position_vel(self.robot, self.robot_status, reset_pose ,MAX_JOINT_VELs=1.3)
        # print("here", self.robot.endpoint_pose()["orientation"])

        # stop the robot
        self.apply_joint_vel(np.zeros((7,)))

        # get the observation
        obs_robot = self.get_state()
        # self.obs_object = self.camera.get_state()
        # gripper_joint = self.gripper.joint_positions()['panda_finger_joint1']
        # print("ee", self.ee_position)
        # obs = np.concatenate((obs_robot['joints'],obs_object))

        obs = np.concatenate((obs_robot["joints"], obs_robot["joint_vels"], [0]*7))
        # obs = np.concatenate((self.ee_position_table, [joint_position], self.obs_object))

        # print('reset_done')
        self.time_steps = 0

        # self.camera.empty_q()
        self.tv = time.time()
        # self.monitor.reset()
        return self.camera.get_state(), obs

    def render(self):
        # get camera image
        
        width, height = self.conf_env['image-width'], self.conf_env['image-width']
        '''
        _, _ = self.cam.next()  # skip one frame
        img, meta = self.cam.next()
        pil_img = Image.fromarray(img).convert('L').crop(self.crop_bbox).resize((width, height))
        if self.conf_env['visualization']:
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bgr_img = bgr_img[108:1000 + 108, 468:1000 + 468]
            self.eye_in_hand_cam_pub.publish(self.br.cv2_to_imgmsg(bgr_img, "bgr8"))
        #return np.array(pil_img)
        '''
        return np.zeros((width, height))

    def get_robot_jacobian(self):
        return self.robot.zero_jacobian()
 
    def euler_from_quaternion(self,q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        w,x,y,z = q        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

    def get_state(self):
        # get object state
        # self.obs_object = self.camera.get_state()
        
        # get robot states
        joint_angles = extract_values(self.robot.joint_angles(), self.joint_names)
        joint_velocitys = extract_values(self.robot.joint_velocities(), self.joint_names)
        # joint_efforts = extract_values(self.robot.joint_efforts(), self.joint_names)
        ee_pose = self.robot.endpoint_pose()
        ee_quaternion = [ee_pose['orientation'].w, ee_pose['orientation'].x,
                         ee_pose['orientation'].y, ee_pose['orientation'].z]
        # ee_velocities = self.robot.endpoint_velocity()
        # joint_state_vector = np.concatenate((
        #     joint_angles, joint_velocitys, [j / 100 for j in joint_efforts]
        # )).astype(np.float32)
        # joint_state_no_torques = np.concatenate((
        #     joint_angles, joint_velocitys)).astype(np.float32)
        # ee_state_vector = np.concatenate((
        #     ee_pose['position'], ee_quaternion, ee_velocities['linear'], ee_velocities['angular']
        # )).astype(np.float32)
        # joint_torques = np.array(
        #     [j / 100 for j in joint_efforts]
        # )
        # print(ee_quaternion)
        # t = time.time()
        image = self.camera.get_state()
        # print("time used", time.time() - t)
        self.last_action_history.append(self.prev_action)
        
        observation = {
            
            # 'joint_states': joint_state_vector,
            # 'joint_torques': joint_torques,
            # 'joint_states_no_torques': joint_state_no_torques,
            # 'ee_states': ee_state_vector,
            'image': np.array(image),
            'last_action': self.prev_action,
            'joints': np.array(joint_angles),
            'joint_vels': np.array(joint_velocitys)
        }
        # print('orientation',ee_pose['orientation'])
        self.ee_position = ee_pose['position']
        # print(self.ee_position)
        self.ee_position_table = np.array([1.07-self.ee_position[0], 0.605-self.ee_position[1], self.ee_position[2]])
        self.ee_orientation = ee_quaternion
        #return observation['joints']
        return observation

    def get_reward(self, image, action):
        """
        Calculates the reward based on the current state of the agent and the environment.

        Parameters
        ----------
        info : dict

        Returns
        -------
        float
            Value of the reward.
        """

        if (image.shape[-1] == 1):
            mask = image

        else:
            image = image[:, :, -3:]
            lower = [0, 0, 120]
            upper = [120, 90, 255]
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            mask = cv2.inRange(image, lower, upper)
        # cv2.imshow('mask', mask)
        # # cv2.imwrite("/home/franka/project/franka_async_rl_noni/ur5_async_rl-master/mask/{}.jpg".format(self.counter), mask)
        size_x, size_y = mask.shape
        # reward for reaching task, may not be suitable for tracking
        if 255 in mask:
            xs, ys = np.where(mask == 255.)
            reward_x = 1 / 2  - np.abs(xs - int(size_x / 2)) / size_x
            reward_y = 1 / 2 - np.abs(ys - int(size_y / 2)) / size_y
            reward = np.sum(reward_x * reward_y) / self._image_width / self._image_height
        else:
            reward = 0
        reward *= 800
        reward = np.clip(reward, 0, 4)

        # scale = (np.abs(action[0] + action[4]) + np.abs(np.pi + np.sum(action[1:4])))

        return reward

    def out_of_boundaries(self):
        x, y, z = self.robot.endpoint_pose()['position']
        
        x_bound = self.safe_bound_box[0,:]
        y_bound= self.safe_bound_box[1,:]
        z_bound = self.safe_bound_box[2,:]
        if scalar_out_of_range(x, x_bound):
            print('x out of bound, motion will be aborted! x {}'.format(x))
            return True
        if scalar_out_of_range(y, y_bound):
            print('y out of bound, motion will be aborted! y {}'.format(y))
            return True
        if scalar_out_of_range(z, z_bound):
            print('z out of bound, motion will be aborted!, z {}'.format(z))
            return True
        return False

    def apply_joint_vel(self, joint_vels):
        joint_vels = dict(zip(self.joint_names, joint_vels))
        self.robot.set_joint_velocities(joint_vels)
        # self.rate.sleep()
        # print(time.time()-self.tv)
        
        
        return True

    def step(self, action, agent_started=False, pose_vel_limit=0.3, ignore_safety=False):  # 0.08
        self.tv=time.time()
        self.robot_status.enable()
        
        # limit joint action
        action = np.clip(action, -self.joint_action_limit, self.joint_action_limit)

        # convert joint velocities to pose velocities
        pose_action = np.matmul(self.get_robot_jacobian(), action)

        # limit action
        pose_action[:3] = np.clip(pose_action[:3], -pose_vel_limit, pose_vel_limit)

        # safety
        out_boundary = self.out_of_boundaries()
        if ignore_safety == False:
            pose_action[:3] = self.safe_actions(pose_action[:3])

        # calculate joint actions
        d_angle =  np.array(self.euler_from_quaternion(self.reset_ee_quaternion)) - np.array(self.euler_from_quaternion(self.ee_orientation))
        for i in range(3):
            if d_angle[i] < -np.pi:
                d_angle[i] += 2*np.pi
            elif d_angle[i] > np.pi:
                d_angle[i] -= 2*np.pi
        d_angle *= 0.5

        d_X = pose_action
        if out_boundary:
            action = self.get_joint_vel_from_pos_vel(d_X)

        action = self.handle_joint_angle_in_bound(action)
        self.apply_joint_vel(action)
        self.prev_action = action

        # get next observation
        observation_robot = self.get_state()

        # calculate reward
        reward = self.get_reward(observation_robot["image"], action)

        # termination
        if self.time_steps >= self.max_time_steps:
            done = True
            self.apply_joint_vel(np.zeros((7,)))
        else:
            done = False

        self.time_steps += 1
        
        # construct the state
        obs = np.concatenate((observation_robot["joints"], observation_robot["joint_vels"], action))
        env_observation = {"image": observation_robot["image"], "joints": obs}

        time.sleep(max(0, self.ct - (time.time()-self.tv)))

        return env_observation, reward, done, None
    
    def handle_joint_angle_in_bound(self, action):
        current_joint_angle = self.robot.joint_angles()
        in_bound = [False] * 7
        for i, joint_name in enumerate(self.joint_names):
            if current_joint_angle[joint_name] <= self.joint_angle_bound[i][1] and current_joint_angle[joint_name] >= self.joint_angle_bound[i][0]:
                in_bound[i] = True
        
        is_safe = np.all(in_bound)
        corrective_position = np.array(list(current_joint_angle.values())[:7])

        # print("before", action)
        if not is_safe:
            # print("joint {} out of bound, correcting".format((np.where(np.array(in_bound) == False))[0]))
            corrective_position = np.array(self.return_point) - corrective_position
            if np.linalg.norm(corrective_position) != 0:
                corrective_position /= np.linalg.norm(corrective_position) / 0.1
                c_action = np.clip(corrective_position, -self.joint_action_limit, self.joint_action_limit)
            for i in range(len(in_bound)):
                if not in_bound[i]:
                    action[i] = c_action[i]
            # print("after", action)
        return action

    def get_timeout_reward(self):
        if self.time_out_reward:
            reward = -1
            print('call time out reward {:+.3f}'.format(reward))
            return reward
        else:
            return 0

    def move_to_pose_ee(self, ref_ee_pos, pose_vel_limit=0.2):
        counter = 0
        # print('11111', rospy.Time.now())
        
        while True:
            self.robot_status.enable()
            # print(self.robot_status.state())
            counter += 1
            #action = agent.act(observations['ee_states'], ref_ee_pos, self.get_robot_jacobian(), add_noise=False)
            self.get_state()
            action = np.zeros((4,))
            action[:3] = ref_ee_pos-self.ee_position
            action[-1] = 1
            
            #if max(np.abs(action[:3])) < 0.005 or 
            #print(action)
            if max(np.abs(action[:3])) < 0.005 or counter > 100:
                break

            #self.step(action, ignore_safety=True)
            # limit action
            pose_action = np.clip(action[:3], -pose_vel_limit, pose_vel_limit)

            # calculate joint actions
            d_angle =  np.array(self.euler_from_quaternion(self.reset_ee_quaternion)) - np.array(self.euler_from_quaternion(self.ee_orientation))
            for i in range(3):
                if d_angle[i] < -np.pi:
                    d_angle[i] += 2*np.pi
                elif d_angle[i] > np.pi:
                    d_angle[i] -= 2*np.pi
            d_angle *= 0.5
            #print('d_angle', d_angle)
            d_X = np.array([pose_action[0], pose_action[1], pose_action[2], d_angle[0],d_angle[1],d_angle[2]])
            joints_action = self.get_joint_vel_from_pos_vel(d_X)
            # print('joints_action', joints_action)
            self.apply_joint_vel(joints_action)
            
            # action cycle time
            self.rate.sleep()
        self.apply_joint_vel(np.zeros((7,)))

    def get_joint_vel_from_pos_vel(self, pose_vel):
        return np.matmul(np.linalg.pinv( self.get_robot_jacobian() ), pose_vel)

    def safe_actions(self, action):
        out_boundary = self.out_of_boundaries()
        x, y, z = self.robot.endpoint_pose()['position']
        self.box_Normals = np.zeros((6,3))
        self.box_Normals[0,:] = [1,0,0]
        self.box_Normals[1,:] = [-1,0,0]
        self.box_Normals[2,:] = [0,1,0]
        self.box_Normals[3,:] = [0,-1,0]
        self.box_Normals[4,:] = [0,0,1]
        self.box_Normals[5,:] = [0,0,-1]
        self.planes_d = [   self.safe_bound_box[0][0],
                            -self.safe_bound_box[0][1],
                            self.safe_bound_box[1][0],
                            -self.safe_bound_box[1][1],
                            self.safe_bound_box[2][0],
                            -self.safe_bound_box[2][1]]
        if out_boundary:
            action = np.zeros((3,))
            for i in range(6):
                action += 0.05 * self.box_Normals[i] * ( (self.box_Normals[i].dot(np.array([x,y,z])) - self.planes_d[i]) < 0 ) 
        
        return action

    def close(self):
        # stop the robot
        self.apply_joint_vel(np.zeros((7,)))
    
    def exit_handler(self,signum):
        self.camera.thread.join()
        self.camera.release()
        exit(signum)

    
    def terminate(self):
        self.close()
        self.exit_handler(1)

class camera():
    def __init__(self):

        cv2.setNumThreads(1)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.frame= None

    def get_state(self):
    
        ret, frame = self.cap.read()#cv2.imread('marker2.jpg')
        if ret:
            #wrapped = cv2.warpPerspective(frame, pt, (1523, 1214))
            frame = cv2.resize(frame, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_AREA)
            self.frame = frame
        return frame
        # return np.zeros((120, 160, 3))

import fmq
class camera_with_thread():
    def __init__(self, record=True):
        
        self.record = record
        
        cv2.setNumThreads(1)

        self.width = 640
        self.height = 480
        self.image_size_one_color = self.width * self.height

        self.q = fmq.Queue(3)

        self.lock = multiprocessing.Lock()

        self.reward = multiprocessing.Value('d', 0) 

        # self.thread = multiprocessing.Process(target=self.loop,args=(self.object_position_x, self.object_position_y,))
        self.thread = multiprocessing.Process(target=self.loop, args=(self.q,self.reward, self.lock))
        self.cap = cv2.VideoCapture(0)

    def get_state(self):
        # self.object_position = np.array([self.object_position_x.value, self.object_position_y.value])
        # return self.object_position

        # img =  np.ones([self.height, self.width, 3])
        # img = np.array(self.q.get()).reshape((self.height, self.width, 3))
        return self.q.get()
        # return self.reward.value
    
    def empty_q(self):
        self.q.empty()
    
    def loop(self, frame_, reward_, lock):
        while True:
            # lock.acquire()
            ret, frame = self.cap.read()

            if ret:
                frame = cv2.resize(frame, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_AREA)
                # if self.record:
                #     self.out.write(frame)
                # cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                #wrapped = cv2.warpPerspective(frame, pt, (1523, 1214))
                # array = np.asarray(frame.copy(), dtype='i').flatten()
                
                image = frame[:, :, -3:]
                lower = [0, 0, 120]
                upper = [120, 90, 255]
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")

                mask = cv2.inRange(image, lower, upper)
                cv2.imshow('mask', frame)
                # # cv2.imwrite("/home/franka/project/franka_async_rl_noni/ur5_async_rl-master/mask/{}.jpg".format(self.counter), mask)
                # size_x, size_y = mask.shape
                # # reward for reaching task, may not be suitable for tracking
                # if 255 in mask:
                #     xs, ys = np.where(mask == 255.)
                #     reward_x = 1 / 2  - np.abs(xs - int(size_x / 2)) / size_x
                #     reward_y = 1 / 2 - np.abs(ys - int(size_y / 2)) / size_y
                #     reward = np.sum(reward_x * reward_y) / self.width / self.height
                # else:
                #     reward = 0
                # reward *= 800
                # reward = np.clip(reward, 0, 4)
                # self.reward.value = reward

                frame_.put(frame)

                # lock.release()
                
                
        self.release()
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def exit(self):
        self.thread.join()
        self.thread.close()
        return
   
class monitor():
    def __init__(self, target_type='reaching', width=160, height=90, radius=7):
        mpl.rcParams['toolbar'] = 'None'
        plt.ion()
        self.fig = plt.figure()
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self.fig.canvas.toolbar_visible = False
        self.ax = plt.axes(xlim=(0, width), ylim=(0, height))
        self.target = plt.Circle((0, 0), radius, color='red')
        self.ax.add_patch(self.target)
        plt.axis('off')
        self.radius = radius
        self.width = width
        self.height = height
        self.target_type=target_type
        actuator_args = {
            'array_len': 1,
            'array_type': 'd',
            'np_array_type': 'd',
        }
        self.reset()
        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()

    def reset(self):
        if self.target_type == 'static':
            self.target.set_center((self.width / 2, self.height / 2))
            self.velocity_x, self.velocity_y = 0, 0
        if self.target_type == 'reaching':
            x, y = np.random.random(2)
            self.target.set_center(
                (self.radius + x * (self.width - 2 * self.radius),
                 self.radius + y * (self.height - 2 * self.radius))
            )
            self.velocity_x, self.velocity_y = 0, 0
        elif self.target_type == 'tracking':
            self.target.set_center((self.width / 2, self.height / 2))
            self.velocity_x, self.velocity_y = np.random.random(2) - 0.5
            velocity = np.sqrt(self.velocity_x ** 2 + self.velocity_y ** 2)
            self.velocity_x /= velocity
            self.velocity_y /= velocity

        x, y = self.target.get_center()
        if x + self.velocity_x + self.radius > self.width or \
        x + self.velocity_x - self.radius < 0:
            self.velocity_x = -self.velocity_x
        if y + self.velocity_y + self.radius > self.height or \
        y + self.velocity_y - self.radius < 0:
            self.velocity_y = -self.velocity_y
        self.target.set_center((x + self.velocity_x, y + self.velocity_y))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.032)
