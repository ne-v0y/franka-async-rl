# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import time
import gym
import sys
from multiprocessing import Array, Value

from senseact.rtrl_base_env import RTRLBaseEnv
from senseact.devices.ur import ur_utils  
from envs.visual_franka_reacher.ur_setup import setups
from senseact.sharedbuffer import SharedBuffer
from senseact import utils
import cv2 as cv
from envs.visual_franka_reacher.camera_communicator import CameraCommunicator, DEFAULT_HEIGHT, DEFAULT_WIDTH
from envs.visual_franka_reacher.monitor_communicator import MonitorCommunicator

import rospy
import signal
from sensor_msgs import msg
from franka_interface import ArmInterface, RobotEnable, GripperInterface


class ReacherEnv(RTRLBaseEnv, gym.core.Env):
    """A class implementing Franka Reaching and tracking environments.
    """
    def __init__(self,
                 setup,
                 host=None,
                 dof=7,
                 camera_id=0,
                 image_width=160,
                 image_height=120,
                 channel_first=True,
                 control_type='velocity',
                 derivative_type='none',
                 target_type='reacher',
                 reset_type='random',
                 deriv_action_max=10,
                 first_deriv_max=10,  # used only with second derivative control
                 vel_penalty=0,
                 image_history=3,
                 joint_history=1,
                 actuation_sync_period=1,
                 episode_length_time=4,
                 episode_length_step=None,
                 rllab_box = False,
                 accel_max=None,
                 speed_max=None,
                 dt=0.008,
                 delay=0.0,  # to simulate extra delay in the system
                 ignore_monitor_comm=False,
                 **kwargs):
        """Inits ReacherEnv class with task and robot specific parameters.
        """

        # Robot related initialization
        signal.signal(signal.SIGINT, self.exit_handler)
        rospy.init_node('franka_communicator')
        self.robot = ArmInterface(True)
        self.gripper = GripperInterface()
        self.robot_status = RobotEnable()
        self.control_frequency = 80
        self.rate = rospy.Rate(self.control_frequency)

        self.current_joint_obs = [0] * 7
        self.prev_action = [0] * 7
        # rospy.Subscriber("/joint_states", msg.JointState, self.joint_state_cb)
        self.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']

        self.package_type = np.dtype(
            [
                # ('joint_states', 'f', (21,)),
                # ('joint_torques', 'f', (7,)),
                # ('joint_states_no_torques', 'f', (14,)),
                # ('ee_states', 'f', (13,)),
                # ('last_action', 'f', (7,)),
                ('joints_angles', 'f', (7,)),
                ('joint_vels', 'f',(7,))
            ])
        # Check that the task parameters chosen are implemented in this class
        assert dof in [2, 5, 6, 7]
        assert control_type in ['position', 'velocity', 'acceleration']
        assert derivative_type in ['none', 'first', 'second']

        assert target_type in ['static', 'reaching', 'tracking']
        assert reset_type in ['random', 'zero', 'none']
        assert actuation_sync_period >= 0

        if episode_length_step is not None:
            assert episode_length_time is None
            self._episode_length_step = episode_length_step
            self._episode_length_time = episode_length_step * dt
        elif episode_length_time is not None:
            assert episode_length_step is None
            self._episode_length_time = episode_length_time
            self._episode_length_step = int(episode_length_time / dt)
        else:
            #TODO: should we allow a continuous behaviour case here, with no episodes?
            print("episode_length_time or episode_length_step needs to be set")
            raise AssertionError



        # Task Parameters
        self._host = setups[setup]['host'] if host is None else host
        self._image_history = image_history
        self._joint_history = joint_history
        self._image_width = image_width
        self._image_height = image_height
        self._channel_first = channel_first
        self._dof = dof
        self._control_type = control_type
        self._derivative_type = derivative_type
        self._target_type = target_type
        self._reset_type = reset_type
        self._vel_penalty = vel_penalty # weight of the velocity penalty
        self._deriv_action_max = deriv_action_max
        self._first_deriv_max = first_deriv_max
        # self._speedj_a = speedj_a
        self._delay = delay
        self.return_point = None
        if accel_max==None:
            accel_max = setups[setup]['accel_max']
        if speed_max==None:
            speed_max = setups[setup]['speed_max']
        if self._dof == 5:
            self._joint_indices = [0, 1, 2, 3, 4]
            self._end_effector_indices = [0, 1, 2]
        elif self._dof == 2:
            self._joint_indices = [1, 2]
            self._end_effector_indices = [1, 2]
        elif self._dof == 6:
            self._joint_indices = [0, 1, 2, 3, 4, 5]  # TODO: should define franka to dof7?

        elif self._dof == 7:
            self._joint_indices = [0, 1, 2, 3, 4, 5, 6]

        # TODO
        self.safe_bound_box = [[ 0.3, 0.8 ], [ -0.5, 0.5 ], [ 0.01, 0.2 ]]
        self.target_bound_box = []

        self._disable_gripper = True
        # Arm/Control/Safety Parameters
        self._end_effector_low = setups[setup]['end_effector_low']
        self._end_effector_high = setups[setup]['end_effector_high']
        self._angles_low = setups[setup]['angles_low'][self._joint_indices]
        self._angles_high = setups[setup]['angles_high'][self._joint_indices]
        self._speed_low = -np.ones(self._dof) * speed_max
        self._speed_high = np.ones(self._dof) * speed_max
        self._accel_low = -np.ones(self._dof) * accel_max
        self._accel_high = np.ones(self._dof) * accel_max

        self._box_bound_buffer = setups[setup]['box_bound_buffer']
        self._angle_bound_buffer = setups[setup]['angle_bound_buffer']
        self._q_ref = setups[setup]['q_ref']
        self._ik_params = setups[setup]['ik_params']

        # State Variables
        self._q_ = np.zeros((self._joint_history, self._dof))
        self._qd_ = np.zeros((self._joint_history, self._dof))

        self._episode_steps = 0

        self._pstop_time_ = None
        self._pstop_times_ = []
        # self._safety_mode_ = ur_utils.SafetyModes.NONE
        self._max_pstop = 10
        self._max_pstop_window = 600
        self._clear_pstop_after = 2
        self._x_target_ = np.frombuffer(Array('f', 3).get_obj(), dtype='float32')
        self._x_ = np.frombuffer(Array('f', 3).get_obj(), dtype='float32')
        self._reward_ = Value('d', 0.0)

        # Set up action and observation space
        if self._derivative_type== 'second' or self._derivative_type== 'first':
            self._action_low = -np.ones(self._dof) * self._deriv_action_max
            self._action_high = np.ones(self._dof) * self._deriv_action_max
        else: # derivative_type=='none'
            if self._control_type == 'position':
                self._action_low = self._angles_low
                self._action_high = self._angles_high
            elif self._control_type == 'velocity':
                self._action_low = self._speed_low
                self._action_high = self._speed_high
            elif self._control_type == 'acceleration':
                self._action_low = self._accel_low
                self._action_high = self._accel_high

        # TODO: is there a nicer way to do this?
        if rllab_box:
            from rllab.spaces import Box as RlBox  # use this for rllab TRPO
            Box = RlBox
        else:
            from gym.spaces import Box as GymBox  # use this for baselines algos
            Box = GymBox

        # [[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.0698], [-	2.8973, 2.8973], [-0.01750, 3.75250], [-0.8, -0.8]]
        self._angles_low = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973,-0.0175, -0.8]
        self._angles_high = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, -0.8]

        self._observation_space = Box(
            low=np.array(
                list(self._angles_low * self._joint_history)  # q_actual
                + list(-np.ones(self._dof * self._joint_history))  # qd_actual
                + list(-self._action_low)  # previous action in cont space
            ),
            high=np.array(
                list(self._angles_high * self._joint_history)  # q_actual
                + list(np.ones(self._dof * self._joint_history))  # qd_actual
                + list(self._action_high)    # previous action in cont space
            )
        )
        self._action_space = Box(low=self._action_low, high=self._action_high)

        # print(self._observation_space, self._action_space)
        # self._action_space = Box(
        #     low=np.array([0,0,0,0,0,-0.6,-0.63]),
        #         high=np.array([0,0,0,0,0,0,6.63]),
        #         dtype=np.float32
        # )

        if rllab_box:
            from rllab.envs.env_spec import EnvSpec
            self._spec = EnvSpec(self.observation_space, self.action_space)

        # Only used with second derivative control
        self._first_deriv_ = np.zeros(len(self.action_space.low))

        # Communicator Parameters
        communicator_setups = {
            # 'Franka':
            #                        {
            #                         'num_sensor_packets': joint_history,

            #                         'kwargs': {'host': self._host,
            #                                    'actuation_sync_period': actuation_sync_period,
            #                                    'buffer_len': joint_history + SharedBuffer.DEFAULT_BUFFER_LEN,
            #                                    }
            #                         },
                                'Camera': {
                                    'num_sensor_packets': image_history,
                                    'kwargs': {'res': (640, 480), 'device_id': camera_id}
                                    # 'kwargs': {'device_id': camera_id}
                                    }
                               }

        if not ignore_monitor_comm:
            communicator_setups['Monitor'] = {
                                    'kwargs': {'target_type': target_type}
                                    }
        # if self._delay > 0:
        #     from senseact.devices.franka.franka_communicator_delay import FrankaCommunicator
        #     communicator_setups['Franka']['kwargs']['delay'] = self._delay
        # else:
        #     from senseact.devices.franka.franka_communicator import FrankaCommunicator
        # communicator_setups['Franka']['Communicator'] = FrankaCommunicator
        communicator_setups['Camera']['Communicator'] = CameraCommunicator

        if 'Monitor' in communicator_setups:
            communicator_setups['Monitor']['Communicator'] = MonitorCommunicator

        super(ReacherEnv, self).__init__(communicator_setups=communicator_setups,
                                         action_dim=len(self.action_space.low),
                                         observation_dim=-2, # ignore the _senseation_buffer in base class
                                         dt=dt,
                                         **kwargs)

        if channel_first:
            image_space = gym.spaces.Box(low=0., high=255.,
                                shape=[3 * image_history, image_height, image_width],
                                dtype=np.uint8)
        else:
            image_space = gym.spaces.Box(low=0., high=255.,
                                     shape=[image_height, image_width, 3 * image_history],
                                     dtype=np.uint8)

        self._observation_space = gym.spaces.Dict({
            'joint': self._observation_space,
            'image': image_space
        })

        self._image_buffer = SharedBuffer(
            buffer_len=SharedBuffer.DEFAULT_BUFFER_LEN,
            array_len=int(640 * 480 * 3 * self._image_history),
            array_type='H',
            np_array_type='H',
        )

        # self._joint_buffer = SharedBuffer(
        #     buffer_len=SharedBuffer.DEFAULT_BUFFER_LEN,
        #     array_len=int(np.product(self._observation_space['joint'].shape)),
        #     array_type='d',
        #     np_array_type='d',
        # )

        self._joint_buffer = np.array([0] * 21)

        # The last action
        self._action_ = self._rand_obj_.uniform(self._action_low, self._action_high)

        self._nothing_packet = [0] * 7

        # Defining packet structure for quickly building actions
        # self._reset_packet = np.ones(self._actuator_comms['Franka'].actuator_buffer.array_len) * ur_utils.USE_DEFAULT
        # self._reset_packet[0] = ur_utils.COMMANDS['MOVEJ']['id']
        # self._reset_packet[1:1 + 6] = self._q_ref
        # self._reset_packet[-2] = movej_t

        self.angle_safety_bound = [[0.26, 2.4], [-1.17, 1.17], [-2.38, -1.2], [-2.5, -1.8], [-0.4, 0.4], [1.7, 2.5], [-0.8, -0.8]]
        self._reset_packet = [1.9, 0, -1.93, -1.52, 0.10, 1.52, 0.8]

        self.current_joint_obs = [0] * 7

        self.previous_reward = 0
        # Make sure all communicatators are ready
        time.sleep(2)
        self.counter = 0
    
    # def joint_state_cb(self, data):
    #     self.current_joint_obs = data
    #     print("here")

    def exit_handler(self):
        exit(1)

    def _reset_(self):
        """Resets the environment episode.

        Moves the arm to either fixed reference or random position and
        generates a new target within a safety box.
        """
        print("Resetting")
        if 'Monitor' in self._actuator_comms:
            self._actuator_comms['Monitor'].actuator_buffer.write(0)
        
        # self._q_target_ = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        self._action_ = self._rand_obj_.uniform(self._action_low, self._action_high)
        # self._cmd_prev_ = np.zeros(len(self._action_low))  # to be used with derivative control of velocity
        # if self._reset_type != 'none':
        #     if self._reset_type == 'random':
        #         reset_angles, _ = self._pick_random_angles_()
        #     elif self._reset_type == 'zero':
        #         reset_angles = self._q_ref[self._joint_indices]
        #     self._reset_arm(reset_angles)
        # if self._episode_steps != 0:

        # self._reset_arm()

        # self._actuator_comms['Franka'].actuator_buffer.write([0] * 7)
        # time.sleep(1)
        for i in range(self._image_history):
            self._sensor_to_sensation_()
        rand_state_array_type, rand_state_array_size, rand_state_array = utils.get_random_state_array(
            self._rand_obj_.get_state()
        )
        np.copyto(self._shared_rstate_array_, np.frombuffer(rand_state_array, dtype=rand_state_array_type))
        
        print("Reset done")

    def _pick_random_angles_(self):
        """Generates a set of random angle positions for each joint."""
        movej_q = self._q_ref.copy()
        while True:
            reset_angles = self._rand_obj_.uniform(self._angles_low, self._angles_high)
            movej_q[self._joint_indices] = reset_angles
            inside_bound, inside_buffer_bound, mat, xyz = self._check_bound(movej_q)
            if inside_buffer_bound:
                break
        return reset_angles, xyz

    def _reset_arm(self, reset_angles=[]):
        """Sends reset packet to communicator and sleeps until executed."""
        print("resetting arm")
        self._reset_packet = [1.9, 0, -1.93, -1.52, 0.10, 1.52, 0.8]
        # self._actuator_comms['Franka'].actuator_buffer.write(self._reset_packet)
        # robot = self._actuator_comms['Franka'].robot

        max_iterations = int(4 * 40)

        speed_clip = 0.1
        
        t0 = time.time()
        ratio = 2
        for _ in range(max_iterations):
            vals = self._q_[0]
            errors = self._q_[0]
            #ratio = 2
            for i in range(7):
                errors[i] = (self._reset_packet[i] - vals[i])
                if np.abs(errors[i]) < 0.05:
                    ratio = 2 #np.abs(40*errors[j])
                    #print('ration is ', ratio)
                else:
                    ratio = 2
                vals[i] = (self._reset_packet[i] - vals[i])*ratio
                vals[i] = np.clip(vals[i], -speed_clip, speed_clip)
            # robot.set_joint_velocities(vals)
            # time.sleep(max(self._reset_packet[-2] * 1.5, 2.0))
            # robot.set_joint_positions(target_joints[j])
            # print(errors)
            self._actuator_comms['Franka'].actuator_buffer.write(vals)
            time.sleep(1/40)
            ## added this line to finish earlier
            if max([np.abs(errors[j]) for j in range(7)]) < 0.01:
                ratio = 0
                for j in range(7):
                
                    vals[j] = 0
                    # print(vals)
                    # robot.set_joint_velocities(vals)
                    self._actuator_comms['Franka'].actuator_buffer.write(vals)
                    time.sleep(1/160)
                break
            
        # print(time.time()-t0 )
        print("reset arm done")

        # self._reset_packet[1:1 + 6][self._joint_indices] = reset_angles

    def _write_actuation_(self):
        """Overwrite the base method, only handle Franka action"""
        # self._actuator_comms['Franka'].actuator_buffer.write(self._actuation_packet_['Franka'])
        if self._control_type == "velocity":
            print("write actuation")
            self.robot.set_joint_velcities(dict(zip(self.joint_names, self._action_)))
            t = time.time()
            self.rate.sleep()
            print("sleep time done", time.time() - t)
        else:
            raise NotImplementedError

    def _sensor_to_sensation_(self):
        """ Overwrite the original function to support image input
        """
        for name, comm in self._sensor_comms.items():
            if comm.sensor_buffer.updated():
                sensor_window, timestamp_window, index_window = comm.sensor_buffer.read_update(
                    self._num_sensor_packets[name])
                # if name == 'Franka':
                #     s = self._compute_joint_(name, sensor_window, timestamp_window, index_window)
                #     self._joint_buffer.write(s)
                if name == 'Camera':
                    s = self._compute_image_(name, sensor_window, timestamp_window, index_window)
                    self._image_buffer.write(s)
                else:
                    raise NotImplementedError()
    
    def extract_values(self, dict, ordered_keys):
        values = []
        for key in ordered_keys:
            values.append(dict[key])
        return values

    def _read_sensation(self):
        """Overwrite this method to support images
        Returns:
            A tuple (observation, reward, done)
        """

        # joint_sensation, joint_timestamp, _ = self._joint_buffer.read_update()
        try:
            self.robot_status.enable()
        except TimeoutError as e:
            print(e)
        joint_angles = self.extract_values(self.robot.joint_angles(), self.joint_names)
        joint_velocities = self.extract_values(self.robot.joint_velocities(), self.joint_names)
        joint_timestamp = [time.time()]
        joint_sensation = np.array([(
                # (joint_state_vector,
                # joint_torques,
                # joint_state_no_torques,
                # ee_state_vector,
                # self.prev_action,
                joint_angles,
                joint_velocities)], dtype=self.package_type
            )
        self.current_joint_obs = joint_sensation
        image_sensation, image_timestamp, _ = self._image_buffer.read_update()
        if np.abs(joint_timestamp[-1] - image_timestamp[-1]) > 0.04:
            print(f'Warning: Image received is delayed by: {np.abs(joint_timestamp[-1] - image_timestamp[-1])}!')
            time.sleep(0.04)
            # joint_sensation, joint_timestamp, _ = self._joint_buffer.read_update()
            joint_angles = self.extract_values(self.robot.joint_angles(), self.joint_names)
            joint_velocities = self.extract_values(self.robot.joint_velocities(), self.joint_names)
            joint_timestamp = [time.time()]
            joint_sensation = np.array([(
                    # (joint_state_vector,
                    # joint_torques,
                    # joint_state_no_torques,
                    # ee_state_vector,
                    # self.prev_action,
                    joint_angles,
                    joint_velocities)], dtype=self.package_type
                )
            self.current_joint_obs = joint_sensation
            image_sensation, image_timestamp, _ = self._image_buffer.read_update()
        # reshape flattened images
        images = []
        image_length = DEFAULT_WIDTH * DEFAULT_HEIGHT * 3
        for i in range(self._image_history):
            images.append(image_sensation[0][i * image_length : (i + 1) * image_length].reshape(DEFAULT_HEIGHT, DEFAULT_WIDTH, 3))
        
        image_sensation = np.concatenate(images, axis=-1).astype(np.uint8)
        image_sensation = image_sensation[::DEFAULT_HEIGHT // self._image_height, ::DEFAULT_WIDTH // self._image_width, :]
        print(joint_sensation, joint_sensation[0])
        reward = self._compute_reward_(image_sensation, joint_sensation[0][self._joint_indices])
        done = self._check_done()
        if self._channel_first:
            image_sensation = np.rollaxis(image_sensation, 2, 0)
        print("end of read sensation")
        return {'image': image_sensation, 'joint': joint_sensation[0]}, reward, done

    def _compute_image_(self, name, sensor_window, timestamp_window, index_window):
        index_end = len(sensor_window)
        index_start = index_end - self._image_history
        images = np.array([sensor_window[i] for i in range(index_start, index_end)])
        return images.flatten()

    def _compute_joint_(self, name, sensor_window, timestamp_window, index_window):

        index_end = len(sensor_window)
        index_start = index_end - self._joint_history
        # q_acutal: position, ind 0
        # q_target
        print(self._q_)
        self._q_ = self.robot.
        self._qd_ = np.array([sensor_window[i][-1][0] for i in range(index_start,index_end)])
        # self._qd_ = np.array([sensor_window[i]['qd_actual'][0] for i in range(index_start,index_end)])
        # self._qdt_ = np.array([sensor_window[i]['qd_target'][0] for i in range(index_start,index_end)])
        # self._qddt_ = np.array([sensor_window[i]['qdd_target'][0] for i in range(index_start,index_end)])

        # self._current_ = np.array([sensor_window[i]['i_actual'][0] for i in range(index_start,index_end)])
        # self._currentt_ = np.array([sensor_window[i]['i_target'][0] for i in range(index_start,index_end)])
        # self._currentc_ = np.array([sensor_window[i]['i_control'][0] for i in range(index_start,index_end)])
        # self._mt_ = np.array([sensor_window[i]['m_target'][0] for i in range(index_start,index_end)])
        # self._voltage_ = np.array([sensor_window[i]['v_actual'][0] for i in range(index_start,index_end)])

        # self._safety_mode_ = np.array([sensor_window[i]['safety_mode'][0] for i in range(index_start,index_end)])
        self._joint_angles_obs = np.array([sensor_window[i][-2][0] for i in range(index_start,index_end)])
        self._joint_velocity_obs = np.array([sensor_window[i][-1][0] for i in range(index_start,index_end)])
        # print("current joint pos and vel", self._joint_velocity_obs.shape, self._joint_angles_obs.shape)
        # print("action", self._action_)

        self.current_joint_obs = self._joint_angles_obs[:, self._joint_indices].reshape((7,))

        return np.concatenate((self._joint_angles_obs[:, self._joint_indices].reshape((7,)),
                               self._joint_velocity_obs[:, self._joint_indices].reshape((7,)) / self._speed_high,
                               self._action_ / self._action_high,))

    def _compute_actuation_(self, action, timestamp, index):
        """Creates and sends actuation packets to the communicator.

        Computes actuation commands based on agent's action and
        control type and writes actuation packets to the
        communicators' actuation buffers. In case of safety box or
        angle joints safety limits being violated overwrites agent's
        actions with actuations that return the arm back within the box.
        Clears p-stops if any.

        Args:
            action: a numpoy array containing agent's action
            timestamp: a float containing action timestamp
            index: an integer containing action index
        """
        print("action", action)
        self._action_ = action
        # self._actuation_packet_['Franka'] = action
        # self._handle_speed_and_joint_limit(action)
    
    def _handle_speed_and_joint_limit(self, action):
        predict = action
        for i, ac in enumerate(action):
            if ac * 50 > self.angle_safety_bound[i][1] or ac * 50 < self.angle_safety_bound[i][0]:
                predict[i] = np.clip(-predict[i]*10, -0.1, 0.1) 
                # print("correct movement")
        # print(predict)
        self._actuation_packet_['Franka'] = predict

    def _check_bound(self, q):
        """Checks whether given arm joints configuration is within box.

        Args:
            q: a numpy array of joints angle positions.self._cmd_
            distance to the closest bound, mat is a 4x4 position matrix
            returned by solving forward kinematics equations, xyz are
            the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        mat = ur_utils.forward(q, self._ik_params)
        xyz = mat[:3, 3]
        inside_bound = np.all(self._end_effector_low <= xyz) and np.all(xyz <= self._end_effector_high)
        inside_buffer_bound = (np.all(self._end_effector_low + self._box_bound_buffer <= xyz) and \
                               np.all(xyz <= self._end_effector_high - self._box_bound_buffer))
        return inside_bound, inside_buffer_bound, mat, xyz

    def _compute_reward_(self, image, joint):
        """Computes reward at a given time step.

        Returns:
            A float reward.
        """
        print("compute reward")
        image = image[:, :, -3:]
        lower = [0, 0, 120]
        upper = [80, 80, 255]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv.inRange(image, lower, upper)
        # cv.imwrite("/home/franka/project/franka_async_rl_noni/ur5_async_rl-master/mask/{}.jpg".format(self.counter), mask)
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

        '''
        When the joint 4 is perpendicular to the mounting ground:
            joint 0 + joint 4 == 0
            joint 1 + joint 2 + joint 3 == -pi
        '''

        # TODO: on franka we should penaltize when joint4 and joint5 are perpendicular
        scale = (np.abs(joint[0] + joint[4]) + np.abs(np.pi + np.sum(joint[1:4])))
        return reward - scale
        # return reward - (abs(joint[5] - self.angle_safety_bound[5][1]) + abs(joint[5] - self.angle_safety_bound[5][1]))
        return reward



    def _check_done(self):
        """Checks whether the episode is over.


        Args:
            env_done:  a bool specifying whether the episode should be ended.

        Returns:
            A bool specifying whether the episode is over.
        """
        self._episode_steps += 1
        self.counter += 1

        if (self._episode_steps >= self._episode_length_step):# or env_done:
            self.robot.set_joint_velocities(dict(zip(self.joint_names, [0] * 7)))
            return True
        else:
            return False

    def reset(self, blocking=True):
        """Resets the arm, optionally blocks the environment until done."""
        ret = super(ReacherEnv, self).reset(blocking=blocking)
        self._episode_steps = 0
        return ret

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def terminate(self):
        """Gracefully terminates environment processes."""
        super(ReacherEnv, self).close()


if __name__ == '__main__':
    env = ReacherEnv(setup='Franka', target_type='reaching', dt = 0.1, episode_length_time=4)
    env.start()
    epi = 0
    for i in range(10):
        img = env.reset(blocking=True)
        done = False
        while not done:
            action = env.action_space.sample()
            img, rewards, done, _ = env.step(action)
        epi += 1
        print("exit epsiode", epi)
    print("done")