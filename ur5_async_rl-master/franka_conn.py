from franka_interface import ArmInterface, RobotEnable, GripperInterface
import rospy

rospy.init_node('franka_communicator')

robot = ArmInterface(True)
gripper = GripperInterface()
robot_status = RobotEnable()
control_frequency = 40
rate = rospy.Rate(control_frequency)
joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
joint_vels = dict(zip(joint_names, [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0]))

while not rospy.is_shutdown():
    robot.set_joint_velocities(joint_vels)
    rate.sleep()