import rospy
from sensor_msgs.msg import JointState

bound = [[-0.5, 0.2], [0, 1.2], [-0.3, 0.3], [-2.2, 0], [0, 0.4], [1.7, 2.5], [0.8, 0.9]]

angle = JointState()

def cb(data):
    angle.position = data.position
    current_joint_angle = angle.position[:7]
    in_bound = [False] * 7
    for i in range(7):
        if current_joint_angle[i] <= bound[i][1] and current_joint_angle[i] >= bound[i][0]:
            in_bound[i] = True
    print(in_bound, current_joint_angle)


rospy.init_node("test_bound")
rospy.Subscriber("/joint_states", JointState ,cb)

rospy.spin()
