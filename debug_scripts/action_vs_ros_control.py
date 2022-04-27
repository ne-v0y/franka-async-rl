from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

if __name__ == "__main__":
    action = "/home/noni/dev/franka-async-rl/debug_scripts/txt/action_time.txt"
    joint_cmd = "/home/noni/dev/franka-async-rl/debug_scripts/txt/joint_cmd.txt"

    action_t = []
    with open(action, "r") as f:
        while True:
            line = f.readline().replace('\n', '')
            if not line:
                break
            line = line.split(' ')
            action_t.append(float(line[-1]))
    

    joint_t = []
    with open(joint_cmd, 'r') as f:
        time = ''
        while True:
            line = f.readline()
            if not line:
                break
            line = line.replace('\n', '')
            if "secs" in line and not "n" in line:
                time += line.split(' ')[-1]
            elif "nsecs" in line:
                time += '.'
                time += line.split(' ')[-1]
                joint_t.append(float(time))
                time = ''


    plt.clf()
    plt.plot(range(len(action_t)), action_t, color='red')
    plt.plot(range(len(joint_t)), joint_t, color='g')
    plt.pause(0.001)
    plt.show()
    

