# Asynchronous Reinforcement Learning for UR5 Robotic Arm

This is the implementation for asynchronous reinforcement learning for UR5 robotic arm. This repo consists of two parts, the vision-based UR5 environment, which is based on the [SenseAct](https://github.com/kindredresearch/SenseAct) framework, and an asynchronous learning architecture for Soft-Actor-Critic. (Our implementation of SAC is partly borrowed from [here](https://sites.google.com/view/sac-ae/home))

## Trained results:
| ![UR-Reacher-2](figs/reaching.GIF) <br> Reaching | ![UR-Reacher-6](figs/tracking.GIF) <br /> Tracking |
| --- | --- |

## Required Packages
Use `pip3` to install the Pytorch packages and not `conda`.

* Python 3.6.9
* Numpy 1.19.5
* Pytorch 1.9.0+cuda 11.1
* Pytorch vision 0.10.0
* Pytorch audio 0.9.0
* OpenCV 4.1.2.30
* Matplotlib 3.3.4
* SenseAct 0.1.2
* Gym 0.17.3
* Termcolor 1.1.0
* Latest NVidia driver on Linux

## Required Hardware
* Graphic card with cuda 11.1 support
* Two monitors

## Installation:
If you have the physical setup shown above (a UR5 robotic arm, a USB camera, and a monitor which are all connected to a Linux workstaion with GPU support), follow the steps below to install
1. Install [SenseAct](https://github.com/kindredresearch/SenseAct/blob/master/README.md#installation) framework.
2. `git clone https://github.com/Yan-Wang88/ur5_async_rl`

## Instructions
Make sure the computer does not go to sleep or a screensaver when the task is running.
### To run the reaching task
1. Open a terminal (task can’t be run in pycharm due to its restrictions).
2. Cd to the `ur5_async_rl` directory.
3. Type `python3 ur5_train.py --target_type reaching --camera_id 0 (or 1, based on your actual camera id) --async_mode (for parallel mode, or ignore it for serial mode)`
### To run the tracking task
#### Run the Monitor communicator on the same computer
* Type `python3 ur5_train.py --target_type tracking --camera_id 0 (or 1, based on your actual camera id) --async_mode (for parallel mode, or ignore it for serial mode)`.
#### Run the Monitor communicator on a different computer (if the above method doesn’t train)
1. Connect the second monitor to a different Linux computer with Matplotlib 3.3.4, python 3.6.9, and [SenseAct](https://github.com/kindredresearch/SenseAct/blob/master/README.md#installation) installed.
2. Copy `ur5_async_rl/envs/visual_ur5_reacher/monitor_communicator.py` to the second computer.
3. Type `python3 monitor_communicator.py` on the second computer to display the moving red circle on the second screen.
4. On the main computer, type `python3 ur5_train.py --target_type tracking --camera_id 0 (or 1, based on your actual camera id) --async_mode (for parallel mode, or ignore it for serial mode) --ignore_monitor_comm`.
### Output format
The console output is available in the form:

```
| train | E: 1 | S: 1000 | D: 0.8 s | R: 0.0000 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | NUM: 0.0000
```

and a training entry decodes as:

```
E - total number of episodes 
S - total number of environment steps
D - duration in seconds of 1 episode
R - episode reward
BR - average reward of sampled batch
ALOSS - average loss of actor
CLOSS - average loss of critic
NUM - number of gradient updates performed so far
```
## Troubleshoot
If the red circle doesn’t show up, try the following:
1. Look for camera id error.
2. Look at the log screen on the UR5 control panel. If there is a message like “error code 7xxx”, then the joint may be at fault and the UR5 environment start-up may halt (without any warnings).
3. If you see the camera LED is on before you run `ur5_train.py`, then the camera isn’t setup properly and the start-up process will halt (without any warnings). On our lab's workstation, most likely the Motion app is using the camera for live streaming to the network. The Motion app might start after every reboot. You can kill the app by typing `sudo service motion stop`. If that doesn't work, you can try to remove the camera and plug it in again or restart the computer.
4. If the red circle is displayed on the main screen, move it to the secondary screen facing the camera.
