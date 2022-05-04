# Asynchronous Reinforcement Learning for Franka Robotic Arm

This is the implementation for asynchronous reinforcement learning for Franka robotic arm. This repo consists of two parts, the vision-based Franka environment, which is based on the [OpenAI Gym](https://gym.openai.com/) framework, [Franka ROS Interface](https://projects.saifsidhik.page/franka_ros_interface/DOC.html), and an asynchronous learning architecture for Soft-Actor-Critic. (Our implementation of SAC is partly borrowed from [here](https://sites.google.com/view/sac-ae/home))

## Trained results:
| ![Franka-Reacher-2](figs/initial.gif) <br> Initial policy, frame dropped | ![UR-Reacher-6](figs/400epi.gif) <br /> After about 400 episodes, frame dropped |
| --- | --- |

[Recording of the initial policy](https://drive.google.com/file/d/18pT0DcMYoXoaTt9tQhxNGVmc43GN6xQB/view?usp=sharing)  
[Recording of the learned behaviour at 400 episodes](https://drive.google.com/file/d/1uyR8kreh1iPXroCcL6Z2FWB_0MBeBPGL/view?usp=sharing)

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
* USB camera

## Instructions
### To run without SenceAct communicator
1. Initialize a Python virtual enviroment `python3 -m venv`
1. install dependencies from `requirement.txt`
2. install ROS dependencies
3. activate local virtual environment `source venv/bin/activate`
4. in the same terminal, go to your `~/catkin_ws` and run `.franka.sh remote`
5. go back to the project root and run `python franka_train.py`

#### Misc
1. test your camera feed with `v4l2-ctl -d /dev/video0 --list-formats-ext`
2. if your virtual env complains about ros dependencies, set your `PYTHONPATH` to your virtual env python. e.g. `export PYTHONPATH="$PYTHONPATH:<path-to-root>"`
or run with this command `<path-to-root>/env/bin/python <path-to-file>/check_bound.py`

---

### To run the reaching task
1. Open a terminal (task can’t be run in pycharm due to its restrictions).
2. Cd to the `ur5_async_rl` directory.
3. Type `python3 ur5_train.py --target_type reaching --camera_id 0 (or 1, based on your actual camera id) --async_mode (for parallel mode, or ignore it for serial mode)`
### To run the tracking task
- TBD

### To run with SenseAct communicator 
- TBD
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
