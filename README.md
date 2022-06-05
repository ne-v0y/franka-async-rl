# Asynchronous Reinforcement Learning for Franka Robotic Arm

This is the implementation for asynchronous reinforcement learning for Franka robotic arm. This repo consists of two parts, the vision-based Franka environment, which is based on the [OpenAI Gym](https://gym.openai.com/) framework, [Franka ROS Interface](https://projects.saifsidhik.page/franka_ros_interface/DOC.html), and an asynchronous learning architecture for Soft-Actor-Critic. (Our implementation of SAC is partly borrowed from [here](https://sites.google.com/view/sac-ae/home))

## Trained results:
| ![Franka-Reacher-2](figs/initial.gif) <br> Initial policy, frame dropped | ![UR-Reacher-6](figs/400epi.gif) <br /> After about 400 episodes, frame dropped |
| --- | --- |

[Recording of the initial policy](https://drive.google.com/file/d/18pT0DcMYoXoaTt9tQhxNGVmc43GN6xQB/view?usp=sharing)  
[Recording of the learned behaviour at 400 episodes](https://drive.google.com/file/d/1uyR8kreh1iPXroCcL6Z2FWB_0MBeBPGL/view?usp=sharing)

## Required Packages
Use `pip3` to install the Pytorch packages and not `conda`.

* Python 3.7+
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
1. Install dependencies from `requirement.txt`
2. Install ROS dependencies
3. Activate local virtual environment `source venv/bin/activate`
4. In the same terminal, go to your `~/catkin_ws` and run `.franka.sh remote`
5. Go back to the project root and run `python franka_train.py --async_mode (for parallel mode, or ignore it for serial mode)`. Arguments can be found defined in this file.

#### Misc
1. Test your camera feed with `v4l2-ctl -d /dev/video0 --list-formats-ext`
2. If your virtual env complains about ROS dependencies, set your `PYTHONPATH` to your virtual env python. e.g. `export PYTHONPATH="$PYTHONPATH:<path-to-root>"`
3. If you have issues with the environment, check with `<root>/debug_scripts/collect_env.py`

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
1. For Python3.7+ you will run into this issue:
    ```
    multiprocessing: TypeError: cannot pickle 'weakref' object
    ```
    Please check out [this thread](https://stackoverflow.com/questions/71945399/python-3-8-multiprocessing-typeerror-cannot-pickle-weakref-object) and follow fix suggested [here](https://github.com/python/cpython/pull/31701/files).

2. When using the `fmq` library, you will find it not compatible with python3. Please follow [this fix](https://github.com/WeiTang114/FMQ/pull/1).
