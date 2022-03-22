import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def plot():
    _f = "./results/SACv0_reaching_dt=0.04_bs=64_dim=160*120_9/train.log"
    rets = []
    batch_re = []
    lines = None
    with open(_f) as f:
        lines = f.readlines()

    for line in lines:
        j = json.loads(line)
        rets.append(j["episode_reward"])
        
    plt.clf()
    plt.plot(range(len(rets)), rets)
    plt.plot(range(len(batch_re)), batch_re)
    plt.pause(0.001)
    plt.show()

if __name__ == "__main__":
    plot()    