import gym
import envs.visual_ur5_reacher.reacher_env_visual
import time

if __name__ == "__main__":
    env = gym.make('franka_agile_grasping-v0')
    # env.start()
    epi = 0
    for i in range(10):
        img = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            t = time.time()
            img, rewards, done, _ = env.step(action)
            print(time.time() - t)
        epi += 1
        print("exit epsiode", epi)
    print("done")