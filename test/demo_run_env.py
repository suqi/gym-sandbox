# -*- coding: utf-8 -*-
import gym
import gym_sandbox
import time

# choose a env name from env_list
# env = gym.make("police-killall-ravel-v0")
# env = gym.make("police-killall-static-cords-500-v0")
# env = gym.make("police-killall-trigger-3dravel-v0")
env = gym.make("police-bokeh-server-v0")

env.env.init_params(show_dashboard=True)

print("action shape >>> ", env.action_space.n)

s_init = env.reset()
print("state shape >>> ", s_init.shape)

for i in range(10):
    # choose your action
    a = 1
    s_, r, done, info = env.step(a)
    time.sleep(10)
    print(s_, r, done, info)
    if done:
        break


