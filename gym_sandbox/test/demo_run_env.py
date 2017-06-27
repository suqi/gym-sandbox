# -*- coding: utf-8 -*-
import gym
import gym_sandbox

# choose a env name from env_list
env = gym.make("police-killall-ravel-v0")

print("action shape >>> ", env.action_space)

s_init = env.reset()
print("state shape >>> ", s_init.shape)

for i in range(10):
    # choose your action
    a = 1
    s_, r, done, info = env.step(a)
    if done:
        break


