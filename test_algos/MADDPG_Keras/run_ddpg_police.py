import gym
import gym_sandbox
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit

OU = OU()       #Ornstein-Uhlenbeck Process
GAME = 'police-maddpg-continous-v0'

def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    episode_count = 50000
    max_steps = 100
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0
    GLOBAL_RUNNING_R = []

    # create env
    env = gym.make(GAME)
    env.env.init_params(show_dashboard=True, bokeh_output='standalone')
    # env = env.unwrapped

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high


    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    #Now load the weight
    # print("Now we load the weight")
    # try:
    #     actor.model.load_weights(".keras/actormodel.h5")
    #     critic.model.load_weights(".keras/criticmodel.h5")
    #     actor.target_model.load_weights(".keras/actormodel.h5")
    #     critic.target_model.load_weights(".keras/criticmodel.h5")
    #     print("Weight load successfully")
    # except:
    #     print("Cannot find the weight")

    print("GYM Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        ob = env.reset()

        s_t = ob
     
        total_reward = 0.
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            # noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            # noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = (a_t_original[0][0] + noise_t[0][0]) * action_bound
            # a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            # a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            ob, r_t, done, info = env.step(a_t[0])

            env.render()

            s_t1 = ob
        
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
           
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            # print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
        
            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights(".keras/actormodel.h5", overwrite=True)
                with open(".keras/actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights(".keras/criticmodel.h5", overwrite=True)
                with open(".keras/criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
            GLOBAL_RUNNING_R.append(total_reward)
        else:
            GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * total_reward)

        print("Episode REWARD #{} >>> {}".format(i, total_reward))
        print("Total Step: " + str(step))
        print("")

    env.close()
    print("Finish.")

if __name__ == "__main__":
    playGame(train_indicator=1)
