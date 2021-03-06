import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
import gym_sandbox
import multiprocessing
import time

GAME = 'police-killall-trigger-3dgrid-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 30000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 20
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

RUN_MODE = "training" # execution

env = gym.make(GAME)
_s = env.reset()
N_S = list(_s.shape) #
N_A = env.action_space.n
WIDTH = _s.shape[0]

class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None]+N_S, 'S')
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None]+N_S, 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v = self._build_net()
                self.a_prob = tf.clip_by_value(self.a_prob, 1e-6,1)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32),
                                             axis=1, keep_dims=True)
                    exp_v = log_prob * td
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob), axis=1,
                                             keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            conv1 = tf.layers.conv2d(self.s,32,[3,3],[2,2],kernel_initializer=w_init,activation=tf.nn.relu6,name='conv1')
            #conv2 = tf.layers.conv2d(conv1,32,[2,2],[1,1],kernel_initializer=w_init,activation=tf.nn.relu6,name='conv2')
            W = int((WIDTH-3)/2+1)
            conv1 = tf.reshape(conv1,[-1,32*W*W])
            f1 = tf.layers.dense(conv1,200,kernel_initializer=w_init,activation=tf.nn.relu6,name='f1')
            a_prob = tf.layers.dense(f1, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            conv1 = tf.layers.conv2d(self.s,32,[3,3],[2,2],kernel_initializer=w_init,activation=tf.nn.relu6,name='conv1')
            #conv2 = tf.layers.conv2d(conv1,32,[2,2],[1,1],kernel_initializer=w_init,activation=tf.nn.relu6,name='conv2')
            W = int((WIDTH-3)/2+1)
            conv1 = tf.reshape(conv1,[-1,32*W*W])
            f1 = tf.layers.dense(conv1,100,kernel_initializer=w_init,activation=tf.nn.relu6,name='f1')
            v = tf.layers.dense(f1,1, kernel_initializer=w_init, name='v')  # state value
        return a_prob, v

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),  # first digit is batch size, drop it
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME)
        self.env.env.init_params(show_dashboard=name == 'W_0')
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0

            # 每100回合保存训练参数
            # if GLOBAL_EP % 100 == 0 and RUN_MODE == 'training':
            #     saver.save(SESS, "models-ma-balls/a3c-1thread-1v1-dynamic", global_step=GLOBAL_EP)

            while True:
                a = self.AC.choose_action(s)

                s_, r, done, info = self.env.step(a)

                if self.name == 'W_0':
                    show_interval = GLOBAL_EP+1 % 10000 == 0
                    nice = GLOBAL_RUNNING_R and GLOBAL_RUNNING_R[-1] >= -10
                    if show_interval or nice or RUN_MODE=='execution':
                        self.env.render()

                        time.sleep(0.2)

                        if done:
                            time.sleep(2)  # 回合结束给点时间看看效果


                # print('>>>>', 's:', s, ' s_:', s_,  'action:', a, '    -- reward:', r, ' -- done:', done, )

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                # 只在训练模式下进行learning
                if RUN_MODE == "training" and (total_step % UPDATE_GLOBAL_ITER == 0 or done):  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(
                        buffer_v_target)
                    buffer_s = np.reshape(buffer_s,[-1]+N_S)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                    )
                    GLOBAL_EP += 1
                    break


SESS = tf.Session()

with tf.device("/cpu:0"):
    OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
    OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
    GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
    workers = []
    # Create worker
    for i in range(N_WORKERS):
        i_name = 'W_%i' % i  # worker name
        workers.append(Worker(i_name, GLOBAL_AC))

COORD = tf.train.Coordinator()
saver = tf.train.Saver()
SESS.run(tf.global_variables_initializer())

# 先加载训练过的参数
# saver.restore(SESS, 'models-ma-balls/a3c-1thread-1v1-dynamic-29900')

if OUTPUT_GRAPH:
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    tf.summary.FileWriter(LOG_DIR, SESS.graph)

worker_threads = []
for worker in workers:
    job = lambda: worker.work()
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)
COORD.join(worker_threads)
