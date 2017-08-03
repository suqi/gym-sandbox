"""
MADDPG solution to solve MA
Based on Morvan's DDPG v1 (A/C is seperate so that MADDPG is easier to implement):
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/9_Deep_Deterministic_Policy_Gradient_DDPG/DDPG.py
Actually v1 has a bug where Morvan-DDPG-v2 has solve:
    This alog will firstly update critic q network and then update actor,
    the bug is that after critic update q, the q is changed,
    then actor calculate dq/da * da/dw, the dq/da is a updated q, not the original q
"""

import time
import tensorflow as tf
import numpy as np
import gym
import gym_sandbox

np.random.seed(1)
tf.set_random_seed(1)

# -------------------  hyper parameters  -------------------
MAX_EPISODES = 1000000
LR_A = 0.0005  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.95  # reward discount
TAU = 0.01  # Soft update for target param, but this is computationally expansive
# so we use replace_iter instead
REPLACE_ITER_A = 500  # how many iter to update target
REPLACE_ITER_C = 300
MEMORY_CAPACITY = 6000
BATCH_SIZE = 20
LEARN_HZ = 1  # how frequent to learn

var = 1  # control exploration, w.r.t action_bound
var_decay = 0.99998 #0.999998    # decay the action randomness

RENDER = False
start_render_ep = 50000

ENV_NAME = 'police-maddpg-continous-vector-v0'


# -------------------  Actor  -------------------


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter,
                 agent_id, agent_num):
        self.agent_id = agent_id
        self.agent_num = agent_num

        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor{}'.format(agent_id)):
            # input s, output a
            self.a = self._build_actor_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_actor_net(S2, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope='Actor{}/eval_net'.format(agent_id))
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope='Actor{}/target_net'.format(agent_id))

    def _build_actor_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            # init_w = tf.random_normal_initializer(0., 0.3)
            # init_b = tf.constant_initializer(0.1)
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0)
            l = tf.layers.dense(s, 1000, activation=tf.nn.relu,
                                 kernel_initializer=init_w, bias_initializer=init_b,
                                 trainable=trainable)
            l = tf.layers.dense(l, 500, activation=tf.nn.relu,
                                 kernel_initializer=init_w, bias_initializer=init_b,
                                 trainable=trainable)
            l = tf.layers.dense(l, 200, activation=tf.nn.relu,
                                 kernel_initializer=init_w, bias_initializer=init_b,
                                 trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(l, self.a_dim, activation=tf.nn.tanh,
                                          kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                # Scale [-1,1] to [-action_bound ~ action_bound]
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')
        return scaled_a

    def learn_actor(self, s, x_ma, epoch):   # batch update
        _, police_grads = self.sess.run(self.train_ops, feed_dict={S: s, X_MA: x_ma})
        # the following method for soft replace target params is computational expansive
        # target_params = (1-tau) * target_params + tau * eval_params
        # self.sess.run([tf.assign(t, (1 - self.tau) * t + self.tau * e) for t, e in zip(self.t_params, self.e_params)])

        summary = tf.Summary()
        # summary.value.add(tag='info/c_gradient{}'.format(self.agent_id), simple_value=float(_c_grad))
        summary.value.add(tag='info/police_grads{}'.format(self.agent_id), simple_value=np.mean([np.mean(_) for _ in police_grads]))
        writer.add_summary(summary, epoch)
        writer.flush()

        # instead of above method, I use a hard replacement here
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads{}'.format(self.agent_id)):
            # ys = policy;
            # xs = policy's parameters;
            # self.a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys,
            # so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            # use minus lr to maximize Q
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_ops = []
            self.train_ops.append(opt.apply_gradients(zip(self.policy_grads, self.e_params)))
            self.train_ops.append(self.policy_grads)


# -------------------  Critic  -------------------

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter,
                 a_ma, a2_ma, agent_id, agent_num):
        self.agent_id = agent_id
        self.agent_num = agent_num

        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic{}'.format(self.agent_id)):
            # Input (s, a), output q
            local_a = a_ma[agent_id]
            self.a_ma = tf.concat(a_ma, axis=1)
            self.q = self._build_critic_net(X_MA, self.a_ma, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            a2_ma = tf.concat(a2_ma, axis=1)
            self.q_ = self._build_critic_net(X2_MA, a2_ma, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic{}/eval_net'.format(agent_id))
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic{}/target_net'.format(agent_id))

        with tf.variable_scope('target_q{}'.format(self.agent_id)):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error{}'.format(self.agent_id)):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))  # MSE

        with tf.variable_scope('C_train{}'.format(self.agent_id)):
            self.train_ops = []
            self.train_ops.append(tf.train.AdamOptimizer(self.lr).minimize(
                self.loss, var_list=self.e_params))  # C train only update c network, don't update a
            self.train_ops.append(self.loss)  # for tf.summary

        with tf.variable_scope('a_grad{}'.format(self.agent_id)):
            # tensor of gradients of each sample (None, a_dim)
            self.a_grads = tf.gradients(self.q, local_a)[0]  # only get dq/da, throw dq/dw
            self.train_ops.append(self.a_grads)

    def _build_critic_net(self, x_ma, a_ma, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 2000  # 30
                w1_x = tf.get_variable('w1_x', [self.s_dim * self.agent_num, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim * self.agent_num, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                l = tf.nn.relu(tf.matmul(x_ma, w1_x) + tf.matmul(a_ma, w1_a) + b1)
                l = tf.layers.dense(l, 1000, activation=tf.nn.relu,
                                     kernel_initializer=init_w, bias_initializer=init_b,
                                     trainable=trainable)
                l = tf.layers.dense(l, 500, activation=tf.nn.relu,
                                     kernel_initializer=init_w, bias_initializer=init_b,
                                     trainable=trainable)
                l = tf.layers.dense(l, 200, activation=tf.nn.relu,
                                     kernel_initializer=init_w, bias_initializer=init_b,
                                     trainable=trainable)

            with tf.variable_scope('q'):
                q = tf.layers.dense(l, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn_critic(self, x_ma, a_ma, r, x2_ma, s, a, s2, epoch=0):
        # ATTENTION!!!!
        # the key point is that we use constant a_ma to replace critic's tensor: self.a_ma
        # here we must replace this tensor, otherwise whole network crash
        # because critic must use constant a_ma to do gradient,
        # while actor must use its network tensor a_ma to do gradient
        # this is the trick!!
        _c_grad, _c_loss, _a_grads = self.sess.run(
            self.train_ops, feed_dict={X_MA: x_ma, self.a_ma: a_ma,
                                       R: r, X2_MA: x2_ma,
                                       S: s, S2: s2})

        summary = tf.Summary()
        # summary.value.add(tag='info/c_gradient{}'.format(self.agent_id),
        #                   simple_value=float(_c_grad))
        summary.value.add(tag='info/c_loss{}'.format(self.agent_id), simple_value=float(_c_loss))
        writer.add_summary(summary, epoch)
        writer.flush()



        # the following method for soft replace target params is computational expansive
        # target_params = (1-tau) * target_params + tau * eval_params
        # self.sess.run([tf.assign(t, (1 - self.tau) * t + self.tau * e) for t, e in zip(self.t_params, self.e_params)])

        # instead of above method, we use a hard replacement here
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


# -------------------  Memory -------------------

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, x_ma, a_ma, r_ma, x2_ma):
        transition = np.hstack([_.ravel() for _ in (x_ma, a_ma, r_ma, x2_ma)])
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        # assert self.pointer >= n, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


# -------------------  Env runner main entry -------------------

env = gym.make(ENV_NAME)
env.env.init_params(show_dashboard=True, bokeh_output='standalone')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high
AGENT_NUM = env.env.agent_num

# all placeholder for tf
with tf.name_scope('transition'):
    S = tf.placeholder(tf.float32, [None, state_dim], 's')
    S2 = tf.placeholder(tf.float32, [None, state_dim], 's2')
    X_MA = tf.placeholder(tf.float32, [None, state_dim * AGENT_NUM], 'x_ma')
    X2_MA = tf.placeholder(tf.float32, [None, state_dim * AGENT_NUM], 'x2_ma')
    A_MA = tf.placeholder(tf.float32, [None, 1 * AGENT_NUM], 'a_ma')
    A2_MA = tf.placeholder(tf.float32, [None, 1 * AGENT_NUM], 'a2_ma')
    R = tf.placeholder(tf.float32, [None, 1], 'r')

sess = tf.Session()

# Create actor and critic.
# They are actually connected to each other, details can be seen in tensorboard or in this picture:
multi_actors = [Actor(sess, action_dim, action_bound, LR_A, REPLACE_ITER_A,
                      _, AGENT_NUM)
                for _ in range(AGENT_NUM)]
multi_critics = [Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACE_ITER_C,
                        [actor.a for actor in multi_actors],
                        [actor.a_ for actor in multi_actors],
                        _, AGENT_NUM)
                 for _ in range(AGENT_NUM)]

for _i in range(AGENT_NUM):
    multi_actors[_i].add_grad_to_graph(multi_critics[_i].a_grads)

sess.run(tf.global_variables_initializer())

M = Memory(MEMORY_CAPACITY, dims=(state_dim * 2 + action_dim + 1) * AGENT_NUM)

writer = tf.summary.FileWriter("logs/", sess.graph)

saver = tf.train.Saver()
saver.restore(sess, '.tf-models/maddpg-vector-1503-501')


for i in range(MAX_EPISODES):
    if i > start_render_ep or var <=0.05:
        RENDER = True

    if i % 501 == 0:
        saver.save(sess, ".tf-models/maddpg-vector", global_step=i)

    x_ma = env.reset()
    ep_reward = 0

    while True:
        # Add exploration noise
        a_ma = [actor.choose_action(x_ma[actor.agent_id]) for actor in multi_actors]
        a_ma = np.random.normal(a_ma, var)    # add randomness to action selection for exploration
        x2_ma, r_ma, done, info = env.step(a_ma)

        if RENDER:
            env.render()
            time.sleep(0.1)

        M.store_transition(x_ma, a_ma, r_ma, x2_ma)

        if M.pointer > MEMORY_CAPACITY and M.pointer % LEARN_HZ == 0:
            var *= var_decay
            b_M = M.sample(BATCH_SIZE)

            _x_len = state_dim * AGENT_NUM
            _a_len = action_dim * AGENT_NUM
            bx_ma = b_M[:, :_x_len]
            ba_ma = b_M[:, _x_len:_x_len + _a_len]
            br_ma = b_M[:, _x_len + _a_len:_x_len + _a_len + AGENT_NUM]
            bx2_ma = b_M[:, -_x_len:]

            for _i in range(AGENT_NUM):
                b_s = np.split(bx_ma, AGENT_NUM, axis=1)[_i]
                b_a = np.split(ba_ma, AGENT_NUM, axis=1)[_i]
                b_r = np.split(br_ma, AGENT_NUM, axis=1)[_i]
                b_s2 = np.split(bx2_ma, AGENT_NUM, axis=1)[_i]

                learn_epoch = M.pointer-MEMORY_CAPACITY

                # TODO: maybe actor should learn befor critic
                # so that actor's Q gradient is from old critic, which should be same as Paper.
                multi_critics[_i].learn_critic(bx_ma, ba_ma, b_r, bx2_ma, b_s, b_a, b_s2, learn_epoch)
                multi_actors[_i].learn_actor(b_s, bx_ma, learn_epoch)

        x_ma = x2_ma
        ep_reward += np.sum(r_ma)

        if done:
            if RENDER:
                time.sleep(0.3)
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            break
