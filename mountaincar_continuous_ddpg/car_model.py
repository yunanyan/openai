import os
import numpy as np
import tensorflow as tf
from tensorflow.initializers import random_uniform
from collections import OrderedDict

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.3, theta=.2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev)  + \
            self.sigma * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class Actor(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, units,
                 f_dense, f_output ,action_bound, batch_size=64, chkpt_dir='./ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.units = units
        self.chkpt_dir = chkpt_dir
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.f_dense = f_dense
        self.f_output = f_output
        self.sess = sess
        self.action_bound = action_bound
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name +'_ddpg.ckpt')

        self.unnormalized_actor_gradients = tf.gradients(
            self.mu, self.params, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size),
                                        self.unnormalized_actor_gradients))

        self.optimize = tf.train.AdamOptimizer(self.lr).\
                    apply_gradients(zip(self.actor_gradients, self.params))

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims],
                                        name='inputs')

            self.action_gradient = tf.placeholder(tf.float32,
                                          shape=[None, self.n_actions],
                                          name='gradients')
            x = self.input

            
            for unit in range(len(self.units)):
                x = tf.layers.dense(x, units=self.units[unit],
                                     kernel_initializer=random_uniform(-self.f_dense, self.f_dense),
                                     bias_initializer=random_uniform(-self.f_dense, self.f_dense))
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)

           
            mu = tf.layers.dense(x, units=self.n_actions,
                            activation='tanh',
                            kernel_initializer= random_uniform(-self.f_output, self.f_output),
                            bias_initializer=random_uniform(-self.f_output, self.f_output))
            self.mu = tf.multiply(mu, self.action_bound)

    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.input: inputs})

    def train(self, inputs, gradients):
        self.sess.run(self.optimize,
                      feed_dict={self.input: inputs,
                                 self.action_gradient: gradients})

    def load_checkpoint(self):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("...Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

class Critic(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, units,f_dense, f_output, regularizer,
                 batch_size=64, chkpt_dir='./ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.units = units
        self.chkpt_dir = chkpt_dir
        self.input_dims = input_dims
        self.f_dense = f_dense
        self.f_output = f_output
        self.regularizer = regularizer
        self.batch_size = batch_size
        self.sess = sess
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name +'_ddpg.ckpt')

        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.action_gradients = tf.gradients(self.q, self.actions)

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims],
                                        name='inputs')

            self.actions = tf.placeholder(tf.float32,
                                          shape=[None, self.n_actions],
                                          name='actions')

            self.q_target = tf.placeholder(tf.float32,
                                           shape=[None,1],
                                           name='targets')


            y = self.input
            for unit in range(len(self.units)-1):
                y = tf.layers.dense(self.input, units=self.units[unit],
                                     kernel_initializer=random_uniform(-self.f_dense, self.f_dense),
                                     bias_initializer=random_uniform(-self.f_dense, self.f_dense))

                y = tf.layers.batch_normalization(y)
                y = tf.nn.relu(y)

            

            layer_before_action = tf.layers.dense(y, units=self.units[0],
                                     kernel_initializer=random_uniform(-self.f_dense, self.f_dense),
                                     bias_initializer=random_uniform(-self.f_dense, self.f_dense))
            batch_final = tf.layers.batch_normalization(layer_before_action)

            action_in = tf.layers.dense(self.actions, units=self.units[0],
                                        activation='relu')
            state_actions = tf.add(batch_final, action_in)
            state_actions = tf.nn.relu(state_actions)

            f3 = self.f_output
            regularizer = self.regularizer
            self.q = tf.layers.dense(state_actions, units=1,
                               kernel_initializer=random_uniform(-f3, f3),
                               bias_initializer=random_uniform(-f3, f3),
                               kernel_regularizer=tf.keras.regularizers.l2(regularizer))

            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

    def predict(self, inputs, actions):
        return self.sess.run(self.q,
                             feed_dict={self.input: inputs,
                                        self.actions: actions})
    def train(self, inputs, actions, q_target):
        return self.sess.run(self.optimize,
                      feed_dict={self.input: inputs,
                                 self.actions: actions,
                                 self.q_target: q_target})

    def get_action_gradients(self, inputs, actions):
        return self.sess.run(self.action_gradients,
                             feed_dict={self.input: inputs,
                                        self.actions: actions})
    def load_checkpoint(self):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("...Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env, n_actions,
                 e, e_decay , e_min ,units, f_dense, f_output, regularizer,
                 max_size=1000000, layer1_size=400, layer2_size=300,gamma=0.99,
                 batch_size=64):

        self.gamma = gamma
        self.tau = tau
        self.e = e
        self.e_decay = e_decay
        self.e_min = e_min
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.sess = tf.Session()
        
        self.actor = Actor(alpha, n_actions, 'Actor', input_dims, self.sess, units,
                            f_dense, f_output, env.action_space.high)
        
        self.critic = Critic(beta, n_actions, 'Critic', input_dims,self.sess,
                             units,layer2_size, f_dense, f_output,regularizer)

        self.target_actor = Actor(alpha, n_actions, 'TargetActor',
                                  input_dims, self.sess, units, f_dense, f_output, env.action_space.high)
        
        self.target_critic = Critic(beta, n_actions, 'TargetCritic', input_dims,
                                    self.sess, units, f_dense, f_output,regularizer)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))
  
        # define ops here in __init__ otherwise time to execute the op
        # increases with each execution.
        self.update_critic = \
        [self.target_critic.params[i].assign(
                      tf.multiply(self.critic.params[i], self.tau) \
                    + tf.multiply(self.target_critic.params[i], 1. - self.tau))
         for i in range(len(self.target_critic.params))]

        self.update_actor = \
        [self.target_actor.params[i].assign(
                      tf.multiply(self.actor.params[i], self.tau) \
                    + tf.multiply(self.target_actor.params[i], 1. - self.tau))
         for i in range(len(self.target_actor.params))]

        self.sess.run(tf.global_variables_initializer())

        self.update_network_parameters(first=True)

    def update_network_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau
        else:
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state, low, high):
        state = state[np.newaxis, :]
        mu = self.actor.predict(state) # returns list of list
        noise = self.noise()
        mu_prime = np.clip(mu + self.e*noise,low, high)
        

        return mu_prime[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
                                      self.memory.sample_buffer(self.batch_size)

        critic_value_ = self.target_critic.predict(new_state,
                                           self.target_actor.predict(new_state))
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = np.reshape(target, (self.batch_size, 1))

        _ = self.critic.train(state, action, target)

        a_outs = self.actor.predict(state)
        grads = self.critic.get_action_gradients(state, a_outs)

        self.actor.train(state, grads[0])

        self.update_network_parameters()

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint() 
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
