import os
import numpy as np
import gym
import tensorflow as tf
from tensorflow.initializers import random_uniform
from collections import OrderedDict
from functionTools.loadSaveModel import saveToPickle, loadFromPickle
from functionTools.loadSaveModel import saveVariables

class GetNoise():
    def __init__(self,noiseDecay,minVar,noiseDacayStep,initnoisevar):
        self.noiseDecay = noiseDecay
        self.minVar = minVar
        self.noiseDacayStep = noiseDacayStep
        self.initnoisevar = initnoisevar
    def __call__(self,runtime):
        if runtime > self.noiseDacayStep:
            self.initnoisevar = self.initnoisevar-self.noiseDecay if self.initnoisevar > self.minVar else self.minVar 
        noise = np.random.normal(0, self.initnoisevar)
        if runtime % 10000 == 0:
            print('noise Variance', self.initnoisevar)
        return noise

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
    def __init__(self, lr, n_actions, name, input_dims, sess, units1,
                 action_bound, batch_size, actoractivationfunction, actorHiddenLayersWeightInit,actorHiddenLayersBiasInit
                 ,actorOutputWeightInit,actorOutputBiasInit,path):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.units = units1
        self.chkpt_dir = path
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.sess = sess
        self.action_bound = action_bound
        
        self.actoractivationfunction =  actoractivationfunction
        self.actorHiddenLayersWeightInit = actorHiddenLayersWeightInit
        self.actorHiddenLayersBiasInit = actorHiddenLayersBiasInit
        self.outputinit = actorOutputWeightInit
        self.outputbiasinit = actorOutputBiasInit

        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(self.chkpt_dir, name +'_ddpg.ckpt')

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
                activationfunction =  self.actoractivationfunction[unit]
                x = tf.layers.dense(x, units=self.units[unit],
                                     kernel_initializer=self.actorHiddenLayersWeightInit[unit],
                                      bias_initializer=self.actorHiddenLayersBiasInit[unit],
                                     activation = activationfunction)
                x = tf.layers.batch_normalization(x)
                

            outputactivationfunction = self.actoractivationfunction[-1]
            mu = tf.layers.dense(x, units=self.n_actions,
                            activation= outputactivationfunction,
                            kernel_initializer= self.outputinit,
                            bias_initializer=self.outputbiasinit)
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
    def __init__(self, lr, n_actions, name, input_dims, sess, units2,
                 batch_size, criticHiddenLayersWidths,criticActivFunction,criticHiddenLayersWeightInit,
                 criticHiddenLayersBiasInit,criticOutputWeightInit,criticOutputBiasInit,path):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        
        self.units = criticHiddenLayersWidths
        self.criticActivFunction = criticActivFunction
        self.criticHiddenLayersWeightInit = criticHiddenLayersWeightInit
        self.criticHiddenLayersBiasInit = criticHiddenLayersBiasInit
        self.criticOutputWeightInit = criticOutputWeightInit
        self.criticOutputBiasInit = criticOutputBiasInit
        
        self.chkpt_dir = path
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.sess = sess
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(self.chkpt_dir, name +'_ddpg.ckpt')

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
            for unit in range(len(self.units)):
                activationfunction = self.criticActivFunction[unit]
                y = tf.layers.dense(y, units=self.units[unit],activation = activationfunction,
                                     kernel_initializer = self.criticHiddenLayersWeightInit[unit],
                                     bias_initializer = self.criticHiddenLayersBiasInit[unit])

                y = tf.layers.batch_normalization(y)
                

            

            layer_before_action = tf.layers.dense(y, units=self.units[0],
                                     kernel_initializer = self.criticHiddenLayersWeightInit[0],
                                     bias_initializer = self.criticHiddenLayersBiasInit[0])
            batch_final = tf.layers.batch_normalization(layer_before_action)

            action_in = tf.layers.dense(self.actions, units=self.units[0],
                                        activation='relu')
            state_actions = tf.add(batch_final, action_in)
            state_actions = tf.nn.relu(state_actions)



    
            outputactivationfunction = self.criticActivFunction[-1]
           
            self.q = tf.layers.dense(state_actions, units=1,
                               kernel_initializer = self.criticOutputWeightInit,activation = outputactivationfunction,
                               bias_initializer = self.criticOutputBiasInit)

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
        

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau,gamma, n_actions,env,
                 max_size, batch_size,units1,units2,
                 noiseDecay,minVar,noiseDacayStep,initnoisevar, actoractivationfunction,actorHiddenLayersWeightInit,actorHiddenLayersBiasInit,
                 actorOutputWeightInit,actorOutputBiasInit,criticHiddenLayersWidths,criticActivFunction,criticHiddenLayersWeightInit,
                 criticHiddenLayersBiasInit,criticOutputWeightInit,criticOutputBiasInit,path ):
        self.gamma = gamma
        self.tau = tau
        self.runtime = 0
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.actor = Actor(alpha, n_actions, 'Actor', input_dims, self.sess, units1,
                            env.action_space.high, batch_size, actoractivationfunction,actorHiddenLayersWeightInit,actorHiddenLayersBiasInit,
                            actorOutputWeightInit,actorOutputBiasInit,path)
         
        self.critic = Critic(beta, n_actions, 'Critic', input_dims,self.sess,
                             units2,batch_size,criticHiddenLayersWidths,criticActivFunction,criticHiddenLayersWeightInit,
                             criticHiddenLayersBiasInit,criticOutputWeightInit,criticOutputBiasInit,path )

        self.target_actor = Actor(alpha, n_actions, 'TargetActor',
                                  input_dims, self.sess, units1,env.action_space.high,batch_size, actoractivationfunction,
                                  actorHiddenLayersWeightInit,actorHiddenLayersBiasInit,actorOutputWeightInit,actorOutputBiasInit,path)
        
        self.target_critic = Critic(beta, n_actions, 'TargetCritic', input_dims,
                                    self.sess, units2,batch_size,criticHiddenLayersWidths,criticActivFunction,
                                    criticHiddenLayersWeightInit,criticHiddenLayersBiasInit,criticOutputWeightInit,criticOutputBiasInit,path )

        self.noise = GetNoise(noiseDecay,minVar,noiseDacayStep,initnoisevar)
  
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
        noise = self.noise(self.runtime)
        mu_prime = np.clip(mu + noise,low, high)
        self.runtime +=1

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

def env_norm(env):
    '''Normalize states (observations) and actions to [-1, 1]'''
    action_space = env.action_space
    state_space = env.observation_space

    env_type = type(env)

    class EnvNormalization(env_type):
        def __init__(self):
            self.__dict__.update(env.__dict__)
            # state (observation - o to match Gym environment class)
            if np.any(state_space.high < 1e10):
                h = state_space.high
                l = state_space.low
                self.o_c = (h+l)/2.
                self.o_sc = (h-l)/2.
            else:
                self.o_c = np.zeros_like(state_space.high)
                self.o_sc = np.ones_like(state_space.high)

            # action
            h = action_space.high
            l = action_space.low
            self.a_c = (h+l)/2.
            self.a_sc = (h-l)/2.

            # reward
            self.r_sc = 0.1
            self.r_c = 0.

            self.observation_space = gym.spaces.Box(self.filter_observation(state_space.low), self.filter_observation(state_space.high))

        def filter_observation(self, o):
            return (o - self.o_c)/self.o_sc

        def filter_action(self, a):
            return self.a_sc*a + self.a_c

        def filter_reward(self, r):
            return self.r_sc*r + self.r_c

        def step(self, a):
            ac_f = np.clip(self.filter_action(a), self.action_space.low, self.action_space.high)
            o, r, done, info = env_type.step(self, ac_f)
            o_f = self.filter_observation(o)

            return o_f, r, done, info
    fenv = EnvNormalization()
    return fenv


class PhilDDPG(object):
    def __init__(self, hyperparamDict):
        self.hyperparamDict = hyperparamDict
        self.runtime = 0

    def __call__(self, env):
        agent = Agent(
        alpha = self.hyperparamDict['actorLR'],
        beta = self.hyperparamDict['criticLR'],
        input_dims = [env.observation_space.shape[0]],
        tau = self.hyperparamDict['tau'],
        env = env_norm(env) if self.hyperparamDict['normalizeEnv'] else env,
        n_actions = env.action_space.shape[0],
        units1 = self.hyperparamDict['actorHiddenLayersWidths'],
        units2 = self.hyperparamDict['criticHiddenLayersWidths'],

        actoractivationfunction = self.hyperparamDict['actorActivFunction'],
        actorHiddenLayersWeightInit = self.hyperparamDict['actorHiddenLayersWeightInit'],
        actorHiddenLayersBiasInit = self.hyperparamDict['actorHiddenLayersBiasInit'],
        actorOutputWeightInit = self.hyperparamDict['actorOutputWeightInit'],
        actorOutputBiasInit = self.hyperparamDict['actorOutputBiasInit'],
        
        criticHiddenLayersWidths = self.hyperparamDict['criticHiddenLayersWidths'],
        criticActivFunction = self.hyperparamDict['criticActivFunction'],
        criticHiddenLayersBiasInit = self.hyperparamDict['criticHiddenLayersBiasInit'],
        criticHiddenLayersWeightInit = self.hyperparamDict['criticHiddenLayersWeightInit'],
        criticOutputWeightInit = self.hyperparamDict['criticOutputWeightInit'],
        criticOutputBiasInit = self.hyperparamDict['criticOutputBiasInit'],

        
        
        

        max_size = self.hyperparamDict['bufferSize'],
        gamma = self.hyperparamDict['gamma'],
        batch_size = self.hyperparamDict['minibatchSize'],
        initnoisevar = self.hyperparamDict['noiseInitVariance'],
        noiseDecay = self.hyperparamDict['varianceDiscount'],
        noiseDacayStep = self.hyperparamDict['noiseDecayStartStep'],
        minVar = self.hyperparamDict['minVar'],

        path = self.hyperparamDict['modelSavePathPhil']
        )

        score_history = []
        
    
        print(self.hyperparamDict['modelSavePathPhil'])
        for i in range(self.hyperparamDict['maxTimeStep']):
            obs = env.reset()
            done = False
            count = 0
            score = 0
            a = []
            b = []
            while not done:
                act = agent.choose_action(obs,env.action_space.low, env.action_space.high)
                new_state, reward, done, info = env.step(act)
                agent.remember(obs, act, reward, new_state, int(done))
                agent.learn()
                score += reward
                obs = new_state
                count+=1
                #a.append(reward)
                #b.append(act)
      
                #env.render()
                
            score_history.append(score)
            print('episode ', i, 'score %.2f' % score,
                  'trailing 100 games avg %.3f' % np.mean(score_history[-100:]),
                  'timestep:', count)
            if np.mean(score_history[-150:])>=90:
                break
            
        agent.save_models()
        saveToPickle(score_history, self.fixedParameters['rewardSavePathPhil'])
        return score_history
    

