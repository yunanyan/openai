import gym
import os
import numpy as np
import tensorflow as tf
from tensorflow.initializers import random_uniform
import matplotlib.pyplot as plt
from utils import plotLearning

       

class Agent(object):
    def __init__ (self,lr, gamma,n_actions, input_dims):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.action_space = [i for i in range(n_actions)]
        self.state_memory = []
        self.reward_memory = []
        self.action_memory = []
        self.sess = tf.Session()
        self.build_network()
        
        
        self.sess.run(tf.global_variables_initializer())
        
        
        
        
    def build_network(self):
        with tf.variable_scope('network'):
            self.input = tf.placeholder(tf.float32, shape = [None, self.input_dims], name = 'inputs')
            self.actions = tf.placeholder(tf.int32, shape = [None, ], name = 'actions')
            self.advantages = tf.placeholder(tf.float32, shape = [None,], name = 'advantages')
            

            l1 = tf.layers.dense(inputs=self.input, units=64,
                                 activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

        
            l2 = tf.layers.dense(inputs=l1, units=64,
                                 activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            l3 = tf.layers.dense(inputs=l2, units=self.n_actions,
                                 activation=None,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.probs = tf.nn.softmax(l3, name='probs')
                                         


            likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                    logits=l3, labels=self.actions)

            self.loss = likelihood*self.advantages
            self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        
    
    def memorize(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
    

        
        
    
    def act(self,state):
        state = state[np.newaxis,:]
        softmax_out = self.sess.run(self.probs, feed_dict={self.input: state})[0]
        selected_action = np.random.choice(self.action_space, p=softmax_out)
        return selected_action
    
    def learn(self):
        state_memory = np.array(self.state_memory)
        reward_memory = np.array(self.reward_memory)
        action_memory = np.array(self.action_memory)
        print(action_memory)
        G = np.zeros_like(reward_memory)
   
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for i in range(t, len(reward_memory)):
                G_sum += reward_memory[i]*discount
                discount = discount*self.gamma
            G[t] = G_sum
        
        mean = np.mean(G)
        std = np.std(G) if np.std(G)>0 else 1
        G = (G-mean)/std
        
        _,lossval = self.sess.run([self.optimize,self.loss],
                            feed_dict={self.input: state_memory,
                                       self.actions: action_memory,
                                       self.advantages: G})

       
       
            
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

def main():
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    n_action = env.action_space.n
    agent = Agent(lr=0.0005, gamma=0.95,n_actions=n_action, input_dims=input_dim)
    
  
    score_history = []
    score = 0
    num_episodes = 1
    #env = wrappers.Monitor(env, "tmp/lunar-lander",
    #                        video_callable=lambda episode_id: True, force=True)
    for i in range(num_episodes):
        print('episode: ', i,'score: ', score)
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.act(observation)
            observation_, reward, done, info = env.step(action)
            agent.memorize(observation, action, reward)
            observation = observation_
            score += reward
        score_history.append(score)
        agent.learn()

    plotLearning(score_history, filename='reinforce-cartpole.png', window=25)
        

main()
        
                
