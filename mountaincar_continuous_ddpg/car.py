import os
import gym
import numpy as np
from car_model import Agent
from utils import plotLearning
from collections import OrderedDict
# Uncomment the lines below to specify which gpu to run on
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    fixedParameters = OrderedDict()
    fixedParameters['alpha'] = 0.001
    fixedParameters['beta'] = 0.001
    fixedParameters['input_dims'] = [env.observation_space.shape[0]]
    fixedParameters['gamma'] = 0.99
    fixedParameters['tau'] = 0.001
    fixedParameters['env'] = env
    fixedParameters['batch_size'] = 128
    fixedParameters['units'] = [30,30]
    fixedParameters['n_actions'] = env.action_space.shape[0]
    fixedParameters['e'] = 0.9
    fixedParameters['e_decay'] = 1
    fixedParameters['e_min'] = 0.005
    fixedParameters['max_size']=1000000 
    fixedParameters['batchsize'] = 128
    fixedParameters['f_dense'] = 0.25
    fixedParameters['f_output'] = 0.003
    fixedParameters['regularizer'] = 0.01

    
    agent = Agent(
        alpha = fixedParameters['alpha'],
        beta = fixedParameters['beta'],
        input_dims = fixedParameters['input_dims'],
        tau =fixedParameters['tau'],
        env = fixedParameters['env'],
        n_actions = fixedParameters['n_actions'],
        e =fixedParameters['e'],
        units = fixedParameters['units'],
        e_decay = fixedParameters['e_decay'],
        e_min = fixedParameters['e_min'],
        f_dense =fixedParameters['f_dense'],
        f_output = fixedParameters['f_output'],
        regularizer = fixedParameters['regularizer'],
        max_size=fixedParameters['max_size'],
        gamma=fixedParameters['gamma'],
        batch_size=fixedParameters['batchsize'])





    np.random.seed(19990525)
    max_step = 1000
    score_history = []
    
    
    for i in range(1000):
        env.reset()
        obs = np.array([-1, 0])
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
        #print(a)
        #print(b)
    filename = 'car.png'
    plotLearning(score_history, filename, window=100)
