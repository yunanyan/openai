import os
import gym
import numpy as np
from car_model import Agent
from utils import plotLearning

# Uncomment the lines below to specify which gpu to run on
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    agent = Agent(alpha=0.001, beta=0.001, input_dims=[2], tau=0.001,
                  env=env, batch_size=128, layer1_size=30, layer2_size=30,
                  n_actions=1, e = 0.9)
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
    filename = 'car-alpha00005-beta0005-800-600-optimized.png'
    plotLearning(score_history, filename, window=100)
