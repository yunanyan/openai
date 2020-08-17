import os
import gym
import numpy as np
from car_model_for_evluation import Agent
from utils import plotLearning
from time import time
import pandas as pd
import pylab as plt
import seaborn as sns
import tensorflow as tf

# Uncomment the lines below to specify which gpu to run on
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def moving_avg(list,window_size=10):
    i = 0
    moving_averages = []
    while i < len(list) - window_size + 1:
        this_window = list[i : i + window_size]

        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1

    return moving_averages

def evaluateModel(modelDf):
    tf.reset_default_graph()
    treatment_level = modelDf.index[0]
    learning_rate = treatment_level[0]
    e_ = treatment_level[1]
    print([learning_rate,e_])
    env = gym.make('MountainCarContinuous-v0')
    agent = Agent(alpha=learning_rate, beta=learning_rate, input_dims=[2], tau=0.001,
                  env=env, batch_size=128, layer1_size=30, layer2_size=30,
                  n_actions=1)
    np.random.seed(19990525)
    max_step = 1000
    score_history = []
    count_epi =[]
    count_game = 0
    e = e_
    
    for i in range(500):
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
        count_game= count_game+1
        count_epi.append(count_game)
        print('episode ', i, 'score %.2f' % score,
              'trailing 100 games avg %.3f' % np.mean(score_history[-100:]),
              'timestep:', count)
        #if np.mean(score_history[-150:])>=90:
            #break


    moving_average = moving_avg(score_history,window_size=50)
    count_x = [i for i in range(len(moving_average))]
    resultSe = pd.DataFrame({'score':moving_average, 'count':count_x})

    return resultSe

    #agent.save_models()
        #print(a)
        #print(b)
    #filename = 'car-alpha00005-beta0005-800-600-optimized.png'
    #plotLearning(score_history, filename, window=100)

def drawLinePlot(plotDf, ax):
    for learning_rate, subDf in plotDf.groupby('learning_rate'):
        subDf = subDf.droplevel('learning_rate')
        subDf.plot.line(ax = ax, label = 'learning_rate = {}'.format(learning_rate), y = 'reward', marker = 'o')

def main():
    learning_rate= [0.001,0.002,0.005]
    e = [0.9,0.75,0.5]

    levelValues = [learning_rate, e]
    levelNames = ['learning_rate', 'e']

    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)

    toSplitFrame = pd.DataFrame(index = modelIndex)

    modelResultDf = toSplitFrame.groupby(levelNames).apply(evaluateModel)

    fig = plt.figure()
    plotLevels = ['learning_rate']
    plotColNum = len(e)
    plotCounter = 1

    for (key, plotDf) in modelResultDf.groupby(['e']):
        plotDf.index = plotDf.index.droplevel([ 'e'])
        for learning_rate, subDf in plotDf.groupby('learning_rate'):
            subDf = subDf.droplevel('learning_rate')
            ax = fig.add_subplot(3,plotColNum, plotCounter)
            subDf.plot.line(ax = ax, y='score',x='count', title = learning_rate, legend = False,marker=',')
            plotCounter+=1


    plt.show()
                                                

                        
main()                                     
