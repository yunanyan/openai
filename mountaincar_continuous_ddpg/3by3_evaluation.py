import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from gym import wrappers
from time import time
import pandas as pd
import pylab as plt
import seaborn as sns


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

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
    treatment_level = modelDf.index[0]
    batch_size = treatment_level[0]
    learning_rate = treatment_level[1]
    print([batch_size,learning_rate])
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    EPISODES = 10
    track_scores = []
    count_epi =[]
    count = 0
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        

        track_scores.append(time)
        count= count+1
        count_epi.append(count)
        

    moving_average = moving_avg(track_scores, window_size=1)
    mean = sum(moving_average)/len(moving_average)
resultSe = pd.DataFrame({'score':track_scores, 'count':count_epi})
        
    
    
    return resultSe

def drawLinePlot(plotDf, ax):
    for learning_rate, subDf in plotDf.groupby('learning_rate'):
        subDf = subDf.droplevel('learning_rate')
   
        subDf.plot.line(ax = ax, y='score',x='count',label = 'batch_size = {}'.format(batch_size), marker = '.')
       

def main():
    batch_size = [16,32,48]
    learning_rate= [0.001,0.002,0.003]

    levelValues = [batch_size, learning_rate]
    levelNames = ['batch_size', 'learning_rate']

    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)

    toSplitFrame = pd.DataFrame(index = modelIndex)

    modelResultDf = toSplitFrame.groupby(levelNames).apply(evaluateModel)

    fig = plt.figure()
    plotLevels = ['batch_size']
    plotColNum = len(learning_rate)
    plotCounter = 1

    for (key, plotDf) in modelResultDf.groupby(['learning_rate']):
        plotDf.index = plotDf.index.droplevel([ 'learning_rate'])
        for learning_rate, subDf in plotDf.groupby('batch_size'):
            subDf = subDf.droplevel('batch_size')
            ax = fig.add_subplot(3,plotColNum, plotCounter)
            subDf.plot.line(ax = ax, y='score',x='count', title = learning_rate, legend = False,marker=',')
            plotCounter+=1
        

    plt.show()
                                                

                        
main()                                     

