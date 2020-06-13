import numpy as np
import gym
import random
from tensorflow.contrib.framework.python.ops import add_arg_scope
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from collections import Counter, defaultdict

Learningrate = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_step = 300
score_requirement = 80
initial_games = 10000

def initial_population():
    training_data = []
    scores = []
    scores_above = []
    for i in range(initial_games):
        score = 0 
        game_memory = []
        ob = env.reset()
        for j in range(goal_step):
            observation = ob
            action = env.action_space.sample()
            observation_next, reward, done, info = env.step(action)
            game_memory.append([observation,action])
            score += reward
            if not done:
                observation = observation_next
            if done:
                break
        if score >= score_requirement:
            scores_above.append(score)
            for data in game_memory:
                if data[1]==1:
                    output = [1,0]
                elif data[1]==0:
                    output = [0,1]
                training_data.append([data[0],output])
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)
    print('average scores accepted =',sum(scores_above)/len(scores_above))
    return training_data

def neural_network(input_size):
    network = input_data(shape=[None,input_size,1],name = 'input')
    network = fully_connected(network,128, activation = 'relu')
    network = fully_connected(network,256, activation = 'relu')
    network = fully_connected(network,512, activation = 'relu')
    network = fully_connected(network,256, activation = 'relu')
    network = fully_connected(network,128, activation = 'relu')
    
    network = fully_connected(network,2,activation = 'softmax')
    network = regression(network,optimizer= 'adam', learning_rate = learningrate,
                        loss = 'categorical_crossentropy',name = 'targets')
    model = tflearn.DNN(network,tensorboard_dir = 'log')
    return model

    
def train_model(training_data,model=False):
    X = np.array([i[0]for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    Y = [i[1]for i in training_data]
    
    if not model:
        model = neural_network(input_size=len(X[0]))
    model.fit({'input':X},{'targets':Y},n_epoch = 3,snapshot_step = 500,show_metric = True,run_id = 'openai')
    return model

def main():
    training_data = initial_population()
    model = train_model(training_data)
    scores = 0
    for each_game in range(15):
        score = 0
        game_memory = []
        prev_obs = []
        env.rest()
        for i in range(goals_steps):
            env.render()
            if len(prev_obs)==0:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            new_observation,reward,done,info=env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation,action])
            score += reward
            if done:
                break
        scores.append(score)
    print('average socre = ', sum(scores)/len(scores))

main()     
