import gym
import numpy as np
from collections import defaultdict
env = gym.make('CartPole-v0')
epsilon = 0.1
actions = [0,1]
qtable = defaultdict(float)

def act(ob):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    qval = {a:q[ob,a] for a in actions}
    max_q = max(qvals.values())
    actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
    return np.random.choice(actions_with_max_q)

def update_Q(s,r,a,s_next,done):
    max_q_next = max([qtable[s_next, a] for a in actions])
    qtable[s, a] += alpha * (r + gamma * max_q_next * (1.0 - done) - qtable[s, a])
    
