# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/12/2017

import numpy as np
from qlearning import *

# state action  next  reward  terminal 
#   0     up      0      0      No
#   0     down    1      0      No
#   1     up      0      1      No 
#   1     down    0      0      No

# environment with 2 states (continuing task)
class continuing_task_example_environment:
    def __init__(self):
        self.n_states = 2       # number of states
        self.n_actions = 2      # number of actions
        self.next_state = np.array([[0,1],[0,0]], dtype=np.int)    # next_state
        self.reward = np.array([[0.,0.],[1.,0.]])            # reward for each (state,action)
        self.terminal = np.array([0,0], dtype=np.int)                          # 1 if terminal state, 0 otherwise
        self.init_state = 0     # initial state

# an instance of the environment
env = continuing_task_example_environment()

n_episodes = 1       # number of episodes to run, 1 for continuing task
max_steps = 1000     # max. # of steps to run in each episode
alpha = 0.2          # learning rate
gamma = 0.9          # discount factor

class epsilon_profile: pass
epsilon = epsilon_profile()
epsilon.init = 1.    # initial epsilon in e-greedy
epsilon.final = 0.   # final epsilon in e-greedy
epsilon.dec_episode = 0.               # amount of decrement in each episode
epsilon.dec_step = 1. / max_steps      # amount of decrement in each step

Q, n_steps, sum_rewards = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon)
print('Q(s,a)')
print(Q)
print(sum_rewards[0])

test_n_episodes = 1  # number of episodes to run
test_max_steps = 100 # max. # of steps to run in each episode
test_epsilon = 0.    # test epsilon
test_n_steps, test_sum_rewards, s, a, sn, r  = Q_test(Q, env, test_n_episodes, test_max_steps, test_epsilon)
print(test_sum_rewards[0])

