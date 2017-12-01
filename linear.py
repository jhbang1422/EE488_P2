# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/12/2017

import numpy as np
from qlearning import *
import matplotlib.pyplot as plt
# state action  next  reward  terminal 
#   0     up      1      0      Yes
#   0     down    2      0      No
#   2     up      3      1      Yes
#   2     down    4      0      Yes

# environment with 5 states where 3 of them are terminal states (episodic task)
class linear_environment:
    def __init__(self):
        self.n_states = 21       # number of states
        self.n_actions = 2      # number of actions
        self.next_state = np.array([[0,0],[0,2],[1,3],[2,4],[3,5],[4,6],[5,7],[6,8],[7,9],[8,10],[9,11],[10,12],[11,13],[12,14],[13,15],[14,16],[15,17],[16,18],[17,19],[18,20],[20,20]], dtype=np.int)    # next_state
        self.reward = np.array([[0.,0.],[1.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,1.],[0.,0.]])            # reward for each (state,action)
        self.terminal = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], dtype=np.int)                          # 1 if terminal state, 0 otherwise
        self.init_state = 10     # initial state

# an instance of the environment
env = linear_environment()

n_episodes = 100      # number of episodes to run
max_steps = 1000     # max. # of steps to run in each episode
alpha = 0.2          # learning rate
gamma = 0.9          # discount factor

class epsilon_profile: pass
epsilon = epsilon_profile()
epsilon.init = 1.    # initial epsilon in e-greedy
epsilon.final = 0.   # final epsilon in e-greedy
epsilon.dec_episode = 1. / n_episodes  # amount of decrement in each episode
#epsilon.dec_episode=0.
epsilon.dec_step = 0.                  # amount of decrement in each step

Q, n_steps, sum_rewards = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon)
print('Q(s,a)')
print(Q)
for k in range(n_episodes):
    print('%2d: %.2f' % (k, sum_rewards[k]))

plt.plot(range(n_episodes), n_steps)
plt.show()

test_n_episodes = 1  # number of episodes to run
test_max_steps =1000 # max. # of steps to run in each episode
test_epsilon = 0.    # test epsilon
test_n_steps, test_sum_rewards, s, a, sn, r = Q_test(Q, env, test_n_episodes, test_max_steps, test_epsilon)
print(test_sum_rewards[0])
print(test_n_steps[0])

