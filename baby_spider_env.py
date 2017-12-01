# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/12/2017

import numpy as np

# baby spider environment
class baby_spider_environment: 
    def __init__(self):
        self.n_states = 4         # number of states: leg up/down, forward/backward
        self.n_actions = 4        # number of actions 
        self.reward = np.zeros([self.n_states, self.n_actions])
        self.terminal = np.zeros(self.n_states, dtype=np.int)          # 1 if terminal state, 0 otherwise
        self.next_state = np.zeros([self.n_states, self.n_actions], dtype=np.int)        # next_state
        self.init_state = 0b00    # initial state
        transition = [[1,0,2,0],[1,0,3,1],[3,2,2,0],[3,2,3,1]]
        for s in range(4):
            up = s & 1
            fw = (s >> 1) & 1
            for a in range(4):
                action_up = (a & 3) == 0
                action_dn = (a & 3) == 1
                action_fw = (a & 3) == 2
                action_bw = (a & 3) == 3
                self.next_state[s,a] = transition[s][a]
                self.reward[s,a] = (up == 0 and fw == 1 and action_bw == 1) - (up == 0 and fw == 0 and action_fw == 1)

