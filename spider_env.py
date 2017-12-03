# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/12/2017

import numpy as np

# baby spider environment
class spider_environment: 
    def __init__(self):
        self.n_states = 256         # number of states: leg up/down, forward/backward
        self.n_actions = 256        # number of actions 
        self.reward = np.zeros([self.n_states, self.n_actions])
        self.terminal = np.zeros(self.n_states, dtype=np.int)          # 1 if terminal state, 0 otherwise
        self.next_state = np.zeros([self.n_states, self.n_actions], dtype=np.int)        # next_state
        self.init_state = 0b00001010    # initial state
        transition = [[1,0,2,0],[1,0,3,1],[3,2,2,0],[3,2,3,1]]
        for s in range(256):
            s_rb = (s & (3<<6))>>6
            s_lb = (s & (3<<4))>>4
            s_rf = (s & (3<<2))>>2
            s_lf = (s & 3) 
            up_rb = s_rb & 1
            fw_rb = (s_rb >> 1) & 1
            up_lb = s_lb & 1
            fw_lb = (s_lb >> 1) & 1
            up_rf = s_rf & 1
            fw_rf = (s_rf >> 1) & 1
            up_lf = s_lf & 1
            fw_lf = (s_lf >> 1) & 1

            for a in range(256):
                a_rb = (a & (3<<6)) >> 6
                a_lb = (a & (3<<4)) >> 4
                a_rf = (a & (3<<2)) >> 2
                a_lf = a & 3
                
                rb_action_up = (a_rb & 3) == 0
                rb_action_dn = (a_rb & 3) == 1
                rb_action_fw = (a_rb & 3) == 2
                rb_action_bw = (a_rb & 3) == 3
                
                lb_action_up = (a_lb & 3) == 0
                lb_action_dn = (a_lb & 3) == 1 
                lb_action_fw = (a_lb & 3) == 2
                lb_action_bw = (a_lb & 3) == 3

                rf_action_up = (a_rf & 3) == 0
                rf_action_dn = (a_rf & 3) == 1
                rf_action_fw = (a_rf & 3) == 2
                rf_action_bw = (a_rf & 3) == 3

                lf_action_up = (a_lf & 3) == 0
                lf_action_dn = (a_lf & 3) == 1
                lf_action_fw = (a_lf & 3) == 2
                lf_action_bw = (a_lf & 3) == 3

                # print(str([s,a]) + ":" +str([s_rb, s_lb, s_rf, s_lf]) + "," + str([a_rb, a_lb, a_rf, a_lf]))
                self.next_state[s,a] = (transition[s_rb][a_rb]<<6) + (transition[s_lb][a_lb]<<4) +\
                (transition[s_rf][a_rf]<<2) + transition[s_lf][a_lf]
                # calculate total_down
                total_down=0
                for i in range(4) :
                    if (s & 1<<(2*i)) >> (2*i) == (self.next_state[s,a] & 1<<(2*i)) >> (2*i):
                        total_down = total_down+1
                # print ('total_down : ' + str(total_down))

                total_force = (up_rb==0 and fw_rb==1 and rb_action_bw==1) - (up_rb==0 and fw_rb==0 and rb_action_fw==1)  + \
                (up_lb==0 and fw_lb==1 and lb_action_bw ==1) - (up_lb==0 and fw_lb==0 and lb_action_fw==1) + \
                (up_rf==0 and fw_rf==1 and rf_action_bw ==1) - (up_rf==0 and fw_rf==0 and rf_action_fw==1) + \
                (up_lf==0 and fw_lf==1 and lf_action_bw ==1) - (up_lf==0 and fw_lf==0 and lf_action_fw==1)

                if total_down==0 : 
                    self.reward[s,a] = 0
                elif total_down >= 3:
                    self.reward[s,a]= 1.0*total_force/total_down
                elif total_down == 2 and ((up_lf==0 and up_rb==0) or (up_lb==0 and up_rf==0)):
                    self.reward[s,a] = 1.0*total_force/total_down
                else :
                    self.reward[s,a] = 0.25 * total_force/total_down

                # print('reward :' + str(self.reward[s,a]))

