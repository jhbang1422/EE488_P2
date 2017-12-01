# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/19/2017

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from breakout_env import *
from breakout_ani import *
from wait import *

env = breakout_environment(5, 8, 3, 1, 2)

ani = breakout_animation(env, 20)
#ani.save('breakout.mp4', dpi=200)
plt.show(block=False)
wait('Press enter to quit')
 
