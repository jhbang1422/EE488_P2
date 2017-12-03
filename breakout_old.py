# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/19/2017

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import random
from breakout_env import *
from breakout_ani import *
from wait import *


env = breakout_environment(5, 8, 3, 1, 2)
FLAG = 'TEST'

# Make model 
hidden_neuron = 1000
learning_rate = 0.01
num_state = env.nx * env.ny * env.nf
X = tf.placeholder(tf.float32, [None,num_state])

W1 = tf.Variable(tf.truncated_normal([num_state, hidden_neuron], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[hidden_neuron]))
input_layer = tf.nn.relu(tf.matmul(X, W1) + b1)
W2 = tf.Variable(tf.truncated_normal([hidden_neuron, hidden_neuron], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[hidden_neuron]))
hidden_layer = tf.nn.relu(tf.matmul(input_layer, W2)+b2)
W3 = tf.Variable(tf.truncated_normal([hidden_neuron, env.na], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[env.na]))
#output_layer = tf.nn.softmax(tf.matmul(hidden_layer, W3)+b3)
output_layer = tf.nn.softmax(tf.matmul(hidden_layer, W3) + b3)


# Make model 
# learning_rate = 0.01 
# X= tf.placeholder(tf.float32, [None, env.ny, env.nx, env.nf])


Y = tf.placeholder(tf.float32, [None, env.na])
cost = tf.reduce_sum(tf.square(Y-output_layer))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

n_episodes = 2000
max_step = 200
discount = 0.995

class epsilon_profile: pass
epsilon = epsilon_profile()
epsilon.init = 1.    # initial epsilon in e-greedy
epsilon.final = 0.   # final epsilon in e-greedy
epsilon.dec_episode = 1. / n_episodes  # amount of decrement in each episode
epsilon.dec_step = 0.   # amount of decrement in each step
if FLAG == 'TRAIN':
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	for i in range(n_episodes):
		current_state = np.reshape(env.reset(), (1,num_state))
		# current_state = env.reset()
		err = 0
		terminal = 0
		target_reward = np.zeros((1,env.na))
		reward_sum = 0
		for j in range(max_step):
			if terminal == 1:
				break
			# target_reward = np.zeros((1,env.na))
			if (random.uniform(0,1) <= epsilon):
				idx = random.randrange(0,env.na)
			else :
				q = sess.run(output_layer, feed_dict={X: np.reshape(current_state, [1, num_state])})
				idx = np.random.choice(np.where(q[0]==np.max(q))[0])
			action = idx-1

			if not (action == 1 or action == 0 or action == -1):
				print('action error')
				exit()
			next_state, reward, terminal, p0, p, bx0, by0, vx0, vy0, rx, ry = env.run(action)
			print('action : '+ str(action) + ' reward : ' + str(reward))			

			next_state = np.reshape(next_state, [1, num_state])


			# target_reward[0,idx] = reward
			_,loss = sess.run([optimizer, cost],feed_dict={X: current_state, Y:target_reward})

			
			err = err + loss

			target_reward = q
			if (terminal == 1):
				print('terminate')
				target_reward[0,idx] = reward
			else :
				print('continue')
				next_state_q = sess.run(output_layer, feed_dict={X: np.reshape(next_state, [1, num_state])})
				next_state_max_q = np.amax(next_state_q)
				print('idx:'+str(idx) + 'reward' + str(reward) + 'max q:' + str(next_state_max_q) +\
					'q: ' + str(next_state_q))
				target_reward[0,idx] = reward + discount * next_state_max_q
				#target_reward[0,idx] = reward

			current_state = next_state
			reward_sum = reward_sum + reward
		print("Episode " + str(i) + " : err = " + str(err) + " reward sum = " + str(reward_sum))

	save_path = saver.save(sess, "./breakout.ckpt")

if FLAG == 'TEST':
	ani = breakout_animation(env, max_step)
	#ani.save('breakout.mp4', dpi=200)
	plt.show(block=False)
	wait('Press enter to quit')
 
