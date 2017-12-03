# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/19/2017

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import random
import math
from breakout_env import *
#from breakout_ani import *
from wait import *


env = breakout_environment(5, 8, 3, 1, 2)
#FLAG = 'TEST'

epsilon = 1  # The probability of choosing a random action (in training). This decays as iterations increase. (0 to 1)
epsilonMinimumValue = 0.001 # The minimum value we want epsilon to reach in training. (0 to 1)
nbActions = 3 # The number of actions. Since we only have left/stay/right that means 3 actions.
epoch = 300 # The number of games we want the system to run for.
hiddenSize = 100 # Number of neurons in the hidden layers.
maxMemory = 500 # How large should the memory be (where it stores its past experiences).
batchSize = 50 # The mini-batch size for training. Samples are randomly taken from memory till mini-batch size.
nbStates = env.ny*env.nx*env.nf # We eventually flatten to a 1d tensor to feed the network.
discount = 0.9 # The discount is used to force the network to choose states that lead to the reward quicker (0 to 1)  
learningRate = 0.2 # Learning Rate for Stochastic Gradient Descent (our optimizer).
maxStep = 200

# Create the base model.
X = tf.placeholder(tf.float32, [None, nbStates])
W1 = tf.Variable(tf.truncated_normal([nbStates, hiddenSize], stddev=1.0 / math.sqrt(float(nbStates))))
b1 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.01))  
input_layer = tf.nn.relu(tf.matmul(X, W1) + b1)
W2 = tf.Variable(tf.truncated_normal([hiddenSize, hiddenSize],stddev=1.0 / math.sqrt(float(hiddenSize))))
b2 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.01))
hidden_layer = tf.nn.relu(tf.matmul(input_layer, W2) + b2)
W3 = tf.Variable(tf.truncated_normal([hiddenSize, nbActions],stddev=1.0 / math.sqrt(float(hiddenSize))))
b3 = tf.Variable(tf.truncated_normal([nbActions], stddev=0.01))
#output_layer = tf.nn.softmax(tf.matmul(hidden_layer, W3) + b3)
output_layer = tf.matmul(hidden_layer, W3) + b3
# True labels
Y = tf.placeholder(tf.float32, [None, nbActions])

# Mean squared error cost function
cost = tf.reduce_sum(tf.square(Y-output_layer)) / (2*batchSize)

# Stochastic Gradient Decent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)


# Helper function: Chooses a random value between the two boundaries.
def randf(s, e):
  return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s;


class ReplayMemory:
  def __init__(self, maxMemory, discount):
    self.maxMemory = maxMemory
    self.nbStates = nbStates
    self.discount = discount
    self.inputState = np.empty((self.maxMemory, self.nbStates), dtype = np.float32)
    self.actions = np.zeros(self.maxMemory, dtype = np.int8)
    self.nextState = np.empty((self.maxMemory, self.nbStates), dtype = np.float32)
    self.gameOver = np.empty(self.maxMemory, dtype = np.bool)
    self.rewards = np.empty(self.maxMemory, dtype = np.int8) 
    self.count = 0
    self.current = 0

  # Appends the experience to the memory.
  def remember(self, currentState, action, reward, nextState, gameOver):
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.inputState[self.current, ...] = currentState
    self.nextState[self.current, ...] = nextState
    self.gameOver[self.current] = gameOver
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.maxMemory

  def getBatch(self, model, batchSize, nbActions, nbStates, sess, X):
    
    # We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
    # batch we can (at the beginning of training we will not have enough experience to fill a batch).
    memoryLength = self.count
    chosenBatchSize = min(batchSize, memoryLength)

    inputs = np.zeros((chosenBatchSize, nbStates))
    targets = np.zeros((chosenBatchSize, nbActions))

    # Fill the inputs and targets up.
    for i in xrange(chosenBatchSize):
      if memoryLength == 1:
        memoryLength = 2
      # Choose a random memory experience to add to the batch.
      randomIndex = random.randrange(1, memoryLength)
      current_inputState = np.reshape(self.inputState[randomIndex], (1, self.nbStates))

      target = sess.run(model, feed_dict={X: current_inputState})
      
      current_nextState =  np.reshape(self.nextState[randomIndex], (1, self.nbStates))
      current_outputs = sess.run(model, feed_dict={X: current_nextState})      
      
      # Gives us Q_sa, the max q for the next state.
      nextStateMaxQ = np.amax(current_outputs)
      if (self.gameOver[randomIndex] == True):
      	# print (str(self.actions) + "              " + str(self.rewards[randomIndex]))
        target[0, self.actions[randomIndex]+1] = self.rewards[randomIndex]
      else:
        # reward + discount(gamma) * max_a' Q(s',a')
        # We are setting the Q-value for the action to  r + gamma*max a' Q(s', a'). The rest stay the same
        # to give an error of 0 for those outputs.
        # print (str(self.actions) + "             " + str(self.rewards[randomIndex]) + "              " + str(self.discount) + "                  " + str(nextStateMaxQ))
        target[0, self.actions[randomIndex]+1] = self.rewards[randomIndex] + self.discount * nextStateMaxQ

      # Update the inputs and targets.
      inputs[i] = current_inputState
      targets[i] = target

    return inputs, targets

    
def main(_):
	print("Training new model")


  	# Define Replay Memory
  	memory = ReplayMemory(maxMemory, discount)

  	# Add ops to save and restore all the variables.
  	saver = tf.train.Saver()
  
  	
  	with tf.Session() as sess:   
  		tf.global_variables_initializer().run() 

  		for i in xrange(epoch):
      		# Initialize the environment.
  			err = 0
  			currentState=np.reshape(env.reset(),[1,nbStates])
  			isGameOver = False
			winCount = 0            
			#while (isGameOver != True):
			for j in range(maxStep):
				if isGameOver == True:
					break
				global epsilon
				if (randf(0, 1) <= epsilon):
					action = random.randrange(0, nbActions)-1
					# print('action : ' + str(action))
				else:          
					# Forward the current state through the network.
					q = sess.run(output_layer, feed_dict={X: currentState})          
					# Find the max index (the chosen action).
					index = q.argmax()
					action = index-1     
					# print ('action : ' + str(action))

				# Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
				if (epsilon > epsilonMinimumValue):
					epsilon = epsilon * 0.999
				
				nextState, reward, gameOver, _,_,_,_,_,_,_,_ = env.run(action)
				nextState = np.reshape(nextState, [1,nbStates])    
				if (reward == 1):
					winCount = winCount + 1

				memory.remember(currentState, action, reward, nextState, gameOver)
				
				# Update the current state and if the game is over.
				currentState = nextState
				isGameOver = gameOver
					
				# We get a batch of training data to train the model.
				inputs, targets = memory.getBatch(output_layer, batchSize, nbActions, nbStates, sess, X)
				
				# Train the network which returns the error.
				_, loss = sess.run([optimizer, cost], feed_dict={X: inputs, Y: targets})  
				err = err + loss

			print("Epoch " + str(i) + ": err = " + str(err) + ": Win count = " + str(winCount) + " Win ratio = " + str(float(winCount)/float(i+1)*100))
    	# Save the variables to disk.
    		save_path = saver.save(sess, "./breakout.ckpt")
   	 	print("Model saved in file: %s" % save_path)

if __name__ == '__main__':
	tf.app.run()
