import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import pickle
#one hot implies that only one out of every feature is "hot" or on
#mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
#from create_sentiment_featuresets import create_feature_sets_and_labels
import numpy as np


train_x, train_y, test_x, test_y, = pickle.load(open("sentiment_set.pickle", "rb"))
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100 #bathces of data being fed through the network at a time

print("Len of train_x[0] and train_x is ", len(train_x[0]), " ", len(train_x))

#heght x width
x = tf.placeholder('float',[None, len(train_x[0])]) #28*28
y = tf.placeholder('float') #label

def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
					 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					 'biases':tf.Variable(tf.random_normal([n_classes]))}					 					 					

	#(i/p data * weights) + biases , incase data is all 0's, adds dynamic quotient
 
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1) #activation fn

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2) #activation fn

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3) #activation fn

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	#cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y, name=None))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 10

	with tf.Session() as sess:
		#sess.run(tf.global_variables_initializer())﻿
		sess.run(tf.global_variables_initializer())
		for epoch in range(hm_epochs):
			epoch_loss = 0
			
			#for _ in range(int(mnist.train.num_examples/batch_size)):
			#	epoch_x, epoch_y = mnist.train.next_batch(batch_size)
			
			i = 0
			while i < len(train_x):
				start = i
				end = i + batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])


				_, c = sess.run([optimizer, cost], feed_dict = {x : batch_x, y : batch_y})
				epoch_loss += c
				i += batch_size
			print("Epoch: " , epoch, "loss: ", epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print("Accuracy: ", accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)