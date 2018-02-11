# -*- coding:utf-8 -*-
"""
@author:zzh
@file:tf5_add_layer.py
@time:2018/2/915:10
"""
import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
	# add one more layer and return the output of this layer
	with tf.name_scope('layer'):
		with tf.name_scope('Weights'):
			Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='w')
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.matmul(inputs, Weights) + biases
		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)
		return outputs


# # make up some real data
# x_data = np.linspace(-1, 1, 300).reshape([300, 1])
# # print(x_data)
# noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
	ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediction and real data
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# important step
init = tf.global_variables_initializer()
with tf.Session() as sess:
	writer = tf.summary.FileWriter("I:/PycharmProjects/tensorflow/tf1_regression/graph/", sess.graph)
	sess.run(init)
	writer.close()

# 启动tensorboard，路径必须为双引号或不带引号，单引号打开后网址内不显示内容
# tensorboard --logdir="I:\PycharmProjects\tensorflow\tf1_regression\graph"

# 举例：
# I:\PycharmProjects\tensorflow\tf1_regression>tensorboard --logdir="I:\PycharmProjects\tensorflow\tf1_regression\graph"
# Starting TensorBoard b'54' at http://DESKTOP-KDG6NEO:6006
# (Press CTRL+C to quit)
