# -*- coding:utf-8 -*-
"""
@author:zzh
@file:tf4_placeholder.py
@time:2018/2/914:28
"""
import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
ouput = tf.multiply(input1, input2)

with tf.Session() as sess:
	print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))
