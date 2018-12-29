#!/usr/bin/env python  
# _*_ coding:utf-8 _*_  
#  
# @Version : 1.0  
# @Time    : 2017/1/9 10:14  
# @Author  : dongyouyuan  
# @File    : dyy_test.py  
#  
# 此脚本仅为一个演示脚本

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

# Input
x = tf.placeholder(tf.float32, [None, in_units])
y_ = tf.placeholder(tf.float32, [None, 10])


keep_prob = tf.placeholder(tf.float32)
h1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
h1_drop = tf.nn.dropout(h1, keep_prob=keep_prob)
y = tf.nn.softmax(tf.add(tf.matmul(h1_drop, W2), b2))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
global_step =tf.Variable(0, dtype=tf.int64, trainable=False)
train_step = tf.train.AdagradDAOptimizer(0.3, global_step=global_step).minimize(cross_entropy,global_step=global_step)

tf.global_variables_initializer().run()

for i in range(3000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
	print(sess.run(global_step))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1}))
