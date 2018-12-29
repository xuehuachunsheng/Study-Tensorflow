#!/usr/bin/env python  
# _*_ coding:utf-8 _*_  
#  
# @Version : 1.0  
# @Time    : 2017/1/9 10:14  
# @Author  : yanxuewu
# @File    : alexnet.py
#  
# 此脚本仅为一个演示脚本

from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100

def print_activation(t):
	print(t.op.name, '', t.get_shape().as_list())

def inference(images):
	parameters = []
	with tf.name_scope('conv1') as scope:
		kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), name='biases')
		bias_conv = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(bias_conv, name=scope)
		parameters += [kernel, biases]
		print_activation(conv1)

	# Local response normalization
	lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=1e-3/9.0, beta=0.75, name='lrn1')
	pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
	print_activation(pool1)

	with tf.name_scope('conv2') as scope:
		kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(bias, name=scope)
		parameters += [kernel, biases]
		print_activation(conv2)

	lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=1e-3/9.0, beta=0.75, name='lrn2')
	pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
	print_activation(pool2)

	with tf.name_scope('conv3') as scope:
		kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), name='biases')
		bias_conv = tf.nn.bias_add(conv, biases)
		conv3 = tf.nn.relu(bias_conv, name=scope)
		parameters += [kernel, biases]
		print_activation(conv3)

	with tf.name_scope('conv4') as scope:
		kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), name='biases')
		bias_conv = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(bias_conv, name=scope)
		parameters += [kernel, biases]
		print_activation(conv4)

	with tf.name_scope('conv5') as scope:
		kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), name='biases')
		bias_conv = tf.nn.bias_add(conv, biases)
		conv5 = tf.nn.relu(bias_conv, name=scope)
		parameters += [kernel, biases]
		print_activation(conv5)

	pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
	print_activation(pool5)
	return pool5, parameters

def time_tensorflow_run(session, target, info_str):
	num_steps_burn_in = 10
	total_duration = 0.0
	total_duration_squared = 0.0
	for i in range(num_batches + num_steps_burn_in):
		start_time = time.time()
		session.run(target)
		duration = time.time() - start_time
		if i >= num_steps_burn_in:
			if i % 10 != 0:
				print('{}: step {}, duration = {:.3f}'.format(datetime.now(), i - num_steps_burn_in, duration))

			total_duration += duration
			total_duration_squared += duration * duration

	mn = total_duration / num_batches
	vr = total_duration_squared / num_batches - mn**2
	sd = math.sqrt(vr)
	print('{}: {} across {} steps, {:.3f} +/- {:.3f} sec / batch'.format(datetime.now(), info_str, num_batches, mn, sd))

def run_benchmark():
	g = tf.Graph()
	with g.as_default():
		image_size = 224
		# Generate random images
		images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))
		pool5, parameters = inference(images)
		init = tf.global_variables_initializer()

	with tf.Session(graph=g) as sess:
		sess.run(init)
		time_tensorflow_run(sess, pool5, 'Forward')
		objective = tf.nn.l2_loss(pool5)
		grad = tf.gradients(objective, parameters)
		time_tensorflow_run(sess, grad, 'Forward-backward')

run_benchmark()