#!/usr/bin/env python  
# _*_ coding:utf-8 _*_  


from datetime import datetime
import math
import time
import tensorflow as tf

def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
	n_in = input_op.get_shape()[-1].value

	with tf.name_scope(name) as scope:
		print('this scope is ', scope)

		kernel = tf.get_variable(scope + 'w', shape=[kh, kw, n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
		conv = tf.nn.conv2d(input_op, kernel, strides=[1, dh, dw, 1], padding='SAME')
		bias_init_val = tf.constant(0, shape=[n_out], dtype=tf.float32)
		biases = tf.Variable(bias_init_val, name='b')
		activation = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
		p += [kernel, biases]
		return activation

def fc_op(input_op, name, n_out, p):
	n_in = input_op.get_shape()[-1].value
	with tf.name_scope(name) as scope:
		print('this scope is ', scope)
		kernel = tf.get_variable(scope + 'w', shape=[n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
		bias_init_val = tf.constant(0, shape=[n_out], dtype=tf.float32)
		biases = tf.Variable(bias_init_val, name='b')
		activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
		p += [kernel, biases]
		return activation

def max_pool_op(input_op, name, kh, kw, dh, dw):
	return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)


def inference_op(input_op, keep_prob=0.5):
	p = []
	# block 1 --output 112x112x64
	conv1_1 = conv_op(input_op, name='conv1_1', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
	conv1_2 = conv_op(conv1_1, name='conv1_2', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
	pool1 = max_pool_op(conv1_2, name='pool1', kh=2, kw=2, dw=2, dh=2)

	# block 2 --output 56x56x128
	conv2_1 = conv_op(pool1, name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
	conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
	pool2 = max_pool_op(conv2_2, name='pool2', kh=2, kw=2, dw=2, dh=2)

	# block 3 --output 28x28x256
	conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
	conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
	conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
	pool3 = max_pool_op(conv3_3, name='pool3', kh=2, kw=2, dw=2, dh=2)

	# block 4 --output 14x14x512
	conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	pool4 = max_pool_op(conv4_3, name='pool4', kh=2, kw=2, dw=2, dh=2)

	# block 5 --output 7x7x512
	conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	pool5 = max_pool_op(conv5_3, name='pool5', kh=2, kw=2, dw=2, dh=2)

	# flatten
	shp = pool5.get_shape()
	flattened_shape = shp[1].value * shp[2].value * shp[3].value
	resh1 = tf.reshape(pool5, [-1, flattened_shape], name='reshape1')

	# fc6
	fc6 = fc_op(resh1, name='fc6', n_out=4096, p=p)
	fc6_drop = tf.nn.dropout(fc6, keep_prob, name='fc6_drop')

	# fc7
	fc7 = fc_op(fc6_drop, name='fc7', n_out=4096, p=p)
	fc7_drop = tf.nn.dropout(fc7, keep_prob, name='fc7_drop')

	# fc8
	fc8 = fc_op(fc7_drop, name='fc8', n_out=1000, p=p)
	softmax = tf.nn.softmax(fc8)
	predictions = tf.argmax(softmax, 1)
	return predictions, softmax, fc8, p

def time_tensorflow_run(session, target, feed, info_str):
	total_duration = 0.0
	total_duration_squared = 0.0
	for i in range(num_batches + num_steps_burn_in):
		start_time = time.time()
		session.run(target, feed_dict=feed)
		duration = time.time() - start_time
		if i >= num_steps_burn_in:

			print('{}: step {}, duration = {:.3f}'.format(datetime.now(), i - num_steps_burn_in, duration))

			total_duration += duration
			total_duration_squared += duration * duration

	mn = total_duration / num_batches
	vr = total_duration_squared / num_batches - mn**2
	sd = math.sqrt(vr)
	print('{}: {} across {} steps, {:.3f} +/- {:.3f} sec / batch'.format(datetime.now(), info_str, num_batches, mn, sd))


def run_benchmark():
	with tf.Graph().as_default():
		image_size = 224
		images = tf.Variable(tf.random_normal([batch_size,
												image_size,
												image_size, 3],
												dtype=tf.float32,
												stddev=1e-1))

		keep_prob = tf.placeholder(tf.float32)
		predictions, softmax, fc8, p = inference_op(images, keep_prob)
		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, "Forward")
		objective = tf.nn.l2_loss(fc8)
		grad = tf.gradients(objective, p)
		time_tensorflow_run(sess, grad, {keep_prob: 0.5}, "Forward-backward")

batch_size = 32
num_batches = 10
num_steps_burn_in = 1
run_benchmark()

