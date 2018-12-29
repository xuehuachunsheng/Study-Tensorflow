#!/usr/bin/env python
# _*_ coding:utf-8 _*_
#
# @Version : 1.0
# @Time    : 2017/1/9 10:14
# @Author  : dongyouyuan
# @File    : dyy_test.py
#
# 此脚本仅为一个演示脚本

import tensorflow as tf
img1 = tf.constant(value=[[[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]]]],dtype=tf.float32)
img2 = tf.constant(value=[[[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]]]],dtype=tf.float32)
img = tf.concat(values=[img1,img2],axis=3)
filter1 = tf.constant(value=0, shape=[3,3,1,1],dtype=tf.float32)
filter2 = tf.constant(value=1, shape=[3,3,1,1],dtype=tf.float32)
filter3 = tf.constant(value=2, shape=[3,3,1,1],dtype=tf.float32)
filter4 = tf.constant(value=3, shape=[3,3,1,1],dtype=tf.float32)
filter_out1 = tf.concat(values=[filter1,filter2],axis=2)
filter_out2 = tf.concat(values=[filter3,filter4],axis=2)
filter = tf.concat(values=[filter_out1,filter_out2],axis=3)

point_filter = tf.constant(value=1, shape=[1,1,4,4],dtype=tf.float32)
out_img1 = tf.nn.depthwise_conv2d(input=img, filter=filter, strides=[1,1,1,1],rate=[1,1], padding='VALID')

out_img2 = tf.nn.conv2d(input=out_img1, filter=point_filter, strides=[1,1,1,1], padding='VALID')

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	print(sess.run(out_img1))
	print(sess.run(tf.shape(out_img1)))
	print(sess.run(out_img2))
	print(sess.run(tf.shape(out_img2)))
