# -*- coding:utf-8 -*-
# Tensorflow linear regression

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
X = np.random.rand(100).astype(np.float32)
Y = 0.1*X*X + 0.3*X + 1
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
a = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))
y = W * X * X + a * X + b
# Define loss function
loss = tf.reduce_mean(tf.square(y - Y))
# Gradient descent to reduce loss
train = tf.train.GradientDescentOptimizer(0.5)
# Init variable
init = tf.global_variables_initializer()

train = train.minimize(loss)
print(init)

sess = tf.Session()
sess.run(init)
for i in range(2001):
    print(sess.run(train))
    # if(i % 20 == 0):
    #     print(train)
    #     print(i, sess.run(W), sess.run(a), sess.run(b))