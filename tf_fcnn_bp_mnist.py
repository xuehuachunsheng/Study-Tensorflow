import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 500
n_batch = mnist.train.num_examples // batch_size
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# Define a full connected network
# 784 inputs --> 300 hidden nodes
w1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1))
# 300 hiden nodes with 300 offset
b1 = tf.Variable(tf.zeros([300]))
# 300 hidden nodes --> 10 outputs
w2 = tf.Variable(tf.zeros([300, 10]))
# 10 output nodes with 10 offset 
b2 = tf.Variable(tf.zeros([10]))
# activate function in Hidden layer
L1 = tf.nn.relu(tf.matmul(X, w1) + b1)
# Define output discriminative funtion
y = tf.nn.softmax(tf.matmul(L1, w2) + b2)

# Define loss
loss = tf.reduce_mean(tf.square(Y - y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
print('Two tensor equals? ', tf.argmax(y, 1) == tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run a session
# Init
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

# the number of iterations
for i in range(21):
    for batch in range(n_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        print(type(batch_xs))
        sess.run(train_step, feed_dict={X:batch_xs, Y:batch_ys})
        acc = sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels})
    print('Iter: ', i, ', Testing accuracy: ', acc)
    print('X: ', X)
    print('Y: ', Y)