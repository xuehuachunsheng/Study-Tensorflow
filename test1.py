import tensorflow as tf
import numpy as np
a = tf.constant(2, tf.int16)
b = tf.constant(4, tf.float32)
c = tf.constant(8, tf.float32)
print(a, b, c)
d = tf.Variable(2, tf.int64)
e = tf.Variable(4, tf.float32)
f = tf.Variable(8, tf.float32)
print(d, e, f)
g = tf.constant(np.zeros(shape=(2,2), dtype=np.float32))
print(g)
h = tf.zeros([11], tf.int16)
i = tf.ones([2, 2], tf.float32)
j = tf.zeros([1000, 4, 3], tf.float64)
print(h, i, j)
k = tf.Variable(tf.zeros([2, 2], tf.float32))
l = tf.Variable(tf.zeros([5, 6, 5], tf.float32))
print(k, l)

weights = tf.Variable(tf.truncated_normal([256*256, 10]))
biases = tf.Variable(tf.zeros([10]))
print(weights.get_shape().as_list())
print(biases.get_shape().as_list())

graph = tf.Graph()
with graph.as_default():
    a = tf.Variable(8, tf.float32)
    b = tf.Variable(tf.zeros([2, 2], tf.float32))

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    print(sess.run(a))
    print(sess.run(b))

list_of_points1_ = [[1,2], [3,4], [5,6], [7,8]]
list_of_points2_ = [[15,16], [13,14], [11,12], [9,10]]
# 升维处理
list_of_points1 = np.array([np.array(elem).reshape(1,2) for elem in list_of_points1_])
list_of_points2 = np.array([np.array(elem).reshape(1,2) for elem in list_of_points2_])

graph = tf.Graph()
print(type(graph.as_default()))
#graph.as_default().__enter__()
with graph.as_default(): # Enter a context
    point1 = tf.placeholder(tf.float32, shape=(1, 2))
    point2 = tf.placeholder(tf.float32, shape=(1, 2))
    
    def eu_dist(point1, point2):
        # point1 - point2
        diff = tf.subtract(point1, point2)
        power2 = tf.pow(diff, tf.constant(2.0, shape=(1, 2)))
        add = tf.reduce_sum(power2)
        return tf.sqrt(add)
    
    dist = eu_dist(point1, point2)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    for ele_pair in zip(list_of_points1, list_of_points2):
        distance = sess.run(point1, feed_dict={point1:ele_pair[0], point2:ele_pair[1]})
        print('Distance between {} and {} is {}'.format(ele_pair[0], ele_pair[1], np.squeeze(distance)))
