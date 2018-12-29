
import tensorflow as tf

# Create some variables.
v1 = tf.Variable([1], name="v1")
v2 = tf.Variable([2], name="v2")

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:

  # Do some work with the model.
  sess.run(init_op)

  saver.restore(sess, "/tmp/model.ckpt")

  print(sess.run(v1))

  # Save the variables to disk.
  #save_path = saver.save(sess, "/tmp/model.ckpt")
