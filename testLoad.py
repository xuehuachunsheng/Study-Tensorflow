import tensorflow as tf
with tf.Session() as sess:  
    new_saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
    print(new_saver)
