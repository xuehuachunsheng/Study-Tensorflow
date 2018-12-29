# coding=utf-8
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils

mnist = keras.datasets.mnist

class MyLayer(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        super(MyLayer, self).build(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=[input_shape[1], self.output_dim],
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=[self.output_dim,],
                                    initializer='zeros',
                                    trainable=True)

    def call(self, inputs):
        x = tf.matmul(inputs, self.kernel)
        return tf.nn.bias_add(x, self.bias)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = keras.Sequential([
    keras.layers.Flatten(),
    MyLayer(512),
    keras.layers.Activation('relu'),
    MyLayer(10),
    keras.layers.Activation('softmax')
])
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# The compile step specifies the training configuration
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(x_train, np_utils.to_categorical(y_train), batch_size=32, epochs=1)

test_loss, test_acc = model.evaluate(x_test, np_utils.to_categorical(y_test), steps=1)

print(test_loss, test_acc)