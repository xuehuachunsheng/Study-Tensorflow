# coding=utf-8

import tensorflow as tf

from tensorflow import keras
from keras.utils import np_utils
import numpy as np
class MyModel(keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        self.flatten_1 = keras.layers.Flatten()
        self.dense_1 = keras.layers.Dense(512, activation='relu')
        self.dense_2 = keras.layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        x = self.flatten_1(inputs)
        x = self.dense_1(x)
        return self.dense_2(x)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = MyModel(num_classes=10)

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(x_train, np_utils.to_categorical(y_train), batch_size=32, epochs=1)

test_loss, test_acc = model.evaluate(x_test, np_utils.to_categorical(y_test), steps=1)

print(test_loss, test_acc)