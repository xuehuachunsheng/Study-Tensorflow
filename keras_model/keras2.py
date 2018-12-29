import tensorflow as tf
print(tf.VERSION)
from tensorflow import keras
print(keras.__version__)
import numpy as np
inputs = keras.Input(shape=(32,))
x = keras.layers.Dense(64, activation=keras.activations.relu)(inputs)
x = keras.layers.Dense(64, activation=keras.activations.relu)(x)
predictions = keras.layers.Dense(10, activation=keras.activations.softmax)(x)
model = keras.Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((1000, 32))
val_labels = np.random.random((1000, 10))


model.fit(x=data,
          y=labels,
          batch_size=32,
          epochs=5,validation_data=(val_data, val_labels))

