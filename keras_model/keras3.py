import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
import numpy as np

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255., x_test / 255.

dataset = tf.data.Dataset.from_tensor_slices((x_train, np_utils.to_categorical(y_train)))
dataset = dataset.batch(32).shuffle(buffer_size=300).repeat()

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, np_utils.to_categorical(y_test)))
test_dataset = test_dataset.batch(32).shuffle(buffer_size=300).repeat()


def get_model():
    inputs = keras.Input(shape=(28, 28), dtype=tf.float32)
    x = keras.layers.Flatten(data_format='channels_last')(inputs)
    x = keras.layers.Dense(512, activation=keras.activations.relu, name='dense_1')(x)
    x = keras.layers.Dropout(rate=0.2)(x)
    predictions = keras.layers.Dense(10, activation=keras.activations.softmax, name='dense_3')(x)
    model = keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])
    return model


model = get_model()
model.fit(dataset,
          epochs=1,
          steps_per_epoch=int(np.shape(x_train)[0] / 32))
test_loss, test_acc = model.evaluate(test_dataset, steps=int(np.shape(x_test)[0] / 32))
print(test_loss, test_acc)





