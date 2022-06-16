# dataset: https://github.com/zalandoresearch/fashion-mnist

import tensorflow as tf
from tensorflow import keras
import numpy as np

from mnist import MNIST
import matplotlib.pyplot as plt

from visualize import plot


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print('we are in the Callback func.')
        if logs.get('sparse_categorical_accuracy') > 0.8:
            self.model.stop_training = True
        elif logs.get('loss') < 0.7:
            print('we are in the Callback func.')
            self.model.stop_training = True


data = MNIST("./data/")
train_img, train_lbl = data.load_training()
test_img, test_lbl = data.load_testing()


train_img = np.array(train_img).reshape(60000, 28, 28, 1)
train_lbl = np.array(train_lbl)

test_img = np.array(test_img).reshape(10000, 28, 28, 1)
test_lbl = np.array(test_lbl)

train_img = train_img / 255.0
test_img = test_img / 255.0


model = keras.Sequential([
    keras.layers.Conv2D(10, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Flatten(),
    # keras.layers.Dense(328, activation=tf.nn.relu),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)    
])

mcb = MyCallback()

model.compile(optimizer = keras.optimizers.SGD(), # Adam()
                loss = keras.losses.SparseCategoricalCrossentropy(),
                metrics= [keras.metrics.SparseCategoricalAccuracy()])

model.fit(train_img, train_lbl, batch_size=200, epochs=5, callbacks=[mcb])                

print('-'*12, 'prediction', '-'*12)
train_pre = model.predict(train_img)
print('model prediction: \n', train_pre[:2])
train_pre = np.argmax(train_pre, axis=1)
print('model prediction-->label: \n', train_pre[:2])

test_pre = model.predict(test_img)
test_pre = np.argmax(test_pre, axis=1)

print('-'*12, 'accuraciy', '-'*12)
print('train accuracy:', sum(train_lbl==train_pre)/len(train_lbl))
print('test accuracy:', sum(test_lbl==test_pre)/len(test_lbl))

plot(train_img, train_lbl, train_pre)
plot(test_img, test_lbl, test_pre)
