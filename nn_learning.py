# -*- coding: utf-8 -*-
from pickletools import uint8
import numpy
import tensorflow
from tensorflow import keras as keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)
print("x_train.size=",x_train.size)
print("y_train.size=",y_train.size)
print("x_valid.size=",x_valid.size)
print("y_valid.size=",y_valid.size)
print("x_test.size=",x_test.size)
print("y_test.size=",y_test.size)

# 前処理
x_train = x_train.reshape(y_train.size, 784)
x_valid = x_valid.reshape(y_valid.size, 784)
x_test = x_test.reshape(y_test.size, 784)

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_valid /= 255.0
x_test /= 255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_valid = keras.utils.to_categorical(y_valid, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print("x_train.shape=",x_train.shape)
print("x_valid.shape=",x_valid.shape)
print("x_train.shape=",x_test.shape)


# モデル構築
model = Sequential()
model.add(InputLayer(input_shape=(784,)))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# 学習
epochs = 20
batch_size = 128
history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=True,
                    validation_data=(x_valid, y_valid))


# 検証
score = model.evaluate(x_test, y_test, verbose=1)
print()
print('Test loss:', score[0])
print('Test accuracy:', score[1])
