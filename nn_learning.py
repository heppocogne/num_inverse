# -*- coding: utf-8 -*-
import sys
from pickletools import uint8
import numpy
import tensorflow
from tensorflow import keras as keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.optimizers import RMSprop, Adagrad, Adam
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)
    print("x_train.size=",x_train.size)
    print("y_train.size=",y_train.size)
    print("x_valid.size=",x_valid.size)
    print("y_valid.size=",y_valid.size)
    print("x_test.size=",x_test.size)
    print("y_test.size=",y_test.size)

    # 前処理(1)
    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_valid /= 255.0
    x_test /= 255.0

    train_size = y_train.size
    valid_size = y_valid.size
    test_size = y_test.size

    y_train = keras.utils.to_categorical(y_train, 10)
    y_valid = keras.utils.to_categorical(y_valid, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # print("x_train.shape=",x_train.shape)
    # print("x_valid.shape=",x_valid.shape)
    # print("x_train.shape=",x_test.shape)


    # モデル構築
    model = Sequential()
    model_name = ""
    if len(sys.argv)==1 or sys.argv[1]=="0":
        # 前処理(2)
        x_train = x_train.reshape(train_size, 784)
        x_valid = x_valid.reshape(valid_size, 784)
        x_test = x_test.reshape(test_size, 784)

        model.add(InputLayer(input_shape=(784,)))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        model_name = "simple-dense"
    elif sys.argv[1]=="1":
        # 前処理(2)
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_valid = x_valid.reshape(x_valid.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

        model_name = "cnn-and-dense"
    elif sys.argv[1]=="2":
        # 前処理(2)
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_valid = x_valid.reshape(x_valid.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        model.add(Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D())
        model.add(Conv2D(16, (3,3), activation='relu'))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model_name='simple-cnn'


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
    model.save("models/"+model_name)
