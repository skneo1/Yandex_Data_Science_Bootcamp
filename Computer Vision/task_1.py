from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


def load_train(path):
    X_train = np.load(path + 'train_features.npy')
    y_train = np.load(path + 'train_target.npy')
    X_train = X_train.reshape(
        X_train.shape[0], 28 * 28
    ) / 255.
    return X_train, y_train


def create_model(input_shape):
    model = Sequential()
    model.add(
        Dense(units=512, activation='relu', input_shape=input_shape)
    )
    model.add(
        Dense(units=512, activation='relu')
    )
    model.add(
        Dense(10, activation='softmax')
    )
    model.compile(
        optimizer = 'sgd', 
        loss = 'sparse_categorical_crossentropy',
        metrics = ['acc']
    )
    return model

def train_model(
    model, 
    train_data, 
    test_data, 
    batch_size = 32, 
    epochs = 20,
    steps_per_epoch = None, 
    validation_steps = None
):

    X_train, y_train = train_data
    X_test, y_test = test_data

    model.fit(
        X_train, y_train,
        validation_data = (X_test, y_test),
        batch_size = batch_size, 
        epochs = epochs,
        steps_per_epoch = steps_per_epoch,
        validation_steps = validation_steps,
        verbose = 2, shuffle = True
    )
    return model