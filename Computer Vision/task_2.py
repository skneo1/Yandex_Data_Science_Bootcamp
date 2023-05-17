from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import(
    Conv2D,
    AveragePooling2D,
    Dense,
    Flatten
)
from tensorflow.keras.optimizers import Adam
import numpy as np



def load_train(path):
    X_train = np.load(path + 'train_features.npy')
    y_train = np.load(path + 'train_target.npy')
    X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
    return X_train, y_train


def create_model(input_shape):

    model = Sequential()
    optimizer = Adam(learning_rate = 0.001)
    model.add(
        Conv2D(
            10,
            kernel_size = (5, 5),
            padding = 'same',
            activation = 'relu',
            input_shape = input_shape
        )
    )
    model.add(
        AveragePooling2D(pool_size = (2, 2))
    )

    model.add(
        Conv2D(
            15,
            kernel_size = (5, 5),
            padding = 'same',
            activation = 'relu'
        )
    )
    model.add(
        AveragePooling2D(pool_size = (2, 2))
    )

    model.add(
        Conv2D(
            25, 
            kernel_size = (5, 5),
            padding = 'valid',
            activation = 'relu'
        )
    )
    model.add(
        AveragePooling2D(pool_size = (2, 2))
    )

    model.add(Flatten())

    model.add(
        Dense(500, activation = 'relu')
    )
    model.add(
        Dense(250, activation = 'relu')
    )
    model.add(Dense(10, activation = 'softmax'))

    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = optimizer,
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

