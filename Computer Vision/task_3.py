from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense, 
    Conv2D,
    AveragePooling2D,
    Flatten
)


def load_train(path):
    datagen = ImageDataGenerator(
        rescale = 1 / 255.
    )

    flow = datagen.flow_from_directory(
        path,
        target_size = (150, 150),
        batch_size = 16,
        class_mode = 'sparse',
        seed = 12345
    )

    return flow


def create_model(input_shape):

    model = Sequential()
    optimizer = Adam(learning_rate = 0.001)

    model.add(
        Conv2D(
            15, 
            (5, 5),
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
            30, 
            (3, 3),
            strides = 1,
            activation = 'relu',
            padding = 'same'
        )
    )
    model.add(
        AveragePooling2D(pool_size = (2, 2))
    )

    model.add(
        Conv2D(
            50,
            (3, 3),
            padding = 'same',
            strides = 2,
            activation = 'relu'
        )
    )
    model.add(
        AveragePooling2D(pool_size = (2, 2))
    )

    model.add(Flatten())
    model.add(
        Dense(
            300,
            activation = 'relu'
        )
    )
    model.add(
        Dense(
            10,
            activation = 'softmax'
        )
    )

    model.compile(
        optimizer = optimizer,
        loss='sparse_categorical_crossentropy',
        metrics = ['acc']
    )

    return model


def train_model(
    model,
    train_data,
    test_data,
    batch_size = None,
    epochs = 10,
    steps_per_epoch = None,
    validation_steps = None
):
    model.fit(
        train_data,
        validation_data = test_data,
        epochs = epochs,
        steps_per_epoch = steps_per_epoch,
        batch_size = batch_size,
        validation_steps = validation_steps,
        verbose = 2,
        shuffle = True
    )

    return model 