import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_train(path):
    labels = pd.read_csv(path + 'labels.csv')
    train_data_gen = ImageDataGenerator(
        rescale = 1. / 255,
        vertical_flip = False,
        validation_split = 0.2
    )

    train_gen_flow = train_data_gen.flow_from_dataframe(
        dataframe = labels,
        directory = path + 'final_files/',
        x_col = 'file_name',
        y_col = 'real_age',
        target_size = (224, 224),
        batch_size = 32,
        class_mode = 'raw',
        subset = 'training',
        seed = 12345
    )

    return train_gen_flow

def load_test(path):
    labels = pd.read_csv(path + 'labels.csv')
    test_datagen = ImageDataGenerator(
        rescale = 1. / 255, 
        validation_split = 0.2
    )

    test_gen_flow = test_datagen.flow_from_dataframe(
        dataframe=labels,
        directory=path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=12345)

    return test_gen_flow

def create_model(input_shape):

    optimizer = Adam(learning_rate = 0.0001)
    backbone = ResNet50(
        input_shape = input_shape, 
        weights = 'imagenet', 
        include_top = False
    )
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation = 'relu'))
    model.compile(
        loss = 'mean_squared_error',
        optimizer = optimizer,
        metrics = ['mae']
    )

    return model


def train_model(
    model, 
    train_gen_flow,
    test_gen_flow,
    batch_size = None, 
    epochs = 20,
    steps_per_epoch = None,
    validation_steps = None
):
    if steps_per_epoch is None:
        steps_per_epoch = len(train_gen_flow)
    
    if validation_steps is None:
        validation_steps = len(test_gen_flow)

    model.fit(
        train_gen_flow,
        validation_data = test_gen_flow,
        batch_size = batch_size,
        epochs = epochs,
        validation_steps = validation_steps,
        verbose = 2,
        shuffle = True
    )

    return model 