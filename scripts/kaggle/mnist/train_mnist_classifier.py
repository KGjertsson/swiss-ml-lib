from pathlib import Path

from keras.layers import Conv2D, Input, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

if __name__ == '__main__':
    # Load the data
    root_data_dir = Path('../../../data/mnist/')
    train_raw = pd.read_csv(root_data_dir / "train.csv")
    test_raw = pd.read_csv(root_data_dir / "test.csv")
    epochs = 50
    validation_split = 0.7
    batch_size = 64
    dropout_factor = 0.3

    train_targets = train_raw["label"]

    # Drop 'label' column
    train_images = train_raw.drop(labels=["label"], axis=1)

    # reshape
    train_images = train_images.values.reshape(-1, 28, 28, 1)
    test_images = test_raw.values.reshape(-1, 28, 28, 1)

    # train_images = np.stack([train_images, train_images, train_images], -1)
    # train_images = np.squeeze(train_images, -2)

    # normalize
    train_images = train_images / 255
    test_images = test_images / 255

    train_targets = to_categorical(train_targets, num_classes=10)

    random_seed = 2
    train_images, val_images, train_targets, val_targets = \
        train_test_split(train_images, train_targets,
                         test_size=0.1, random_state=random_seed)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)

    datagen.fit(train_images)

    input_layer = Input(shape=train_images.shape[1:])

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = Dropout(dropout_factor)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(dropout_factor)(x)
    x = MaxPool2D()(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(dropout_factor)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(dropout_factor)(x)
    x = MaxPool2D()(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(dropout_factor)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(dropout_factor)(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)

    x = Dense(1024, activation="relu")(x)
    x = Dropout(dropout_factor)(x)

    x = Dense(1024, activation="relu")(x)
    x = Dropout(dropout_factor)(x)

    predictions = Dense(10, activation="softmax")(x)
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    model.fit_generator(
        datagen.flow(train_images, train_targets, batch_size=batch_size),
        validation_data=(val_images, val_targets), epochs=epochs,
        steps_per_epoch=train_targets.shape[0] // batch_size,
        callbacks=[learning_rate_reduction], verbose=1)

    # predict results
    results = model.predict(test_images)

    # select the indix with the maximum probability
    results = np.argmax(results, axis=1)

    results = pd.Series(results, name="Label")

    submission = pd.concat(
        [pd.Series(range(1, 28001), name="ImageId"), results], axis=1)

    submission.to_csv("cnn_mnist_datagen.csv", index=False)
