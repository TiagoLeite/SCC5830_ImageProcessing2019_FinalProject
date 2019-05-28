import keras
import numpy as np
from matplotlib import pyplot as plt
import imageio
import cv2
from scipy import signal
from keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 4


def create_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
    x = MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    encoder = MaxPool2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    decoder = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input=input_img, output=decoder)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder


def main():
    train_datagen = ImageDataGenerator(preprocessing_function=None,
                                       rescale=1.0/255.0,
                                       rotation_range=180,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       validation_split=0.2)

    train_gen = train_datagen.flow_from_directory('data/train/',
                                                  target_size=(256, 536),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='input',
                                                  subset='training')

    val_gen = train_datagen.flow_from_directory('data/train/',
                                                target_size=(256, 536),
                                                batch_size=BATCH_SIZE,
                                                class_mode='input',
                                                subset='validation')

    autoencoder = create_autoencoder(input_shape=(256, 536, 3))

    print(autoencoder.summary())

    autoencoder.fit_generator(train_gen,
                              steps_per_epoch=train_gen.samples//BATCH_SIZE,
                              validation_data=val_gen,
                              validation_steps=val_gen.samples//BATCH_SIZE,
                              epochs=100,
                              verbose=1)

    autoencoder.save('autoencoder.h5')


if __name__ == '__main__':
    main()
