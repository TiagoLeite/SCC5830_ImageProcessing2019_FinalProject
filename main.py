import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping
from PIL import Image
import glob

BATCH_SIZE = 4

train_images = glob.glob('data/train/*.png')
target_images = glob.glob('data/train_cleaned/*.png')
test_images = glob.glob('data/test/*.png')


def load_image(path):
    all_images = np.zeros((len(path), 256, 544, 1))

    for index, fig in enumerate(path):
        img = image.load_img(fig, color_mode='grayscale', target_size=(256, 544))
        x = image.img_to_array(img).astype('float32') / 255.0
        all_images[index] = x

    return all_images


def normalize(image, normalize_min=0, normalize_max=255):
    min = np.min(image)
    max = np.max(image)
    image = (image - min) * ((normalize_max - normalize_min) / (max - min)) + normalize_min
    image = image.astype(np.uint8)
    return image


def create_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(input_img)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    encoder = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input=input_img, output=decoder)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder


def load_autoencoder(filename):
    return load_model(filename)


def split_train_val(x_train, y_train):
    rand = np.random.RandomState(seed=2019)
    perm = rand.permutation(len(x_train))
    train_idx = perm[:int(0.8 * len(x_train))]
    val_idx = perm[int(0.8 * len(x_train)):]
    return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]


def train_autoencoder():

    x_train = load_image(train_images)
    y_train = load_image(target_images)
    # x_test = load_image(test_images)

    autoencoder = create_autoencoder(input_shape=(256, 544, 1))
    print(autoencoder.summary())

    x_train, y_train, x_val, y_val = split_train_val(x_train, y_train)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=5,
                                   verbose=1,
                                   mode='auto')

    autoencoder.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=100,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping])

    autoencoder.save('autoencoder.h5')


def main():

    # Uncomment the next line if you want to train the model (it will take some time), otherwise a previously trained model will be loaded
    # train_autoencoder()

    autoencoder = load_model(filepath='autoencoder.h5')
    print(autoencoder.summary())

    test_image = Image.open('data/test/58.png')  # unknown image
    test_image = test_image.resize((544, 256))
    test_image = np.array(test_image) / 255.0

    test_image = np.reshape(test_image, newshape=[1, 256, 544, 1])
    pred = autoencoder.predict(test_image, verbose=1)[0]
    pred = np.reshape(pred, newshape=[256, 544])
    pred = normalize(pred)

    plt.figure(figsize=(10, 10))
    plt.subplot(122)
    plt.title('Clean image')
    plt.imshow(pred, cmap='gray')

    test_image = np.reshape(test_image, newshape=[256, 544])
    plt.subplot(121)
    plt.title('Noisy image')
    plt.imshow(test_image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
