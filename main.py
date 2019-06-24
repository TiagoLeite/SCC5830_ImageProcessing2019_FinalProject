import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.layers import *
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping
from PIL import Image
import glob
from scipy import ndimage

BATCH_SIZE = 4

train_images = glob.glob('data/train/*.png')
target_images = glob.glob('data/train_cleaned/*.png')
test_images = glob.glob('data/test/*.png')


def interquartile_range(array, axis):
    q_75, q_50, q_25 = np.percentile(array, [75, 50, 25], interpolation='midpoint')
    iqr = q_75 - q_25
    return iqr, q_50


def adap_denoising(image, gamma, mode, kernel_size):
    assert mode in ['average', 'robust']

    image_w = np.shape(image)[0]
    image_h = np.shape(image)[1]

    if mode == 'average':
        centr_measure = np.mean
        disp_measure = np.std
        disp_n = disp_measure(image[:image_w // 6, :image_h // 6], axis=None)
    else:
        centr_measure = np.median
        disp_measure = interquartile_range
        disp_n = disp_measure(image[:image_w // 6, :image_h // 6], axis=None)[0]

    disp_n = 1 if disp_n == 0 else disp_n

    new_image = np.copy(image)

    # TODO: pad the image
    # Main loop for applying the filter through the image:
    for i in range(kernel_size // 2, image_w - kernel_size // 2):

        for j in range(kernel_size // 2, image_h - kernel_size // 2):

            if mode == 'robust':

                sliced_image = image[i - kernel_size // 2: i + kernel_size // 2 + 1,
                               j - kernel_size // 2: j + kernel_size // 2 + 1]

                disp_l, centr_l = disp_measure(sliced_image, axis=None)

                disp_l = disp_n if disp_l == 0 else disp_l

            else:

                sliced_image = image[i - kernel_size // 2: i + kernel_size // 2 + 1,
                               j - kernel_size // 2: j + kernel_size // 2 + 1]

                disp_l = disp_measure(sliced_image, axis=None)

                disp_l = disp_n if disp_l == 0 else disp_l

                centr_l = centr_measure(sliced_image, axis=None)

            new_image[i][j] = image[i][j] - gamma * (disp_n / disp_l) * (image[i][j] - centr_l)

    mask = image < new_image - 0.003
    # return the input value for all pixels in the mask or pure white otherwise
    return np.where(mask, image, 1.0)
    # return new_image


def median_filter(image, filter_shape):
    padding_x = filter_shape[0] // 2
    padding_y = filter_shape[1] // 2
    image_shape = np.shape(image)
    # Fills the boarders of the images with zeros
    image = np.pad(image, (padding_x, padding_y), 'constant', constant_values=(0, 0))

    # Applies the median filter through the image
    bg = np.asarray([[np.median(image[i:i + filter_shape[0], j:j + filter_shape[1]])
                      for j in range(np.shape(image)[1] - 2 * padding_y)]
                     for i in range(np.shape(image)[0] - 2 * padding_x)])

    # image = image[padding_x:-padding_x, padding_y:-padding_y]

    # mask = image < bg - 0.1
    # return the input value for all pixels in the mask or pure white otherwise
    # return np.where(mask, image, 1.0)

    return np.reshape(255 * bg, newshape=image_shape).astype(np.int)


def denoise_image(inp):
    # estimate 'background' color by a median filter
    bg = ndimage.median_filter(inp, size=[5, 5])
    # compute 'foreground' mask as anything that is significantly darker than
    # the background
    mask = inp < bg - 0.1
    # return the input value for all pixels in the mask or pure white otherwise
    return np.where(mask, inp, 1.0)


def load_image(path):
    images_1 = []
    images_2 = []
    images_ids_1 = []
    images_ids_2 = []
    for index, fig in enumerate(path):
        img = image.load_img(fig, color_mode='grayscale')
        x = image.img_to_array(img).astype('float32') / 255.0
        x_shape = np.shape(x)
        if x_shape == (420, 540, 1):
            images_1.append(x)
            images_ids_1.append((fig.split('/')[-1]).split('.')[0])
        else:
            images_2.append(x)
            images_ids_2.append((fig.split('/')[-1]).split('.')[0])

    return np.asarray(images_1), np.asarray(images_2), images_ids_1, images_ids_2


def normalize(image, normalize_min=0, normalize_max=255):
    min = np.min(image)
    max = np.max(image)
    image = (image - min) * ((normalize_max - normalize_min) / (max - min)) + normalize_min
    image = image.astype(np.uint8)
    return image


def create_autoencoder_1(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(input_img)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    encoder = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),
                     activation='relu', padding='same', name='encoder')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    # x = Cropping2D(cropping=(3, 2))(x)
    x = Cropping2D(cropping=(2, 2))(x)

    decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input=input_img, output=decoder)
    autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return autoencoder


def create_autoencoder_2(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(input_img)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    encoder = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),
                     activation='relu', padding='same', name='encoder')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Cropping2D(cropping=(3, 2))(x)
    # x = Cropping2D(cropping=(2, 2))(x)

    decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input=input_img, output=decoder)
    autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
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

    x_train_1, x_train_2, _, _ = load_image(train_images)
    y_train_1, y_train_2, _, _ = load_image(target_images)

    # x_test = load_image(test_images)

    print(np.shape(x_train_1))
    print(np.shape(y_train_1))

    autoencoder_1 = create_autoencoder_1(input_shape=(None, None, 1))
    autoencoder_2 = create_autoencoder_2(input_shape=(None, None, 1))

    print(autoencoder_1.summary())

    x_train_1, y_train_1, x_val_1, y_val_1 = split_train_val(x_train_1, y_train_1)
    x_train_2, y_train_2, x_val_2, y_val_2 = split_train_val(x_train_2, y_train_2)

    autoencoder_1.fit(x_train_1, y_train_1,
                      batch_size=BATCH_SIZE,
                      epochs=100,
                      validation_data=(x_val_1, y_val_1))
    autoencoder_1.save('autoencoder_1.h5')

    autoencoder_2.fit(x_train_2, y_train_2,
                      batch_size=BATCH_SIZE,
                      epochs=100,
                      validation_data=(x_val_2, y_val_2))
    autoencoder_2.save('autoencoder_2.h5')


def encode_images_for_submission(list_images, list_images_ids):
    lines = list()
    for index, image in enumerate(list_images):
        image_shape = np.shape(image)
        for i in range(image_shape[1]):
            for j in range(image_shape[0]):
                # print('>>', image[j][i][0])
                lines.append(str(str(list_images_ids[index]) + '_' + str(j + 1) +
                             '_' + str(i + 1) + ',' + str(image[j][i][0])))
    return lines


def evaluate_autoencoder():
    # train_autoencoder()
    autoencoder_1 = load_model(filepath='autoencoder_1.h5')
    autoencoder_2 = load_model(filepath='autoencoder_2.h5')
    # print(autoencoder_1.summary())

    image_test_1, image_test_2, ids1, ids2 = load_image(test_images)

    pred_images = list()
    for test_image in image_test_1:
        img_shape = np.shape(test_image)
        test_image = np.array(test_image)
        test_image = np.reshape(test_image, newshape=[1, img_shape[0], img_shape[1], 1])
        pred = autoencoder_1.predict(test_image, verbose=0)[0]
        pred_images.append(pred)

    print(np.shape(pred_images))
    sub = encode_images_for_submission(pred_images, ids1)
    for line in sub:
        print(line)

    pred_images = list()
    for test_image in image_test_2:
        img_shape = np.shape(test_image)
        test_image = np.array(test_image)
        test_image = np.reshape(test_image, newshape=[1, img_shape[0], img_shape[1], 1])
        pred = autoencoder_2.predict(test_image, verbose=0)[0]
        pred_images.append(pred)

    sub = encode_images_for_submission(pred_images, ids2)
    for line in sub:
        print(line)


    '''
    encode = encode_images_for_submission(pred, 154)

    print(encode[0])
    print(encode[1])
    print(encode[-2])
    print(encode[-1])

    input()

    pred = np.reshape(pred, newshape=[img_shape[0], img_shape[1]])
    pred = normalize(pred)

    plt.figure(figsize=(10, 10))
    plt.subplot(122)
    plt.title('Clean image')
    plt.imshow(pred, cmap='gray')

    test_image = np.reshape(test_image, newshape=[img_shape[0], img_shape[1]])
    plt.subplot(121)
    plt.title('Noisy image')
    plt.imshow(test_image, cmap='gray')
    plt.show()'''


def main():
    # Uncomment the next line if you want to train the model (it will take some time), otherwise a previously trained model will be loaded
    # train_autoencoder()
    evaluate_autoencoder()

    test_image = Image.open('data/test/205.png')  # unknown image
    test_image = np.array(test_image) / 255.0

    # pred = median_filter(test_image, [5, 5])

    '''pred = adap_denoising(test_image, gamma=.1, mode='robust',
                          kernel_size=13)

    plt.figure(figsize=(10, 10))
    plt.subplot(122)
    plt.title('Clean image')
    plt.imshow(pred, cmap='gray')

    plt.subplot(121)
    plt.title('Noisy image')
    plt.imshow(test_image, cmap='gray')
    plt.show()'''


if __name__ == '__main__':
    main()
