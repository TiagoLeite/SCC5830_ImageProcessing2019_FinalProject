import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.layers import *
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping
from PIL import Image
import glob
import pandas as pd
from scipy import ndimage
import csv

BATCH_SIZE = 4

train_images = glob.glob('data/train/*.png')
target_images = glob.glob('data/train_cleaned/*.png')
test_images = glob.glob('data/test/*.png')


def ensemble_models(list_csv_files):

    dfs = [pd.read_csv(file) for file in list_csv_files]
    ids = dfs[0]['id']

    values = [data['value'] for data in dfs]

    print(np.shape(dfs[0]['value']))

    mean_values = np.mean(values, axis=0)

    dataframe = pd.DataFrame(data={'id': ids, 'value': mean_values})
    dataframe.to_csv('sub_ensemble.csv', index=False)
    print('CSV file saved!')


def filter(image, filtering_function, filter_shape):
    padding_x = filter_shape[0] // 2
    padding_y = filter_shape[1] // 2
    image_shape = np.shape(image)
    # Fills the boarders of the images with zeros
    image = np.pad(image, (padding_x, padding_y), 'constant', constant_values=(0, 0))

    # Applies the median filter through the image
    filtered = np.asarray([[filtering_function(image[i:i + filter_shape[0], j:j + filter_shape[1]])
                            for j in range(np.shape(image)[1] - 2 * padding_y)]
                           for i in range(np.shape(image)[0] - 2 * padding_x)])

    return np.reshape(filtered, newshape=image_shape)


def denoise_image_filtering(input_image, filtering_function, filter_shape):
    img_shape = np.shape(input_image)
    input_image = np.reshape(input_image, newshape=[img_shape[0], img_shape[1]])
    print('input shape:', np.shape(input_image))
    # estimate 'background' color by a median filter
    background = filter(input_image, filtering_function, filter_shape=filter_shape)

    # compute 'foreground' mask as anything that is significantly darker than
    # the background
    mask = input_image < background - 0.1
    # return the input value for all pixels in the mask or pure white otherwise
    return np.where(mask, input_image, 1.0), background


def load_image(path, use_padding=True):
    images_1 = []
    images_2 = []
    images_ids_1 = []
    images_ids_2 = []
    for index, fig in enumerate(path):
        img = image.load_img(fig, color_mode='grayscale')
        x = image.img_to_array(img).astype('float32') / 255.0
        x_shape = np.shape(x)
        if x_shape == (420, 540, 1):
            if use_padding:
                img = np.pad(img, [(2, 2), (2, 2)], 'constant', constant_values=(0, 0))
            x = image.img_to_array(img).astype('float32') / 255.0
            images_1.append(x)
            images_ids_1.append((fig.split('/')[-1]).split('.')[0])
        else:
            if use_padding:
                img = np.pad(img, [(3, 3), (2, 2)], 'constant', constant_values=(0, 0))
            x = image.img_to_array(img).astype('float32') / 255.0
            images_2.append(x)
            images_ids_2.append((fig.split('/')[-1]).split('.')[0])

    return np.asarray(images_1), np.asarray(images_2), images_ids_1, images_ids_2


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
    encoder = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),
                     activation='relu', padding='same', name='encoder')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)

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
    autoencoder = create_autoencoder(input_shape=(None, None, 1))
    print(autoencoder.summary())

    autoencoder.fit(x_train_2, y_train_2,
                    batch_size=BATCH_SIZE,
                    epochs=100)

    autoencoder.fit(x_train_1, y_train_1,
                    batch_size=BATCH_SIZE,
                    epochs=100)

    autoencoder.fit(x_train_2, y_train_2,
                    batch_size=BATCH_SIZE,
                    epochs=40)

    autoencoder.fit(x_train_1, y_train_1,
                    batch_size=BATCH_SIZE,
                    epochs=40)

    autoencoder.save('autoencoder.h5')


def encode_images_for_submission(list_images, list_images_ids):
    lines = list()
    for index, image in enumerate(list_images):
        image_shape = np.shape(image)
        for i in range(image_shape[1]):
            for j in range(image_shape[0]):
                lines.append(str(str(list_images_ids[index]) + '_' + str(j + 1) +
                                 '_' + str(i + 1) + ',' + str(image[j][i][0]) + '\n'))
    return lines


def evaluate_autoencoder():
    autoencoder_1 = load_model(filepath='autoencoder.h5')
    image_test_1, image_test_2, ids1, ids2 = load_image(test_images)
    pred_images = list()

    for test_image in image_test_1:
        img_shape = np.shape(test_image)
        test_image = np.array(test_image)
        test_image = np.reshape(test_image, newshape=[1, img_shape[0], img_shape[1], 1])
        pred = autoencoder_1.predict(test_image, verbose=0)[0]
        pred = pred[2:-2, 2:-2]
        pred_images.append(pred)

    print(np.shape(pred_images))

    rows = encode_images_for_submission(pred_images, ids1)

    pred_images = list()
    for test_image in image_test_2:
        img_shape = np.shape(test_image)
        test_image = np.array(test_image)
        test_image = np.reshape(test_image, newshape=[1, img_shape[0], img_shape[1], 1])
        pred = autoencoder_1.predict(test_image, verbose=0)[0]
        pred = pred[3:-3, 2:-2]
        pred_images.append(pred)

    rows += encode_images_for_submission(pred_images, ids2)

    file = open("submission_ae.csv", "w")
    file.write('id,value\n')
    file.writelines(rows)
    file.close()


def evaluate_filter_denoising(filtering_function, submission_filename):

    count = 0
    image_test_1, image_test_2, ids1, ids2 = load_image(test_images, use_padding=False)

    pred_images = list()
    for test_image in image_test_1:
        print(count)
        count += 1
        pred = denoise_image_filtering(test_image, filtering_function, filter_shape=[5, 5])
        img_shape = np.shape(pred)
        pred = np.reshape(pred, newshape=[img_shape[0], img_shape[1], 1])
        pred_images.append(pred)

    rows = encode_images_for_submission(pred_images, ids1)

    pred_images = list()
    for test_image in image_test_2:
        print(count)
        count += 1
        pred = denoise_image_filtering(test_image, filtering_function, filter_shape=[7, 7])
        img_shape = np.shape(pred)
        pred = np.reshape(pred, newshape=[img_shape[0], img_shape[1], 1])
        pred_images.append(pred)

    rows += encode_images_for_submission(pred_images, ids2)

    file = open(submission_filename, "w")
    file.write('id,value\n')
    file.writelines(rows)
    file.close()



def main():

    # ensemble_models(['sub_mean.csv', 'sub_median.csv', 'submission_ae.csv'])
    # input()

    '''
    # Uncomment the next line if you want to train the model (it will take some time), otherwise a previously trained model will be loaded
    train_autoencoder()

    print('AE filtering:')
    evaluate_autoencoder()

    print('Median filtering:')
    evaluate_filter_denoising(np.median, 'sub_median.csv')

    print('Mean filtering:')
    evaluate_filter_denoising(np.mean, 'sub_mean.csv')'''

    test_image = Image.open('data/test/49.png')  # unknown image
    test_image = np.array(test_image) / 255.0

    plt.subplot(155)
    plt.title('Test Image')
    plt.imshow(test_image, cmap='gray')

    autoencoder = load_model(filepath='autoencoder.h5')
    test_image = np.pad(test_image, [(3, 3), (2, 2)], 'constant', constant_values=(0, 0))
    img_shape = np.shape(test_image)
    test_image = np.reshape(test_image, newshape=[1, img_shape[0], img_shape[1], 1])
    pred = autoencoder.predict(test_image, verbose=0)[0]
    img_shape = np.shape(pred)
    pred = np.reshape(pred, newshape=[img_shape[0], img_shape[1]])
    pred = pred[3:-3, 2:-2]
    img_shape = np.shape(pred)
    test_image = np.reshape(test_image, newshape=[img_shape[0]+6, img_shape[1]+4])

    # plt.figure(figsize=(10, 10))
    plt.subplot(151)
    plt.title('AE')
    plt.imshow(pred, cmap='gray')

    pred2, _ = denoise_image_filtering(test_image, np.median, filter_shape=[11, 11])
    pred2 = pred2[3:-3, 2:-2]
    plt.subplot(152)
    plt.title('Median')
    plt.imshow(pred2, cmap='gray')

    pred3, _ = denoise_image_filtering(test_image, np.mean, filter_shape=[11, 11])
    pred3 = pred3[3:-3, 2:-2]
    plt.subplot(153)
    plt.title('Mean')
    plt.imshow(pred3, cmap='gray')

    ens = (pred + pred2)/2
    plt.subplot(154)
    plt.title('Ensemble')
    plt.imshow(ens, cmap='gray')

    plt.show()


if __name__ == '__main__':
    main()
