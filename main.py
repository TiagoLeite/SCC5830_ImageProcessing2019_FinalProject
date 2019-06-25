# Tiago de Miranda Leite, 7595289

import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.layers import *
from keras.models import Model
from keras.models import load_model
from PIL import Image
import glob
import pandas as pd

# image batch size for training autoencoder
BATCH_SIZE = 4

# getting image files as lists
train_images = glob.glob('data/train/*.png')
target_images = glob.glob('data/train_cleaned/*.png')
test_images = glob.glob('data/test/*.png')


# filtering operation through the image:
def filter(image, filtering_function, filter_shape):
    padding_x = filter_shape[0] // 2
    padding_y = filter_shape[1] // 2
    image_shape = np.shape(image)

    # fills the boarders of the images with zeros
    image = np.pad(image, (padding_x, padding_y), 'constant', constant_values=(0, 0))

    # Applies the median filter through the image
    filtered = np.asarray([[filtering_function(image[i:i + filter_shape[0], j:j + filter_shape[1]])
                            for j in range(np.shape(image)[1] - 2 * padding_y)]
                           for i in range(np.shape(image)[0] - 2 * padding_x)])

    return np.reshape(filtered, newshape=image_shape)


# applies the suggested denoising approach (filtering and mask)
def denoise_image_filtering(input_image, filtering_function, filter_shape):
    img_shape = np.shape(input_image)
    input_image = np.reshape(input_image, newshape=[img_shape[0], img_shape[1]])

    # estimates the background color by a median filter
    background = filter(input_image, filtering_function, filter_shape=filter_shape)

    # calculates the foreground mask as being everything that is significantly darker than the background,
    # using a margin of 0.15
    mask = input_image < background - 0.15

    # return the input value for each pixel in the mask or white otherwise
    return np.where(mask, input_image, 1.0), background


# loads list of images, pads and returns them grouped by format, as well as their ids
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


# ensemble the models into a submission csv file
def ensemble_models(list_csv_files):
    dfs = [pd.read_csv(file) for file in list_csv_files]
    ids = dfs[0]['id']

    values = [data['value'] for data in dfs]

    mean_values = np.mean(values, axis=0)

    dataframe = pd.DataFrame(data={'id': ids, 'value': mean_values})
    dataframe.to_csv('sub_ensemble.csv', index=False)
    print('CSV file saved!')


# creates the autoencoder
def create_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(input_img)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    # bottleneck layer:
    encoder = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),
                     activation='relu', padding='same', name='encoder')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    # final(output) layer
    decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input=input_img, output=decoder)
    # using rmsprop optimizer and binary crossentropy as loss function
    autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return autoencoder


# loads autoencoder from file
def load_autoencoder(filename):
    return load_model(filename)


# train and validation split for evaluating model
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

    # train the autoencoder during 100 epochs for each set of images:

    autoencoder.fit(x_train_2, y_train_2,
                    batch_size=BATCH_SIZE,
                    epochs=100)

    autoencoder.fit(x_train_1, y_train_1,
                    batch_size=BATCH_SIZE,
                    epochs=100)
    # saving to file
    autoencoder.save('autoencoder.h5')


# generates submission encoding as required by Kaggle
def encode_images_for_submission(list_images, list_images_ids):
    lines = list()
    for index, image in enumerate(list_images):
        image_shape = np.shape(image)
        for i in range(image_shape[1]):
            for j in range(image_shape[0]):
                lines.append(str(str(list_images_ids[index]) + '_' + str(j + 1) +
                                 '_' + str(i + 1) + ',' + str(image[j][i][0]) + '\n'))
    return lines


# loads autoencoder from file and runs inference for test image set
def run_autoencoder():
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

    # saving results to submission file
    file = open("submission_ae.csv", "w")
    file.write('id,value\n')
    file.writelines(rows)
    file.close()


# performs denoising by filtering, for all test image set
def filter_denoising(filtering_function, submission_filename):
    image_test_1, image_test_2, ids1, ids2 = load_image(test_images, use_padding=False)

    pred_images = list()
    for test_image in image_test_1:
        pred, _ = denoise_image_filtering(test_image, filtering_function, filter_shape=[11, 11])
        img_shape = np.shape(pred)
        pred = np.reshape(pred, newshape=[img_shape[0], img_shape[1], 1])
        pred_images.append(pred)

    rows = encode_images_for_submission(pred_images, ids1)

    pred_images = list()
    for test_image in image_test_2:
        pred, _ = denoise_image_filtering(test_image, filtering_function, filter_shape=[11, 11])
        img_shape = np.shape(pred)
        pred = np.reshape(pred, newshape=[img_shape[0], img_shape[1], 1])
        pred_images.append(pred)

    rows += encode_images_for_submission(pred_images, ids2)

    file = open(submission_filename, "w")
    file.write('id,value\n')
    file.writelines(rows)
    file.close()


# runs each proposed method on one image, showing the results
def run_test_on_image(image_file):
    test_image = Image.open(image_file)
    test_image = np.array(test_image) / 255.0

    fig = plt.figure(figsize=(10, 10))

    # shows original test image
    fig.add_subplot(2, 3, 5)
    plt.title('Test: ' + image_file)
    plt.imshow(test_image, cmap='gray')

    autoencoder = load_model(filepath='autoencoder.h5')
    img_original_shape = np.shape(test_image)
    # pad the image according to its shape
    if img_original_shape == (420, 540):
        test_image = np.pad(test_image, [(2, 2), (2, 2)], 'constant', constant_values=(0, 0))
    else:
        test_image = np.pad(test_image, [(3, 3), (2, 2)], 'constant', constant_values=(0, 0))

    img_shape = np.shape(test_image)
    test_image = np.reshape(test_image, newshape=[1, img_shape[0], img_shape[1], 1])
    # run autoencoder prediction
    pred = autoencoder.predict(test_image, verbose=0)[0]
    img_shape = np.shape(pred)
    pred = np.reshape(pred, newshape=[img_shape[0], img_shape[1]])
    # unpad the image, back to its initial shape
    if img_original_shape == (420, 540):
        pred = pred[2:-2, 2:-2]
    else:
        pred = pred[3:-3, 2:-2]

    # shows autoencoder image result
    fig.add_subplot(2, 3, 1)
    plt.title('AE')
    plt.imshow(pred, cmap='gray')

    test_image = Image.open(image_file)
    test_image = np.array(test_image) / 255.0

    # filtering and masking with median filter
    pred2, _ = denoise_image_filtering(test_image, np.median, filter_shape=[11, 11])
    fig.add_subplot(2, 3, 2)
    plt.title('Median')
    plt.imshow(pred2, cmap='gray')

    # filtering and masking with mean filter
    pred3, _ = denoise_image_filtering(test_image, np.mean, filter_shape=[11, 11])
    fig.add_subplot(2, 3, 3)
    plt.title('Mean')
    plt.imshow(pred3, cmap='gray')

    # ensembling all previous results
    ens = (pred + pred2 + pred3) / 3
    fig.add_subplot(2, 3, 4)
    plt.title('Ensemble')
    plt.imshow(ens, cmap='gray')

    plt.show()


def main():
    # Uncomment the next line if you want to train the model (it will take some time), otherwise a previously trained model will be loaded
    # train_autoencoder()

    # Run all the methods, generating submission file (it will take time!)
    # run_autoencoder()
    # filter_denoising(np.median, 'sub_median.csv')
    # filter_denoising(np.mean, 'sub_mean.csv')
    # ensemble_models(['sub_median.csv', 'sub_mean.csv', 'submission_ae.csv'])

    # run all methods on one image and shows results:
    run_test_on_image('data/test/94.png')


if __name__ == '__main__':
    main()
