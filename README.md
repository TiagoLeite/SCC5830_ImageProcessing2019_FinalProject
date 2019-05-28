# Denoising Dirty Documents
### Tiago de Miranda Leite, NUSP: 7595289

#### Abstract

Old printed documents often have imperfections due to the passage of time or poor care. Thus, yellow or crumpled pages, faded letters and symbols, spots of liquids and fungi, among others, correspond to noise, which makes it hard to properly manipulate and use the digitized versions of those documents. The proposal of this project is to apply and compare some image restoration techniques, such as noise reduction filters and, as a deep learning approach, convolutional autoencoders, for the restoration of such documents. The dataset that will be used contains images that have been produced in a synthetic way, with noise added in order to simulate the mentioned imperfections. This project aims to contribute to the cleaning of ancient documents after they have been digitized, allowing their use in digital readers and facilitating the later application of techniques such as optical character recognition (OCR).

#### Dataset description
The dataset to be utilized is available in a Kaggle's contenst (https://www.kaggle.com/c/denoising-dirty-documents). It's composed of two image datasets: one for training, which contains 144 png images with sizes 540x258 or 540x420, and another for testing, with 72 images and having the same format and sizes as those ones in training dataset. These images feature a variety of text styles, to which synthetic noise has been added to simulate the imperfections present in ancient real-world documents. 
The training dataset also has a subset containing the clean (no noise) versions of the each training image, in order to allow  the use of supervised machine learning algorithms for cleaning those images.

Examples of noisy image and their clean version:

<img src="/data/train/59.png?raw=true" width="270" height="129" align='top'> <img src="/data/train_cleaned/59.png?raw=true" width="270" height="129" align='top'>

<img src="/data/train/146.png?raw=true" width="270" height="210" align='top'> <img src="/data/train_cleaned/146.png?raw=true" width="270" height="210" align='top'>

#### Steps

##### 1 Image enhancement

Image enhancement techniques, such as adaptive threshold and median filter, will initially be employed for noise removal and segmentation. The main idea is to segment the letters from the noisy background.

##### 2 Convolutional Autoencoder

Techniques based on deep learning, in particular convolutional autoencoder, will also be used through supervised learning, since the dataset contains the clean versions of each training image. With noisy images being fed as input, the main ideia is to train the autoencoder to recover their noise free version, by providing as label the respective clean version of each image, in a supervised machine learning approach.

##### 3 Comparing methods

Finally, for the purpose of comparison between those methods, we'll calculate the root mean squared error between the noisy images and their respective clean version that was provided by each  method, in order to establish which one will produce the best image. In addition, each method will be used to clean the images from the provided test set, whose results will be submitted to the Kaggle's correction system in order to figure out which method will get the best result in the test set. As a last step, an ensemble model including all the used models can be figured out, in order to check if it woould improve the score on Kaggle's leaderboard.
