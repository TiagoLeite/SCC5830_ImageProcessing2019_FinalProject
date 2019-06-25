# Denoising Dirty Documents
### Tiago de Miranda Leite, NUSP: 7595289

#### Abstract

Old printed documents often have imperfections due to the passage of time or poor care. Thus, yellow or crumpled pages, faded letters and symbols, spots of liquids and fungi, among others, correspond to noise, which makes it hard to properly manipulate and use the digitized versions of those documents. The proposal of this project is to apply and compare some image enhancement techniques, such as threshold filters and, as a deep learning approach, convolutional autoencoders, for the restoration of such documents. The dataset that will be used contains images that have been produced in a synthetic way, with noise added in order to simulate the mentioned imperfections. This project aims to contribute to the cleaning of ancient documents after they have been digitized, allowing their use in digital readers and facilitating the later application of techniques such as optical character recognition (OCR).

#### Dataset description
The dataset to be utilized is available in a Kaggle's contenst (https://www.kaggle.com/c/denoising-dirty-documents/data). It's composed of two image datasets: one for training, which contains 144 png images with sizes 540x258 or 540x420, and another for testing, with 72 images and having the same format and sizes as those ones in training dataset. These images feature a variety of text styles, to which synthetic noise has been added to simulate the imperfections present in ancient real-world documents. 
The training dataset also has a subset containing the clean (no noise) versions of the each training image, in order to allow  the use of supervised machine learning algorithms for cleaning those images.

Examples of noisy images and their clean version:

<img src="/data/train/59.png?raw=true" width="270" height="129" align='top'> <img src="/data/train_cleaned/59.png?raw=true" width="270" height="129" align='top'>

<img src="/data/train/146.png?raw=true" width="270" height="210" align='top'> <img src="/data/train_cleaned/146.png?raw=true" width="270" height="210" align='top'>

#### Steps

#### 1 Image filtering/segmentation

Image filtering/segmentation techniques, such as mean and median filter, will initially be employed for noise removal and segmentation. The main idea is to segment the letters from the noisy background.

#### 2 Convolutional Autoencoder

Techniques based on deep learning, in particular convolutional autoencoder, will also be used through supervised learning, since the dataset contains the clean version of each training image. With noisy images being fed as input, the main ideia is to train the autoencoder to recover their noise free version, by providing as label the respective clean version of each image, in a supervised machine learning approach.

#### 3 Comparing methods

Finally, for the purpose of comparison between those methods, we'll first calculate the root mean squared error between the clean images provided by each method and their truly clean version provided by the training dataset, in order to establish which technique will produce the best images. In addition, each method will be used to clean the images from the provided test set, whose results will be submitted to the Kaggle's correction system in order to figure out which method will get the best result in the test set. 

#### 4 Ensembled model

As a last step, an ensembled model including all the used models will be implemented, in order to check if it would improve the score on Kaggle's leaderboard.


### Image Filtering
The first idea was to estimate the background color of the image by filteing it using mean and median filter. Since we have the estimated background color, and considering the fact that the letters are usually darker than the background, we can compute a foreground mask as everything that is significantly darker than the background. The following image shows the mean and filter and the result after applying the suggested idea:
<img src="/sample_images/3_images_mean.png?raw=true">

Using the median filter, the results look slightly better:
<img src="/sample_images/3_images_median.png?raw=true">

For both previous cases, a 5x5 sized filter were used. We noticed that changing the filter size didn't improve the results significantly.

### Convolutional Autoencoder
An Autoencoder is a type of neural network structured in such a way that it aims to create an internal and more compact representation of the input data. It has two components: an encoder followed by a decoder. The encoder aims to create a new rerpesentation of the data in a smaller space whereas the decoder reconstructs the input data to its original format, from the received encoding. 

A Convolutional Autoencoder uses convolutional layers in order to extract features from the input data. For this work, a convolutional autoencoer was implemented, with 3 convolutional layers for the encoder and other 3 convolutional layers for the decoder. More implementation details can be seen in the source code.
The following image shows a noisy image and the clean version obtained by using the suggested autoencoder. One can notice that this technique shows even better results than the previus methods:
<img src="/sample_images/ae.png?raw=true">

### Ensemble 
In order to verify if combining all strategies would improve the quality of the denoised image, all the three techniques were ensembled. After getting the output image from each technique individually, the mean image was computed, pixel by pixel. The following figure shows the result for an image from the test dataset, which seems to look slightly better than the isolated results of each method.

<img src="/sample_images/all_models.png?raw=true" width="800" height="500">

### Submission to Kaggle
For each technique that we used (mask with mean and median filter, and autoencoder), we generated a submission file containing the results after running the methods on the test set of images provided by Kaggle. Submissions were evaluated on the root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities. One can notice that, as expected, the best score was reached by using an ensemble of all methods.

|   | Median | Mean | Autoencoder  | Ensemble |
|---|---|---|---|---|
| Score  | 0.11029 | 0.10768 | 0.07268 | 0.06586 |



