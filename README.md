# Face-Rating-project

This is a project developed for rating face score for Asian people.

## Data

For training and testing dataset, I choosed a new diverse benchmark dataset SCUT-FBP5500 created by HCIILAB.

The link of dataset is here: https://github.com/HCIILAB/SCUT-FBP5500-Database-Release

The SCUT-FBP5500 dataset has totally 5500 frontal faces with diverse properties (male/female, Asian/Caucasian, ages) and diverse labels (face landmarks, beauty scores within [1,~5], beauty score distribution). 

In the dataset, all the images are labeled with beauty scores ranging from [1, 5] by totally 60 volunteers, I average the rating scores of 60 interviewers for one image(except max and min value) and spread them from 0 to 1 (0 means ungly and 1 means beautiful) for easier data processing in code/build_dataset.py.
Also different from training all the images at once for experiment by HCIILAB, since the features of faces from different gender or area are different, I divide the dataset into male and female two parts (fty: asian female face, mty: asian male face) and train them separately. The number of fty and mty images are both 2000 in which 1700 images for training (1400: train, 300: validation), 300 images for test.

## Train Models for Pytorch

Different CNN models (net, Alexnet, Mobilenet, VGG, Resnet) in model folder are trained separately on fty and mty images to see their performance. I choose Smooth Mean Absolute Error as loss function and Adam optimizer with 0.0001 learning rate. When evaluating the model performance on validation and test data, the L1 Loss function is used to calculate the loss beteewn true label and prediction.

After I trained different models and check their performance on the test dataset, I found that it does not seem that the deeper network will have better performance. Overall, the model VGG11 have the best performance on both dataset.

### Requirements
- Python 3.7
- Torch 1.3.1 
- Numpy
- Pandas
- Matplotlib
- Scikit-image

## Result



