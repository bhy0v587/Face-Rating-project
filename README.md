# Face-Rating-project

This is a project developed for rating face score for Asian people.

## Data
For training and testing dataset, I choosed a new diverse benchmark dataset SCUT-FBP5500 created by HCIILAB.
The link of dataset is here: https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
The SCUT-FBP5500 dataset has totally 5500 frontal faces with diverse properties (male/female, Asian/Caucasian, ages) and diverse labels (face landmarks, beauty scores within [1,~5], beauty score distribution). 
In the dataset, all the images are labeled with beauty scores ranging from [1, 5] by totally 60 volunteers, I average the rating scores of 60 interviewers for one image(except max and min value) and spread them from 0 to 1 (0 means ungly and 1 means beautiful) for easier data processing in code/build_dataset.py.
Also different from training all the images at once for experiment by HCIILAB, since the features of faces from different gender or area are different, I divide the dataset into male and female two parts and train them separately.

## Model



## Train



## Result


