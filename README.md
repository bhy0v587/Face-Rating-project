# Face-Rating-project

This is a project developed for rating face beauty for Asian people.

## Prepare Data

For training and testing dataset, they consist of two parts.

Firstly, I choosed a new diverse benchmark dataset SCUT-FBP5500 created by HCIILAB as the orignial dataset.
The link of dataset is here: https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
The SCUT-FBP5500 dataset has totally 5500 frontal faces with diverse properties (male/female, Asian/Caucasian, ages) and diverse labels (face landmarks, beauty scores within [1,~5], beauty score distribution). 

In the dataset, all the images are labeled with beauty scores ranging from [1, 5] by totally 60 volunteers, I average the rating scores of 60 interviewers for one image(except max and min value) and spread them from 0 to 1 (0 means ungly and 1 means beautiful) for easier data processing in code/build_dataset.py.

Also different from training all the images at once for experiment by HCIILAB, since the features of faces from different gender or area are different, I divide the dataset into male and female two parts (fty: asian female face, mty: asian male face) and train them separately. The number of fty and mty images are both 2000.

After training and testing on the orignial dataset, I analyzed the pattern of the orignial dataset and the results on fty and mty data, I found the original dataset is not so evenly distributed and the number of images with high beauty scores are not big enough. Thus, the trained model is not robust enough for evaluating or classify beautiful, normal and ugly face.

Secondly, in order to fix the problems above, I enlarged original dataset by adding other 2000 images (normal faces from AAF dataset (link:https://github.com/JingchunCheng/All-Age-Faces-Dataset) and star faces (http://www.seeprettyface.com/mydataset_page3.html)). Then some preprocessing are done to crop or add boarder to images in code/build_dataset.py.

Since these images have no labels, with the help of 5 lab members (include me) I collected the beauty scores and nomalized the labels.

Btw, the whole dataset are divided into 70% train dataset, 15% validation dataset and 15% test dataset.

## Train Models

Different CNN models (AlexNet, VGG11, ResNet18) in model folder are trained separately on fty and mty images to see their performance. I choose SmoothL1Loss as loss function and Adam optimizer with 0.0001 learning rate. When evaluating the model performance on validation and test data, the maximum absolute error (MAE) and root mean square error (RMSE) are used to calculate the loss between true label and prediction. The code can be found in code/train_model.py.

After I trained different models and check their performance on the test dataset, I found that it does not seem that the deeper network will have better performance. Overall, the model VGG11 have the best performance on both dataset. (Or the reason may be that I did not set appropriate hyperparameters for other networks, I think more attempts are required)

### Requirements
- Python 3.7
- Torch 1.3.1 
- Numpy
- Pandas
- Matplotlib
- Scikit-image

## Results

The evaluation results (MAE and RMSE) of VGG11 model performing on fty data are shown as follows: 

![alt text](https://github.com/bhy0v587/Face-Rating-project/blob/master/result.jpg)

From the results, the model can predict face rating scores from 0 to 1 within the loss of about 6.2% for both male and female face, which means the error range between prediction score and true labels (from 0 to 1) are controlled within about 0.06. Since there is no fixed scoring standard for the face rating, everyone may have his own standard for beauty and ugliness. Thus, the results within such error range show that our model can evaluate and distinguish beauty of the face accurately to some extent.

## Application
The following shows the prediction for other Asian faces (with makeup, edit), which also shows that our model not only has the capability of recognizing and distinguishing beauty but also evaluating the beauty of the same person with makeup or not.

Beautiful woman (without makeup, with makeup, with edit):

![alt text](https://github.com/bhy0v587/Face-Rating-project/blob/master/test11.png)
![alt text](https://github.com/bhy0v587/Face-Rating-project/blob/master/test12.png)
![alt text](https://github.com/bhy0v587/Face-Rating-project/blob/master/test13.png)

Normal woman (without makeup, with makeup):

![alt text](https://github.com/bhy0v587/Face-Rating-project/blob/master/test21.png)
![alt text](https://github.com/bhy0v587/Face-Rating-project/blob/master/test22.png)

Handsome man:

![alt text](https://github.com/bhy0v587/Face-Rating-project/blob/master/test31.png)

Normal man (without makeup, with makeup?):

![alt text](https://github.com/bhy0v587/Face-Rating-project/blob/master/test41.png)
![alt text](https://github.com/bhy0v587/Face-Rating-project/blob/master/test42.png)


