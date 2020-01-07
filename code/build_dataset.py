""" 
build my dataset for face rating

The original contains 5500 frontal, unoccluded faces aged from 15 to 60 with neutral expression. 
It can be divided into four subsets with different races and gender, including 2000 Asian females, 2000 Asian males, 750 Caucasian females and 750 Caucasian males.
About labels: All the images are labeled with beauty scores ranging from [1, 5] by totally 60 volunteers.

I average the rating scores of 60 interviewers for one image(except max and min value) and spread them from 0 to 1
0 means ungly and 1 means beautiful 

In order to increase the robustness of rating model, instead of train ing all the data, I will build dataset and train model based on the gender
"""

from __future__ import print_function, division
import pandas as pd
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

#%% process 750 ftw, mtw images

import csv

for split in ['ftw','mtw']:
    rating_frame = pd.read_csv('face/SCUT-FBP5500_with_Landmarks/data2/{}_image.csv'.format(split))

    sum = 0

    img_name = ''
    rating = 0

    with open('face/SCUT-FBP5500_with_Landmarks/data2/{}_data.csv'.format(split), 'w', newline = '') as csvfile:
        cwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
        cwriter.writerow(['Image name', 'Rating'])
        
        for j in range(1,751):
            max = 0
            min = 5
            for i in range(0,60):
                img_name = rating_frame.iloc[i + 60 * (j - 1), 1]
                rating = rating_frame.iloc[i + 60 * (j - 1), 2]
                if rating > max: 
                    max = rating
                if rating < min:
                    min = rating
                
                sum = rating + sum     
              
            print(max, min)
            rate = round((sum - max - min) / 290, 2)
                
            cwriter.writerow([img_name, rate])
                
            print('Image name: {}'.format(img_name))   
            print('rating: {}'.format(rate)) 
            sum = 0
    
#%% process 2000 fty, mty images
            
for split in ['fty','mty']:
    rating_frame = pd.read_csv('face/SCUT-FBP5500_with_Landmarks/data2/{}_image.csv'.format(split))

    sum = 0

    img_name = ''
    rating = 0

    with open('face/SCUT-FBP5500_with_Landmarks/data2/{}_data.csv'.format(split), 'w', newline = '') as csvfile:
        cwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
        cwriter.writerow(['Image name', 'Rating'])
        
        for j in range(1,2001):
            max = 0
            min = 5
            for i in range(0,60):
                img_name = rating_frame.iloc[i + 60 * (j - 1), 1]
                rating = rating_frame.iloc[i + 60 * (j - 1), 2]
                if rating > max: 
                    max = rating
                if rating < min:
                    min = rating
                    
                sum = rating + sum     
                
            rate = round((sum - max - min) / 290, 2)
                
            cwriter.writerow([img_name, rate])
                
            print('Image name: {}'.format(img_name))   
            print('rating: {}'.format(rate)) 
            j = j + 1
            sum = 0

#%% combine csv files
            
combined_csv = pd.concat([pd.read_csv('face/SCUT-FBP5500_with_Landmarks/data/{}_data.csv'.format(split)) for split in ['ftw','fty','mtw','mty']])

combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8')

print(combined_csv)

#%%

# change the order of image filenames
df = pd.read_csv('face/SCUT-FBP5500_with_Landmarks/data2/mty_data.csv')

with open('face/SCUT-FBP5500_with_Landmarks/data2/mty_data2.csv', 'w', newline = '') as csvfile:
        cwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
        cwriter.writerow(['Image name', 'Rating'])
        for i in range(1,2001):
            filename = 'mty' + str(i) + '.jpg';
            for j in range(0,2000):              
                img_name = df.iloc[j,0]
                if(filename == img_name):
                    cwriter.writerow([filename, df.iloc[j,1]])
                    print(filename)
                    print(df.iloc[j,1])
            


    
    

