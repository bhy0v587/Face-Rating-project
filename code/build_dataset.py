""" 
build my dataset for face rating

Firstly, the original dataset I got from SCUT-FBP5500 dataset contains 5500 frontal, unoccluded faces aged from 15 to 60 with neutral expression. 
It can be divided into four subsets with different races and gender, including 2000 Asian females, 2000 Asian males, 750 Caucasian females and 750 Caucasian males.
About labels: All the images are labeled with beauty scores ranging from [1, 5] by totally 60 volunteers.
I average the rating scores of 60 interviewers for one image(except max and min value) and spread them from 0 to 1
0 means ungly and 1 means beautiful.

Secondly, because of the unevenly distribution of original dataset, I collocted around 2000 other images including normal faces and star faces.  
I used several functions to edit the raw images(crop, add border...) to be similar to the originial dataset.

In order to increase the robustness of rating model, instead of training all the data, I will build dataset and train model based on the gender
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


#%%  change the order of image filenames
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
            

#%%  collect images aged from 15 to 60

data_dir2 = 'face/data/woman'

i = 0
for split in range(15,61):
    #print(split)
    #print(i)
    data_dir = 'face/AFAD-Full/{}/112'.format(split)
    filenames = os.listdir(data_dir)
    for f in filenames:
        if f.endswith('.jpg'):
            im = Image.open(os.path.join(data_dir,f))
            if(im.size[0] >= 224):
                im.save(os.path.join(data_dir2,"img" + str(i) + ".jpg"), "JPEG", quality=80, optimize=True, progressive=True)
                i = i + 1


#%% add border to images

import math

i = 0
for split in ['man', 'woman']:
    data_dir = 'face/star_try/{}'.format(split)
    data_save = 'face/newdata2/{}'.format(split)
    filenames = os.listdir(data_dir)
    for f in filenames:
        if f.endswith('.jpg'):
            im = Image.open(os.path.join(data_dir,f))
            old_size = im.size

            new_size = (600, 600)
            new_im = Image.new("RGB", new_size,color = (255, 255, 255))
            new_im.paste(im, (math.floor((new_size[0]-old_size[0])/2),math.floor((new_size[1]-old_size[1])/2)))
            img = new_im.resize((256, 256), Image.ANTIALIAS)
            img.save(os.path.join(data_save,"img" + str(i) + ".jpg"), "JPEG", quality=80, optimize=True, progressive=True)
            i = i + 1

            
#%% crop the main face part of images

i = 0
data_dir = 'face/newdata/man'
data_save = 'face/newdata/man_after'
filenames = os.listdir(data_dir)
for f in filenames:
   if f.endswith('.jpg'):
      img = Image.open(os.path.join(data_dir,f))
      if((img.size[0]+60) < img.size[1]):
          area = (0, 15, img.size[0], img.size[1]-40)
          cropped_img = img.crop(area)
          cropped_img.save(os.path.join(data_save,"img" + str(i) + ".jpg"), "JPEG", quality=80, optimize=True, progressive=True)
          i = i + 1
      elif((img.size[0]+50) < img.size[1]):
          area = (0, 10, img.size[0], img.size[1]-35)
          cropped_img = img.crop(area)
          cropped_img.save(os.path.join(data_save,"img" + str(i) + ".jpg"), "JPEG", quality=80, optimize=True, progressive=True)
          i = i + 1
      elif((img.size[0]+40) < img.size[1]):
          area = (0, 10, img.size[0], img.size[1]-25)
          cropped_img = img.crop(area)
          cropped_img.save(os.path.join(data_save,"img" + str(i) + ".jpg"), "JPEG", quality=80, optimize=True, progressive=True)
          i = i + 1
      elif((img.size[0]+30) < img.size[1]):
          area = (0, 10, img.size[0], img.size[1]-15)
          cropped_img = img.crop(area)
          cropped_img.save(os.path.join(data_save,"img" + str(i) + ".jpg"), "JPEG", quality=80, optimize=True, progressive=True)
          i = i + 1
      elif((img.size[0]+20) < img.size[1]):
          area = (0, 10, img.size[0], img.size[1]-5)
          cropped_img = img.crop(area)
          cropped_img.save(os.path.join(data_save,"img" + str(i) + ".jpg"), "JPEG", quality=80, optimize=True, progressive=True)
          i = i + 1
      else:
          img.save(os.path.join(data_save,"img" + str(i) + ".jpg"), "JPEG", quality=80, optimize=True, progressive=True)
          i = i + 1

        
#%% modify to square size (crop, border...)
          
import math

data_dir = 'face/newdata/man_after'
data_save = 'face/newdata/man_after2'
filenames = os.listdir(data_dir)
for f in filenames:
    if f.endswith('.jpg'):
        im = Image.open(os.path.join(data_dir,f))
        if(im.size[0] > im.size[1]):
            area = (math.floor((im.size[0]-im.size[1])/2), 0, math.floor((im.size[0]+im.size[1])/2), im.size[1])
            cropped_img = im.crop(area)
            cropped_img.save(os.path.join(data_save,"img" + str(i) + ".jpg"), "JPEG", quality=80, optimize=True, progressive=True)
            i = i + 1
        elif(im.size[0] < im.size[1]):
            old_size = im.size
            new_size = (im.size[1], im.size[1])
            new_im = Image.new("RGB", new_size,color = (255, 255, 255))
            new_im.paste(im, (math.floor((new_size[0]-old_size[0])/2),math.floor((new_size[1]-old_size[1])/2)))
            im = new_im.resize((256, 256), Image.ANTIALIAS)
            im.save(os.path.join(data_save,"img" + str(i) + ".jpg"), "JPEG", quality=80, optimize=True, progressive=True)
            i = i + 1
        else:
            im.save(os.path.join(data_save,"img" + str(i) + ".jpg"), "JPEG", quality=80, optimize=True, progressive=True)
            i = i + 1

    
    

