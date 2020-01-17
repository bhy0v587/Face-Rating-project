# For each gender, I create two folders named fty and mty with images and label data
# divide them into 70% train(2200), 15% val(500), 15% test(500)
# add the corresponding csv file train_data.csv, val_data.csv, test_data.csv

import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# define a training image loader that specifies transforms on images. 
train_transformer = transforms.Compose([
    transforms.Resize(224),  # resize the image to 224x224
    transforms.RandomHorizontalFlip(),   # randomly horizontally flip image
    transforms.ToTensor(),   # transform it into a torch tensor
    transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]) ]) # normalize with ImageNet mean and std

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize((224,224)),  # resize the image to 224x224
    transforms.ToTensor(),    # transform it into a torch tensor
    transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]) ])  


class SIGNSDataset(Dataset):
   
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)      
        self.filenames = [os.path.join(data_dir,f) for f in self.filenames if f.endswith('.jpg')]
        
        df = pd.read_csv(os.path.join(data_dir,'data.csv'), index_col = "Image name")
        self.labels = [df.loc[[os.path.split(filename)[1]],['Rating']].iloc[0,0] for filename in self.filenames]
        
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx]) 
        image = self.transform(image)
        
        return image, self.labels[idx]


def fetch_dataloader(types, data_dir):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}".format(split))
            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(SIGNSDataset(path, train_transformer), batch_size=32, shuffle=True,
                                        num_workers=0)
            else:
                dl = DataLoader(SIGNSDataset(path, eval_transformer), batch_size=32, shuffle=False,
                                num_workers=0)

            dataloaders[split] = dl

    return dataloaders

#%%  show part of image and labels

dataloaders = fetch_dataloader(['train', 'val'], 'data/fty')

train_dl = dataloaders['train']
val_dl = dataloaders['val']

image, labels = next(iter(train_dl))

print(image.size())
print(labels)

show_img = torchvision.utils.make_grid(image, nrow=5).numpy().transpose((1,2,0))
plt.imshow(show_img)


