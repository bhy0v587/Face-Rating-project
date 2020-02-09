import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import data_loader

import matplotlib.pyplot as plt
from PIL import Image

#%% evaluate model on dataset
def evaluate_model(model, data_loader):
  print("Testing the network...")
  model.eval()
  total_num = 0
  loss1_num = 0.0
  loss2_num = 0.0
  for val_iter, val_data in enumerate(data_loader):
    # Get one batch of test samples
    inputs, labels = val_data    
    bch = inputs.size(0)

    # Move inputs and labels into GPU
    inputs = inputs.cuda()
    labels = labels.cuda()

    # Forward
    outputs = model(inputs)   

    mae_loss = np.sum(np.absolute(outputs.cpu().detach().numpy().flatten() - labels.cpu().detach().numpy()))
    rmse_loss = np.sum(np.square(outputs.cpu().detach().numpy().flatten() - labels.cpu().detach().numpy()))
        
    #Record test result
    loss1_num+= mae_loss
    loss2_num+= rmse_loss
    total_num+= bch
    
  #print(total_num)
  print("MAE_LOSS: "+"%.4f"%(loss1_num/float(total_num)))
  print("RMSE_LOSS: "+"%.4f"%(np.sqrt(loss2_num/float(total_num))))
  
#%% evaluate model on one image

def evaluate_image(model, image_name):
    
    test_image = Image.open(image_name)
    plt.imshow(test_image)
    plt.show()
    
    test_image_tensor = data_loader.eval_transformer(test_image)
    
    #print(test_image_tensor.shape)
    
    test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    
    model.eval()
    
    outputs = model(test_image_tensor)
    
    print("Prediction Score: ", "%.2f"%(outputs.reshape(-1).cpu().detach().numpy()[0]))
    
#%% 

data_dense = os.path.join('data_dense','fty')

dataloaders = data_loader.fetch_dataloader(['test'], data_dense)

test_dl = dataloaders['test']
  
#  load saved model
model_path = os.path.join('model','_resnet_Adam_fty_200_0.001.pt')
model = torch.load(model_path)

evaluate_model(model, test_dl)


