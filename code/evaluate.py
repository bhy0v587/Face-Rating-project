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

#%%
def evaluate_model(model, data_loader):
  print("Testing the network...")
  model.eval()
  total_num   = 0
  loss_num = 0.0
  for val_iter, val_data in enumerate(data_loader):
    # Get one batch of test samples
    inputs, labels = val_data    
    bch = inputs.size(0)
    #inputs = inputs.view(bch, -1) <-- We don't need to reshape inputs here (we are using CNNs).

    # Move inputs and labels into GPU
    inputs = inputs.cuda()
    labels = labels.cuda()

    # Forward
    outputs = model(inputs)   

    loss = np.sum(np.absolute(outputs.cpu().detach().numpy().flatten() - labels.cpu().detach().numpy()))
    # Get predicted classes
    #_, pred_cls = torch.max(outputs, 1)
    print("True label:\n", labels)
    print("Prediction:\n", outputs.reshape(-1))
        
        
    #Record test result
    loss_num+= loss
    total_num+= bch

  #model.train()
  
  print("LOSS: "+"%.2f"%(loss_num*100/float(total_num)))
  
def evaluate_image(model, image_name):
    
    test_image = Image.open(image_name)
    plt.imshow(test_image)
    
    test_image_tensor = data_loader.eval_transformer(test_image)
    
    print(test_image_tensor.shape)
    
    test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    
    model.eval()
    
    outputs = model(test_image_tensor)
    
    print("Prediction Score: ", "%.2f"%(outputs.reshape(-1).cpu().detach().numpy()[0]))
    
def evaluate_image_gender(model, image_name):
    
    gender = ''
    
    test_image = Image.open(image_name)
    plt.imshow(test_image)
    plt.show()
    
    test_image_tensor = data_loader.eval_transformer(test_image)
    
    print(test_image_tensor.shape)
    
    test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    
    model.eval()
    
    outputs = model(test_image_tensor)
    
    # Get predicted classes
    _, pred_cls = torch.max(outputs, 1)
    
    
    if pred_cls.cpu().detach().numpy()[0] == 1:
        gender = 'Male'
    else:
        gender = 'Female'
    
    print("Gender: ", str(gender))
    
  
#%%

data_dense = os.path.join('data_dense','mty')

dataloaders = data_loader.fetch_dataloader(['test'], data_dense)

test_dl = dataloaders['test']
  
model_path = os.path.join('model2','_lenet_SGD_gender_200.pt')
model = torch.load(model_path)

#evaluate_model(model, test_dl)
data_dir = 'labfaces'

filenames = os.listdir(data_dir) 
filenames = [os.path.join(data_dir,f) for f in filenames if f.endswith('.jpg')]

for filename in filenames:
    evaluate_image_gender(model, filename)

#evaluate_image_gender(model, 'bhy.jpg')
#evaluate_image_gender(model, 'test.jpg')
#evaluate_image_gender(model, 'female.jpg')


