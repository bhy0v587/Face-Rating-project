"""Train the model"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import model.net as net
import model.Alexnet as alexnet
import model.Resnet as resnet
import model.Mobilenet as mobilenet
import model.vgg as vgg
import data_loader

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

    # Move inputs and labels into GPU
    inputs = inputs.cuda()
    labels = labels.cuda()

    # Forward
    outputs = model(inputs)   

    # Use L1Loss to calculate loss
    loss = np.sum(np.absolute(outputs.cpu().detach().numpy().flatten() - labels.cpu().detach().numpy()))

    # Show one prediction
    if total_num == 0:
        print("True label:\n", labels)
        print("Prediction:\n", outputs.reshape(-1))
        
    loss_num+= loss
    total_num+= bch
  
  # Print loss between 0% to 100%
  print("LOSS: "+"%.2f"%(loss_num*100/float(total_num)))
  
#%%
  
def train_model(model, train_loader, val_loader, test_loader, loss_func, optimizer, epochs=25):
    
    for epoch in range(epochs):
        running_loss = 0.0
        ct_num = 0
        
        model.train()
        
        for iteration, data in enumerate(train_loader):
            # Take the inputs and the labels for 1 batch.
            inputs, labels = data
            #bch = inputs.size(0)
            
            # Move inputs and labels into GPU
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            # Remove old gradients for the optimizer.
            optimizer.zero_grad()
            
            # Compute result (Forward)
            outputs = model(inputs)
                      
            #print(outputs.view(32), labels)
            # Compute loss
            loss    = loss_func(outputs.view(inputs.size(0)), labels)

            # Calculate gradients (Backward)
            loss.backward()

            # Update parameters
            optimizer.step()
  
            #with torch.no_grad():
            running_loss += loss.item()
            ct_num+= 1
            
            if iteration%20 == 19:
                #print("Iteration: "+str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')
                print("[Epoch: "+str(epoch+1)+"]"" --- Iteration: "+str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')
        # Test
        evaluate_model(model, val_loader)
        
        if epoch%50 == 49:
            evaluate_model(model, test_loader)
        
    # Save if the model has best accuracy till now
    torch.save(model, os.path.join('model2','_model_mty_'+str(epoch+1)+'.pt'))
    
#%%
model = net.Net().cuda()
model = alexnet.AlexNet().cuda()
model = mobilenet.mobilenet_v2().cuda()
model = resnet.resnet18().cuda() 
model = vgg.vgg11().cuda()

# Loss function
loss_func = nn.SmoothL1Loss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

data_dense = os.path.join('data_701515','mty')

dataloaders = data_loader.fetch_dataloader(['train', 'val', 'test'], data_dense)

train_dl = dataloaders['train']
val_dl = dataloaders['val']
test_dl = dataloaders['test']

epochs = 200

train_model(model, train_dl, val_dl, test_dl, loss_func, optimizer, epochs)


