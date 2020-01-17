"""Train the model"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import model.net as net
import model.Alexnet as alexnet
import model.Resnet as resnet
import model.vgg as vgg
import data_loader

#%%
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
    #inputs = inputs.view(bch, -1) <-- We don't need to reshape inputs here (we are using CNNs).

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
        
  #model.train()
  #print(total_num)
  print("MAE_LOSS: "+"%.4f"%(loss1_num/float(total_num)))
  print("RMSE_LOSS: "+"%.4f"%(np.sqrt(loss2_num/float(total_num))))
  
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
            torch.save(model, os.path.join('model2','_model_fty_'+str(epoch+1)+'.pt'))
    
    
#%%
model = net.Net().cuda()
model = alexnet.AlexNet().cuda()
model = resnet.resnet18().cuda() 
model = vgg.vgg11().cuda()

# Loss function
loss_func = nn.SmoothL1Loss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

data_dense = os.path.join('data','fty')

dataloaders = data_loader.fetch_dataloader(['train', 'val', 'test'], data_dense)

train_dl = dataloaders['train']
val_dl = dataloaders['val']
test_dl = dataloaders['test']

epochs = 1000

train_model(model, train_dl, val_dl, test_dl, loss_func, optimizer, epochs)


