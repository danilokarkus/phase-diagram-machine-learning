# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 10:22:24 2021

@author: Danilo_Elias
"""
def train_autoencoder(net, trainloader, NUM_EPOCHS,var_break,device,optimizer,criterion):
    train_loss = []
    epoch=0
    while epoch<NUM_EPOCHS:
        running_loss = 0.0
        for data in trainloader:
            img = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            outputs = net(img)
            outputs.int()
            outputs.float()
            loss = criterion(outputs, img)
            loss.backward()
            outputs.float()
            optimizer.step()
            running_loss += loss.item()
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(epoch+1, NUM_EPOCHS, loss)) 
        if loss<var_break:
            break
        elif epoch>0.95*NUM_EPOCHS:
            NUM_EPOCHS+=500
        epoch+=1
    return train_loss