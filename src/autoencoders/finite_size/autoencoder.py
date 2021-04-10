# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 21:09:10 2021

@author: Danilo_Elias
"""

'''This Autoencoder has 9 linear fully conected linear layers in the encoder and is symmetrical in the decoder, we utilize
hyperbolic tangent activation functions on eache layer. The training stage begins by loading all the samples and
iterating for 2000 EPOCHS with adam at learning rate(LR) of 1e-4.
Next Stage we update the ADAM learning rate (LR) to 5e-5 for 6000 EPOCHS, and in the last training stage we reset the LR
to 1e-5 and stop when the loss function reaches 9.5% accuracy'''

import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from ..\train_autoencoder import train_autoencoder
TAM=16
print(torch.cuda.is_available())
NUM_EPOCHS = 7000
LEARNING_RATE = 1e-4
BATCH_SIZE =28
s400=np.loadtxt('C:\\Users\\Danilo_Elias\\Downloads\\Samples_{}x{}.txt'.format(TAM,TAM),dtype='int',max_rows=400)
samples=np.loadtxt('C:\\Users\\Danilo_Elias\\Downloads\\Samples_{}x{}.txt'.format(TAM,TAM),dtype='int',max_rows=1400)
aux1=np.copy(s400)
aux2=np.copy(samples)
while (np.shape(s400)[1]<128*128):
    s400=np.concatenate((s400,aux1),1)
    samples=np.concatenate((samples,aux2),1)

trainset=torch.as_tensor(samples).float()

 
trainloader = DataLoader(
    trainset, 
    batch_size=BATCH_SIZE,
    shuffle=True
)
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
 
        # encoder
        self.enc1 = nn.Linear(in_features=128*128, out_features=750)
        self.enc2 = nn.Linear(in_features=750, out_features=600)
        self.enc3 = nn.Linear(in_features=600,out_features=450)
        self.enc4 = nn.Linear(in_features=450, out_features=300)
        self.enc5 = nn.Linear(in_features=300, out_features=150)
        self.enc6 = nn.Linear(in_features=150, out_features=75)
        self.enc7 = nn.Linear(in_features=75, out_features=30)
        self.enc8 = nn.Linear(in_features=30,out_features=10)
        self.enc9 = nn.Linear(in_features=10,out_features=2)

        # decoder 
        self.dec1 = nn.Linear(in_features=2,out_features=10)
        self.dec2 = nn.Linear(in_features=10, out_features=30)
        self.dec3 = nn.Linear(in_features=30, out_features=75)
        self.dec4 = nn.Linear(in_features=75, out_features=150)
        self.dec5 = nn.Linear(in_features=150, out_features=300)
        self.dec6 = nn.Linear(in_features=300, out_features=450)
        self.dec7 = nn.Linear(in_features=450, out_features=600)
        self.dec8 = nn.Linear(in_features=600, out_features=750)
        self.dec9 = nn.Linear(in_features=750, out_features=128*128)
 
    def forward(self, x):
        x = F.tanh(self.enc1(x))
        x = F.tanh(self.enc2(x))
        x = F.tanh(self.enc3(x))
        x = F.tanh(self.enc4(x))
        x = F.tanh(self.enc5(x))
        x = F.tanh(self.enc6(x))
        x = F.tanh(self.enc7(x))
        x = F.tanh(self.enc8(x))
        x = F.tanh(self.enc9(x))


        x = F.tanh(self.dec1(x))
        x = F.tanh(self.dec2(x))
        x = F.tanh(self.dec3(x))
        x = F.tanh(self.dec4(x))
        x = F.tanh(self.dec5(x))
        x = F.tanh(self.dec6(x))
        x = F.tanh(self.dec7(x))
        x = F.tanh(self.dec8(x))
        x = F.tanh(self.dec9(x))
        return x
    def latent(self,x):
        x = F.tanh(self.enc1(x))
        x = F.tanh(self.enc2(x))
        x = F.tanh(self.enc3(x))
        x = F.tanh(self.enc4(x))
        x = F.tanh(self.enc5(x))
        x = F.tanh(self.enc6(x))
        x = F.tanh(self.enc7(x))
        x = F.tanh(self.enc8(x))
        x = F.tanh(self.enc9(x))
        
        return x

 
net = Autoencoder()
print(net)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


device = get_device()
net.to(device)
trainset_new=torch.as_tensor(s400).float()
trainloader_new = DataLoader(trainset_new,batch_size=16,shuffle=True)
train_loss_new = train_autoencoder(net, trainloader_new, 4000,0.36,device,optimizer,criterion)
print('1st stage DONE, all samples: begin')
train_loss = train_autoencoder(net, trainloader, 6000,0.2,device,optimizer,criterion)
optimizer = optim.Adam(net.parameters(), lr=5e-5)
train_loss = train_autoencoder(net, trainloader, 20000,0.07,device,optimizer,criterion)
plt.figure()
plt.plot(train_loss)
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
net.to('cpu')
LATENT=net.latent(trainset)
np.savetxt('C:\\Users\\Danilo_Elias\\Teste_Auto\\PUD\\{}x{}\\latente_{}_run2_new.txt'.format(TAM,TAM,TAM),LATENT.detach().numpy())