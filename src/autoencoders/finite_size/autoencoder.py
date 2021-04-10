# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 21:09:10 2021

@author: Danilo_Elias
"""

import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
#from torchvision.utils import save_image
TAM=16
print(torch.cuda.is_available())
NUM_EPOCHS = 7000
LEARNING_RATE = 1e-4
BATCH_SIZE =28
#s400=np.loadtxt('/content/drive/My Drive/Tresf10_samples_1000.txt',dtype='int')
#s1000=np.loadtxt('/content/drive/My Drive/Tresf10_samples_denso.txt',dtype='int')
s400=np.loadtxt('C:\\Users\\Danilo_Elias\\Downloads\\Samples_{}x{}.txt'.format(TAM,TAM),dtype='int',max_rows=400)
samples=np.loadtxt('C:\\Users\\Danilo_Elias\\Downloads\\Samples_{}x{}.txt'.format(TAM,TAM),dtype='int',max_rows=1400)
aux1=np.copy(s400)
aux2=np.copy(samples)
while (np.shape(s400)[1]<128*128):
    s400=np.concatenate((s400,aux1),1)
    samples=np.concatenate((samples,aux2),1)

#srand=np.loadtxt('/content/drive/My Drive/Tresf10_samples_denso_novo.txt',dtype='int')
#aux=np.concatenate((srand,s1000),0)
#np.savetxt('samples_rand.txt',aux,fmt='%d')
#samples=np.loadtxt('samples_rand.txt',dtype='int',skiprows=300,max_rows=1400)
#samples=np.loadtxt('/content/drive/Shared drives/Danilo/data/Tresf10_samples_1000.txt',dtype='int')

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
'''
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
 
        # encoder
        self.enc1 = nn.Linear(in_features=900, out_features=750)
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
        self.dec9 = nn.Linear(in_features=750, out_features=900)
 
    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        x = F.relu(self.enc6(x))
        x = F.relu(self.enc7(x))
        x = F.relu(self.enc8(x))
        x = F.relu(self.enc9(x))


        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        x = F.relu(self.dec6(x))
        x = F.relu(self.dec7(x))
        x = F.relu(self.dec8(x))
        x = F.tanh(self.dec9(x))  
        return x
    def latent(self,x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        x = F.relu(self.enc6(x))
        x = F.relu(self.enc7(x))
        x = F.relu(self.enc8(x))
        x = F.relu(self.enc9(x))
        return x
'''
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
#        x = F.hardtanh(x,min_val=0,max_val=0)
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
def train(net, trainloader, NUM_EPOCHS,var_break):
    train_loss = []
    epoch=0
    while epoch<NUM_EPOCHS:
#    for epoch in range(NUM_EPOCHS):
        
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
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, NUM_EPOCHS, loss)) 
        if loss<var_break:
            break
        elif epoch>0.95*NUM_EPOCHS:
            NUM_EPOCHS+=500
        epoch+=1
    return train_loss
'''
def test_image_reconstruction(net, testloader):
     i=1
     
     for batch in testloader:
        
        
        img = batch
        img = img.to(device)
        img = img.view(img.size(0), -1)
        outputs = net(img)
        outputs.int()
        outputs.float()
        #print(outputs.latent)
        outputs = outputs.view(outputs.size(0), 1, 32, 32).cpu().data
        
        img=img.view(img.size(0), 1, 32, 32).cpu().data
        
#            save_image(outputs[0], 'train_reconstruction.svg')
#            save_image(img[0], 'original.png.svg')
        plt.imshow(img[i].view(32,32))
        plt.gray()
        #plt.colorbar(cmap='hot')
        plt.savefig('teste2_{}.svg'.format(i))
        plt.show()
        i+=1
        print(i)
#        save_image(outputs,'/content/drive/Shared drives/Danilo/batches_new/recbatch_40_{}.png'.format(i))
        
#        save_image(img,'/content/drive/Shared drives/Danilo/batches_new/origbatch_40_{}.png'.format(i))

                
'''
device = get_device()
net.to(device)
 
#test_image_reconstruction(net, trainloader)    
#np.random.shuffle(samples)
trainset_new=torch.as_tensor(s400).float()
trainloader_new = DataLoader(trainset_new,batch_size=16,shuffle=True)
train_loss_new = train(net, trainloader_new, 4000,0.36)

print('1st stage DONE, all samples: begin')
 
train_loss = train(net, trainloader, 6000,0.2)
optimizer = optim.Adam(net.parameters(), lr=5e-5)
train_loss = train(net, trainloader, 20000,0.07)
#optimizer = optim.Adam(net.parameters(), lr=1e-5)
#train_loss = train(net, trainloader, 20000,0.01)
plt.figure()
plt.plot(train_loss)
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.savefig('/content/drive/My Drive/Colab Notebooks/loss_sample_rand.png')

#test_image_reconstruction(net, trainloader)
net.to('cpu')
LATENT=net.latent(trainset)
np.savetxt('C:\\Users\\Danilo_Elias\\Teste_Auto\\PUD\\{}x{}\\latente_{}_run2_new.txt'.format(TAM,TAM,TAM),LATENT.detach().numpy())
