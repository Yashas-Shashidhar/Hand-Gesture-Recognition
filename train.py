import torch
import numpy as np

from torchvision import datasets
from torch.utils.data import DataLoader

from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
from data_iterator import Rpsdata


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        
        # Convolutional layers
                            #Init_channels, channels, kernel_size, padding) 
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        # Pooling layers
        self.pool = nn.MaxPool2d(2,2)
        
        # FC layers
        # Linear layer (64x4x4 -> 500)
        self.fc1 = nn.Linear(64 * 9 * 6, 500)
        
        # Linear Layer (500 -> 10)
        self.fc2 = nn.Linear(500, 3)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.pool(F.elu(self.conv3(x)))
        x = self.pool(F.elu(self.conv4(x)))
        x = self.pool(F.elu(self.conv5(x)))
        # Flatten the image
       # print("x===",x.shape)
        x = x.view(-1, 64*9*6)
        x = self.dropout(x)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 3):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #sprint("=====",x.shape)
        x = self.fc(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
        
num_classes = 3
num_epochs = 40
batch_size = 16
learning_rate = 0.01

#model = ResNet(ResidualBlock, [3, 4, 6, 3])
model = CNNNet()
print(model)


'''
==========Data orginal==============
# import torchvision
# # Number of subprocesses to use for data loading
# num_workers = 0

# # How many samples per batch to load
# batch_size = 20

# # Percentage of training set to use as validation
# n_valid = 0.2

# # Convert data to a normalized torch.FloatTensor
# # Data augmentation
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(), # randomly flip and rotate
#     transforms.RandomRotation(10),
#                                 transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                                ])

# # Select training_set and testing_set
# train_data = datasets.CIFAR10("data", 
#                               train= True,
#                              download=True,
#                              transform = transform)

# test_data = datasets.CIFAR10("data", 
#                               train= False,
#                              download=True,
#                              transform = transform)



# # Get indices for training_set and validation_set
# n_train = len(train_data)
# indices = list(range(n_train))
# np.random.shuffle(indices)
# split = int(np.floor(n_valid * n_train))
# train_idx, valid_idx = indices[split:], indices[:split]

# # Define samplers for obtaining training and validation
# train_sampler = SubsetRandomSampler(train_idx)
# valid_sampler = SubsetRandomSampler(valid_idx)

# # Prepare data loaders (combine dataset and sampler)
# train_loader = torch.utils.data.DataLoader(train_data, 
#                                        batch_size = batch_size,
#                                           sampler = train_sampler,
#                                          num_workers = num_workers)

# valid_loader = torch.utils.data.DataLoader(train_data, 
#                                             batch_size = batch_size,
#                                            sampler = valid_sampler,
#                                            num_workers = num_workers)
# test_loader = torch.utils.data.DataLoader(test_data, 
#                                             batch_size = batch_size,
#                                            num_workers = num_workers)
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                          download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                            shuffle=True, num_workers=2)

# #dataiter = iter(trainloader)
# #images, labels = next(dataiter)

# # Specify the image classes
# classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
#           "horse", "ship", "truck"]
=====Data orginal End=======
'''
training_data=Rpsdata("./rps_train_test_val_split/train_1/")
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

validation_data=Rpsdata("./rps_train_test_val_split/val/")
valid_loader=DataLoader(validation_data,batch_size=batch_size,shuffle=True)

criterion = nn.CrossEntropyLoss()

# Specify the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
#optimizer = optim.SGD(model.parameters(), lr=0.001,weight_decay = 0.001, momentum=0.9)
def main():


    n_epochs = 40 # you may increase this number to train a final model

    valid_loss_min = np.Inf # track change in validation loss

    for epoch in range(1, n_epochs+1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            #if train_on_gpu:
            #    data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            #print(output.dtype)
            #print(target.shape)
            #target_loss=target.reshape(target.size(0))
            #print("new shape==",target_loss.shape)
            loss = criterion(output, target.reshape(target.size(0)).long())
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for data, target in valid_loader:
           # print("validation start======")
            # move tensors to GPU if CUDA is available
            #if train_on_gpu:
            #    data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            #print("data==",data.shape)
            output = model(data)
            # calculate the batch loss
            
            loss = criterion(output, target.reshape(target.size(0)).long())
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
        
        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), './model_alexnet_new_data_40epochs.pt')
            valid_loss_min = valid_loss


if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()          
