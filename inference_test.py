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

model = CNNNet()
print(model)
model.load_state_dict(torch.load('/Users/shreenidhir/Documents/Machine learning/project_18nov/hand_gesture_code/output_model/model_alexnet_new_data_40epochs.pt'))
#model.load_state_dict(torch.load('model_cifar_prev.pt'))
criterion = nn.CrossEntropyLoss()
batch_size=1
# track test loss
test_data=Rpsdata("/Users/shreenidhir/Documents/Machine learning/project_18nov/rps_train_test_val_split/test/")
test_loader=DataLoader(test_data,batch_size=30,shuffle=True)
classes=['rock','paper','scissors']

def main():
    test_loss = 0.0
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))

    model.eval()
    # iterate over test data
    for data, target in test_loader:
        # move tensors to GPU if CUDA is available
      #  if train_on_gpu:
      #      data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        #print("data shape==",data.shape)
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target.reshape(target.size(0)).long())
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1) 
        print("pred===",pred.numpy())   
        print("target==",target.numpy())
        # compare predictions to true label
        target=target.reshape(target.size(0))
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(np.squeeze(correct_tensor.cpu().numpy()))
        # calculate test accuracy for each object class 
        for i in range(batch_size):
            label = target.data[i].long()
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(3):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  