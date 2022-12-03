import cv2
import numpy as np
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
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import cm




transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # randomly flip and rotate
    transforms.RandomRotation(10),
                                transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])


def peer2(img):
        b, g, r = cv2.split(img)
        ret, m1 = cv2.threshold(r, 95, 255, cv2.THRESH_BINARY)
        ret, m2 = cv2.threshold(g, 30, 255, cv2.THRESH_BINARY)
        ret, m3 = cv2.threshold(b, 20, 255, cv2.THRESH_BINARY)
        mmax = cv2.max(r, cv2.max(g, b))
        mmin = cv2.min(r, cv2.min(g, b))

        ret, m4 = cv2.threshold(mmax - mmin, 15, 255, cv2.THRESH_BINARY)
        ret, m5 = cv2.threshold(cv2.absdiff(r, g), 15, 255, cv2.THRESH_BINARY)
        m6 = cv2.compare(r, g, cv2.CMP_GE)
        m7 = cv2.compare(r, b, cv2.CMP_GE)
        mask = m1 & m2 & m3 & m6 & m4 & m5 & m7

        return mask

def get_direction(direction,x,y):
  diff_val_x=abs(x_li[-1]-x)
  diff_val_y=abs(y_li[-1]-y)
  if diff_val_x>diff_val_y:
    if(x-x_li[-1])>30:
       direction="RIGHT"
    elif (x-x_li[-1])>-30:
       direction="LEFT"
    else:
      direction=direction

  else:
    if (y-y_li[-1])>-30:
       direction="UP"
    elif (y-y_li[-1])>30:
       direction="DOWN"
    else:
      direction=direction
  return direction



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
model.load_state_dict(torch.load('./model_alexnet_new_data_40epochs.pt'))
classes=['rock','paper','scissors']
class_pred_li=[]
x_li=[]
y_li=[]
direction=" "
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('./rock.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
i=0

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  i=i+1
  print("i===",i)
  if ret == True:
        #cv2.imshow('Frame',frame)
    print("frame shape==",frame.shape)
    frame_re=cv2.resize(frame, (300,200))
    #frame_re=frame_re.reshape(300,200,3)
    print("frame shape==",frame_re.shape)
    im = Image.fromarray(frame_re)
    image = transform(im)
    image = image.reshape(1,3, 300, 200)
    #print("data shape==",data.shape)
    output = model(image)
    _, pred = torch.max(output, 1) 
    print(pred)
    class_name=classes[pred]
    print("====",class_name)
    #print(image.shape)
    font = cv2.FONT_HERSHEY_SIMPLEX
  
  
    # org
    org = (50, 50)
  
    # fontScale
    fontScale = 1
   
    # Blue color in BGR
    color = (255, 0, 0)
  
    # Line thickness of 2 px
    thickness = 2
   
    # Using cv2.putText() method
    frame = cv2.putText(frame, "class pred:"+class_name, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

    #========contour detection================

    #img = cv2.imread("/Users/shreenidhir/Documents/Machine learning/project_18nov/rps_data/scissors/2l1K148aIJHRR1q7.png")
    th3 = peer2(frame)
    kernel = np.ones((7,7), np.uint8)
    closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
    # gradient = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)
    contours, hiearachy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    #print(max_index)
    cnt = contours[max_index]
    cnt_arr=np.array(contours)
    (x,y,w,h) = cv2.boundingRect(cnt)
    #print("x==",x)
    #print("y==",y)
    ##print("w==",w)
    print("h==",h)
    start_point=(x,y)
    end_point=(x+w,y+h)
    color = (255, 0, 0)
    thickness = 2
    class_pred_li.append(pred)
    cx=x+(w/2)
    cy=y+(h/2)
    x_li.append(cx)
    y_li.append(cy)
    if i > 5:
      direction=get_direction(direction,cx,cy)
      print(direction)
    gesture_recognized=" "
    if class_name == 'rock' and direction == 'UP':
      gesture_recognized='ZOOM IN'
    if class_name == 'rock' and direction == 'DOWN':
      gesture_recognized='ZOOM OUT'
    if class_name == 'paper' and direction =='UP':
      gesture_recognized='SWIPE UP'
    if class_name == 'paper' and direction == 'DOWN':
      gesture_recognized='SWIPE DOWN'
    print("gess==",gesture_recognized)
    frame = cv2.putText(frame, "gesture:"+gesture_recognized, (100,100), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    frame = cv2.rectangle(frame, start_point, end_point, color, thickness)           
    out.write(frame)

  else:
    break

cap.release()
cv2.destroyAllWindows()
out.release()
    # Display the resulting frame

 
    # Press Q on keyboard to  exit
    #if cv2.waitKey(25) & 0xFF == ord('q'):
      #//break
 
  # Break the loop
  #else: 
    #break
 
# When everything done, release the video capture object
#cap.release()
 
# Closes all the frames
#cv2.destroyAllWindows()
