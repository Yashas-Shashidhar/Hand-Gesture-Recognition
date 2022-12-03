import torch
import torch
from PIL import Image
import os
import glob 
import numpy as np
import torchvision.transforms as transforms
import cv2
from torch.utils.data import DataLoader
class_folders=['rock','paper','scissors']


#  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # randomly flip and rotate
    transforms.RandomRotation(10),
                                transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])


class Rpsdata(torch.utils.data.Dataset):
    def __init__(self, image_folder):
          # self.image_folder = "/Users/shreenidhir/Documents/Machine learning/project_18nov/rps_train_test_val_split/train/"  
        self.image_folder=image_folder
        self.images = glob.glob(image_folder+"/*/*")
    
        

    # get sample
    def __getitem__(self, idx):
        image_file = self.images[idx]
        image = Image.open((image_file))
        image = image.resize((300, 200))
        #image.save('myimage_myimg.jpg')
        #exit()
        #exit()
        #image = np.array(image)
        #cv2.imwrite("./shree_dat_it_here.png",image)

        image = transform(image)
        #image = (image)
        #image = np.array(image)
        
        #print("hereee====",image.shape)
        #exit()
        # normalize image
        #image = image / 255

        # convert to tensor
        image = image.reshape(3, 300, 200)

        
        # get the label, in this case the label was noted in the name of the image file, ie: 1_image_28457.png where 1 is the label and the number at the end is just the id or something

        target = np.array(class_folders.index((image_file.split("/")[-2])))
        #print(target.shape)
        target=np.expand_dims(target,axis=0)
        target = torch.Tensor(target)

        return image, target

    def __len__(self):
        return len(self.images)
#training_data=Rpsdata("/Users/shreenidhir/Documents/Machine learning/project_18nov/rps_train_test_val_split/train/")
#train_dataloader = DataLoader(training_data, batch_size=10, shuffle=True)
#x,y=next(iter(train_dataloader))
#print(x.shape)
#print(y.shape)
#trail1=x[0].numpy()
#print("p old===",trail1.shape)
#tail2=trail1.reshape(200,300,3)
#print("p===",tail2.shape)
#cv2.imwrite("./shree_dat_it11s.png",tail2)

#test_dataloader = Rpsdata(test_data, batch_size=64, shuffle=True)


    
