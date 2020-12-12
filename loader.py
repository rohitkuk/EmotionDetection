
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np 
import torch
import cv2

import torch
import numpy as np 
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import pandas as pd
#imports
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
import sys

from time import time


# from loader import FER2013Dataset
# Borrowed from a medium article 


class LinearNet(torch.nn.Module):
    def __init__(self, input_size,  num_classes):
        super(LinearNet, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.classifier = torch.nn.Linear(self.input_size, self.num_classes)

    def forward(self, x):
        # Resizing , keeping batch size separate and making everythng else a one D vector 

        x = x.view(x.size()[0] , -1)
        y = self.classifier(x)
        return y

input_size    = 1*48*48
num_classes   = 6
batch_size    = 32
learning_rate = 0.01
num_epochs    = 10

train_file = 'Dataset/Training.csv' 
val_file   = 'Dataset/PrivateTest.csv'
test_file  = 'Dataset/PublicTest.csv'


class FileReader:
    def __init__(self, csv_file_name):
        self._csv_file_name = csv_file_name
    def read(self):
        self._data = pd.read_csv(self._csv_file_name)



file_reader = FileReader('Dataset/Training.csv')
file_reader.read()



class Data:
    """
        Initialize the Data utility.
        :param data:
                    a pandas DataFrame containing data from the
                    FER2013 dataset.
        :type file_path:
                    DataFrame
        class variables:
        _x_train, _y_train:
                    Training data and corresopnding labels
        _x_test, _y_test:
                    Testing data and corresopnding labels
        _x_valid, _y_validation:
                    Validation/Development data and corresopnding labels

    """
    def __init__(self, data):
        self._x_train, self._y_train = [],  []
        self._x_test, self._y_test = [], []
        self._x_valid, self._y_valid = [], []

        for xdx, x in enumerate(data.values):
            pixels = []
            label = None
            for idx, i in enumerate(x[1].split(' ')):
                pixels.append(int(i))
            pixels = np.array(pixels).reshape((1, 48, 48))

            if x[2] == 'Training':
                self._x_train.append(pixels)
                self._y_train.append(int(x[0]))
            elif x[2] == 'PublicTest':
                self._x_test.append(pixels)
                self._y_test.append(int(x[0]))
            else:
                self._x_valid.append(pixels)
                self._y_valid.append(int(x[0]))
        self._x_train, self._y_train = np.array(self._x_train).reshape((len(self._x_train), 1, 48, 48)),\
            np.array(self._y_train, dtype=np.int64)
        self._x_test, self._y_test = np.array(self._x_test).reshape((len(self._x_test), 1, 48, 48)),\
            np.array(self._y_test, dtype=np.int64)
        self._x_valid, self._y_valid = np.array(self._x_valid).reshape((len(self._x_valid), 1, 48, 48)),\
            np.array(self._y_valid, dtype=np.int64)

data = Data(file_reader._data)


class FER2013Dataset(Dataset):
    """FER2013 Dataset."""

    def __init__(self, X, Y, transform=None):
        """
        Args:
            X (np array): Nx1x32x32.
            Y (np array): Nx1.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self._X = X
        self._y = Y

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        if self.transform:
          return {'inputs': self.transform(self._X[idx]), 'labels': self._y[idx]}
        return {'inputs': self._X[idx], 'labels': self._y[idx]}


preprocess = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(6),
    transforms.ColorJitter()
])


train_dataset = FER2013Dataset(data._x_train, data._y_train , transform=preprocess)
test_set = FER2013Dataset(data._x_valid, data._y_valid)


train_loader  = torch.utils.data.DataLoader(train_dataset	, batch_size = batch_size,  shuffle = True)



device = 'cuda'if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# Instantiate the Model
# model = FCNN(input_size, num_classes).to(device)
# model = FashionCNN().to(device)
model = LinearNet(input_size, num_classes).to(device)


print(model)

# Loss Function and Optimizers
criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr = learning_rate)



def train(epoch):
    # Initializing or setting to the training method
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Converting the tensor to be calculated on GPU if available
       	print(data.shape )
        print(target.shape)
        sys.exit()
        data, target = data.to(device), target.to(device)
        # Make sure optimizer has no gradients,  leaning the exisiting gradients.
        optimizer.zero_grad()
        # Giving model the data it calls the forward function and returns the output Forward pass - compute outputs on input data using the model
        output = model(data)
        # Calculate the loss
        loss = criterion(output, target)
        # back propagating the loss calclulating gradients and adjustmrnts
        loss.backward()
        # Updating Optimizing Parameters
        optimizer.step()
        # Logging the training progress to console.
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end = '\r')



def test():
    # Setting it up to the Eval ot test mode
    model.eval()
    # While testing we woould not need to calculte gradients hence it will same some computation.
    test_loss = 0
    correct   = 0
    with torch.no_grad():
        for data, target in val_loader:
            print("Evaluating...", end = '\r' )
            data, target = data.to(device), target.to(device)
            # reshaping the data as it has Color channel things        # 
            # Giving model the data it calls the forward function and returns the output Forward pass - compute outputs on input data using the model
            output = model(data)
            # Calculate the loss and appending the loss to make it a total figure
            test_loss += criterion(output, target).item()
            # Getting the highest probability Output.
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset) 
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def main():
    start = time()
    for i in range(num_epochs):
        train(i+1)
    print('\n Time Taken {:.2f} secs'. format(time()-start))
    start = time()
    print('\n Time Taken {:.2f} secs'. format(time()-start))
    # test()

if __name__ == '__main__':
    main()
    torch.save(model.state_dict(), "mnist_cnn.pt")








# class FER2013Dataset(Dataset):
# 	def __init__(self, file_name):
		# df = pd.read_csv(file_name, sep = ",")
		# df['pixels'] = df.apply(lambda x : np.array(x['pixels'].split(' '), dtype = np.float32).reshape(48,48), axis = 1)
		# self.X =torch.from_numpy(df.iloc[:,1:2].values)
		# self.Y =torch.from_numpy(df.iloc[:,0].values)
		# self.X = torch.tensor(df.iloc[:,1:2].values, dtype = torch.float32)
		# self.Y = torch.tensor(df.iloc[:,0].values)

# data = pd.read_csv('Dataset/Training.csv' )
# data = (data[data['pixels'].notnull()])
# pixels = data['pixels'].tolist()
# width, height = 48, 48
# faces = []
# for pixel_sequence in pixels:
#     face = [int(pixel) for pixel in pixel_sequence.split(' ')]
#     face = np.asarray(face).reshape(width, height)
#     face = cv2.resize(face.astype('uint8'),image_size)
#     faces.append(face.astype('float32'))

# faces = np.asarray(faces)
# faces = np.expand_dims(faces, -1)
# emotions = (data['emotion'])#.values
# # return faces, emotions

# 	def __len__(self):
# 		return len(self.Y)

# 	def __getitem__(self, idx):
# 		return self.X[idx], self.Y[idx] 

