
# TO DO

# -- Download the data 
# -- Extract the file
# -- Split the File

# Pytorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

#General Modules
from PIL import Image
from tqdm import tqdm
from time import time, sleep
import pandas as pd
import numpy as np 
import os
import sys


START_SESSION = time()

class dataParser:
    file_name = 'Dataset/fer2013.csv'
    df = pd.read_csv(file_name)
    train = df[df["Usage"] == "Training"]
    val   = df[df["Usage"] == "PublicTest"]
    test  = df[df["Usage"] == "PublicTest"]


class FER2013Dataset(Dataset):
    def __init__(self, data_obj):
        self.df = data_obj
        self.imgdata = []
        self.arr = None 

        for i in self.df['pixels'].tolist():
            self.arr = np.array([int(j) for j in i.split(' ')])

            self.arr = self.arr.reshape((48,48))
            self.arr = Image.fromarray(self.arr)
            self.arr = self.arr.resize((32, 32), Image.BILINEAR) 
            self.arr = np.array(self.arr.getdata())
            self.arr = self.arr.reshape((1,32,32))
            self.imgdata.append(self.arr)

        self.imgdata = np.array(self.imgdata)

        self.X =torch.tensor(self.imgdata,  dtype = torch.float32)
        self.Y =torch.tensor(self.df['emotion'].values)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  
        return self.X[idx], self.Y[idx] 


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


def save_checkpoint(state, filename):
    print("Saving model checkpoint")
    torch.save(state, filename)


def load_checkpoint(model, optimizer, checkpoint):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


FerData = dataParser()


input_size    = 1*32*32
num_classes   = 7
batch_size    = 32
learning_rate = 0.01
num_epochs    = 10
checkpoint_file = "my_checkpoint.pth.tar"
load_model    = True if os.path.isfile("my_checkpoint.pth.tar") else False


print(FerData.train.shape)
print(FerData.val.shape)
print(FerData.test.shape)


train_dataset = FER2013Dataset(FerData.train)  
val_dataset   = FER2013Dataset(FerData.val)
test_dataset  = FER2013Dataset(FerData.test)

# train_loader  = DataLoader(train_dataset    , batch_size = batch_size,  shuffle = True)
# val_loader    = DataLoader(val_dataset      , batch_size = batch_size,  shuffle = True)
# test_loader   = DataLoader(test_dataset     , batch_size = batch_size,  shuffle = True)



# check if cuda is available
device = 'cuda'if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# Instantiate the Model
# model = FCNN(input_size, num_classes).to(device)
# model = FashionCNN().to(device)   
model = LinearNet(input_size, num_classes).to(device)


# Loss Function and Optimizers





batch_sizes = [32]
learning_rates = [3e-4]



def train(): 
    for batch_size in batch_sizes:
        for learning_rate in learning_rates: 
            criterion  = nn.CrossEntropyLoss()
            optimizer  = optim.Adam(model.parameters(), lr = learning_rate)
            # if load_model:
            #     load_checkpoint(model, optimizer, torch.load(checkpoint_file))
            train_loader  = DataLoader(train_dataset    , batch_size = batch_size,  shuffle = True)
            val_loader    = DataLoader(val_dataset      , batch_size = batch_size,  shuffle = True)
            test_loader   = DataLoader(test_dataset     , batch_size = batch_size,  shuffle = True)

            writer = SummaryWriter(f"runs/EmotionDetection LR {learning_rate} BS {batch_size}") 
            losses = []
            accuracies = []
            step = 0
            print(f"Training for Learning Rate :: {learning_rate} Batch Size :: {batch_size} ")
            for epoch in range(num_epochs):                # Initializing or setting to the training method
                model.train()
                
                loop = tqdm(enumerate(train_loader), total = len(train_loader), leave = False)

                for batch_idx, (data, target) in loop:

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

                    # Appending Loss & Accuracy, Values
                    losses.append(loss.item())
                    # Logging the training progress to console.
                    # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #         epoch, batch_idx * len(data), len(train_loader.dataset),
                    #         100. * batch_idx / len(train_loader), loss.item()), end = '\r')
                    loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                    loop.set_postfix(loss = loss.item())
                    
                    # For tensorboard
                    writer.add_scalar('Trainig loss', loss, global_step = step)
                    step +=1
                # if epoch % 3==0:
                #     checkpoint = {"state_dict": model.state_dict(), "optimizer" : optimizer.state_dict()}
                #     save_checkpoint(checkpoint, checkpoint_file)

        writer.add_hparams({'lr':learning_rate, 'bsize' : batch_size,},
                            {'loss' : sum(losses)/len(losses)}
            )


def test():
    # Setting it up to the Eval ot test mode
    train_loader  = DataLoader(train_dataset    , batch_size = batch_sizes[0],  shuffle = True)
    val_loader    = DataLoader(val_dataset      , batch_size = batch_sizes[0],  shuffle = True)
    test_loader   = DataLoader(test_dataset     , batch_size = batch_sizes[0],  shuffle = True)
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(model.parameters(), lr = learning_rates[0])
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

    test_loss /= len(val_loader.dataset) 
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    sleep(0.002)


def main():
    start = time()
    train()
    print('\n Time Taken {:.2f} secs'. format(time()-start))
    start = time()
    print('\n Time Taken {:.2f} secs'. format(time()-start))
    test()

if __name__ == '__main__':
    main()
    print(f"TOOK {time()-START_SESSION} SECS TO RUN ENTIRE MODULE !!!")
    # torch.save(model.state_dict(), "mnist_cnn.pt")






