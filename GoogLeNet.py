"""

An Implementation of the GoogLeNet in reference to Video explanation:https://www.youtube.com/watch?v=uQc4Fs7yx5I&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=21

PaperLink : https://arxiv.org/pdf/1409.4842.pdf

@Author: Rohit Kukreja
@Email : rohit.kukreja01@gmail.com

"""
import torch
import torch.nn as nn 


# first we will create conv_blocks then inception_block then GoogLeNet



class GoogLeNet(nn.Module):
    def __init__(self, in_channels= 3, num_classes=1000):
        super(GoogLeNet, self).__init__()
        
        self.conv1          = conv_block(in_channels = in_channels, out_channels = 64, kernel_size = (7,7), stride = 2, padding = (3,3))
        self.max_pool       = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = (1,1))

        self.conv2          = conv_block(in_channels = 64, out_channels = 192, kernel_size = (3,3), stride = 1, padding = (1,1))
        self.max_pool1      = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = (1,1))

        self.inception_3a   = InceptionModule(in_channels = 192, out_1x1 = 64, red_3x3 = 96, out_3x3 = 128, red_5x5 = 16, out_5x5= 32, out_pool_1x1 = 32 )
        self.inception_3b   = InceptionModule(in_channels = 256, out_1x1 = 128, red_3x3 = 128, out_3x3 = 192, red_5x5 = 32, out_5x5= 96, out_pool_1x1 = 64 )

        self.max_pool2      = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = (1,1))

        self.inception_4a   = InceptionModule(in_channels = 480, out_1x1 = 192, red_3x3 = 96 , out_3x3 = 208, red_5x5 = 16, out_5x5= 48 , out_pool_1x1 = 64  )
        self.inception_4b   = InceptionModule(in_channels = 512, out_1x1 = 160, red_3x3 = 112, out_3x3 = 224, red_5x5 = 24, out_5x5= 64 , out_pool_1x1 = 64  )
        self.inception_4c   = InceptionModule(in_channels = 512, out_1x1 = 128, red_3x3 = 128, out_3x3 = 256, red_5x5 = 24, out_5x5= 64 , out_pool_1x1 = 64  )
        self.inception_4d   = InceptionModule(in_channels = 512, out_1x1 = 112, red_3x3 = 144, out_3x3 = 288, red_5x5 = 32, out_5x5= 64 , out_pool_1x1 = 64  )
        self.inception_4e   = InceptionModule(in_channels = 528, out_1x1 = 256, red_3x3 = 160, out_3x3 = 320, red_5x5 = 32, out_5x5= 128, out_pool_1x1 = 128 )

        self.max_pool3      = nn.MaxPool2d( kernel_size = (3,3), stride = 2, padding = (1,1))

        self.inception_5a   = InceptionModule(in_channels = 832, out_1x1 = 256, red_3x3 = 160, out_3x3 = 320, red_5x5 = 32, out_5x5= 128 , out_pool_1x1 = 128  )
        self.inception_5b   = InceptionModule(in_channels = 832, out_1x1 = 384, red_3x3 = 192, out_3x3 = 384, red_5x5 = 48, out_5x5= 128 , out_pool_1x1 = 128  )

        self.avgpool        = nn.AvgPool2d(kernel_size = (7,7), stride = 1) #try with padding 1 also
        self.dropout        = nn.Dropout(p=0.4)
        self.fc1            = nn.Linear(1024,1000) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool1(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.max_pool2(x)
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.max_pool3(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool(x)
        print(x.shape)
        x = x.reshape(x.shape[0], -1) #Falttning the data for FC try doing it after dropout
        print(x.shape)
        x = self.dropout(x)
        print(x.shape)

        x = self.fc1(x)
        return x

# Biggst Mistake in my life to assume padding to be default 1 IT IS 0 please  don't forget this 
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool_1x1):
        super(InceptionModule, self).__init__()

        # self.in_channels = in_channels

        self.branch1 = conv_block(in_channels = in_channels, out_channels = out_1x1, kernel_size = (1,1), stride = 1, padding  = 0 ) 

        self.branch2 = nn.Sequential(
            conv_block(in_channels = in_channels, out_channels = red_3x3, kernel_size = (1,1), stride = 1, padding  = 0 ),
            conv_block(in_channels = red_3x3, out_channels = out_3x3, kernel_size = (3,3), stride = 1, padding  = 1 )
            )

        self.branch3 = nn.Sequential(
            conv_block(in_channels = in_channels, out_channels = red_5x5, kernel_size = (1,1), stride = 1, padding  = 0 ),
            conv_block(in_channels = red_5x5, out_channels = out_5x5, kernel_size = (5,5), stride = 1, padding  = 2 ) # for kernel 5 padding will be 5 
            )       
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = (3,3), stride  = 1, padding = 1),
            conv_block(in_channels = in_channels, out_channels = out_pool_1x1, kernel_size = (1,1), stride = 1, padding  = 0 ) # for kernel 5 padding will be 5 
            )        
        
    def forward(self, x):
        # print("I am here ")
        # Concatenating all layers output
        # size  B X N X H X W i.e, Batch, Num_Channels, Height, Width
        # 1 is for dimension 0 is  batch size N for Num_Channels coz we want to combine channels from all layers
        return torch.cat([self.branch1(x), self.branch2(x),  self.branch3(x) ,  self.branch4(x)], 1)  


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x




def test_net():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    x = torch.randn(1, 3, 224, 224).to(device)
    model = GoogLeNet().to(device)
    return model(x)


if __name__ == '__main__':
    out = test_net()
    print(out.shape)
