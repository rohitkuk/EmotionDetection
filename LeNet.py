"""

An Implementation of the LeNet Architecture by Yann lecun in referance to Video explanation: https://youtu.be/fcOW-Zyb5Bo.

PaperLink : http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

@Author: Rohit Kukreja
@Email : rohit.kukreja01@gmail.com

"""

import torch
import torch.nn as nn


class LeNet(nn.Module):
	def __init__(self, num_classes):
		super(LeNet, self).__init__()
		self.relu = nn.ReLU()
		self.pool = nn.AvgPool2d(kernel_size = (2,2), stride = (2,2))
		self.conv1	= nn.Conv2d(
								in_channels  = 1,
								out_channels = 6,
								kernel_size  = (5,5),
								stride       = (1,1),
								padding      = (0,0)
								)
		self.conv2 	= nn.Conv2d(
								in_channels  = 6,
								out_channels = 16,
								kernel_size  = (5,5),
								stride       = (1,1),
								padding      = (0,0)
								)
		self.conv3 	= nn.Conv2d(
								in_channels  = 16,
								out_channels = 120,
								kernel_size  = (5,5),
								stride       = (1,1),
								padding      = (0,0)
								)
		self.linear1     = nn.Linear(120, 84)
		self.linear2     = nn.Linear(84, num_classes)


	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.pool(x)

		x = self.conv2(x)
		x = self.relu(x)
		x = self.pool(x)

		x = self.conv3(x)
		x = self.relu(x)

		x = x.reshape(x.shape[0], -1)

		x = self.linear1(x)
		x = self.relu(x)
		x = self.linear2(x)
		return x


def test_net():
	device = "cuda" if torch.cuda.is_available() else "cpu"
	x = torch.randn(64, 1, 32, 32).to(device)
	model = LeNet(num_classes=10).to(device)
	return model(x)


if __name__ == '__main__':
	out = test_net()
	print(out.shape)