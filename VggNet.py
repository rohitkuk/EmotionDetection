"""

An Implementation of the VGG (Visual Geometry Group) NET by by K. Simonyan and A. Zisserman reference to Video explanation:https://www.youtube.com/watch?v=ACmuBbuXn20&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=20

PaperLink : https://arxiv.org/pdf/1409.1556.pdf

@Author: Rohit Kukreja
@Email : rohit.kukreja01@gmail.com

"""
import torch
import torch.nn as nn 





class VggNet(nn.Module):

	def __init__(self, in_channels, num_classes, architecture_type):
		super(VggNet, self).__init__()
		self.architecture_type = architecture_type
		self.in_channels = in_channels
		self.VGG_types = {
		    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
		    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
		    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
		    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
		}
		self.conv_layers = self.make_conv_setup(self.VGG_types[architecture_type])
		self.fcs = nn.Sequential(

				# """
				# 	512 channels for Convolution layer and 7 X 7 image dimension 
				# 	image size was 224 x 224 & 5 times max pool 224/2/2/2/2/2 = 7 hence 7*7
				# 	That means Current dimension is 7*7 and we are flatting the layers to give 
				# 	it to Linear FC
				# """
				nn.Linear(512 * 7 * 7,4096),
				nn.ReLU(),
				nn.Dropout(p = 0.5),
				nn.Linear(4096,4096),
				nn.ReLU(),
				nn.Dropout(p = 0.5),
				nn.Linear(4096,num_classes)
			)



	def forward(self, x):
		x = self.conv_layers(x)
		x = x.reshape(x.shape[0], -1) # We do this before the first FC layer to give it flatten input
		x = self.fcs(x)
		return x


	def make_conv_setup(self,architecture):
		layers = []
		# The Variable name is kept same for simplicity but it is not same as the class variable. 
		# The scope is different hence only scoped to this function.
		in_channels = self.in_channels
		
		for wall in architecture:
			if isinstance(wall, int):
				layer = nn.Conv2d(
					in_channels  = in_channels,
					out_channels = wall,
					kernel_size  = (3,3),
					padding      = (1,1),
					stride       = (1,1)
					)

				layers.append(layer)
				layers.append(nn.BatchNorm2d(wall))
				layers.append(nn.ReLU())
				in_channels = wall
			elif isinstance(wall, str) and wall =="M":
				layers.append(nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))

		return nn.Sequential(*layers)


def test_net(architecture_type):
	print(f"Running VGG on architecture :: {architecture_type}")
	device = "cuda" if torch.cuda.is_available() else "cpu"
	# device = "cpu"
	# batch size more than 8	 is causing CUDA out of memory error
	x = torch.randn(8, 3, 224, 224).to(device)
	model = VggNet(in_channels = 3, architecture_type = architecture_type, num_classes = 1000).to(device)
	return model(x)


if __name__ == '__main__':
	architectures = ["VGG11","VGG13","VGG16","VGG19"]	
	for architecture in architectures:
		out = test_net(architecture)
		print(out.shape)
