"""Extracting and saving the CORnet-S feature maps of the training and test
images, and of the ILSVRC-2012 validation and test images.
Parameters
----------
project_dir : str
	Directory of the project folder.

"""

import argparse
import math
from torch import nn
import os
from torch.autograd import Variable as V
from torchvision import transforms as trn
import torch
from PIL import Image
import numpy as np


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='/project/directory', type=str)
args = parser.parse_args()

print('\n\n\n>>> Extracting feature maps CORnet-S <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Importing the model
# =============================================================================
class Flatten(nn.Module):
	"""
	Helper module for flattening input tensor to 1-D for the use in Linear modules
	"""
	def forward(self, x):
		return x.view(x.size(0), -1)

class Identity(nn.Module):
	"""
	Helper module that stores the current tensor. Useful for accessing by name
	"""
	def forward(self, x):
		return x

class CORblock_S(nn.Module):
	scale = 4 # scale of the bottleneck convolution channels
	def __init__(self, in_channels, out_channels, times=1):
		super().__init__()
		self.times = times
		self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1,
			bias=False)
		self.skip = nn.Conv2d(out_channels, out_channels,
			kernel_size=1, stride=2, bias=False)
		self.norm_skip = nn.BatchNorm2d(out_channels)
		self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
			kernel_size=1, bias=False)
		self.nonlin1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels * self.scale, 
			out_channels * self.scale, kernel_size=3, stride=2, padding=1,
			bias=False)
		self.nonlin2 = nn.ReLU(inplace=True)
		self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
			kernel_size=1, bias=False)
		self.nonlin3 = nn.ReLU(inplace=True)
		self.output = Identity() # for an easy access to this block's output
		# need BatchNorm for each time step for training to work well
		for t in range(self.times):
			setattr(self, f'norm1_{t}', nn.BatchNorm2d(
				out_channels * self.scale))
			setattr(self, f'norm2_{t}', nn.BatchNorm2d(
				out_channels * self.scale))
			setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

	def forward(self, inp):
		x = self.conv_input(inp)
		for t in range(self.times):
			if t == 0:
				skip = self.norm_skip(self.skip(x))
				self.conv2.stride = (2, 2)
			else:
				skip = x
				self.conv2.stride = (1, 1)
			x = self.conv1(x)
			x = getattr(self, f'norm1_{t}')(x)
			x = self.nonlin1(x)
			x = self.conv2(x)
			x = getattr(self, f'norm2_{t}')(x)
			x = self.nonlin2(x)
			x = self.conv3(x)
			x = getattr(self, f'norm3_{t}')(x)
			x += skip
			x = self.nonlin3(x)
			output = self.output(x)
		return output

class CORnet_S(nn.Module):
	def __init__(self):
		super(CORnet_S, self).__init__()
		self.feat_list = ['V1', 'V2', 'V4', 'IT', 'decoder']
		# V1
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
			bias=False)
		self.norm1 = nn.BatchNorm2d(64)
		self.nonlin1 = nn.ReLU(inplace=True)
		self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.conv2 =  nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
			bias=False)
		self.norm2 = nn.BatchNorm2d(64)
		self.nonlin2 = nn.ReLU(inplace=True)
		self.output = Identity()
		# V2, V4, IT, decoder
		self.V2 = CORblock_S(64, 128, times = 2)
		self.V4 = CORblock_S(128, 256, times = 4)
		self.IT = CORblock_S(256, 512, times = 2)
		self.avgpool = nn.AdaptiveAvgPool2d((1))
		self.flatten = Flatten()
		self.linear = nn.Linear(512, 1000)
		# weight initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			# to add it during the training of this network
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, x):
		x = self.conv1(x)
		x = self.norm1(x)
		x = self.nonlin1(x)
		x = self.pool(x)
		x = self.conv2(x)
		x = self.norm2(x)
		x = self.nonlin2(x)
		x1 = self.output(x)
		x2 = self.V2(x1)
		x3 = self.V4(x2)
		x4 = self.IT(x3)
		x = self.avgpool(x4)
		x = self.flatten(x)
		x5 = self.linear(x)
		return x1, x2, x3, x4, x5

def cornet_s(pretrained=True , **kwargs):
	"""
	Pretrained weights(state_dict) are download from here:
	https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth
	"""
	pretrained_path = '../cornet_s-1d3f7974.pth'
	checkpoint = torch.load(pretrained_path, map_location="cpu")
	state_dict = checkpoint['state_dict']
	model = CORnet_S()
	if pretrained:
		model.load_state_dict(state_dict, strict=False)
	return model

model = cornet_s()
if torch.cuda.is_available():
	model.cuda()
model.eval()


# =============================================================================
# Defining the image preprocessing
# =============================================================================
centre_crop = trn.Compose([
	trn.Resize((224,224)),
	trn.ToTensor(),
	trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =============================================================================
# Loading the images and extracting the corresponding feature maps
# =============================================================================
# Extracting the feature maps of (1) training images, (2) test images,
# (3) ILSVRC-2012 validation images, (4) ILSVRC-2012 test images.
# Image directories
img_set_dir = os.path.join(args.project_dir, 'image_set')
img_partitions = os.listdir(img_set_dir)
for p in img_partitions:
	part_dir = os.path.join(img_set_dir, p)
	image_list = []
	for root, dirs, files in os.walk(part_dir):
		for file in files:
			if file.endswith(".jpg") or file.endswith(".JPEG"):
				image_list.append(os.path.join(root,file))
	image_list.sort()

	# Extracting and saving the feature maps
	idx = 1
	for image in image_list:
		img = Image.open(image).convert('RGB')
		filename=image.split("/")[-1].split(".")[0]
		input_img = V(centre_crop(img).unsqueeze(0))
		if torch.cuda.is_available():
			input_img=input_img.cuda()
		x = model.forward(input_img)
		feats={}
		for i,feat in enumerate(x):
			feats[model.feat_list[i]] = feat.data.cpu().numpy()

		# Creating the directory if not existing and saving
		save_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
			'full_feature_maps', 'cornet_s', p)
		file_name = p + '_' + format(idx, '07')
		if os.path.isdir(save_dir) == False:
			os.makedirs(save_dir)
		np.save(os.path.join(save_dir, file_name), feats)
		idx += 1
