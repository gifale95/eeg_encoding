"""Extracting and saving the CORnet-S feature maps of the training and test
images, and of the ILSVRC-2012 validation and test images.
Parameters
----------
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import cornet
from torchvision import transforms
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
# Get model helper function and importing the model
# =============================================================================
# Used CORnet-S layers
layers = ['V1', 'V2', 'V4', 'IT', 'decoder']
# Time steps of each layer
time_steps = [1, 2, 4, 2, 1]
# Sublayer for all layers
sublayer = 'output'
# Number of GPUs used
n_gpus = 0
# CORnet model used
cornet_model = 'S'

def get_model(pretrained=True):
	map_location = None if n_gpus > 0 else 'cpu'
	model = getattr(cornet, f'cornet_{cornet_model.lower()}')
	if cornet_model.lower() == 'r':
		model = model(pretrained=pretrained, map_location=map_location,
			times=args.times)
	else:
		model = model(pretrained=pretrained, map_location=map_location)
	if n_gpus == 0:
		model = model.module
	if n_gpus > 0:
		model = model.cuda()
	return model

model = get_model(pretrained=True)
model.eval()


# =============================================================================
# Defining the image preprocessing
# =============================================================================
transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =============================================================================
# Get feature maps helper function
# =============================================================================
def get_features(model, layer, sublayer, time_step, transform, img_dir):
	"""
	- model (CORnet model)
	- layers (choose from: V1, V2, V4, IT, decoder)
	- sublayer (e.g., output, conv1, avgpool)
	- time_step (which time step to use for storing features)
	- transform (image preprocessing variable)
	- img_dir (directory of single input images)

	"""

	def _store_feats(layer, inp, output):
		"""An ugly but effective way of accessing intermediate model features
		"""
		output = output.cpu().numpy()
		_model_feats.append(output)

	try:
		m = model.module
	except:
		m = model

	model_layer = getattr(getattr(m, layer), sublayer)
	model_layer.register_forward_hook(_store_feats)
	with torch.no_grad():
		im = Image.open(img_dir)
		im = transform(im)
		im = im.unsqueeze(0) # adding extra dimension for batch size of 1
		_model_feats = []
		model(im)
		model_feats = _model_feats[time_step]
	return model_feats


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
			if file.endswith(".jpg") or file.endswith(".JPEG"):
				image_list.append(os.path.join(root,file))
	image_list.sort()

	# Extracting and saving the feature maps
	idx = 1
	for image in image_list:
		feats = {}
		for l, layer in enumerate(layers):
			for time_step in range(time_steps[l]):
				features = get_features(model, layer, sublayer, time_step,
					transform, image)
				feats[layer+'_'+format(time_step,'02')] = features

		# Creating the directory if not existing and saving
		save_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
			'full_feature_maps', 'cornet-s', p)
		file_name = p + '_' + format(idx, '07')
		if os.path.isdir(save_dir) == False:
			os.makedirs(save_dir)
		np.save(os.path.join(save_dir, file_name), feats)
		idx += 1
