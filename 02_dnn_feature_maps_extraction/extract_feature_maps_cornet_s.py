"""Extract and save the CORnet-S feature maps of the training and test images,
and of the ILSVRC-2012 validation and test images.
Before running, install the "cornet" package from:
https://github.com/dicarlolab/CORnet.

Parameters
----------
layer : str
	Used CORnet-S layer from ['V1', 'V2', 'V4', 'IT', 'decoder'].
pretrained : bool
	If True use a pretrained network, if False a randomly initialized one.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import torch
import cornet
from torch.autograd import Variable as V
from torchvision import transforms as trn
from PIL import Image
import numpy as np


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layer', default='V1', type=str)
parser.add_argument('--project_dir', default='../project/directory', type=str)
args = parser.parse_args()

print('>>> Extract feature maps CORnet-S <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)


# =============================================================================
# Import the model
# =============================================================================
def get_model(pretrained=args.pretrained):
	map_location = None if torch.cuda.is_available() else 'cpu'
	model = getattr(cornet, 'cornet_s')
	model = model(pretrained=pretrained, map_location=map_location)
	if torch.cuda.is_available():
		model = model.cuda()
	else:
		model = model.module # remove DataParallel
	return model

model = get_model()
model.eval()


# =============================================================================
# Define the image preprocessing
# =============================================================================
centre_crop = trn.Compose([
	trn.Resize((224,224)),
	trn.ToTensor(),
	trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =============================================================================
# Load the images and extract the corresponding feature maps
# =============================================================================
# Extracting the feature maps of (1) training images, (2) test images,
# (3) ILSVRC-2012 validation images, (4) ILSVRC-2012 test images.
sublayer = 'output'

def _store_feats(layer, inp, output):
	"""An ugly but effective way of accessing intermediate model features
	"""
	output = output.cpu().numpy()
	#_model_feats = []
	_model_feats.append(np.reshape(output, (len(output), -1)))

try:
	m = model.module
except:
	m = model

model_layer = getattr(getattr(m, args.layer), sublayer)
model_layer.register_forward_hook(_store_feats)

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
	# Create the saving directory if not existing
	save_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
		'full_feature_maps', 'cornet_s', 'pretrained-'+str(args.pretrained), p)
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)

	# Extract and save the feature maps
	for i, image in enumerate(image_list):
		img = Image.open(image).convert('RGB')
		filename=image.split("/")[-1].split(".")[0]
		input_img = V(centre_crop(img).unsqueeze(0))
		if torch.cuda.is_available():
			input_img=input_img.cuda()
		model_feats = {}
		with torch.no_grad():
			_model_feats = []
			model(input_img)
			# Store the feature maps of all time steps
			model_feats[args.layer] = _model_feats
		file_name = p + '_layer-' + args.layer + '_' + format(i+1, '07')
		np.save(os.path.join(save_dir, file_name), model_feats)
