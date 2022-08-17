"""When extracting the CORnet-S feature maps the activations of each layer were
saved independently. Here, for each image, we load the feature maps of all
layers and store them together in one single dictionary. In this way the full
feature maps of all DNNs will be in the same format for the PCA downsampling.

Parameters
----------
pretrained : bool
	If True use a pretrained network, if false a randomly initialized one.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--project_dir', default='../project_directory', type=str)
args = parser.parse_args()

print('>>> Sort feature maps CORnet-S <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load, sort and save the CORnet-S feature maps
# =============================================================================
layers = ['V1', 'V2', 'V4', 'IT', 'decoder']
fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'full_feature_maps', 'cornet_s', 'pretrained-'+str(args.pretrained))
img_partitions = ['training_images', 'test_images', 'ILSVRC2012_img_val',
	'ILSVRC2012_img_test_v10102019']
num_partition_imgs = [16540, 200, 50000, 100000]

for p, part in enumerate(img_partitions):
	save_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
		'full_feature_maps', 'cornet_s', 'pretrained-'+str(args.pretrained),
		part)
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)
	for i in range(num_partition_imgs[p]):
		model_feats = {}
		for l in layers:
			model_feats[l] = np.asarray(np.load(os.path.join(fmaps_dir, part+
				'_individual_layers', part+'_layer-'+l+'_'+format(i+1, '07')+
				'.npy'), allow_pickle=True).item()[l])
		file_name = part + '_' + format(i+1, '07')
		np.save(os.path.join(save_dir, file_name), model_feats)
