"""PCA is performed on the DNN feature maps to reduce their dimensionality.
PCA is applied on either the feature maps of single DNN layers, or on the
appended feature maps of all layers.

Parameters
----------
dnn : str
	Used DNN among 'alexnet', 'resnet50', 'cornet_s', 'moco'.
pretrained : bool
	If True use the pretrained network feature maps, if False use the randomly
	initialized network feature maps.
layers : str
	Whether to use 'all' or 'single' layers.
n_components : int
	Number of PCA components retained.
project_dir : str
	Directory of the project folder.

"""

import argparse
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layers', default='single', type=str)
parser.add_argument('--n_components', default=10000, type=int)
parser.add_argument('--project_dir', default='../project/directory', type=str)
args = parser.parse_args()

print('>>> Apply PCA on the feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220


# =============================================================================
# Apply PCA on the training images feature maps
# =============================================================================
# The standardization and PCA statistics computed on the training images feature
# maps are also applied to the test images feature maps and to the ILSVRC-2012
# images feature maps.

# Load the feature maps
if args.layers == 'all':
	fmaps_train = []
elif args.layers == 'single':
	feats = []
	fmaps_train = {}
fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'full_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained),
	'training_images')
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for f, fmaps in enumerate(fmaps_list):
	fmaps_data = np.load(os.path.join(fmaps_dir, fmaps),
		allow_pickle=True).item()
	if f == 0:
		layer_names = fmaps_data.keys()
	for l, dnn_layer in enumerate(layer_names):
		if args.layers == 'all':
			if l == 0:
				feats = np.reshape(fmaps_data[dnn_layer], -1)
			else:
				feats = np.append(feats, np.reshape(fmaps_data[dnn_layer], -1))
		elif args.layers == 'single':
			if f == 0:
				feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
			else:
				feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
	if args.layers == 'all':
		fmaps_train.append(feats)
if args.layers == 'all':
	fmaps_train = np.asarray(fmaps_train)
elif args.layers == 'single':
	for l, dnn_layer in enumerate(layer_names):
		fmaps_train[dnn_layer] = np.squeeze(np.asarray(feats[l]))

# Standardize the data
if args.layers == 'all':
	scaler = StandardScaler()
	scaler.fit(fmaps_train)
	fmaps_train = scaler.transform(fmaps_train)
elif args.layers == 'single':
	scaler = []
	for l, layer in enumerate(layer_names):
		scaler.append(StandardScaler())
		scaler[l].fit(fmaps_train[layer])
		fmaps_train[layer] = scaler[l].transform(fmaps_train[layer])

# Apply PCA
if args.layers == 'all':
	pca = KernelPCA(n_components=args.n_components, kernel='poly', degree=4,
		random_state=seed)
	pca.fit(fmaps_train)
	fmaps_train = pca.transform(fmaps_train)
elif args.layers == 'single':
	pca = []
	for l, layer in enumerate(layer_names):
		pca.append(KernelPCA(n_components=args.n_components, kernel='poly',
			degree=4, random_state=seed))
		pca[l].fit(fmaps_train[layer])
		fmaps_train[layer] = pca[l].transform(fmaps_train[layer])

# Save the downsampled feature maps
save_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'pca_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained), 'layers-'+
	args.layers)
file_name = 'pca_feature_maps_training'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), fmaps_train)
del fmaps_train


# =============================================================================
# Apply PCA on the test images feature maps
# =============================================================================
# Load the feature maps
if args.layers == 'all':
	fmaps_test = []
elif args.layers == 'single':
	feats = []
	fmaps_test = {}
fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'full_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained),
	'test_images')
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for f, fmaps in enumerate(fmaps_list):
	fmaps_data = np.load(os.path.join(fmaps_dir, fmaps),
		allow_pickle=True).item()
	if f == 0:
		layer_names = fmaps_data.keys()
	for l, dnn_layer in enumerate(layer_names):
		if args.layers == 'all':
			if l == 0:
				feats = np.reshape(fmaps_data[dnn_layer], -1)
			else:
				feats = np.append(feats, np.reshape(fmaps_data[dnn_layer], -1))
		elif args.layers == 'single':
			if f == 0:
				feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
			else:
				feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
	if args.layers == 'all':
		fmaps_test.append(feats)
if args.layers == 'all':
	fmaps_test = np.asarray(fmaps_test)
elif args.layers == 'single':
	for l, dnn_layer in enumerate(layer_names):
		fmaps_test[dnn_layer] = np.squeeze(np.asarray(feats[l]))

# Standardize the data
if args.layers == 'all':
	fmaps_test = scaler.transform(fmaps_test)
elif args.layers == 'single':
	for l, layer in enumerate(layer_names):
		fmaps_test[layer] = scaler[l].transform(fmaps_test[layer])

# Apply PCA
if args.layers == 'all':
	fmaps_test = pca.transform(fmaps_test)
elif args.layers == 'single':
	for l, layer in enumerate(layer_names):
		fmaps_test[layer] = pca[l].transform(fmaps_test[layer])

# Save the downsampled feature maps
file_name = 'pca_feature_maps_test'
np.save(os.path.join(save_dir, file_name), fmaps_test)
del fmaps_test


# =============================================================================
# Apply PCA on the ILSVRC-2012 validation images feature maps
# =============================================================================
# PCA is applied to partitions of 10k images feature maps for memory efficiency.

# Load the feature maps
n_img_part = 10000
fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'full_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained),
	'ILSVRC2012_img_val')
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
if args.layers == 'single':
	fmaps_ilsvrc2012_val = {}
for p in range(0, len(fmaps_list), n_img_part):
	if args.layers == 'all':
		fmaps_part = []
	elif args.layers == 'single':
		feats = []
		fmaps_part = {}
	for f, fmaps in enumerate(fmaps_list[p:p+n_img_part]):
		fmaps_data = np.load(os.path.join(fmaps_dir, fmaps),
			allow_pickle=True).item()
		if f == 0:
			layer_names = fmaps_data.keys()
		for l, dnn_layer in enumerate(layer_names):
			if args.layers == 'all':
				if l == 0:
					feats = np.reshape(fmaps_data[dnn_layer], -1)
				else:
					feats = np.append(feats, np.reshape(fmaps_data[dnn_layer], -1))
			elif args.layers == 'single':
				if f == 0:
					feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
				else:
					feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
		if args.layers == 'all':
			fmaps_part.append(feats)
	if args.layers == 'all':
		fmaps_part = np.asarray(fmaps_part)
	elif args.layers == 'single':
		for l, dnn_layer in enumerate(layer_names):
			fmaps_part[dnn_layer] = np.squeeze(np.asarray(feats[l]))
	
	# Standardize the data
	if args.layers == 'all':
		fmaps_part = scaler.transform(fmaps_part)
	elif args.layers == 'single':
		for l, layer in enumerate(layer_names):
			fmaps_part[layer] = scaler[l].transform(fmaps_part[layer])

	# Apply PCA
	if args.layers == 'all':
		fmaps_part = pca.transform(fmaps_part)
		if p == 0:
			fmaps_ilsvrc2012_val = fmaps_part
		else:
			fmaps_ilsvrc2012_val = np.append(fmaps_ilsvrc2012_val, fmaps_part,
				0)
	elif args.layers == 'single':
		for l, layer in enumerate(layer_names):
			if p == 0:
				fmaps_ilsvrc2012_val[layer] = pca[l].transform(
					fmaps_part[layer])
			else:
				fmaps_ilsvrc2012_val[layer] = np.append(
					fmaps_ilsvrc2012_val[layer], pca[l].transform(
					fmaps_part[layer]))

# Save the downsampled feature maps
file_name = 'pca_feature_maps_ilsvrc2012_val'
np.save(os.path.join(save_dir, file_name), fmaps_ilsvrc2012_val)
del fmaps_ilsvrc2012_val


# =============================================================================
# Apply PCA on the ILSVRC-2012 test images feature maps
# =============================================================================
# PCA is applied to partitions of 10k images feature maps for memory efficiency.

# Load the feature maps
fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'full_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained),
	'ILSVRC2012_img_test_v10102019')
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
if args.layers == 'single':
	fmaps_ilsvrc2012_test = {}
for p in range(0, len(fmaps_list), n_img_part):
	if args.layers == 'all':
		fmaps_part = []
	elif args.layers == 'single':
		feats = []
		fmaps_part = {}
	for f, fmaps in enumerate(fmaps_list[p:p+n_img_part]):
		fmaps_data = np.load(os.path.join(fmaps_dir, fmaps),
			allow_pickle=True).item()
		if f == 0:
			layer_names = fmaps_data.keys()
		for l, dnn_layer in enumerate(layer_names):
			if args.layers == 'all':
				if l == 0:
					feats = np.reshape(fmaps_data[dnn_layer], -1)
				else:
					feats = np.append(feats, np.reshape(fmaps_data[dnn_layer], -1))
			elif args.layers == 'single':
				if f == 0:
					feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
				else:
					feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
		if args.layers == 'all':
			fmaps_part.append(feats)
	if args.layers == 'all':
		fmaps_part = np.asarray(fmaps_part)
	elif args.layers == 'single':
		for l, dnn_layer in enumerate(layer_names):
			fmaps_part[dnn_layer] = np.squeeze(np.asarray(feats[l]))
	
	# Standardize the data
	if args.layers == 'all':
		fmaps_part = scaler.transform(fmaps_part)
	elif args.layers == 'single':
		for l, layer in enumerate(layer_names):
			fmaps_part[layer] = scaler[l].transform(fmaps_part[layer])

	# Apply PCA
	if args.layers == 'all':
		fmaps_part = pca.transform(fmaps_part)
		if p == 0:
			fmaps_ilsvrc2012_test = fmaps_part
		else:
			fmaps_ilsvrc2012_test = np.append(fmaps_ilsvrc2012_test, fmaps_part,
				0)
	elif args.layers == 'single':
		for l, layer in enumerate(layer_names):
			if p == 0:
				fmaps_ilsvrc2012_test[layer] = pca[l].transform(
					fmaps_part[layer])
			else:
				fmaps_ilsvrc2012_test[layer] = np.append(
					fmaps_ilsvrc2012_test[layer], pca[l].transform(
					fmaps_part[layer]))

# Save the downsampled feature maps
file_name = 'pca_feature_maps_ilsvrc2012_test'
np.save(os.path.join(save_dir, file_name), fmaps_ilsvrc2012_test)
