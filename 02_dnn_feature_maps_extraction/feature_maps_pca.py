"""Appending the feature maps of all DNN layers, standardizing and applying PCA.

Parameters
----------
dnn : str
	Used DNN architecture.
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
parser.add_argument('--n_components', default=1000, type=int)
parser.add_argument('--project_dir', default='/project/directory', type=str)
args = parser.parse_args()

print('>>> Applying PCA on the feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Applying PCA on the training images feature maps
# =============================================================================
# The standardization and PCA statistics computed on the training images feature
# maps are also applied to the test images feature maps and to the ILSVRC-2012
# images feature maps.
# Loading the feature maps
fmaps_train = []
fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'full_feature_maps', args.dnn, 'training_images')
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for f in fmaps_list:
	fmaps_data = np.load(os.path.join(fmaps_dir, f), allow_pickle=True).item()
	for l, dnn_layer in enumerate(fmaps_data.keys()):
		if l == 0:
			feats = np.reshape(fmaps_data[dnn_layer], -1)
		else:
			feats = np.append(feats, np.reshape(fmaps_data[dnn_layer], -1))
	fmaps_train.append(feats)
fmaps_train = np.asarray(fmaps_train)

# Standardizing the data
scaler = StandardScaler()
scaler.fit(fmaps_train)
fmaps_train = scaler.transform(fmaps_train)

# Applying PCA
pca = KernelPCA(n_components=args.n_components, kernel='poly', degree=4,
	random_state=20200220)
pca.fit(fmaps_train)
fmaps_train = pca.transform(fmaps_train)

# Creating the directory if not existing and saving
save_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'pca_feature_maps', args.dnn)
file_name = 'pca_feature_maps_training'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), fmaps_train)
del fmaps_train


# =============================================================================
# Applying PCA on the test images feature maps
# =============================================================================
# Loading the feature maps
fmaps_test = []
fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'full_feature_maps', args.dnn, 'test_images')
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for f in fmaps_list:
	fmaps_data = np.load(os.path.join(fmaps_dir, f), allow_pickle=True).item()
	for l, dnn_layer in enumerate(fmaps_data.keys()):
		if l == 0:
			feats = np.reshape(fmaps_data[dnn_layer], -1)
		else:
			feats = np.append(feats, np.reshape(fmaps_data[dnn_layer], -1))
	fmaps_test.append(feats)
fmaps_test = np.asarray(fmaps_test)

# Standardizing the data
fmaps_test = scaler.transform(fmaps_test)

# Applying PCA
fmaps_test = pca.transform(fmaps_test)

# Creating the directory if not existing and saving
save_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'pca_feature_maps', args.dnn)
file_name = 'pca_feature_maps_test'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), fmaps_test)
del fmaps_test


# =============================================================================
# Applying PCA on the ILSVRC-2012 validation images feature maps
# =============================================================================
# PCA is applied to partitions of 10k images feature maps for memory efficiency
# Loading the feature maps
n_img_part = 10000
fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'full_feature_maps', args.dnn, 'ILSVRC2012_img_val')
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for p in range(0, len(fmaps_list), n_img_part):
	fmaps_part = []
	for f in fmaps_list[p:p+n_img_part]:
		fmaps_data = np.load(os.path.join(fmaps_dir, f),
			allow_pickle=True).item()
		for l, dnn_layer in enumerate(fmaps_data.keys()):
			if l == 0:
				feats = np.reshape(fmaps_data[dnn_layer], -1)
			else:
				feats = np.append(feats, np.reshape(fmaps_data[dnn_layer], -1))
		fmaps_part.append(feats)
	fmaps_part = np.asarray(fmaps_part)

	# Standardizing the data
	fmaps_part = scaler.transform(fmaps_part)

	# Applying PCA
	fmaps_part = pca.transform(fmaps_part)
	if p == 0:
		fmaps_ilsvrc2012_val = fmaps_part
	else:
		fmaps_ilsvrc2012_val = np.append(fmaps_ilsvrc2012_val, fmaps_part, 0)

# Creating the directory if not existing and saving
save_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'pca_feature_maps', args.dnn)
file_name = 'pca_feature_maps_ilsvrc2012_val'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), fmaps_ilsvrc2012_val)
del fmaps_ilsvrc2012_val


# =============================================================================
# Applying PCA on the ILSVRC-2012 test images feature maps
# =============================================================================
# PCA is applied to partitions of 10k images feature maps for memory efficiency
# Loading the feature maps
n_img_part = 10000
fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'full_feature_maps', args.dnn, 'ILSVRC2012_img_test_v10102019')
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for p in range(0, len(fmaps_list), n_img_part):
	fmaps_part = []
	for f in fmaps_list[p:p+n_img_part]:
		fmaps_data = np.load(os.path.join(fmaps_dir, f),
			allow_pickle=True).item()
		for l, dnn_layer in enumerate(fmaps_data.keys()):
			if l == 0:
				feats = np.reshape(fmaps_data[dnn_layer], -1)
			else:
				feats = np.append(feats, np.reshape(fmaps_data[dnn_layer], -1))
		fmaps_part.append(feats)
	fmaps_part = np.asarray(fmaps_part)

	# Standardizing the data
	fmaps_part = scaler.transform(fmaps_part)

	# Applying PCA
	fmaps_part = pca.transform(fmaps_part)
	if p == 0:
		fmaps_ilsvrc2012_test = fmaps_part
	else:
		fmaps_ilsvrc2012_test = np.append(fmaps_ilsvrc2012_test, fmaps_part, 0)

# Creating the directory if not existing and saving
save_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'pca_feature_maps', args.dnn)
file_name = 'pca_feature_maps_ilsvrc2012_test'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), fmaps_ilsvrc2012_test)
