def load_dnn_data(args):
	"""Load the DNN feature maps of the training and test images, and of the
	ILSVRC-2012 test and validation images.

	Parameters
	----------
	args : Namespace
		Input arguments.

	Returns
	-------
	X_train : float
		Training images feature maps.
	X_test : float
		Test images feature maps.
	X_ilsvrc2012_val : float
		ILSVRC-2012 validation images feature maps.
	X_ilsvrc2012_test : float
		ILSVRC-2012 test images feature maps.

	"""

	import numpy as np
	import os

	### Load the DNN feature maps ###
	# Feature maps directories
	if args.layers == 'all':
		data_dir = os.path.join('dnn_feature_maps', 'pca_feature_maps',
			args.dnn, 'pretrained-'+str(args.pretrained), 'layers-all')
	else:
		data_dir = os.path.join('dnn_feature_maps', 'pca_feature_maps',
			args.dnn, 'pretrained-'+str(args.pretrained), 'layers-single')
	training_file = 'pca_feature_maps_training.npy'
	test_file = 'pca_feature_maps_test.npy'
	ilsvrc2012_val_file = 'pca_feature_maps_ilsvrc2012_val.npy'
	ilsvrc2012_test_file = 'pca_feature_maps_ilsvrc2012_test.npy'
	# Load the feature maps
	X_train = np.load(os.path.join(args.project_dir, data_dir, training_file),
		allow_pickle=True).item()
	X_test = np.load(os.path.join(args.project_dir, data_dir, test_file),
		allow_pickle=True).item()
	X_ilsvrc2012_val = np.load(os.path.join(args.project_dir, data_dir,
		ilsvrc2012_val_file), allow_pickle=True).item()
	X_ilsvrc2012_test = np.load(os.path.join(args.project_dir, data_dir,
		ilsvrc2012_test_file), allow_pickle=True).item()

	### Append the PCA-downsampled feature maps of different layers ###
	if args.layers == 'appended':
		for l, layer in enumerate(X_train.keys()):
			if l == 0:
				train = X_train[layer]
				test = X_test[layer]
				ilsvrc2012_val = X_ilsvrc2012_val[layer]
				ilsvrc2012_test = X_ilsvrc2012_test[layer]
			else:
				train = np.append(train, X_train[layer], 1)
				test = np.append(test, X_test[layer], 1)
				ilsvrc2012_val = np.append(ilsvrc2012_val,
					X_ilsvrc2012_val[layer], 1)
				ilsvrc2012_test = np.append(ilsvrc2012_test,
					X_ilsvrc2012_test[layer], 1)
		X_train = {'appended_layers': train}
		X_test = {'appended_layers': test}
		X_ilsvrc2012_val = {'appended_layers': ilsvrc2012_val}
		X_ilsvrc2012_test = {'appended_layers': ilsvrc2012_test}

	### Retain only the selected amount of PCA components ###
	for layer in X_train.keys():
		X_train[layer] = X_train[layer][:,:args.n_components]
		X_test[layer] = X_test[layer][:,:args.n_components]
		X_ilsvrc2012_val[layer] = X_ilsvrc2012_val[layer][:,:args.n_components]
		X_ilsvrc2012_test[layer] = X_ilsvrc2012_test[layer][:,:args.n_components]

	### Output ###
	return X_train, X_test, X_ilsvrc2012_val, X_ilsvrc2012_test


def load_eeg_data(args):
	"""Load the EEG within subjects (the training data of the subject of
	interest) or between subjects (the averaged training data of the all other
	subjects except the subject of interest) data.

	Parameters
	----------
	args : Namespace
		Input arguments.

	Returns
	-------
	y_train : float
		Training EEG data.
	ch_names : list of str
		EEG channel names.
	times : float
		EEG time points.

	"""

	import os
	import numpy as np

	### Load the within subjects EEG training data ###
	y_train_within = []
	y_train_between = []
	for s in range(args.n_tot_sub):
		data_dir = os.path.join('eeg_dataset', 'preprocessed_data', 'sub-'+
			format(s+1,'02'), 'preprocessed_eeg_training.npy')
		data = np.load(os.path.join(args.project_dir, data_dir),
			allow_pickle=True).item()
		# Extract the data while averaging across repetitions
		if s+1 == args.sub:
			y_train_within.append(np.mean(data['preprocessed_eeg_data'], 1))
		else:
			y_train_between.append(np.mean(data['preprocessed_eeg_data'], 1))
		ch_names = data['ch_names']
		times = data['times']
		del data
	if args.subjects == 'within':
		y_train = np.asarray(y_train_within[0])
	elif args.subjects == 'between':
		y_train = np.mean(np.asarray(y_train_between), 0)

	### Output ###
	return y_train, ch_names, times


def perform_regression(args, ch_names, times, X_train, X_test, X_ilsvrc2012_val,
	X_ilsvrc2012_test, y_train):
	"""Train a linear regression on the training images DNN feature maps (X)
	and training EEG data (Y), and use the trained weights to synthesize the EEG
	responses to the training and test images (within and between subjects), and
	to the test/validation ILSVRC-2012 images (within subjects).

	Parameters
	----------
	args : Namespace
		Input arguments.
	ch_names : list of str
		EEG channel names.
	times : float
		EEG time points.
	X_train : float
		Training images feature maps.
	X_test : float
		Test images feature maps.
	X_ilsvrc2012_val : float
		ILSVRC-2012 validation images feature maps.
	X_ilsvrc2012_test : float
		ILSVRC-2012 test images feature maps.
	y_train : float
		Training EEG data.

	"""

	import numpy as np
	from ols import OLS_pytorch
	import os

	### Fit the regression at each time-point and channel ###
	eeg_shape = y_train.shape
	y_train = np.reshape(y_train, (y_train.shape[0],-1))
	# Within subjects
	synt_train = {}
	synt_test = {}
	synt_ilsvrc2012_val = {}
	synt_ilsvrc2012_test = {}
	for layer in X_train.keys():
		reg = OLS_pytorch(use_gpu=False)
		reg.fit(X_train[layer], y_train.T)
		synt_train[layer] = np.reshape(reg.predict(X_train[layer]),
			(X_train[layer].shape[0],eeg_shape[1],eeg_shape[2]))
		synt_test[layer] = np.reshape(reg.predict(X_test[layer]),
			(X_test[layer].shape[0],eeg_shape[1],eeg_shape[2]))
		synt_ilsvrc2012_val[layer] = np.reshape(reg.predict(
			X_ilsvrc2012_val[layer]), (X_ilsvrc2012_val[layer].shape[0],
			eeg_shape[1],eeg_shape[2]))
		synt_ilsvrc2012_test[layer] = np.reshape(reg.predict(
			X_ilsvrc2012_test[layer]), (X_ilsvrc2012_test[layer].shape[0],
			eeg_shape[1],eeg_shape[2]))

	### Put the data into dictionaries and save ###
	# Create the saving directories
	save_dir = os.path.join(args.project_dir, 'results', 'sub-'+
		format(args.sub,'02'), 'synthetic_eeg_data', 'encoding-linearizing',
		'subjects-'+args.subjects, 'dnn-'+args.dnn, 'pretrained-'+
		str(args.pretrained), 'layers-'+args.layers, 'n_components-'+
		format(args.n_components,'05'))
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	# Training data
	data_dict = {
		'synthetic_data': synt_train,
		'ch_names': ch_names,
		'times': times
		}
	file_name = 'synthetic_eeg_training.npy'
	np.save(os.path.join(save_dir, file_name), data_dict)
	# Test data
	data_dict = {
		'synthetic_data': synt_test,
		'ch_names': ch_names,
		'times': times
		}
	file_name = 'synthetic_eeg_test.npy'
	np.save(os.path.join(save_dir, file_name), data_dict)
	# ILSVRC-2012 validation data
	data_dict = {
		'synthetic_data': synt_ilsvrc2012_val,
		'ch_names': ch_names,
		'times': times
		}
	file_name = 'synthetic_eeg_ilsvrc2012_val.npy'
	np.save(os.path.join(save_dir, file_name), data_dict)
	# ILSVRC-2012 test data
	data_dict = {
		'synthetic_data': synt_ilsvrc2012_test,
		'ch_names': ch_names,
		'times': times
		}
	file_name = 'synthetic_eeg_ilsvrc2012_test.npy'
	np.save(os.path.join(save_dir, file_name), data_dict)
