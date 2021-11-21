def load_dnn_data(args):
	"""Loading the DNN feature maps of the training and test data, and of the
	ILSVRC-2012 test and validation data.

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

	### Loading the DNN feature maps ###
	# Feature maps directories
	print('\n>>> Loading the DDN feature maps <<<')
	data_dir = os.path.join('dnn_feature_maps', 'pca_feature_maps', args.dnn)
	training_file = 'pca_feature_maps_training.npy'
	test_file = 'pca_feature_maps_test.npy'
	ilsvrc2012_val_file = 'pca_feature_maps_ilsvrc2012_val.npy'
	ilsvrc2012_test_file = 'pca_feature_maps_ilsvrc2012_test.npy'
	# Loading the feature maps
	X_train = np.load(os.path.join(args.project_dir, data_dir, training_file))
	X_test = np.load(os.path.join(args.project_dir, data_dir, test_file))
	X_ilsvrc2012_val = np.load(os.path.join(args.project_dir, data_dir,
		ilsvrc2012_val_file))
	X_ilsvrc2012_test = np.load(os.path.join(args.project_dir, data_dir,
		ilsvrc2012_test_file))

	### Output ###
	return X_train, X_test, X_ilsvrc2012_val, X_ilsvrc2012_test


def load_eeg_data(args):
	"""Loading the EEG within subjects (the training data of the subject of
	interest) and between subjects (the training data of the all other subjects
	except the subject of interest) data.

	Parameters
	----------
	args : Namespace
		Input arguments.

	Returns
	-------
	y_train_within : float
		Within subjects training EEG data.
	y_train_between : float
		Between subjects training EEG data.
	ch_names : list of str
		EEG channel names.
	times : float
		EEG time points.

	"""

	import os
	import numpy as np

	### Loading the EEG data ###
	print('\n>>> Loading the EEG data <<<')
	y_train_within = []
	y_train_between = []
	for s in range(args.n_tot_sub):
		data_dir = os.path.join('eeg_dataset', 'preprocessed_data', 'sub-'+
			format(s+1,'02'), 'preprocessed_eeg_training.npy')
		data = np.load(os.path.join(args.project_dir, data_dir),
			allow_pickle=True).item()
		# Extracting the data while averaging across repetitions
		if s+1 == args.sub:
			y_train_within.append(np.mean(data['preprocessed_eeg_data'], 1))
		else:
			y_train_between.append(np.mean(data['preprocessed_eeg_data'], 1))
		ch_names = data['ch_names']
		times = data['times']
		del data
	# Averaging the between subjects data across subjects
	y_train_within = np.asarray(y_train_within[0])
	y_train_between = np.mean(np.asarray(y_train_between), 0)

	### Output ###
	return y_train_within, y_train_between, ch_names, times


def perform_regression(args, ch_names, times, X_train, X_test, X_ilsvrc2012_val,
	X_ilsvrc2012_test, y_train_within, y_train_between):
	"""Training a linear regression on the training images DNN feature maps (X)
	and training EEG data (Y), and use the trained weights to synthesize the EEG
	training and test images (within and between subjects) and test/validation
	ILSVRC-2012 images (within subjects).

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
	y_train_within : float
		Within subjects training EEG data.
	y_train_between : float
		Between subjects training EEG data.

	"""

	import numpy as np
	from ols import OLS_pytorch
	import os

	### Fitting the regression at each time-point and channel ###
	print('\n>>> Performing the regression <<<')
	eeg_shape = y_train_within.shape
	y_train_within = np.reshape(y_train_within, (y_train_within.shape[0],-1))
	y_train_between = np.reshape(y_train_between, (y_train_between.shape[0],-1))
	# Within subjects
	reg_within = OLS_pytorch(use_gpu=False)
	reg_within.fit(X_train, y_train_within.T)
	synt_train_within = np.reshape(reg_within.predict(X_train),
		(X_train.shape[0],eeg_shape[1], eeg_shape[2]))
	synt_test_within = np.reshape(reg_within.predict(X_test), (X_test.shape[0],
		eeg_shape[1],eeg_shape[2]))
	synt_ilsvrc2012_val_within = np.reshape(reg_within.predict(
		X_ilsvrc2012_val), (X_ilsvrc2012_val.shape[0],eeg_shape[1],
		eeg_shape[2]))
	synt_ilsvrc2012_test_within = np.reshape(reg_within.predict(
		X_ilsvrc2012_test), (X_ilsvrc2012_test.shape[0],eeg_shape[1],
		eeg_shape[2]))
	del reg_within
	# Between subjects
	reg_between = OLS_pytorch(use_gpu=False)
	reg_between.fit(X_train, y_train_between.T)
	synt_train_between = np.reshape(reg_between.predict(X_train),
		(X_train.shape[0],eeg_shape[1],eeg_shape[2]))
	synt_test_between = np.reshape(reg_between.predict(X_test),
		(X_test.shape[0],eeg_shape[1],eeg_shape[2]))
	del reg_between, X_train, X_test, X_ilsvrc2012_val, X_ilsvrc2012_test,\
		y_train_within, y_train_between

	### Putting the data into dictionaries and saving ###
	print('\n>>> Saving the data <<<')
	# Creating the saving directories
	save_dir = os.path.join(args.project_dir, 'results', 'sub-'+
		format(args.sub,'02'), 'synthetic_eeg_data', 'dnn-'+args.dnn)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	# Training data
	data_dict = {
		'synthetic_within_data': synt_train_within,
		'synthetic_between_data': synt_train_between,
		'ch_names': ch_names,
		'times': times
		}
	file_name = 'synthetic_eeg_training.npy'
	np.save(os.path.join(save_dir, file_name), data_dict)
	# Test data
	data_dict = {
		'synthetic_within_data': synt_test_within,
		'synthetic_between_data': synt_test_between,
		'ch_names': ch_names,
		'times': times
		}
	file_name = 'synthetic_eeg_test.npy'
	np.save(os.path.join(save_dir, file_name), data_dict)
	# ILSVRC-2012 validation data
	data_dict = {
		'synthetic_within_data': synt_ilsvrc2012_val_within,
		'ch_names': ch_names,
		'times': times
		}
	file_name = 'synthetic_eeg_ilsvrc2012_val.npy'
	np.save(os.path.join(save_dir, file_name), data_dict)
	# ILSVRC-2012 test data
	data_dict = {
		'synthetic_within_data': synt_ilsvrc2012_test_within,
		'ch_names': ch_names,
		'times': times
		}
	file_name = 'synthetic_eeg_ilsvrc2012_test.npy'
	np.save(os.path.join(save_dir, file_name), data_dict)
