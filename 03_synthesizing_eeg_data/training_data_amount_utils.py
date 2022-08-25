def load_dnn_data(args, cond_idx):
	"""Load the DNN feature maps of the training and test images.

	Parameters
	----------
	args : Namespace
		Input arguments.
	cond_idx : int
		Indices of the used image conditions.

	Returns
	-------
	X_train : dict of float
		Training images feature maps.
	X_test : dict of float
		Test images feature maps.
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
	# Load the feature maps
	X_train = np.load(os.path.join(args.project_dir, data_dir, training_file),
		allow_pickle=True).item()
	X_test = np.load(os.path.join(args.project_dir, data_dir, test_file),
		allow_pickle=True).item()

	### Append the PCA-downsampled feature maps of different layers ###
	if args.layers == 'appended':
		for l, layer in enumerate(X_train.keys()):
			if l == 0:
				train = X_train[layer]
				test = X_test[layer]
			else:
				train = np.append(train, X_train[layer], 1)
				test = np.append(test, X_test[layer], 1)
		X_train = {'appended_layers': train}
		X_test = {'appended_layers': test}

	### Retain only the selected image conditions and PCA components ###
	for layer in X_train.keys():
		X_train[layer] = X_train[layer][cond_idx,:args.n_components]
		X_test[layer] = X_test[layer][cond_idx,:args.n_components]

	### Output ###
	return X_train, X_test


def load_eeg_data(args, cond_idx, rep_idx):
	"""Load the EEG training and test data, and select only the time point
	interval 60-500ms for the subsequent analyses.

	Parameters
	----------
	args : Namespace
		Input arguments.
	cond_idx : int
		Indices of the used image conditions.
	rep_idx : int
		Indices of the used EEG repetitions.

	Returns
	-------
	y_train : float
		Training EEG data.
	y_test : float
		Test EEG data.

	"""

	import os
	import numpy as np

	### Load the EEG training data ###
	data_dir = os.path.join('eeg_dataset', 'preprocessed_data', 'sub-'+
		format(args.sub,'02'))
	training_file = 'preprocessed_eeg_training.npy'
	data = np.load(os.path.join(args.project_dir, data_dir, training_file),
		allow_pickle=True).item()
	y_train = data['preprocessed_eeg_data']
	times = np.round(data['times'], 2)
	# Select the desired amount of training categories
	y_train = y_train[cond_idx]
	# Average across the selected amount of training repetitions
	y_train = np.mean(y_train[:,rep_idx], 1)
	# Select the time points between 60-500ms
	times_start = np.where(times == 0.06)[0][0]
	times_end = np.where(times == 0.51)[0][0]
	y_train = y_train[:,:,times_start:times_end]

	### Load the EEG test data ###
	test_file = 'preprocessed_eeg_test.npy'
	data = np.load(os.path.join(args.project_dir, data_dir, test_file),
		allow_pickle=True).item()
	y_test = data['preprocessed_eeg_data']
	# Select the time points between 60-500ms
	y_test = y_test[:,:,:,times_start:times_end]

	### Output ###
	return y_train, y_test


def perform_regression(X_train, X_test, y_train):
	"""Train a linear regression on the training images DNN feature maps (X) and
	training EEG data (Y), and use the trained weights to synthesize the EEG
	responses to the test images.

	Parameters
	----------
	X_train : dict of float
		Training images feature maps.
	X_test : dict of float
		Test images feature maps.
	y_train : float
		Training EEG data.

	Returns
	-------
	y_test_pred : dict of float
		Predicted test EEG data.

	"""

	import numpy as np
	from ols import OLS_pytorch

	### Fit the regression at each time-point and channel ###
	eeg_shape = y_train.shape
	y_train = np.reshape(y_train, (y_train.shape[0],-1))
	y_test_pred = {}
	for l, layer in enumerate(X_train.keys()):
		reg = OLS_pytorch(use_gpu=False)
		reg.fit(X_train[layer], y_train.T)
		y_test_pred[layer] = np.reshape(reg.predict(X_test[layer]), eeg_shape)

	### Output ###
	return y_test_pred


def correlation_analysis(args, y_test_pred, y_test):
	"""Evaluate the encoding models prediction accuracy by correlating the
	synthetic EEG test data with biological test data.

	Parameters
	----------
	args : Namespace
		Input arguments.
	y_test_pred : dict of float
		Predicted test EEG data.
	y_test : float
		Test EEG data.

	Returns
	-------
	correlation : dict of float
		Correlation results.
	noise_ceiling : float
		Noise ceiling results.

	"""

	import numpy as np
	from tqdm import tqdm
	from sklearn.utils import resample
	from scipy.stats import pearsonr as corr

	### Perform the correlation ###
	# Results matrices of shape:
	# (Iterations ×  EEG channels × EEG time points)
	correlation = {}
	for layer in y_test_pred.keys():
		correlation[layer] = np.zeros((args.n_iter,y_test.shape[2],
			y_test.shape[3]))
	noise_ceiling = np.zeros((args.n_iter,y_test.shape[2],y_test.shape[3]))
	for i in tqdm(range(args.n_iter)):
		# Random data repetitions index
		shuffle_idx = resample(np.arange(0, y_test.shape[1]), replace=False,
			n_samples=int(y_test.shape[1]/2))
		# Average across one half of the biological data repetitions
		bio_data_avg_half_1 = np.mean(np.delete(y_test, shuffle_idx, 1), 1)
		# Average across the other half of the biological data repetitions for
		# the noise ceiling calculation
		bio_data_avg_half_2 = np.mean(y_test[:,shuffle_idx,:,:], 1)
		# Compute the correlation and noise ceiling
		for t in range(y_test.shape[3]):
			for c in range(y_test.shape[2]):
				for layer in y_test_pred.keys():
					correlation[layer][i,c,t] = corr(y_test_pred[layer][:,c,t],
						bio_data_avg_half_1[:,c,t])[0]
				noise_ceiling[i,c,t] = corr(bio_data_avg_half_2[:,c,t],
					bio_data_avg_half_1[:,c,t])[0]
	# Average the results across iterations, EEG channels and time points
	for layer in y_test_pred.keys():
		correlation[layer] = np.mean(correlation[layer])
	noise_ceiling = np.mean(noise_ceiling)

	### Output ###
	return correlation, noise_ceiling


def save_data(args, correlation_results, noise_ceiling):
	"""Save the results.

	Parameters
	----------
	args : Namespace
		Input arguments.
	correlation_results : dict of float
		Correlation results.
	noise_ceiling : float
		Noise ceiling results.

	"""

	import numpy as np
	import os

	### Store the results into a dictionary ###
	results_dict = {
		'correlation_results': correlation_results,
		'noise_ceiling': noise_ceiling
	}

	### Save the results ###
	# Save directories
	save_dir = os.path.join(args.project_dir, 'results', 'sub-'+
		format(args.sub,'02'), 'training_data_amount_analysis', 'dnn-'+args.dnn,
		'pretrained-'+str(args.pretrained), 'layers-'+args.layers,
		'n_components-'+format(args.n_components,'05'))
	file_name = 'training_data_amount_n_img_cond-'+\
		format(args.n_img_cond,'06')+'_n_eeg_rep-'+format(args.n_eeg_rep,'02')
	# Create the directory if not existing and save the data
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	np.save(os.path.join(save_dir, file_name), results_dict)
