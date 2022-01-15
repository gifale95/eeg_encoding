def load_dnn_data(args, cond_idx):
	"""Loading the DNN feature maps of training and test images.

	Parameters
	----------
	args : Namespace
		Input arguments.
	cond_idx : int
		Indices of the used image conditions.

	Returns
	-------
	X_train : float
		Training images feature maps.
	X_test : float
		Test images feature maps.
	"""

	import numpy as np
	import os

	### Loading the DNN feature maps ###
	# Feature maps directories
	data_dir = os.path.join('dnn_feature_maps', 'pca_feature_maps', args.dnn)
	training_file = 'pca_feature_maps_training.npy'
	test_file = 'pca_feature_maps_test.npy'
	# Loading the feature maps
	X_train = np.load(os.path.join(args.project_dir, data_dir, training_file))
	X_test = np.load(os.path.join(args.project_dir, data_dir, test_file))
	# Selecting the desired amount of training categories
	X_train = X_train[cond_idx]

	### Output ###
	return X_train, X_test


def load_eeg_data(args, cond_idx, rep_idx):
	"""Loading the EEG training and test data, and selecting only the time
	point interval 60-500ms for the subsequent analyses.

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

	### Loading the EEG training data ###
	data_dir = os.path.join('eeg_dataset', 'preprocessed_data', 'sub-'+
		format(args.sub,'02'))
	training_file = 'preprocessed_eeg_training.npy'
	data = np.load(os.path.join(args.project_dir, data_dir, training_file),
		allow_pickle=True).item()
	y_train = data['preprocessed_eeg_data']
	# Selecting the desired amount of training categories
	y_train = y_train[cond_idx]
	# Averaging across the selected amount of training repetitions
	y_train = np.mean(y_train[:,rep_idx], 1)
	# Selecting the time points between 60-500ms
	y_train = y_train[:,:,26:71]

	### Loading the EEG test data ###
	test_file = 'preprocessed_eeg_test.npy'
	data = np.load(os.path.join(args.project_dir, data_dir, test_file),
		allow_pickle=True).item()
	y_test = data['preprocessed_eeg_data']
	# Selecting the time points between 60-500ms
	y_test = y_test[:,:,:,26:71]

	### Output ###
	return y_train, y_test


def perform_regression(X_train, X_test, y_train):
	"""Training a linear regression on the training images DNN feature maps (X)
	and training EEG data (Y), and use the trained weights to synthesize the EEG
	responses to the test images.

	Parameters
	----------
	X_train : float
		Training images feature maps.
	X_test : float
		Test images feature maps.
	y_train : float
		Training EEG data.

	Returns
	-------
	y_test_pred : float
		Predicted test EEG data.

	"""

	import numpy as np
	from ols import OLS_pytorch

	### Fitting the model at each time-point and channel ###
	y_train = np.reshape(y_train, (y_train.shape[0],-1))
	reg = OLS_pytorch(False)
	reg.fit(X_train, y_train.T)
	y_test_pred = reg.predict(X_test)
	y_test_pred = np.reshape(y_test_pred, (200,17,45))

	### Output ###
	return y_test_pred


def correlation_analysis(args, y_test_pred, y_test):
	"""Correlation of predicted data with biological data.

	Parameters
	----------
	args : Namespace
		Input arguments.
	y_test_pred : float
		Predicted test EEG data.
	y_test : float
		Test EEG data.

	Returns
	-------
	correlation : float
		Correlation results.
	noise_ceiling : float
		Noise ceiling results.

	"""

	import numpy as np
	from tqdm import tqdm
	from sklearn.utils import resample
	from scipy.stats import pearsonr as corr

	### Performing the correlation ###
	# Correlation matrices of shape:
	# (Iterations ×  EEG channels × EEG time points)
	correlation = np.zeros((args.n_iter, y_test.shape[2], y_test.shape[3]))
	noise_ceiling = np.zeros((args.n_iter, y_test.shape[2], y_test.shape[3]))
	for i in tqdm(range(args.n_iter)):
		# Random data repetitions index
		shuffle_idx = resample(np.arange(0, y_test.shape[1]), replace=False)\
			[:int(y_test.shape[1]/2)]
		# Averaging across one half of the biological data repetitions
		bio_data_avg_half_1 = np.mean(np.delete(y_test, shuffle_idx, 1), 1)
		# Averaging across the other half of the biological data repetitions for the
		# noise ceiling calculation
		bio_data_avg_half_2 = np.mean(y_test[:,shuffle_idx,:,:], 1)
		# Computing the correlation and noise ceilings
		for t in range(y_test.shape[3]):
			for c in range(y_test.shape[2]):
				correlation[i,c,t] = corr(y_test_pred[:,c,t],
					bio_data_avg_half_1[:,c,t])[0]
				noise_ceiling[i,c,t] = corr(bio_data_avg_half_2[:,c,t],
					bio_data_avg_half_1[:,c,t])[0]
	# Averaging the results across iterations, EEG channels and time points
	correlation = np.mean(correlation)
	noise_ceiling = np.mean(noise_ceiling)

	### Output ###
	return correlation, noise_ceiling


def save_data(args, correlation_results, noise_ceiling):
	"""Saving the results.

	Parameters
	----------
	args : Namespace
		Input arguments.
	correlation_results : float
		Correlation results.
	noise_ceiling : float
		Noise ceiling results.

	"""

	import numpy as np
	import os

	### Storing the results into a dictionary ###
	results_dict = {
		'correlation_results' : correlation_results,
		'noise_ceiling' : noise_ceiling
	}

	### Saving the results ###
	# Save directories
	save_dir = os.path.join(args.project_dir, 'results', 'sub-'+
		format(args.sub,'02'), 'linearizing_encoding',
		'training_data_amount_analysis', 'dnn-'+args.dnn)
	file_name = 'training_data_amount_n_img_cond-'+\
		format(args.n_img_cond,'06')+'_n_eeg_rep-'+format(args.n_eeg_rep,'02')
	# Creating the directory if not existing and saving the data
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	np.save(os.path.join(save_dir, file_name), results_dict)
