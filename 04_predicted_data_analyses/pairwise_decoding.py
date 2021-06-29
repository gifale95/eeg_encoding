"""Pairwise decoding of the predicted EEG test data, and noise ceiling
calculation. For each EEG time point, a SVM classifier is trained to correctly
decode between two biological data image-conditions (using the EEG channels
data), and is then tested on the corresponding two predicted data
image-conditions.

Parameters
----------
sub : int
	Used subject.
sfreq : int
	Downsampling frequency.
dnn : str
	Used DNN network.
n_iter : int
	Number of analysis iterations.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.svm import SVC


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--sfreq', default=100, type=int)
parser.add_argument('--net', default='CORnet-S', type=str)
parser.add_argument('--n_iter', default=100, type=int)
parser.add_argument('--project_dir', default='/project/directory', type=str)
args = parser.parse_args()

# Printing the arguments
print('\n\n\n>>> Pairwise decoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
np.random.seed(seed=20200220)


# =============================================================================
# Loading the biological test data
# =============================================================================
data_dir = os.path.join('eeg_dataset', 'preprocessed_data', 'sub-'+
	format(args.sub,'02'), 'hz-'+format(args.sfreq,'04'),
	'preprocessed_eeg_test.npy')
bio_data = np.load(os.path.join(args.project_dir, data_dir),
	allow_pickle=True).item()
times = bio_data['info']['times']
bio_data = bio_data['preprocessed_data']


# =============================================================================
# Loading the predicted test data
# =============================================================================
data_dir = os.path.join('predicted_eeg_data', 'sub-'+format(args.sub,'02'),
	'dnn-' + args.dnn, 'hz-'+format(args.sfreq,'04'), 'predicted_data.npy')
pred_data = np.load(os.path.join(args.project_dir, data_dir),
	allow_pickle=True).item()
pred_data = pred_data['predicted_test_data']


# =============================================================================
# Computing the pairwise decoding and noise ceiling
# =============================================================================
# Results and noise ceiling (NC) matrices of shape:
# (Iterations x Image_conditions x Image_conditions x EEG_time_points)
pair_dec = np.zeros((args.n_iter, bio_data.shape[0],bio_data.shape[0],
	bio_data.shape[3]))
pair_dec_nc_lower = np.zeros((args.n_iter, bio_data.shape[0],bio_data.shape[0],
	bio_data.shape[3]))
pair_dec_nc_upper = np.zeros((args.n_iter, bio_data.shape[0],bio_data.shape[0],
	bio_data.shape[3]))

# Averaging across all biological data repetitions for the NC upper bound
# calculation
bio_data_avg_all = np.mean(bio_data, 1)

# Loop over iterations
for i in tqdm(range(args.n_iter)):
	# Random data repetitions index
	shuffle_idx = resample(np.arange(0, bio_data.shape[1]), replace=False)\
		[:int(bio_data.shape[1]/2)]
	# Selecting one half of the biological data repetitions for training the
	# classifier, and avraging them into 10 pseudo-trials of 4 repetitions
	bio_data_avg_half_1 = np.zeros((bio_data.shape[0],10,bio_data.shape[2],
			bio_data.shape[3]))
	bio_data_provv = np.delete(bio_data, shuffle_idx, 1)
	for r in range(bio_data_avg_half_1.shape[1]):
		bio_data_avg_half_1[:,r,:,:] = np.mean(bio_data_provv[:,r*4:r*4+4,:,:],
		1)
	del bio_data_provv
	# Averaging across the other half of the biological data repetitions for the
	# NC lower bound calculation
	bio_data_avg_half_2 = np.mean(bio_data[:,shuffle_idx,:,:], 1)

	# Classifier target vectors
	y_train = np.zeros((bio_data_avg_half_1.shape[1])*2)
	y_train[bio_data_avg_half_1.shape[1]:(bio_data_avg_half_1.shape[1])*2] = 1
	y_test = np.asarray((0, 1))

	# Loop over image-conditions and EEG time points
	for i1 in range(pred_data.shape[0]):
		for i2 in range(pred_data.shape[0]):
			if i1 < i2:
				for t in range(pred_data.shape[2]):
					# Defining the training/test partitions
					X_train = np.append(bio_data_avg_half_1[i1,:,:,t], \
						bio_data_avg_half_1[i2,:,:,t], 0)
					X_test_pred_data = np.append(np.expand_dims(
						pred_data[i1,:,t], 0), np.expand_dims(
						pred_data[i2,:,t], 0), 0)
					X_test_avg_half = np.append(np.expand_dims(
						bio_data_avg_half_2[i1,:,t], 0), np.expand_dims(
						bio_data_avg_half_2[i2,:,t], 0), 0)
					X_test_avg_all = np.append(np.expand_dims(
						bio_data_avg_all[i1,:,t], 0), np.expand_dims(
						bio_data_avg_all[i2,:,t], 0), 0)
					# Training the classifier
					dec_svm = SVC(kernel="linear")
					dec_svm.fit(X_train, y_train)
					# Testing the classifier
					y_pred = dec_svm.predict(X_test_pred_data)
					y_pred_nc_lower = dec_svm.predict(X_test_avg_half)
					y_pred_nc_upper = dec_svm.predict(X_test_avg_all)
					# Storing the accuracy
					pair_dec[i,i2,i1,t] = sum(y_pred == y_test) / len(y_test)
					pair_dec_nc_lower[i,i2,i1,t] = sum(y_pred_nc_lower ==
						y_test) / len(y_test)
					pair_dec_nc_upper[i,i2,i1,t] = sum(y_pred_nc_upper ==
						y_test) / len(y_test)


# =============================================================================
# Averaging the results across iterations and pairwise comparisons
# =============================================================================
# Averaging across iterations
pair_dec = np.mean(pair_dec, 0)
pair_dec_nc_lower = np.mean(pair_dec_nc_lower, 0)
pair_dec_nc_upper = np.mean(pair_dec_nc_upper, 0)

# Averaging across pairwise comparisons
idx = np.tril_indices(pair_dec.shape[0], -1)
pair_dec = pair_dec[idx]
pair_dec_nc_lower = pair_dec_nc_lower[idx]
pair_dec_nc_upper = pair_dec_nc_upper[idx]
pair_dec = np.mean(pair_dec, axis=0)
pair_dec_nc_lower = np.mean(pair_dec_nc_lower, axis=0)
pair_dec_nc_upper = np.mean(pair_dec_nc_upper, axis=0)


# =============================================================================
# Saving the results
# =============================================================================
# Storing the results into a dictionary
results_dict = {
	'pair_dec': pair_dec,
	'pair_dec_nc_lower': pair_dec_nc_lower,
	'pair_dec_nc_upper': pair_dec_nc_upper,
	'times': times
}

# Saving directory
save_dir = os.path.join('results', 'sub-'+format(args.sub,'02'),
	'pairwise_decoding', 'dnn-'+args.dnn, 'hz-'+format(args.sfreq,'04'))
file_name = 'pairwise_decoding.npy'

# Creating the directory if not existing and saving
if os.path.isdir(os.path.join(args.project_dir, save_dir)) == False:
	os.makedirs(os.path.join(args.project_dir, save_dir))
np.save(os.path.join(args.project_dir, save_dir, file_name), results_dict)
