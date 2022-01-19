"""Pairwise decoding of the synthetic EEG test data, and noise ceiling
calculation. For each EEG time point, a SVM classifier is trained to decode
between each combination of two biological data image conditions (using the EEG
channels data), and is then tested on the corresponding combinations of two
synthetic data image conditions.
The analysis is performed on both the within and between subjects synthesized
data.

Parameters
----------
sub : int
	Used subject.
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
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--n_iter', default=100, type=int)
parser.add_argument('--project_dir', default='/project/directory', type=str)
args = parser.parse_args()

print('>>> Pairwise decoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
np.random.seed(seed=20200220)


# =============================================================================
# Loading the biological EEG test data
# =============================================================================
data_dir = os.path.join('eeg_dataset', 'preprocessed_data', 'sub-'+
	format(args.sub,'02'), 'preprocessed_eeg_test.npy')
bio_data = np.load(os.path.join(args.project_dir, data_dir),
	allow_pickle=True).item()
bio_test = bio_data['preprocessed_eeg_data']
times = bio_data['times']
ch_names = bio_data['ch_names']
del bio_data


# =============================================================================
# Loading the synthetic EEG test data
# =============================================================================
# Linearizing encoding synthetic data
data_dir = os.path.join('results', 'sub-'+format(args.sub,'02'),
	'synthetic_eeg_data', 'linearizing_encoding', 'dnn-' + args.dnn,
	'synthetic_eeg_test.npy')
synt_data = np.load(os.path.join(args.project_dir, data_dir),
	allow_pickle=True).item()
synt_test_within = synt_data['synthetic_within_data']
synt_test_between = synt_data['synthetic_between_data']

# End-to-end encoding synthetic data
if args.dnn == 'alexnet':
	data_dir = os.path.join('results', 'sub-'+format(args.sub,'02'),
		'synthetic_eeg_data', 'end_to_end_encoding', 'dnn-' + args.dnn,
		'synthetic_eeg_test.npy')
	synt_data = np.load(os.path.join(args.project_dir, data_dir),
		allow_pickle=True).item()
	synt_test_end = synt_data['synthetic_data']
else:
	synt_test_end = np.zeros(synt_test_within.shape)
del synt_data


# =============================================================================
# Computing the pairwise decoding and noise ceilings
# =============================================================================
# Results and noise ceiling matrices of shape:
# (Iterations × Image conditions × Image conditions × EEG time points)
pair_dec_within = np.zeros((args.n_iter, bio_test.shape[0],bio_test.shape[0],
	bio_test.shape[3]))
pair_dec_between = np.zeros((args.n_iter, bio_test.shape[0],bio_test.shape[0],
	bio_test.shape[3]))
pair_dec_end = np.zeros((args.n_iter, bio_test.shape[0],bio_test.shape[0],
	bio_test.shape[3]))
noise_ceiling_low = np.zeros((args.n_iter, bio_test.shape[0],bio_test.shape[0],
	bio_test.shape[3]))
noise_ceiling_up = np.zeros((args.n_iter, bio_test.shape[0],bio_test.shape[0],
	bio_test.shape[3]))

# Averaging across all the biological data repetitions for the noise ceiling
# upper bound calculation
bio_data_avg_all = np.mean(bio_test, 1)

# Loop over iterations
for i in tqdm(range(args.n_iter)):
	# Random data repetitions index
	shuffle_idx = resample(np.arange(0, bio_test.shape[1]), replace=False)\
		[:int(bio_test.shape[1]/2)]
	# Selecting one half of the biological data repetitions for training the
	# classifier, and averaging them into 10 pseudo-trials of 4 repetitions
	bio_data_avg_half_1 = np.zeros((bio_test.shape[0],10,bio_test.shape[2],
			bio_test.shape[3]))
	bio_data_provv = np.delete(bio_test, shuffle_idx, 1)
	for r in range(bio_data_avg_half_1.shape[1]):
		bio_data_avg_half_1[:,r,:,:] = np.mean(bio_data_provv[:,r*4:r*4+4,:,:],
		1)
	del bio_data_provv
	# Averaging across the other half of the biological data repetitions for the
	# noise ceiling lower bound calculation
	bio_data_avg_half_2 = np.mean(bio_test[:,shuffle_idx,:,:], 1)

	# Classifier target vectors
	y_train = np.zeros((bio_data_avg_half_1.shape[1])*2)
	y_train[bio_data_avg_half_1.shape[1]:(bio_data_avg_half_1.shape[1])*2] = 1
	y_test = np.asarray((0, 1))

	# Loop over image-conditions and EEG time points
	for i1 in range(bio_test.shape[0]):
		for i2 in range(bio_test.shape[0]):
			if i1 < i2:
				for t in range(bio_test.shape[3]):
					# Defining the training/test partitions
					X_train = np.append(bio_data_avg_half_1[i1,:,:,t], \
						bio_data_avg_half_1[i2,:,:,t], 0)
					X_test_synt_data_within = np.append(np.expand_dims(
						synt_test_within[i1,:,t], 0), np.expand_dims(
						synt_test_within[i2,:,t], 0), 0)
					X_test_synt_data_between = np.append(np.expand_dims(
						synt_test_between[i1,:,t], 0), np.expand_dims(
						synt_test_between[i2,:,t], 0), 0)
					X_test_synt_data_end = np.append(np.expand_dims(
						synt_test_end[i1,:,t], 0), np.expand_dims(
						synt_test_end[i2,:,t], 0), 0)
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
					y_pred_within = dec_svm.predict(X_test_synt_data_within)
					y_pred_between = dec_svm.predict(X_test_synt_data_between)
					y_pred_end = dec_svm.predict(X_test_synt_data_end)
					y_pred_noise_ceiling_low = dec_svm.predict(X_test_avg_half)
					y_pred_noise_ceiling_up = dec_svm.predict(X_test_avg_all)
					# Storing the accuracy
					pair_dec_within[i,i2,i1,t] = sum(
						y_pred_within == y_test) / len(y_test)
					pair_dec_between[i,i2,i1,t] = sum(
						y_pred_between == y_test) / len(y_test)
					pair_dec_end[i,i2,i1,t] = sum(
						y_pred_end == y_test) / len(y_test)
					noise_ceiling_low[i,i2,i1,t] = sum(
						y_pred_noise_ceiling_low == y_test) / len(y_test)
					noise_ceiling_up[i,i2,i1,t] = sum(
						y_pred_noise_ceiling_up == y_test) / len(y_test)


# =============================================================================
# Averaging the results across iterations and pairwise comparisons
# =============================================================================
# Averaging across iterations
pair_dec_within = np.mean(pair_dec_within, 0)
pair_dec_between = np.mean(pair_dec_between, 0)
pair_dec_end = np.mean(pair_dec_end, 0)
noise_ceiling_low = np.mean(noise_ceiling_low, 0)
noise_ceiling_up = np.mean(noise_ceiling_up, 0)

# Averaging across pairwise comparisons
idx = np.tril_indices(pair_dec_within.shape[0], -1)
pair_dec_within = pair_dec_within[idx]
pair_dec_between = pair_dec_between[idx]
pair_dec_end = pair_dec_end[idx]
noise_ceiling_low = noise_ceiling_low[idx]
noise_ceiling_up = noise_ceiling_up[idx]
pair_dec_within = np.mean(pair_dec_within, axis=0)
pair_dec_between = np.mean(pair_dec_between, axis=0)
pair_dec_end = np.mean(pair_dec_end, axis=0)
noise_ceiling_low = np.mean(noise_ceiling_low, axis=0)
noise_ceiling_up = np.mean(noise_ceiling_up, axis=0)


# =============================================================================
# Saving the results
# =============================================================================
# Storing the results into a dictionary
results_dict = {
	'pairwise_decoding_within': pair_dec_within,
	'pairwise_decoding_between': pair_dec_between,
	'pairwise_decoding_end': pair_dec_end,
	'noise_ceiling_low': noise_ceiling_low,
	'noise_ceiling_up': noise_ceiling_up,
	'times': times,
	'ch_names': ch_names
}

# Saving directory
save_dir = os.path.join(args.project_dir, 'results', 'sub-'+
	format(args.sub,'02'), 'pairwise_decoding', 'dnn-'+args.dnn)
file_name = 'pairwise_decoding.npy'

# Creating the directory if not existing and saving
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), results_dict)
