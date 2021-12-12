"""Correlation of each synthetic test EEG data feature (EEG_channels x
EEG_time_points) with the corresponding biological test EEG data feature (across
the 200 test image conditions), and noise ceiling calculation.
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
from scipy.stats import pearsonr as corr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--n_iter', default=100, type=int)
parser.add_argument('--project_dir', default='/project/directory', type=str)
args = parser.parse_args()

print('>>> Correlation <<<')
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
data_dir = os.path.join('results', 'sub-'+format(args.sub,'02'),
	'synthetic_eeg_data', 'dnn-' + args.dnn, 'synthetic_eeg_test.npy')
synt_data = np.load(os.path.join(args.project_dir, data_dir),
	allow_pickle=True).item()
synt_test_within = synt_data['synthetic_data_within']
synt_test_between = synt_data['synthetic_data_between']
del synt_data


# =============================================================================
# Computing the correlation and noise ceilings
# =============================================================================
# Results and noise ceiling matrices of shape:
# (Iterations × EEG channels × EEG time points)
correlation_within = np.zeros((args.n_iter,bio_test.shape[2],
	bio_test.shape[3]))
correlation_between = np.zeros((args.n_iter,bio_test.shape[2],
	bio_test.shape[3]))
noise_ceiling = np.zeros((args.n_iter,bio_test.shape[2], bio_test.shape[3]))

# Loop over iterations
for i in tqdm(range(args.n_iter)):
	# Random data repetitions index
	shuffle_idx = resample(np.arange(0, bio_test.shape[1]), replace=False)\
		[:int(bio_test.shape[1]/2)]
	# Averaging across one half of the biological data repetitions
	bio_data_avg_half_1 = np.mean(np.delete(bio_test, shuffle_idx, 1), 1)
	# Averaging across the other half of the biological data repetitions for the
	# noise ceiling calculation
	bio_data_avg_half_2 = np.mean(bio_test[:,shuffle_idx,:,:], 1)

	# Loop over EEG time points and channels
	for t in range(bio_test.shape[3]):
		for c in range(bio_test.shape[2]):
			# Computing the correlation and noise ceilings
			correlation_within[i,c,t] = corr(synt_test_within[:,c,t],
				bio_data_avg_half_1[:,c,t])[0]
			correlation_between[i,c,t] = corr(synt_test_between[:,c,t],
				bio_data_avg_half_1[:,c,t])[0]
			noise_ceiling[i,c,t] = corr(bio_data_avg_half_2[:,c,t],
				bio_data_avg_half_1[:,c,t])[0]

# Averaging the results across iterations
correlation_within = np.mean(correlation_within, 0)
correlation_between = np.mean(correlation_between, 0)
noise_ceiling = np.mean(noise_ceiling, 0)


# =============================================================================
# Saving the results
# =============================================================================
# Storing the results into a dictionary
results_dict = {
	'correlation_within' : correlation_within,
	'correlation_between' : correlation_between,
	'noise_ceiling': noise_ceiling,
	'times': times,
	'ch_names': ch_names
}

# Saving directory
save_dir = os.path.join(args.project_dir, 'results', 'sub-'+
	format(args.sub,'02'), 'correlation', 'dnn-'+args.dnn)
file_name = 'correlation.npy'

# Creating the directory if not existing and saving
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), results_dict)
