"""Correlation of each predicted EEG data feature (EEG_channels x
EEG_time_points) with the corresponding biological EEG data feature, and noise
ceiling calculation.

Parameters
----------
sub : int
		Used subject.
sfreq : int
		Downsampling frequency.
dnn : str
		Used DNN network.
n_pca : int
		PCA downsampling dimensionality of DNN activations.
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
parser.add_argument('--sfreq', default=100, type=int)
parser.add_argument('--dnn', default='CORnet-S', type=str)
parser.add_argument('--n_pca', default=1000, type=int)
parser.add_argument('--n_iter', default=100, type=int)
parser.add_argument('--project_dir', default='/project/directory', type=str)
args = parser.parse_args()

# Printing the arguments
print('\n\n\n>>> Correlation <<<')
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
ch_names = bio_data['info']['ch_names']
bio_data = bio_data['preprocessed_data']


# =============================================================================
# Loading the predicted test data
# =============================================================================
data_dir = os.path.join('predicted_eeg_data', 'sub-'+format(args.sub,'02'),
	'dnn-' + args.dnn, 'pca-'+format(args.n_pca,'05'), 'hz-'+
	format(args.sfreq,'04'), 'predicted_data.npy')
pred_data = np.load(os.path.join(args.project_dir, data_dir),
	allow_pickle=True).item()
pred_data = pred_data['predicted_test_data']


# =============================================================================
# Computing the correlation and noise ceilings
# =============================================================================
# Results and noise ceiling (NC) matrices of shape:
# (Iterations x EEG_channels x EEG_time_points)
correlation = np.zeros((args.n_iter, bio_data.shape[2], bio_data.shape[3]))
noise_ceiling_lower_bound = np.zeros((args.n_iter,bio_data.shape[2],
	bio_data.shape[3]))
noise_ceiling_upper_bound = np.zeros((args.n_iter,bio_data.shape[2],
	bio_data.shape[3]))

# Averaging across all biological data repetitions for the NC upper bound
# calculation
bio_data_avg_all = np.mean(bio_data, 1)

# Loop over iterations
for i in tqdm(range(args.n_iter)):
	# Random data repetitions index
	shuffle_idx = resample(np.arange(0, bio_data.shape[1]), replace=False)\
		[:int(bio_data.shape[1]/2)]
	# Averaging across one half of the biological data repetitions
	bio_data_avg_half_1 = np.mean(np.delete(bio_data, shuffle_idx, 1), 1)
	# Averaging across the other half of the biological data repetitions for the
	# NC lower bound calculation
	bio_data_avg_half_2 = bio_data[:,shuffle_idx,:,:]

	# Loop over EEG time points and channels
	for t in range(bio_data.shape[3]):
		for c in range(bio_data.shape[2]):
			# Computing the correlation and noise ceilings
			correlation[i,c,t] = corr(pred_data[:,c,t],
				bio_data_avg_half_1[:,c,t])[0]
			corr_nc_lower[i,c,t] = corr(bio_data_avg_half_2[:,c,t],
				bio_data_avg_half_1[:,c,t])[0]
			corr_nc_upper[i,c,t] = corr(bio_data_avg_all[:,c,t],
				bio_data_avg_half_1[:,c,t])[0]

# Averaging the results across iterations
correlation = np.mean(correlation, 0)
corr_nc_lower = np.mean(corr_nc_lower, 0)
corr_nc_upper = np.mean(corr_nc_upper, 0)


# =============================================================================
# Saving the results
# =============================================================================
# Storing the results into a dictionary
results_dict = {
	'correlation' : correlation,
	'correlation_nc_lower': corr_nc_lower,
	'correlation_nc_upper': corr_nc_upper,
	'times': times,
	'ch_names': ch_names
}

# Saving directory
save_dir = os.path.join('results', 'sub-'+format(args.sub,'02'), 'correlation',
	'dnn-'+args.dnn, 'pca-'+format(args.n_pca,'05'), 'hz-'+
	format(args.sfreq,'04'))
file_name = 'correlation.npy'

# Creating the directory if not existing and saving
if os.path.isdir(os.path.join(args.project_dir, save_dir)) == False:
	os.makedirs(os.path.join(args.project_dir, save_dir))
np.save(os.path.join(args.project_dir, save_dir, file_name), corr_dict)
