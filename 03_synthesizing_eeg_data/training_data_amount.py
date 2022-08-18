"""Fit a linear regression to predict EEG data using the DNN feature maps as
predictors. The linear regression is trained using varying amounts (25%, 50%,
75%, 100%) of image conditions and EEG repetitions of the training images EEG
data (Y) and feature maps (X). The learned weights are used to synthesize the
EEG data of the test images. The synthetic EEG test data is then correlated with
the biological EEG test data.

Parameters
----------
sub : int
	Used subject.
dnn : str
	Used DNN network.
pretrained : bool
	If True use the pretrained network feature maps, if False use the randomly
	initialized network feature maps.
layers : str
	If 'all', the EEG data will be predicted using the feature maps downsampled
	through PCA applied across all DNN layers. If 'single', the EEG data will be
	independently predicted using the PCA-downsampled feature maps of each DNN
	layer independently. If 'appended', the EEG data will be predicted using the
	PCA-downsampled feature maps of each DNN layer appended onto each other.
n_components : int
	Number of DNN feature maps PCA components retained.
n_img_cond : int
	Number of used image conditions.
n_eeg_rep : int
	Number of used EEG repetitions.
n_iter : int
	Number of analysis iterations.
project_dir : str
	Directory of the project folder.


Returns
-------
Saves the predicted test EEG data.

"""

import argparse
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from training_data_amount_utils import load_dnn_data
from training_data_amount_utils import load_eeg_data
from training_data_amount_utils import perform_regression
from training_data_amount_utils import correlation_analysis
from training_data_amount_utils import save_data


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layers', default='all', type=str)
parser.add_argument('--n_components', default=1000, type=int)
parser.add_argument('--n_img_cond', default=4135, type=int)
parser.add_argument('--n_eeg_rep', default=1, type=int)
parser.add_argument('--n_iter', default=100, type=int)
parser.add_argument('--project_dir', default='../project_directory', type=str)
args = parser.parse_args()

print('>>> Training data amount <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Used image conditions and EEG repetitions combinations
# =============================================================================
tot_img_conditions = 16540
tot_eeg_repetitions = 4

# Loop across iterations
for i in tqdm(range(args.n_iter)):
	# Randomly select the image conditions and repetitions
	cond_idx = np.sort(resample(np.arange(0, tot_img_conditions),
		replace=False, n_samples=args.n_img_cond))
	rep_idx = np.sort(resample(np.arange(0, tot_eeg_repetitions),
		replace=False, n_samples=args.n_eeg_rep))


# =============================================================================
# Load the DNN feature maps
# =============================================================================
	X_train, X_test = load_dnn_data(args, cond_idx)


# =============================================================================
# Load the EEG data
# =============================================================================
	y_train, y_test = load_eeg_data(args, cond_idx, rep_idx)


# =============================================================================
# Train a linear regression to predict the EEG data
# =============================================================================
	y_test_pred = perform_regression(X_train, X_test, y_train)


# =============================================================================
# Test the encoding prediction accuracy through a correlation
# =============================================================================
	corr_res, noise_ceil = correlation_analysis(args, y_test_pred, y_test)

	# Results matrices of shape: Iterations
	if i == 0:
		correlation_results = {}
		noise_ceiling = {}
		for layer in y_test_pred.keys():
			correlation_results[layer] = np.zeros(args.n_iter)
			noise_ceiling[layer] = np.zeros(args.n_iter)

	# Store the results
	for layer in correlation_results.keys():
		correlation_results[layer] = corr_res[layer]
		noise_ceiling[layer] = noise_ceil[layer]

# Average the results across iterations
for layer in correlation_results.keys():
	correlation_results[layer] = np.mean(correlation_results[layer])
	noise_ceiling[layer] = np.mean(noise_ceiling[layer])


# =============================================================================
# Save the correlation results
# =============================================================================
save_data(args, correlation_results, noise_ceiling)
