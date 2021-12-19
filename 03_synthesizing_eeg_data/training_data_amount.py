"""Fitting a linear regression to predict EEG data using the DNN feature maps as
predictors. The linear regression is trained using varying amounts (25%, 50%,
75%, 100%) of image conditions and EEG repetitions of the the training images
EEG data (Y) and feature maps (X). The learned weights are used to synthesize
the EEG data of the test images. The synthetic EEG test data is then correlated
with the biological EEG test data.

Parameters
----------
sub : int
	Used subject.
dnn : str
	Used DNN network.
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
parser.add_argument('--n_img_cond', default=4135, type=int)
parser.add_argument('--n_eeg_rep', default=1, type=int)
parser.add_argument('--n_iter', default=100, type=int)
parser.add_argument('--project_dir', default='/project/directory', type=str)
args = parser.parse_args()

print('>>> Training data amount <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
np.random.seed(seed=20200220)


# =============================================================================
# Used image conditions and EEG repetitions combinations
# =============================================================================
tot_img_conditions = 16540
tot_eeg_repetitions = 4

# Correlation results matrix of shape: Iterations
correlation_results = np.zeros((args.n_iter))
for i in tqdm(range(args.n_iter)):
	# Randomly selecting image conditions and repetitions
	cond_idx = np.sort(resample(np.arange(0, tot_img_conditions),
		replace=False)[:args.n_img_cond])
	rep_idx = np.sort(resample(np.arange(0, tot_eeg_repetitions),
		replace=False)[:args.n_eeg_rep])


# =============================================================================
# Loading the DNN feature maps
# =============================================================================
	X_train, X_test = load_dnn_data(args, cond_idx)


# =============================================================================
# Loading the EEG data
# =============================================================================
	y_train, y_test = load_eeg_data(args, cond_idx, rep_idx)


# =============================================================================
# Training a linear regression and predicting the EEG test data
# =============================================================================
	y_test_pred = perform_regression(X_train, X_test, y_train)
	del X_train, X_test, y_train


# =============================================================================
# Performing the correlation
# =============================================================================
	correlation_results[i], noise_ceiling[i] = correlation_analysis(args,
		y_test_pred, y_test)

# Averaging the results across iterations
correlation_results = np.mean(correlation_results, 0)
noise_ceiling = np.mean(noise_ceiling, 0)


# =============================================================================
# Saving the predicted test data
# =============================================================================
save_data(args, correlation_results, noise_ceiling)
