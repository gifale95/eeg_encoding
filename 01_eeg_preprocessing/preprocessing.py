"""Preprocessing of raw EEG data: channel selection, epoching, frequency
downsampling, baseline correction, multivariate noise normalization (MVNN),
sorting of the data conditions and reshaping the data to:
Image conditions × EEG repetitions × EEG channels × EEG time points.
Then, the data of both test and training EEG partitions is saved.

Parameters
----------
sub : int
	Used subject.
n_ses : int
	Number of EEG sessions.
sfreq : int
	Downsampling frequency.
mvnn_dim : str
	Whether to compute the MVNN covariace matrices for each time point
	('time') or for each epoch/repetition ('epochs').
project_dir : str
	Directory of the project folder.

"""

import argparse
import numpy as np
from preprocessing_utils import epoching
from preprocessing_utils import mvnn
from preprocessing_utils import save_prepr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--n_ses', default=4, type=int)
parser.add_argument('--sfreq', default=100, type=int)
parser.add_argument('--mvnn_dim', default='time', type=str)
parser.add_argument('--project_dir', default='/project/directory', type=str)
args = parser.parse_args()

print('\n\n\n>>> EEG data preprocessing <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
np.random.seed(seed=20200220)


# =============================================================================
# Epoching and sorting the data
# =============================================================================
# Channel selection, epoching, baseline correcting and frequency downsampling of
# the test and training data partitions.
# Then, the conditions are sorted and the EEG data is reshaped to:
# Image conditions × EGG repetitions × EEG channels × EEG time points
# This step is applied independently to the data of each partition and session.
epoched_test, _, ch_names, times = epoching(args, 'test')
epoched_train, img_conditions_train, _, _ = epoching(args, 'training')


# =============================================================================
# Multivariate Noise Normalization
# =============================================================================
# MVNN is applied independently to the data of each session.
whitened_test, whitened_train = mvnn(args, epoched_test, epoched_train)
del epoched_test, epoched_train


# =============================================================================
# Merging and saving the preprocessed data
# =============================================================================
# In this step the data of all sessions is merged into the shape:
# Image conditions × EGG repetitions × EEG channels × EEG time points
# Then, the preprocessed data of the test and training data partitions is saved.
save_prepr(args, whitened_test, whitened_train, img_conditions_train, ch_names,
	times)
