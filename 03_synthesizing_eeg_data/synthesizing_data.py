"""Fitting a linear regression to predict EEG data using the DNN feature maps as
predictors. The linear regression is trained using the training images EEG data
(Y) and feature maps (X), and the learned weights are used to synthesize the EEG
data of the training and test images, and also of the ILSVRC-2012 test and
validation images. The linear regression is trained both within and between
subjects in a leave-one-subject out fashion.

Parameters
----------
sub : int
	Used subject.
n_tot_sub : int
	Total number of subjects.
dnn : str
	Used DNN network.
project_dir : str
	Directory of the project folder.

"""

import argparse
from synthesizing_data_utils import load_dnn_data
from synthesizing_data_utils import load_eeg_data
from synthesizing_data_utils import perform_regression


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--n_tot_sub', default=10, type=int)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--project_dir', default='/project/directory', type=str)
args = parser.parse_args()

print('>>> Predicting the EEG data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Loading the DNN feature maps
# =============================================================================
X_train, X_test, X_ilsvrc2012_val, X_ilsvrc2012_test = load_dnn_data(args)


# =============================================================================
# Loading the EEG data
# =============================================================================
y_train_within, y_train_between, ch_names, times = load_eeg_data(args)


# =============================================================================
# Training the linear regression and saving the predicted data
# =============================================================================
perform_regression(args, ch_names, times, X_train, X_test, X_ilsvrc2012_val,
	X_ilsvrc2012_test, y_train_within, y_train_between)
