"""Fit a linear regression to predict EEG data using the DNN feature maps as
predictors. The linear regression is trained using the training images EEG data
(Y) and feature maps (X), and the learned weights are used to synthesize the EEG
data of the training and test images, and also of the ILSVRC-2012 validation and
test images. The linear regression is trained both within and between subjects
in a leave-one-subject-out fashion.

Parameters
----------
sub : int
	Used subject.
subjects : str
	If 'within', the linearizing encoding model is fit using the training data
	of the subject of interest. If 'between', the linearizing encoding model is
	fit using the training data of N-1 subjects, and then tested on the subject
	of interest.
all_sub : int
	List of all subjects.
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
project_dir : str
	Directory of the project folder.

"""

import argparse
from linearizing_encoding_utils import load_dnn_data
from linearizing_encoding_utils import load_eeg_data
from linearizing_encoding_utils import perform_regression


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--subjects', default='within', type=str)
parser.add_argument('--all_sub', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
	type=list)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layers', default='all', type=str)
parser.add_argument('--n_components', default=1000, type=int)
parser.add_argument('--project_dir', default='../project_directory', type=str)
args = parser.parse_args()

print('>>> Training linearizing encoding model <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the DNN feature maps
# =============================================================================
X_train, X_test, X_ilsvrc2012_val, X_ilsvrc2012_test = load_dnn_data(args)


# =============================================================================
# Load the EEG data
# =============================================================================
y_train, ch_names, times = load_eeg_data(args)


# =============================================================================
# Train the linear regression and save the predicted data
# =============================================================================
perform_regression(args, ch_names, times, X_train, X_test, X_ilsvrc2012_val,
	X_ilsvrc2012_test, y_train)
