"""Zero-shot identification of the biological test data image conditions using
the synthetic test and ILSVRC-2012 data.

Parameters
----------
sub : int
	Used subject.
dnn : str
	Used DNN network.
n_used_features : int
	Number of best features used for the zero-shot decoding.
n_iter : int
	Number of analysis iterations.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr as corr
from sklearn.utils import resample


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--n_used_features', default=300, type=int)
parser.add_argument('--n_iter', default=100, type=int)
parser.add_argument('--project_dir', default='/project/directory/'+
		'studies/eeg_encoding/paradigm_3', type=str)
args = parser.parse_args()

print('>>> Zero-shot identification <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
np.random.seed(seed=20200220)


# =============================================================================
# Loading the biological EEG training/test data
# =============================================================================
# Loading the biological data, averaging across repetitions and reshaping it to
# Samples × Features format
# Training data
train_dir = os.path.join('eeg_dataset', 'preprocessed_data', 'sub-'+
		format(args.sub,'02'), 'preprocessed_eeg_training.npy')
bio_train = np.load(os.path.join(args.project_dir, train_dir),
		allow_pickle=True).item()['preprocessed_eeg_data']
bio_train = np.reshape(np.mean(bio_train, 1), (bio_train.shape[0],-1))

# Test data
test_dir = os.path.join('eeg_dataset', 'preprocessed_data', 'sub-'+
		format(args.sub,'02'), 'preprocessed_eeg_test.npy')
bio_test = np.load(os.path.join(args.project_dir, test_dir),
		allow_pickle=True).item()['preprocessed_eeg_data']
bio_test = np.reshape(np.mean(bio_test, 1), (bio_test.shape[0],-1))


# =============================================================================
# Loading the synthetic EEG training/test/ILSVRC2012 data
# =============================================================================
# Loading the synthetic data and reshaping it to Samples × Features format
# Training data
train_dir = os.path.join('results', 'sub-'+format(args.sub,'02'),
	'linearizing_encoding', 'synthetic_eeg_data', 'dnn-' + args.dnn,
	'synthetic_eeg_training.npy')
data = np.load(os.path.join(args.project_dir, train_dir),
	allow_pickle=True).item()
synt_train = np.reshape(data['synthetic_data_within'],
	(data['synthetic_data_within'].shape[0],-1))

# Test data
test_dir = os.path.join('results', 'sub-'+format(args.sub,'02'),
	'linearizing_encoding', 'synthetic_eeg_data', 'dnn-' + args.dnn,
	'synthetic_eeg_test.npy')
data = np.load(os.path.join(args.project_dir, test_dir),
	allow_pickle=True).item()
synt_test = np.reshape(data['synthetic_data_within'],
	(data['synthetic_data_within'].shape[0],-1))

# ILSVRC-2012 validation data
ilsvrc2012_val_dir = os.path.join('results', 'sub-'+format(args.sub,'02'),
	'linearizing_encoding', 'synthetic_eeg_data', 'dnn-' + args.dnn,
	'synthetic_eeg_ilsvrc2012_val.npy')
data = np.load(os.path.join(args.project_dir, ilsvrc2012_val_dir),
	allow_pickle=True).item()
synt_ilsvrc2012_val = np.reshape(data['synthetic_data_within'],
	(data['synthetic_data_within'].shape[0],-1))

# ILSVRC-2012 test data
ilsvrc2012_test_dir = os.path.join('results', 'sub-'+format(args.sub,'02'),
	'linearizing_encoding', 'synthetic_eeg_data', 'dnn-' + args.dnn,
	'synthetic_eeg_ilsvrc2012_test.npy')
data = np.load(os.path.join(args.project_dir, ilsvrc2012_test_dir),
	allow_pickle=True).item()
synt_ilsvrc2012_test = np.reshape(data['synthetic_data_within'],
	(data['synthetic_data_within'].shape[0],-1))

# Appending the ILSVRC-2012 data across the image conditions
synt_ilsvrc2012 = np.append(synt_ilsvrc2012_val, synt_ilsvrc2012_test, 0)
del data, synt_ilsvrc2012_val, synt_ilsvrc2012_test


# =============================================================================
# Feature selection
# =============================================================================
# Computing the correlation between the training biological and synthetic data
# features, and use it to select the best features
# Correlation matrix of shape: Features
correlation = np.zeros(bio_train.shape[1])
for f in range(bio_train.shape[1]):
	correlation[f] = corr(bio_train[:,f], synt_train[:,f])[0]
del bio_train, synt_train

# Sorting the features based on their correlation index, and selecting the
# maximum number of features
idx = np.argsort(correlation)[::-1]
bio_test = bio_test[:,idx[:args.n_used_features]]
synt_test = synt_test[:,idx[:args.n_used_features]]
synt_ilsvrc2012 = synt_ilsvrc2012[:,idx[:args.n_used_features]]
del correlation, idx


# =============================================================================
# Correlating the biological test data with the candidate data conditions
# =============================================================================
tot_images = synt_test.shape[0] + synt_ilsvrc2012.shape[0]
# Correlation matrix of shape: Test images × Tot Images
correlation = np.zeros((synt_test.shape[0],tot_images))

# Performing the correlation
for bd in tqdm(range(bio_test.shape[0]), position=0):
	for pd in range(tot_images):
		if pd < 200:
			correlation[bd,pd] = corr(bio_test[bd,:], synt_test[pd,:])[0]
		else:
			correlation[bd,pd] = corr(bio_test[bd,:],
				synt_ilsvrc2012[pd-200,:])[0]
del bio_test, synt_test, synt_ilsvrc2012


# =============================================================================
# Sorting the correlation values
# =============================================================================
steps = np.arange(0, correlation.shape[1]+1, 1000)
n_steps = len(steps)
# Sorted data matrix of shape: Iterations × Test images × Steps
zero_shot_identification = np.zeros((args.n_iter,correlation.shape[0],n_steps))

for i in tqdm(range(args.n_iter)):
	for s in range(n_steps):
		# Randomly selecting the additional image conditions EEG data
		idx_ilsvrc2012 = resample(np.arange(200, correlation.shape[1]),
			replace=False)[:steps[s]]
		corr = np.append(correlation[:,:200], correlation[:,idx_ilsvrc2012], 1)
		for bd in range(correlation.shape[0]):
			# Sorting the correlation values
			idx = np.argsort(corr[bd,:])[::-1]
			# Storing the results
			zero_shot_identification[i,bd,s] = np.where(idx == bd)[0][0]
del correlation, corr


# =============================================================================
# Saving the results
# =============================================================================
# Storing the results into a dictionary
results_dict = {
'zero_shot_identification': zero_shot_identification,
'steps': steps
}

# Saving directory
save_dir = os.path.join(args.project_dir, 'results', 'sub-'+
	format(args.sub,'02'), 'linearizing_encoding', 'zero_shot_identification',
	'dnn-'+args.dnn)
file_name = 'zero_shot_identification.npy'

# Creating the directory if not existing and saving
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), results_dict)
