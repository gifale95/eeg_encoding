"""Zero-shot identification of the biological test data image conditions using
the linearizing synthetic test and ILSVRC-2012 data.

Parameters
----------
sub : int
	Used subject.
n_used_features : int
	Number of best EEG features used for the zero-shot decoding.
dnn : str
	Used DNN network.
pretrained : bool
	If True, analyze the data synthesized through pretrained (linearizing or
	end-to-end) models. If False, analyze the data synthesized through randomly
	initialized (linearizing or end-to-end) models.
subjects : str
	Whether to analyze the 'within' or 'between' subjects linearizing encoding
	synthetic data.
layers : str
	Whether to analyse the data synthesized using 'all', 'single' or 'appended'
	DNN layers feature maps.
n_components : int
	Number of DNN feature maps PCA components retained for synthesizing the EEG
	data.
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
parser.add_argument('--n_used_features', default=300, type=int)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--subjects', default='within', type=str)
parser.add_argument('--layers', default='all', type=str)
parser.add_argument('--n_components', default=1000, type=int)
parser.add_argument('--n_iter', default=100, type=int)
parser.add_argument('--project_dir', default='../project_directory', type=str)
args = parser.parse_args()

print('>>> Zero-shot identification <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Load the biological EEG training/test data
# =============================================================================
# Load the biological data, average it across repetitions and reshape it to
# (Samples × Features) format.

# Training data
train_dir = os.path.join('eeg_dataset', 'preprocessed_data', 'sub-'+
	format(args.sub,'02'), 'preprocessed_eeg_training.npy')
bio_train = np.load(os.path.join(args.project_dir, train_dir),
	allow_pickle=True).item()['preprocessed_eeg_data']
n_eeg_chan = bio_train.shape[2]
n_eeg_time = bio_train.shape[3]
bio_train = np.reshape(np.mean(bio_train, 1), (bio_train.shape[0],-1))

# Test data
test_dir = os.path.join('eeg_dataset', 'preprocessed_data', 'sub-'+
	format(args.sub,'02'), 'preprocessed_eeg_test.npy')
bio_test_provv = np.load(os.path.join(args.project_dir, test_dir),
	allow_pickle=True).item()['preprocessed_eeg_data']
bio_test_provv = np.reshape(np.mean(bio_test_provv, 1),
	(bio_test_provv.shape[0],-1))
n_test_img = bio_test_provv.shape[0]


# =============================================================================
# Load the synthetic EEG training/test/ILSVRC-2012 data
# =============================================================================
# Load the synthetic data
data_dir = os.path.join(args.project_dir, 'results', 'sub-'+
	format(args.sub,'02'), 'synthetic_eeg_data', 'encoding-linearizing',
	'subjects-'+args.subjects, 'dnn-'+args.dnn, 'pretrained-'+
	str(args.pretrained), 'layers-'+args.layers, 'n_components-'+
	format(args.n_components,'05'))
synt_train = np.load(os.path.join(args.project_dir, data_dir,
	'synthetic_eeg_training.npy'), allow_pickle=True).item()['synthetic_data']
synt_test = np.load(os.path.join(args.project_dir, data_dir,
	'synthetic_eeg_test.npy'), allow_pickle=True).item()['synthetic_data']
synt_ilsvrc2012_val = np.load(os.path.join(args.project_dir, data_dir,
	'synthetic_eeg_ilsvrc2012_val.npy'),
	allow_pickle=True).item()['synthetic_data']
synt_ilsvrc2012_test = np.load(os.path.join(args.project_dir, data_dir,
	'synthetic_eeg_ilsvrc2012_test.npy'),
	allow_pickle=True).item()['synthetic_data']

# Reshape the data to (Samples × Features) format
for layer in synt_test.keys():
	synt_train[layer] = np.reshape(synt_train[layer],
		(synt_train[layer].shape[0],-1))
	synt_test[layer] = np.reshape(synt_test[layer],
		(synt_test[layer].shape[0],-1))
	synt_ilsvrc2012_val[layer] = np.reshape(synt_ilsvrc2012_val[layer],
		(synt_ilsvrc2012_val[layer].shape[0],-1))
	synt_ilsvrc2012_test[layer] = np.reshape(synt_ilsvrc2012_test[layer],
		(synt_ilsvrc2012_test[layer].shape[0],-1))

# Append the ILSVRC-2012 data across the image conditions
synt_ilsvrc2012 = {}
for layer in synt_test.keys():
	synt_ilsvrc2012[layer] = np.append(synt_ilsvrc2012_val[layer],
		synt_ilsvrc2012_test[layer], 0)
del synt_ilsvrc2012_val, synt_ilsvrc2012_test


# =============================================================================
# Feature selection
# =============================================================================
# Compute the correlation between the training biological and synthetic data
# features, and use it to select the best features.

# Correlation matrix of shape: (Features)
correlation = {}
for layer in synt_test.keys():
	correlation[layer] = np.zeros(bio_train.shape[1])

# Compute the correlation
for layer in synt_test.keys():
	for f in range(bio_train.shape[1]):
		correlation[layer][f] = corr(bio_train[:,f], synt_train[layer][:,f])[0]
del bio_train, synt_train

# Sort the features based on their correlation index, and select the maximum
# number of features
bio_test = {}
best_features_masks = {}
for layer in synt_test.keys():
	idx = np.argsort(correlation[layer])[::-1]
	bio_test[layer] = bio_test_provv[:,idx[:args.n_used_features]]
	synt_test[layer] = synt_test[layer][:,idx[:args.n_used_features]]
	synt_ilsvrc2012[layer] = synt_ilsvrc2012[layer][:,idx[:args.n_used_features]]
	# Create the best features masks in (EEG channels × EEG time points) format
	mask = np.zeros((n_eeg_chan*n_eeg_time), dtype=int)
	mask[idx[:args.n_used_features]] = 1
	best_features_masks[layer] = np.reshape(mask, (n_eeg_chan,n_eeg_time))
del bio_test_provv


# =============================================================================
# Correlate the biological test data with the candidate synthetic data
# conditions
# =============================================================================
tot_images = synt_test.shape[0] + synt_ilsvrc2012.shape[0]
# Correlation matrix of shape: (Test images × Tot Images)
correlation = {}
for layer in synt_test.keys():
	correlation[layer] = np.zeros((n_test_img,tot_images))

# Perform the correlation
for b in tqdm(range(n_test_img)):
	for p in range(tot_images):
		for layer in synt_test.keys():
			if p < n_test_img:
				correlation[layer][b,p] = corr(bio_test[layer][b,:],
					synt_test[layer][p,:])[0]
			else:
				correlation[layer][b,p] = corr(bio_test[layer][b,:],
					synt_ilsvrc2012[layer][p-n_test_img,:])[0]
del bio_test, synt_test, synt_ilsvrc2012


# =============================================================================
# Sort the correlation values for the identification
# =============================================================================
steps = np.arange(0, tot_images, 1000)
n_steps = len(steps)
# Sorted data matrix of shape: (Iterations × Test images × Steps)
zero_shot_identification = {}
for layer in correlation.keys():
	zero_shot_identification[layer] = np.zeros((args.n_iter,n_test_img,n_steps),
		dtype=int)

for i in tqdm(range(args.n_iter)):
	for s in range(n_steps):
		# Randomly select the additional ILSVRC-2012 candidate synthetic data
		# conditions
		idx_ilsvrc2012 = resample(np.arange(n_test_img, tot_images),
			replace=False, n_samples=steps[s])
		for layer in correlation.keys():
			corr = np.append(correlation[layer][:,:n_test_img],
				correlation[layer][:,idx_ilsvrc2012], 1)
			for b in range(n_test_img.shape[0]):
				# Sort the correlation values
				idx = np.argsort(corr[b,:])[::-1]
				# Sort the results
				zero_shot_identification[layer][i,b,s] = np.where(idx == b)[0][0]


# =============================================================================
# Save the results
# =============================================================================
# Store the results into a dictionary
results_dict = {
	'best_features_masks' :best_features_masks,
	'zero_shot_identification': zero_shot_identification,
	'steps': steps
}

# Saving directory
save_dir = os.path.join(args.project_dir, 'results', 'sub-'+
	format(args.sub,'02'), 'zero_shot_identification', 'encoding-linearizing',
	'subjects-'+args.subjects, 'dnn-'+args.dnn, 'pretrained-'+
	str(args.pretrained), 'layers-'+args.layers, 'n_components-'+
	format(args.n_components,'05'))
file_name = 'zero_shot_identification'

# Create the directory if not existing and save
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), results_dict)
