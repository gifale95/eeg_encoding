"""Pairwise decoding of the synthetic EEG test data, and noise ceiling
calculation. For each EEG time point, a SVM classifier is trained to decode
between each combination of two biological data image conditions (using the EEG
channels data), and is then tested on the corresponding combinations of two
synthetic data image conditions.

Parameters
----------
sub : int
	Used subject.
encoding_type : str
	Whether to analyze the 'linearizing' or 'end-to-end' encoding synthetic
	data.
dnn : str
	Used DNN network.
pretrained : bool
	If True, analyze the data synthesized through pretrained (linearizing or
	end-to-end) models. If False, analyze the data synthesized through randomly
	initialized (linearizing or end-to-end) models.
subjects : str
	If 'linearizing' encoding_type is chosen, whether to analyze the 'within' or
	'between' subjects linearizing encoding synthetic data.
layers : str
	If 'linearizing' encoding_type is chosen, whether to analyse the data
	synthesized using 'all', 'single' or 'appended' DNN layers feature maps.
n_components : int
	If 'linearizing' encoding_type is chosen, number of DNN feature maps PCA
	components retained for synthesizing the EEG data.
modeled_time_points : str
	If 'end_to_end' encoding_type is chosen, whether to analyze the synthetic
	data of end-to-end models trained to predict 'single' or 'all' time points.
lr : float
	If 'end_to_end' encoding_type is chosen, learning rate used to train the
	end-to-end encoding models.
weight_decay : float
	If 'end_to_end' encoding_type is chosen, weight decay coefficint used to
	train the end-to-end encoding models.
batch_size : int
	If 'end_to_end' encoding_type is chosen, batch size used to train the
	end-to-end encoding models.
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
parser.add_argument('--encoding_type', default='linearizing', type=str)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--subjects', default='within', type=str)
parser.add_argument('--layers', default='all', type=str)
parser.add_argument('--n_components', default=1000, type=int)
parser.add_argument('--modeled_time_points', type=str, default='single')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_iter', default=100, type=int)
parser.add_argument('--project_dir', default='../project/directory', type=str)
args = parser.parse_args()

print('>>> Pairwise decoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Load the biological EEG test data
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
# Load the synthetic EEG test data
# =============================================================================
if args.encoding_type == 'linearizing':
	data_dir = os.path.join(args.project_dir, 'results', 'sub-'+
		format(args.sub,'02'), 'synthetic_eeg_data', 'encoding-linearizing',
		'subjects-'+args.subjects, 'dnn-'+args.dnn, 'pretrained-'+
		str(args.pretrained), 'layers-'+args.layers, 'n_components-'+
		format(args.n_components,'05'), 'synthetic_eeg_test.npy')
elif args.encoding_type == 'end_to_end':
	data_dir = os.path.join(args.project_dir, 'results', 'sub-'+
		format(args.sub,'02'), 'synthetic_eeg_data', 'encoding-end_to_end',
		'dnn-'+args.dnn, 'modeled_time_points-'+args.modeled_time_points,
		'pretrained-'+str(args.pretrained), 'lr-{:.0e}'.format(args.lr)+
		'__wd-{:.0e}'.format(args.weight_decay)+'__bs-'+
		format(args.batch_size,'03'), 'synthetic_eeg_test.npy')

synt_test = np.load(os.path.join(args.project_dir, data_dir),
	allow_pickle=True).item()
synt_test = synt_test['synthetic_data']


# =============================================================================
# Compute the pairwise decoding and noise ceiling
# =============================================================================
# Results and noise ceiling matrices of shape:
# (Iterations × Image conditions × Image conditions × EEG time points)
pair_dec = {}
for layer in synt_test.keys():
	pair_dec[layer] = np.zeros((args.n_iter,bio_test.shape[0],bio_test.shape[0],
		bio_test.shape[3]))
noise_ceiling_low = np.zeros((args.n_iter,bio_test.shape[0],bio_test.shape[0],
	bio_test.shape[3]))
noise_ceiling_up = np.zeros((args.n_iter,bio_test.shape[0],bio_test.shape[0],
	bio_test.shape[3]))

# Average across all the biological data repetitions for the noise ceiling
# upper bound calculation
bio_data_avg_all = np.mean(bio_test, 1)

# Loop over iterations
for i in tqdm(range(args.n_iter)):
	# Random data repetitions index
	shuffle_idx = resample(np.arange(0, bio_test.shape[1]), replace=False,
		n_samples=int(bio_test.shape[1]/2))
	# Select one half of the biological data repetitions for training the
	# classifier, and average them into 10 pseudo-trials of 4 repetitions
	ptrial_rep = 4
	ptrail_num = int((bio_test.shape[1]/2) / ptrial_rep)
	bio_data_avg_half_1 = np.zeros((bio_test.shape[0],ptrail_num,
		bio_test.shape[2],bio_test.shape[3]))
	bio_data_provv = np.delete(bio_test, shuffle_idx, 1)
	for r in range(bio_data_avg_half_1.shape[1]):
		bio_data_avg_half_1[:,r] = np.mean(
			bio_data_provv[:,r*ptrial_rep:r*ptrial_rep+ptrial_rep], 1)
	del bio_data_provv
	# Average across the other half of the biological data repetitions for the
	# noise ceiling lower bound calculation
	bio_data_avg_half_2 = np.mean(bio_test[:,shuffle_idx], 1)

	# Classifier target vectors
	y_train = np.zeros((bio_data_avg_half_1.shape[1])*2)
	y_train[bio_data_avg_half_1.shape[1]:(bio_data_avg_half_1.shape[1])*2] = 1
	y_test = np.asarray((0, 1))

	# Loop over image-conditions and EEG time points
	for i1 in range(bio_test.shape[0]):
		for i2 in range(bio_test.shape[0]):
			if i1 < i2:
				for t in range(bio_test.shape[3]):
					# Train the classifier
					X_train = np.append(bio_data_avg_half_1[i1,:,:,t],
						bio_data_avg_half_1[i2,:,:,t], 0)
					dec_svm = SVC(kernel="linear")
					dec_svm.fit(X_train, y_train)
					# Test the classifier on the synthetic data
					for layer in synt_test.keys():
						X_test_synt = np.append(np.expand_dims(
							synt_test[layer][i1,:,t], 0), np.expand_dims(
							synt_test[layer][i2,:,t], 0), 0)
						y_pred = dec_svm.predict(X_test_synt)
						pair_dec[layer][i,i2,i1,t] = sum(y_pred == y_test) / len(y_test)
					# Compute the noise ceiling
					X_test_avg_half = np.append(np.expand_dims(
						bio_data_avg_half_2[i1,:,t], 0), np.expand_dims(
						bio_data_avg_half_2[i2,:,t], 0), 0)
					X_test_avg_all = np.append(np.expand_dims(
						bio_data_avg_all[i1,:,t], 0), np.expand_dims(
						bio_data_avg_all[i2,:,t], 0), 0)
					y_pred_noise_ceiling_low = dec_svm.predict(X_test_avg_half)
					y_pred_noise_ceiling_up = dec_svm.predict(X_test_avg_all)
					noise_ceiling_low[i,i2,i1,t] = sum(
						y_pred_noise_ceiling_low == y_test) / len(y_test)
					noise_ceiling_up[i,i2,i1,t] = sum(
						y_pred_noise_ceiling_up == y_test) / len(y_test)


# =============================================================================
# Average the results across iterations and pairwise comparisons
# =============================================================================
# Average across iterations
for layer in synt_test.keys():
	pair_dec[layer] = np.mean(pair_dec[layer], 0)
noise_ceiling_low = np.mean(noise_ceiling_low, 0)
noise_ceiling_up = np.mean(noise_ceiling_up, 0)

# Average across pairwise comparisons
idx = np.tril_indices(pair_dec[layer].shape[0], -1)
for layer in synt_test.keys():
	pair_dec[layer] = np.mean(pair_dec[layer][idx], 0)
noise_ceiling_low = np.mean(noise_ceiling_low[idx], 0)
noise_ceiling_up = np.mean(noise_ceiling_up[idx], 0)


# =============================================================================
# Save the results
# =============================================================================
# Store the results into a dictionary
results_dict = {
	'pairwise_decoding': pair_dec,
	'noise_ceiling_low': noise_ceiling_low,
	'noise_ceiling_up': noise_ceiling_up,
	'times': times,
	'ch_names': ch_names
}

# Saving directory
if args.encoding_type == 'linearizing':
	save_dir = os.path.join(args.project_dir, 'results', 'sub-'+
		format(args.sub,'02'), 'pairwise_decoding', 'encoding-linearizing',
		'subjects-'+args.subjects, 'dnn-'+args.dnn, 'pretrained-'+
		str(args.pretrained), 'layers-'+args.layers, 'n_components-'+
		format(args.n_components,'05'))
elif args.encoding_type == 'end_to_end':
	save_dir = os.path.join(args.project_dir, 'results', 'sub-'+
		format(args.sub,'02'), 'pairwise_decoding', 'encoding-end_to_end',
		'dnn-'+args.dnn, 'modeled_time_points-'+args.modeled_time_points,
		'pretrained-'+str(args.pretrained), 'lr-{:.0e}'.format(args.lr)+
		'__wd-{:.0e}'.format(args.weight_decay)+'__bs-'+
		format(args.batch_size,'03'))
file_name = 'pairwise_decoding.npy'

# Create the directory if not existing and save
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), results_dict)
