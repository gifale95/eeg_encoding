"""Calculate the confidence intervals (through bootstrap tests) and significance
(through sign permutation tests) of the training data amount analysis.

Parameters
----------
used_subs : list
	List of subjects used for the stats.
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
n_iter : int
	Number of iterations for the bootstrap test.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from sklearn.utils import resample
import pandas as pd
import pingouin as pg
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--used_subs', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
	type=list)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layers', default='all', type=str)
parser.add_argument('--n_components', default=1000, type=int)
parser.add_argument('--n_boot_iter', default=10000, type=int)
parser.add_argument('--project_dir', default='../project/directory', type=str)
args = parser.parse_args()

print('>>> Training data amount stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Load the correlation results
# =============================================================================
# Used image conditions, EEG repetitions and DNNs
used_img_cond = [4135, 8270, 12405, 16540]
used_eeg_rep = [1, 2, 3, 4]
dnns = ['alexnet', 'resnet50', 'cornet_s', 'moco']
correlation = {}
noise_ceiling = []

# Load the correlation results
for s, sub in enumerate(args.used_subs):
	corr = {}
	noise_ceil = []
	for c, img_cond in enumerate(used_img_cond):
		for r, eeg_rep in enumerate(used_eeg_rep):
			for d, dnn in enumerate(dnns):
				data_dir = os.path.join('results', 'sub-'+format(sub,'02'),
					'training_data_amount_analysis', 'dnn-'+dnn,
					'pretrained-'+str(args.pretrained), 'layers-'+args.layers,
					'n_components-'+format(args.n_components,'05'),
					'training_data_amount_n_img_cond-'+format(img_cond,'06')+
					'_n_eeg_rep-'+format(eeg_rep,'02')+'.npy')
				results_dict = np.load(os.path.join(args.project_dir, data_dir),
					allow_pickle=True).item()
				for layer in results_dict['correlation'].keys():
					if c == 0 and r == 0 and d == 0:
						# Results matrix of shape:
						# (Used image conditions × Used EEG repetitions × DNNs)
						corr[layer] = np.zeros((len(used_img_cond),
							len(used_eeg_rep),len(dnns)))
					corr[layer][c,r,d] = results_dict['correlation'][layer]
				noise_ceil.append(results_dict['noise_ceiling'])
			# Average the correlation results across DNNs
			for layer in corr.keys():
				corr[layer] = np.mean(corr[layer], 2)
	# Average the noise ceiling across conditions, repetitions and DNNs
	noise_ceil = np.mean(np.asarray(noise_ceil))
	# Append the data across subjects
	for layer in corr.keys():
		if s == 0:
			correlation[layer] = np.expand_dims(corr[layer], 0)
		else:
			correlation[layer] = np.append(correlation[layer], np.expand_dims(
				corr[layer], 0), 0)
	noise_ceiling.append(noise_ceil)
noise_ceiling = np.asarray(noise_ceiling)


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
ci_lower = {}
ci_upper = {}
for layer in correlation.keys():
	# CI matrices of shape: (Used image condition × Used EEG repetitions)
	ci_lower[layer] = np.zeros((correlation[layer].shape[1],
		correlation[layer].shape[2]))
	ci_upper[layer] = np.zeros((correlation[layer].shape[1],
		correlation[layer].shape[2]))
	# Calculate the CIs
	for c in range(len(used_img_cond)):
		for r in range(len(used_eeg_rep)):
			sample_dist = np.zeros(args.n_boot_iter)
			for i in range(args.n_boot_iter):
				# Calculate the sample distribution of the correlation results
				sample_dist[i] = np.mean(resample(correlation[layer][:,c,r]))
			# Calculate the 95% confidence intervals
			ci_lower[layer][c,r] = np.percentile(sample_dist, 2.5)
			ci_upper[layer][c,r] = np.percentile(sample_dist, 97.5)


# =============================================================================
# Perform the two-way ANOVA
# =============================================================================
anova_summary = {}
for layer in correlation.keys():
	# Fisher transform the correlation values
	corr_shape = correlation[layer].shape
	fisher_vaules = np.arctanh(np.reshape(correlation[layer], -1))
	# Organizing the data for the two-way ANOVA
	df = pd.DataFrame({
		'subs': np.repeat(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], 16),
		'img_cond': np.tile(np.repeat(['4135', '8270', '12405', '16540'], 4), 10),
		'eeg_rep': np.tile(np.tile(['1', '2', '3', '4'], 4), 10),
		'fisher_vaules': list(np.reshape(fisher_vaules, -1))
		})

# Two-way ANOVA
anova_summary = pg.rm_anova(data=df, dv='fisher_vaules', within=['img_cond',
	'eeg_rep'], subject='subs')


# =============================================================================
# Perform the t-tests & multiple comparisons correction
# =============================================================================
# Select the correlation results of the encoding models built using varying
# amounts of training data (25%, 50%, 75%, 100%) while using either all the
# image conditions and a fractions of the EEG repetitions, or all of the EEG
# repetitions and a fraction of the image conditions
corr_res_all_img_cond = {}
corr_res_all_eeg_rep = {}
for layer in correlation.keys():
	corr_res_all_img_cond[layer] = correlation[layer][:,3,:]
	corr_res_all_eeg_rep[layer] = correlation[layer][:,:,3]

# t-test
p_values = {}
for layer in corr_res_all_img_cond.keys():
	# p-values matrices of shape: (Training data amounts)
	p_values[layer] = np.ones((corr_res_all_img_cond[layer].shape[1]))
	for a in range(corr_res_all_img_cond.shape[1]):
		# Fisher transform the correlation values and perform the t-tests
		fisher_all_img_cond = np.arctanh(corr_res_all_img_cond[layer])
		fisher_all_eeg_rep = np.arctanh(corr_res_all_eeg_rep[layer])
		p_values[layer] = ttest_rel(fisher_all_img_cond[:,a],
			fisher_all_eeg_rep[:,a], alternative='two-sided')[1]

# Correct for multiple comparisons
significance = {}
for layer in p_values.keys():
	significance[layer] = multipletests(p_values[layer], 0.05, 'bonferroni')[0]


# =============================================================================
# Save the results
# =============================================================================
# Store the results into a dictionary
stats_dict = {
	'correlation': correlation,
	'noise_ceiling': noise_ceiling,
	'ci_lower': ci_lower,
	'ci_upper': ci_upper,
	'anova_summary': anova_summary,
	'corr_res_all_img_cond': corr_res_all_img_cond,
	'corr_res_all_eeg_rep': corr_res_all_eeg_rep,
	'significance': significance
}

# Saving directory
save_dir = os.path.join(args.project_dir, 'results', 'stats',
	'training_data_amount_analysis', 'dnn-'+args.dnn, 'pretrained-'+
	str(args.pretrained), 'layers-'+args.layers, 'n_components-'+
	format(args.n_components,'05'))
file_name = 'training_data_amount_analysis_stats.npy'

# Create the directory if not existing and save
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), stats_dict)
