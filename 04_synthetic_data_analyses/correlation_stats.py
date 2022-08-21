"""Calculate the confidence intervals (through bootstrap tests) and significance
(through sign permutation tests) of the correlation analysis results, and of the
differences between the results and the noise ceiling.

Parameters
----------
used_subs : list
	List of subjects used for the stats.
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
	Number of iterations for the bootstrap test.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
import itertools
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--used_subs', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
	type=int)
parser.add_argument('--encoding_type', default='linearizing', type=str)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--subjects', default='within', type=str)
parser.add_argument('--layers', default='all', type=str)
parser.add_argument('--n_components', default=1000, type=int)
parser.add_argument('--modeled_time_points', type=str, default='single')
parser.add_argument('--lr', type=float, default=1e-7)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_iter', default=100, type=int)
parser.add_argument('--project_dir', default='../project/directory', type=str)
args = parser.parse_args()

print('>>> Correlation stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Load the correlation results
# =============================================================================
correlation = {}
noise_ceiling_low = []
noise_ceiling_up = []
for s in range(len(args.used_subs)):
	if args.encoding_type == 'linearizing':
		data_dir = os.path.join(args.project_dir, 'results', 'sub-'+
			format(args.sub,'02'), 'correlation', 'encoding-linearizing',
			'subjects-'+s+1, 'dnn-'+args.dnn, 'pretrained-'+
			str(args.pretrained), 'layers-'+args.layers, 'n_components-'+
			format(args.n_components,'05'), 'correlation.npy')
	elif args.encoding_type == 'end-to-end':
		data_dir = os.path.join(args.project_dir, 'results', 'sub-'+
			format(s+1,'02'), 'correlation', 'encoding-end_to_end', 'dnn-'+
			args.dnn, 'modeled_time_points-'+args.modeled_time_points,
			'pretrained-'+str(args.pretrained), 'lr-{:.0e}'.format(args.lr)+
			'__wd-{:.0e}'.format(args.weight_decay)+'__bs-'+
			format(args.batch_size,'03'), 'correlation.npy')
	results_dict = np.load(os.path.join(args.project_dir, data_dir),
		allow_pickle=True).item()
	for layer in results_dict['correlation'].keys():
		if s == 0:
			correlation[layer] = np.expand_dims(
				results_dict['correlation'][layer], 0)
		else:
			correlation[layer] = np.append(correlation[layer], np.expand_dims(
				results_dict['correlation'][layer], 0), 0)
	noise_ceiling_low.append(results_dict['noise_ceiling_low'])
	noise_ceiling_up.append(results_dict['noise_ceiling_up'])
	times = results_dict['times']
	ch_names = results_dict['ch_names']
del results_dict

# Difference between noise ceiling and predicted data results
noise_ceiling_low = np.asarray(noise_ceiling_low)
noise_ceiling_up = np.asarray(noise_ceiling_up)
diff_noise_ceiling = {}
for layer in correlation.keys():
	diff_noise_ceiling[layer] = noise_ceiling_low - correlation[layer]


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
ci_lower = {}
ci_upper = {}
ci_lower_diff_noise_ceiling = {}
ci_upper_diff_noise_ceiling = {}
# Calculate the CIs independently at each time point
for layer in correlation.keys():
	# CI matrices of shape: (Time)
	ci_lower[layer] = np.zeros((correlation[layer].shape[2]))
	ci_upper[layer] = np.zeros((correlation[layer].shape[2]))
	ci_lower_diff_noise_ceiling[layer] = np.zeros((
		diff_noise_ceiling[layer].shape[2]))
	ci_upper_diff_noise_ceiling[layer] = np.zeros((
		diff_noise_ceiling[layer].shape[2]))
	for t in tqdm(range(correlation[layer].shape[2])):
		sample_dist = np.zeros(args.n_boot_iter)
		sample_dist_diff = np.zeros(args.n_boot_iter)
		for i in range(args.n_boot_iter):
			# Calculate the sample distribution of the correlation values
			# averaged across channels
			sample_dist[i] = np.mean(resample(np.mean(
				correlation[layer][:,:,t], 1)))
			sample_dist_diff[i] = np.mean(resample(np.mean(
				diff_noise_ceiling[layer][:,:,t], 1)))
		# Calculate the 95% confidence intervals
		ci_lower[layer][t] = np.percentile(sample_dist, 2.5)
		ci_upper[layer][t] = np.percentile(sample_dist, 97.5)
		ci_lower_diff_noise_ceiling[layer][t] = np.percentile(
			sample_dist_diff, 2.5)
		ci_upper_diff_noise_ceiling[layer][t] = np.percentile(
			sample_dist_diff, 97.5)


# =============================================================================
# Sign permutation test for significance & multiple comparisons correction
# =============================================================================
# Sign permutation test
sign_permutations = list(itertools.product([-1, 1], repeat=10))
sign_permutations = np.asarray(sign_permutations)
p_values = {}
p_values_diff_noise_ceiling = {}
for layer in correlation.keys():
	# p-values matrices of shape: (Time)
	p_values[layer] = np.ones((correlation[layer].shape[2]))
	p_values_diff_noise_ceiling[layer] = np.ones((
		diff_noise_ceiling[layer].shape[2]))
	for t in tqdm(range(correlation[layer].shape[2])):
		# Create the sign permutation distributions
		permutation_dist = np.zeros(len(sign_permutations))
		permutation_dist_diff = np.zeros(len(sign_permutations))
		for p in range(len(sign_permutations)):
			permutation_dist[p] = np.mean(np.mean(
				correlation[layer][:,:,t], 1) * sign_permutations[p])
			permutation_dist_diff[p] = np.mean(np.mean(
				diff_noise_ceiling[layer][:,:,t], 1) * sign_permutations[p])
		# Calculate the p-values
		p_values[layer][t] = (sum(permutation_dist >= np.mean(
			correlation[layer][:,:,t])) + 1) / (len(permutation_dist) + 1)
		p_values_diff_noise_ceiling[layer][t] = (sum(permutation_dist_diff >= \
			np.mean(diff_noise_ceiling[layer][:,:,t])) + 1) / (len(
			permutation_dist) + 1)

# Correct for multiple comparisons
significance = {}
significance_diff_noise_ceiling = {}
for layer in p_values.keys():
	significance[layer] = multipletests(p_values[layer], 0.05, 'fdr_bh')[0]
	significance_diff_noise_ceiling[layer] = multipletests(
		p_values_diff_noise_ceiling[layer], 0.05, 'fdr_bh')[0]


# =============================================================================
# Save the results
# =============================================================================
# Store the results into a dictionary
stats_dict = {
	'correlation': correlation,
	'ci_lower': ci_lower,
	'ci_upper': ci_upper,
	'p_values': p_values,
	'significance': significance,
	'noise_ceiling_low': noise_ceiling_low,
	'noise_ceiling_up': noise_ceiling_up,
	'diff_noise_ceiling': diff_noise_ceiling,
	'ci_lower_diff_noise_ceiling': ci_lower_diff_noise_ceiling,
	'ci_upper_diff_noise_ceiling': ci_upper_diff_noise_ceiling,
	'p_values_diff_noise_ceiling': p_values_diff_noise_ceiling,
	'significance_diff_noise_ceiling': significance_diff_noise_ceiling,
	'times': times,
	'ch_names': ch_names
}

# Saving directory
if args.encoding_type == 'linearizing':
	save_dir = os.path.join(args.project_dir, 'results', 'stats', 'correlation',
		'encoding-linearizing', 'subjects-'+args.subjects, 'dnn-'+args.dnn,
		'pretrained-'+str(args.pretrained), 'layers-'+args.layers,
		'n_components-'+format(args.n_components,'05'))
elif args.encoding_type == 'end-to-end':
	save_dir = os.path.join(args.project_dir, 'results', 'stats', 'correlation',
		'encoding-end_to_end', 'dnn-'+args.dnn, 'modeled_time_points-'+
		args.modeled_time_points, 'pretrained-'+str(args.pretrained),
		'lr-{:.0e}'.format(args.lr)+'__wd-{:.0e}'.format(args.weight_decay)+
		'__bs-'+format(args.batch_size,'03'))
file_name = 'correlation_stats.npy'

# Create the directory if not existing and save
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), stats_dict)
