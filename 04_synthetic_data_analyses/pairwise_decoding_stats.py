"""Calculate the confidence intervals (through bootstrap tests) and significance
(through one-sample t-tests) of the pairwise decoding analysis results, and of
the differences between the results and the noise ceiling.

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
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--used_subs', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
	type=list)
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

print('>>> Pairwise decoding stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Load the pairwise_decoding results
# =============================================================================
decoding = {}
noise_ceiling_low = []
noise_ceiling_up = []
for s, sub in enumerate(args.used_subs):
	if args.encoding_type == 'linearizing':
		data_dir = os.path.join(args.project_dir, 'results', 'sub-'+
			format(sub,'02'), 'pairwise_decoding', 'encoding-linearizing',
			'subjects-'+args.subjects, 'dnn-'+args.dnn, 'pretrained-'+
			str(args.pretrained), 'layers-'+args.layers, 'n_components-'+
			format(args.n_components,'05'), 'pairwise_decoding.npy')
	elif args.encoding_type == 'end_to_end':
		data_dir = os.path.join(args.project_dir, 'results', 'sub-'+
			format(sub,'02'), 'pairwise_decoding', 'encoding-end_to_end',
			'dnn-'+args.dnn, 'modeled_time_points-'+args.modeled_time_point,
			'pretrained-'+str(args.pretrained), 'lr-{:.0e}'.format(args.lr)+
			'__wd-{:.0e}'.format(args.weight_decay)+'__bs-'+
			format(args.batch_size,'03'), 'pairwise_decoding.npy')
	results_dict = np.load(os.path.join(args.project_dir, data_dir),
		allow_pickle=True).item()
	for layer in results_dict['pairwise_decoding'].keys():
		if s == 0:
			decoding[layer] = np.expand_dims(
				results_dict['pairwise_decoding'][layer], 0)
		else:
			decoding[layer] = np.append(decoding[layer], np.expand_dims(
				results_dict['pairwise_decoding'][layer], 0), 0)
	noise_ceiling_low.append(results_dict['noise_ceiling_low'])
	noise_ceiling_up.append(results_dict['noise_ceiling_up'])
	times = results_dict['times']
	ch_names = results_dict['ch_names']
del results_dict

# Difference between noise ceiling and predicted data results
noise_ceiling_low = np.asarray(noise_ceiling_low)
noise_ceiling_up = np.asarray(noise_ceiling_up)
diff_noise_ceiling = {}
for layer in decoding.keys():
	diff_noise_ceiling[layer] = noise_ceiling_low - decoding[layer]


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
ci_lower = {}
ci_upper = {}
ci_lower_diff_noise_ceiling = {}
ci_upper_diff_noise_ceiling = {}
# Calculate the CIs independently at each time point
for layer in decoding.keys():
	# CI matrices of shape: (Time)
	ci_lower[layer] = np.zeros((decoding[layer].shape[1]))
	ci_upper[layer] = np.zeros((decoding[layer].shape[1]))
	ci_lower_diff_noise_ceiling[layer] = np.zeros((
		diff_noise_ceiling[layer].shape[1]))
	ci_upper_diff_noise_ceiling[layer] = np.zeros((
		diff_noise_ceiling[layer].shape[1]))
	for t in tqdm(range(decoding[layer].shape[1])):
		sample_dist = np.zeros(args.n_iter)
		sample_dist_diff = np.zeros(args.n_iter)
		for i in range(args.n_iter):
			# Calculate the sample distribution of the pairwise deocoding
			# results
			sample_dist[i] = np.mean(resample(decoding[layer][:,t]))
			sample_dist_diff[i] = np.mean(resample(
				diff_noise_ceiling[layer][:,t]))
		# Calculate the 95% confidence intervals
		ci_lower[layer][t] = np.percentile(sample_dist, 2.5)
		ci_upper[layer][t] = np.percentile(sample_dist, 97.5)
		ci_lower_diff_noise_ceiling[layer][t] = np.percentile(
			sample_dist_diff, 2.5)
		ci_upper_diff_noise_ceiling[layer][t] = np.percentile(
			sample_dist_diff, 97.5)


# =============================================================================
# One-sample t-tests for significance & multiple comparisons correction
# =============================================================================
p_values = {}
p_values_diff_noise_ceiling = {}
for layer in decoding.keys():
	# p-values matrices of shape: (Time)
	p_values[layer] = np.ones((decoding[layer].shape[1]))
	p_values_diff_noise_ceiling[layer] = np.ones((
		diff_noise_ceiling[layer].shape[1]))
	for t in range(decoding[layer].shape[1]):
		# Fisher transform the pairwise decoding values and perform the t-tests
		fisher_vaules = np.arctanh(decoding[layer][:,t])
		fisher_vaules_diff_nc = np.arctanh(diff_noise_ceiling[layer][:,t])
		p_values[layer][t] = ttest_1samp(fisher_vaules, .5,
			alternative='greater')[1]
		p_values_diff_noise_ceiling[layer][t] = ttest_1samp(
			fisher_vaules_diff_nc, 0, alternative='greater')[1]

# Correct for multiple comparisons
significance = {}
significance_diff_noise_ceiling = {}
for layer in p_values.keys():
	significance[layer] = multipletests(p_values[layer], 0.05, 'bonferroni')[0]
	significance_diff_noise_ceiling[layer] = multipletests(
		p_values_diff_noise_ceiling[layer], 0.05, 'bonferroni')[0]


# =============================================================================
# Save the results
# =============================================================================
# Store the results into a dictionary
stats_dict = {
	'decoding': decoding,
	'ci_lower': ci_lower,
	'ci_upper': ci_upper,
	'significance': significance,
	'noise_ceiling_low': noise_ceiling_low,
	'noise_ceiling_up': noise_ceiling_up,
	'diff_noise_ceiling': diff_noise_ceiling,
	'ci_lower_diff_noise_ceiling': ci_lower_diff_noise_ceiling,
	'ci_upper_diff_noise_ceiling': ci_upper_diff_noise_ceiling,
	'significance_diff_noise_ceiling': significance_diff_noise_ceiling,
	'times': times,
	'ch_names': ch_names
}

# Saving directory
if args.encoding_type == 'linearizing':
	save_dir = os.path.join(args.project_dir, 'results', 'stats',
		'pairwise_decoding', 'encoding-linearizing', 'subjects-'+args.subjects,
		'dnn-'+args.dnn, 'pretrained-'+str(args.pretrained), 'layers-'+
		args.layers, 'n_components-'+format(args.n_components,'05'))
elif args.encoding_type == 'end_to_end':
	save_dir = os.path.join(args.project_dir, 'results', 'stats',
		'pairwise_decoding', 'encoding-end_to_end', 'dnn-'+args.dnn,
		'modeled_time_points-'+args.modeled_time_points, 'pretrained-'+
		str(args.pretrained), 'lr-{:.0e}'.format(args.lr)+
		'__wd-{:.0e}'.format(args.weight_decay)+'__bs-'+
		format(args.batch_size,'03'))
file_name = 'pairwise_decoding_stats.npy'

# Create the directory if not existing and save
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), stats_dict)
