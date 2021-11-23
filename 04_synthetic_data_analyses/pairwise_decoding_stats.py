"""Confidence intervals and significance of the pairwise decoding analysis
results, and of the differences between the results and the noise ceiling.

Parameters
----------
n_tot_sub : int
	Number of total subjects used.
dnn : str
	Used DNN network.
n_boot_iter : int
	Number of bootstrap iterations for the confidence intervals.
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
parser.add_argument('--n_tot_sub', default=10, type=int)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--n_boot_iter', default=10000, type=int)
parser.add_argument('--project_dir', default='/project/directory', type=str)
args = parser.parse_args()

print('\n\n\n>>> Pairwise decoding stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
np.random.seed(seed=20200220)


# =============================================================================
# Loading the correlation results
# =============================================================================
pairwise_decoding_within = []
pairwise_decoding_between = []
noise_ceiling = []
for s in range(args.n_tot_sub):
	data_dir = os.path.join('results', 'sub-'+format(s+1,'02'),
		'pairwise_decoding', 'dnn-'+args.dnn, 'pairwise_decoding.npy')
	results_dict = np.load(os.path.join(args.project_dir, data_dir),
		allow_pickle=True).item()
	pairwise_decoding_within.append(results_dict['pairwise_decoding_within'])
	pairwise_decoding_between.append(results_dict['pairwise_decoding_between'])
	noise_ceiling.append(results_dict['noise_ceiling'])
	times = results_dict['times']
	ch_names = results_dict['ch_names']

# Averaging the results across EEG channels
pairwise_decoding_within = np.mean(np.asarray(pairwise_decoding_within), 1)
pairwise_decoding_between = np.mean(np.asarray(pairwise_decoding_between), 1)
noise_ceiling = np.mean(np.asarray(noise_ceiling), 1)
del results_dict

# Difference between noise ceiling and predicted data results
diff_noise_ceiling = noise_ceiling - pairwise_decoding_within

# =============================================================================
# Bootstrapping the confidence intervals (CIs)
# =============================================================================
# CI matrices of shape: Time
ci_lower_within = np.zeros((pairwise_decoding_within.shape[1]))
ci_upper_within = np.zeros((pairwise_decoding_within.shape[1]))
ci_lower_between = np.zeros((pairwise_decoding_between.shape[1]))
ci_upper_between = np.zeros((pairwise_decoding_between.shape[1]))
ci_lower_diff_noise_ceiling = np.zeros((diff_noise_ceiling.shape[1]))
ci_upper_diff_noise_ceiling = np.zeros((diff_noise_ceiling.shape[1]))

# Calculating the CIs independently at each time point
for t in tqdm(range(pairwise_decoding_within.shape[1])):
	sample_dist_within = np.zeros(args.n_boot_iter)
	sample_dist_between = np.zeros(args.n_boot_iter)
	sample_dist_diff = np.zeros(args.n_boot_iter)
	for i in range(args.n_boot_iter):
		# Calculating the sample distribution
		sample_dist_within[i] = np.mean(resample(pairwise_decoding_within[:,t]))
		sample_dist_between[i] = np.mean(resample(
			pairwise_decoding_between[:,t]))
		sample_dist_diff[i] = np.mean(resample(diff_noise_ceiling[:,t]))
	# Calculating the confidence intervals
	ci_lower_within[t] = np.percentile(sample_dist_within, 2.5)
	ci_upper_within[t] = np.percentile(sample_dist_within, 97.5)
	ci_lower_between[t] = np.percentile(sample_dist_between, 2.5)
	ci_upper_between[t] = np.percentile(sample_dist_between, 97.5)
	ci_lower_diff_noise_ceiling[t] = np.percentile(sample_dist_diff, 2.5)
	ci_upper_diff_noise_ceiling[t] = np.percentile(sample_dist_diff, 97.5)


# =============================================================================
# Performing the t-tests & multiple comparisons correction
# =============================================================================
# p-values matrices of shape: Time
p_values_within = np.ones((pairwise_decoding_within.shape[1]))
p_values_between = np.ones((pairwise_decoding_between.shape[1]))
p_values_difference_noise_ceiling = np.ones((diff_noise_ceiling.shape[1]))
for t in range(pairwise_decoding_within.shape[1]):
	_, p_values_within[t] = ttest_1samp(pairwise_decoding_within[:,t], 0,
		alternative='greater')
	_, p_values_between[t] = ttest_1samp(pairwise_decoding_between[:,t], 0,
		alternative='greater')
	_, p_values_difference_noise_ceiling[t] = ttest_1samp(
		diff_noise_ceiling[:,t], 0, alternative='greater')

# Correcting for multiple comparisons
results_within = multipletests(p_values_within, 0.05, 'bonferroni')
significance_within = results_within[0]
results_between = multipletests(p_values_between, 0.05, 'bonferroni')
significance_between = results_between[0]
results_diff_noise_ceiling = multipletests(p_values_difference_noise_ceiling,
	0.05, 'bonferroni')
significance_diff_noise_ceiling = results_diff_noise_ceiling[0]


# =============================================================================
# Saving the results
# =============================================================================
# Storing the results into a dictionary
stats_dict = {
	'pairwise_decoding_within': pairwise_decoding_within,
	'ci_lower_within': ci_lower_within,
	'ci_upper_within': ci_upper_within,
	'significance_within': significance_within,
	'pairwise_decoding_between': pairwise_decoding_between,
	'ci_lower_between': ci_lower_between,
	'ci_upper_between': ci_upper_between,
	'significance_between': significance_between,
	'noise_ceiling': noise_ceiling,
	'diff_noise_ceiling': diff_noise_ceiling,
	'ci_lower_diff_noise_ceiling': ci_lower_diff_noise_ceiling,
	'ci_upper_diff_noise_ceiling': ci_upper_diff_noise_ceiling,
	'significance_diff_noise_ceiling': significance_diff_noise_ceiling,
	'times': times,
	'ch_names': ch_names
}

# Saving directory
save_dir = os.path.join(args.project_dir, 'results', 'stats',
	'pairwise_decoding', 'dnn-'+args.dnn)
file_name = 'pairwise_decoding_stats.npy'

# Creating the directory if not existing and saving
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), stats_dict)
