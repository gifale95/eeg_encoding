"""Confidence intervals, significance and function fitting of the zero-shot
decoding analysis results.

Parameters
----------
n_tot_sub : int
	Number of total subjects used.
dnn : str
	Used DNN network.
rank_correct : int
	Accepted correlation rank of the correct synthetic data image condition.
n_boot_iter : int
	Number of bootstrap iterations for the confidence intervals.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from scipy.optimize import curve_fit
from sklearn.utils import resample
from tqdm import tqdm
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--n_tot_sub', default=10, type=int)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--rank_correct', default=1, type=int)
parser.add_argument('--n_boot_iter', default=10000, type=int)
parser.add_argument('--project_dir', default='/project/directory', type=str)
args = parser.parse_args()

print('\n\n\n>>> Zero-shot decoding stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
np.random.seed(seed=20200220)


# =============================================================================
# Loading the zero-shot decoding results
# =============================================================================
zero_shot_decoding = []
for s in range(args.n_tot_sub):
	data_dir = os.path.join('results', 'sub-'+format(s+1,'02'),
		'zero_shot_decoding', 'dnn-'+args.dnn, 'zero_shot_decoding.npy')
	results_dict = np.load(os.path.join(args.project_dir, data_dir),
		allow_pickle=True).item()
	zero_shot_decoding.append(results_dict['zero_shot_decoding'])
	steps = results_dict['steps']
zero_shot_decoding = np.asarray(zero_shot_decoding)
del results_dict


# =============================================================================
# Calculating the accuracy
# =============================================================================
# Accuracy matrix of shape: Subjects × Iterations × Steps
decoding_accuracy = np.zeros((zero_shot_decoding.shape[0],
	zero_shot_decoding.shape[1], zero_shot_decoding.shape[3]))
for s in range(decoding_accuracy.shape[0]):
	for i in range(decoding_accuracy.shape[1]):
		for st in range(decoding_accuracy.shape[2]):
			decoding_accuracy[s,i,st] = len(np.where(
				zero_shot_decoding[s,i,:,st] <= args.rank_correct-1)[0]) /\
				zero_shot_decoding.shape[2] * 100
del zero_shot_decoding

# Averaging across iterations
decoding_accuracy = np.mean(decoding_accuracy, 1)


# =============================================================================
# Bootstrapping the confidence intervals
# =============================================================================
sample_dist = np.zeros((decoding_accuracy.shape[1],args.n_boot_iter))
for st in tqdm(range(decoding_accuracy.shape[1])):
	for i in range(args.n_boot_iter):
		# Calculating the sample distribution
		sample_dist[st,i] = np.mean(resample(decoding_accuracy[:,st]))
# Calculating the confidence intervals
ci_lower = np.percentile(sample_dist, 2.5, axis=1)
ci_upper= np.percentile(sample_dist, 97.5, axis=1)


# =============================================================================
# Performing the t-tests & multiple comparisons correction
# =============================================================================
p_values = np.zeros((decoding_accuracy.shape[1]))
for st in range(decoding_accuracy.shape[1]):
	chance = args.rank_correct / (200+steps[st]) * 100
	_, p_val = ttest_1samp(decoding_accuracy[:,st], chance,
		alternative='greater')
	p_values[st] = p_val

# Correcting for multiple comparisons
results = multipletests(p_values, 0.05, 'bonferroni')
significance = results[0]


# =============================================================================
# Fitting a power-law function to the data and extrapolating performance drop
# =============================================================================
# Defining the power-law function
def power_law(x, a, b):
	return a*np.power(x, b)

# Fitting the power-law function for each subject, using the zero-shot decoding
# results of candidate image set sizes 50,200 to 150,200
popt_pow = []
for s in range(decoding_accuracy.shape[0]):
	popt_pow_sub, _ = curve_fit(power_law, steps[50:]+200,
		decoding_accuracy[s,50:])
	popt_pow.append(popt_pow_sub)

# Extrapolating how many image conditions are required for the decoding
# accuracy to drop below 10% (with steps of 1000 images)
extr_10_percent = np.zeros(decoding_accuracy.shape[0])
for s in range(len(popt_pow)):
	n = 0
	acc = 100
	while acc >= 10:
		acc = power_law(n+200, *popt_pow[s])
		n += 1000
	extr_10_percent[s] = n

# Extrapolating how many image conditions are required for the decoding
# accuracy to drop below 0.5% (with steps of 1000 images)
extr_0point5_percent = np.zeros(decoding_accuracy.shape[0])
for s in range(len(popt_pow)):
	n = 0
	acc = 100
	while acc >= 0.5:
		acc = power_law(n+200, *popt_pow[s])
		n += 1000
	extr_0point5_percent[s] = n


# =============================================================================
# Saving the results
# =============================================================================
# Storing the results into a dictionary
stats_dict = {
	'decoding_accuracy': decoding_accuracy,
	'ci_lower': ci_lower,
	'ci_upper': ci_upper,
	'significance': significance,
	'extr_10_percent': extr_10_percent,
	'extr_0point5_percent': extr_0point5_percent,
	'steps': steps
}

# Saving directory
save_dir = os.path.join(args.project_dir, 'results', 'stats',
	'zero_shot_decoding', 'dnn-'+args.dnn, 'rank_correct-'+
	format(args.rank_correct,'02'))
file_name = 'zero_shot_decoding_stats.npy'

# Creating the directory if not existing and saving
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), stats_dict)
