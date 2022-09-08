"""Calculate the confidence intervals (through bootstrap tests) and significance
(through sign permutation tests) of the zero-shot identification analysis
results, and of the differences between the results and the noise ceiling.
Furthermore, fit a power-law function to the identification results to
extrapolate the image set sizes needed for the identification accuracy to fall
below certain thresholds.

Parameters
----------
used_subs : list
	List of subjects used for the stats.
rank_correct : int
	Accepted correlation rank of the correct synthetic data image condition.
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
from scipy.optimize import curve_fit


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--used_subs', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
	type=list)
parser.add_argument('--rank_correct', default=1, type=int)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--subjects', default='within', type=str)
parser.add_argument('--layers', default='all', type=str)
parser.add_argument('--n_components', default=1000, type=int)
parser.add_argument('--n_iter', default=100, type=int)
parser.add_argument('--project_dir', default='../project/directory', type=str)
args = parser.parse_args()

print('>>> Zero-shot identification stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Load the zero-shot decoding results
# =============================================================================
zero_shot_identification = {}
best_features_masks = {}
for s, sub in enumerate(args.used_subs):
	data_dir = os.path.join('results', 'sub-'+format(sub,'02'),
		'zero_shot_identification', 'encoding-linearizing','subjects-'+
		args.subjects, 'dnn-'+args.dnn, 'pretrained-'+str(args.pretrained),
		'layers-'+args.layers, 'n_components-'+format(args.n_components,'05'),
		'zero_shot_identification.npy')
	results_dict = np.load(os.path.join(args.project_dir, data_dir),
		allow_pickle=True).item()
	for layer in results_dict['zero_shot_identification'].keys():
		if s == 0:
			zero_shot_identification[layer] = np.expand_dims(
				results_dict['zero_shot_identification'][layer], 0)
			best_features_masks[layer] = np.expand_dims(
				results_dict['best_features_masks'][layer], 0)
		else:
			zero_shot_identification[layer] = np.append(
				zero_shot_identification[layer], np.expand_dims(
				results_dict['zero_shot_identification'][layer], 0), 0)
			best_features_masks[layer] = np.append(
				best_features_masks[layer], np.expand_dims(
				results_dict['best_features_masks'][layer], 0), 0)
	steps = results_dict['steps']
del results_dict


# =============================================================================
# Calculate the accuracy
# =============================================================================
identification_accuracy = {}
for layer in zero_shot_identification.keys():
	# Accuracy matrix of shape: (Subjects × Iterations × Steps)
	identification_accuracy[layer] = np.zeros((
		zero_shot_identification[layer].shape[0],
		zero_shot_identification[layer].shape[1],
		zero_shot_identification[layer].shape[3]))
	for s in range(identification_accuracy[layer].shape[0]):
		for i in range(identification_accuracy[layer].shape[1]):
			for st in range(identification_accuracy[layer].shape[2]):
				identification_accuracy[layer][s,i,st] = len(np.where(
					zero_shot_identification[layer][s,i,:,st] <= \
					args.rank_correct-1)[0]) / \
					zero_shot_identification[layer].shape[2]
del zero_shot_identification

# Average the accuracy across iterations
for layer in identification_accuracy.keys():
	identification_accuracy[layer] = np.mean(identification_accuracy[layer], 1)


# =============================================================================
# Bootstrap the confidence intervals
# =============================================================================
ci_lower = {}
ci_upper = {}
# Calculate the CIs independently at each step
for layer in identification_accuracy.keys():
	# CI matrices of shape: (Steps)
	ci_lower[layer] = np.zeros((identification_accuracy[layer].shape[1]))
	ci_upper[layer] = np.zeros((identification_accuracy[layer].shape[1]))
	for st in tqdm(range(identification_accuracy[layer].shape[1])):
		sample_dist = np.zeros(args.n_iter)
		for i in range(args.n_iter):
			# Calculate the sample distribution of the identification results
			sample_dist[i] = np.mean(resample(
				identification_accuracy[layer][:,st]))
		# Calculate the 95% confidence intervals
		ci_lower[layer][st] = np.percentile(sample_dist, 2.5)
		ci_upper[layer][st] = np.percentile(sample_dist, 97.5)


# =============================================================================
# One-sample t-tests for significance & multiple comparisons correction
# =============================================================================
p_values = {}
for layer in identification_accuracy.keys():
	# p-values matrices of shape: (Steps)
	p_values[layer] = np.ones((identification_accuracy[layer].shape[1]))
	for st in range(identification_accuracy[layer].shape[1]):
		# Perform the t-tests
		p_values[layer][st] = ttest_1samp(identification_accuracy[layer][:,st],
			0, alternative='greater')[1]

# Correct for multiple comparisons
significance = {}
for layer in p_values.keys():
	significance[layer] = multipletests(p_values[layer], 0.05, 'bonferroni')[0]


# =============================================================================
# Fit a power-law function to the data and extrapolate performance drop
# =============================================================================
# Define the power-law function
def power_law(x, a, b):
	return a*np.power(x, b)

# Fit the power-law function for each subject, using the zero-shot
# identification results of candidate image set sizes 50,200 to 150,200
popt_pow = {}
for layer in identification_accuracy.keys():
	popt_pow[layer] = []
	for s in range(identification_accuracy[layer].shape[0]):
		popt_pow[layer].append(curve_fit(power_law, steps[50:]+200,
			identification_accuracy[layer][s,50:])[0])

# Extrapolate how many image conditions are required for the identification
# accuracy to drop to 10% and 0.5%
extr_10_percent = {}
extr_0point5_percent = {}
for layer in popt_pow.keys():
	for s in range(len(popt_pow[layer])):
		if s == 0:
			extr_10_percent[layer] = np.zeros(len(popt_pow[layer]))
			extr_0point5_percent[layer] = np.zeros(len(popt_pow[layer]))
		# Invert the fit power law function to isolate the image conditions
		a = popt_pow[layer][s][0]
		b = popt_pow[layer][s][1]
		extr_10_percent[layer][s] = (.1 / a) ** (1.0 / b)
		extr_0point5_percent[layer][s] = (.005 / a) ** (1.0 / b)


# =============================================================================
# Saving the results
# =============================================================================
# Storing the results into a dictionary
stats_dict = {
	'identification_accuracy': identification_accuracy,
	'ci_lower': ci_lower,
	'ci_upper': ci_upper,
	'significance': significance,
	'extr_10_percent': extr_10_percent,
	'extr_0point5_percent': extr_0point5_percent,
	'best_features_masks': best_features_masks,
	'steps': steps
}

# Saving directory
save_dir = os.path.join(args.project_dir, 'results', 'stats',
	'zero_shot_identification', 'encoding-linearizing', 'subjects-'+
	args.subjects, 'dnn-'+args.dnn, 'pretrained-'+str(args.pretrained),
	'layers-'+args.layers, 'n_components-'+format(args.n_components,'05'),
	'rank_correct-'+format(args.rank_correct,'02'))
file_name = 'zero_shot_identification_stats.npy'

# Creating the directory if not existing and saving
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), stats_dict)
