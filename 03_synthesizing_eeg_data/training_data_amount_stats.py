"""Confidence intervals and significance of the training data amount analysis.

Parameters
----------
n_tot_sub : int
	Number of total subjects used.
n_boot_iter : int
	Number of bootstrap iterations for the confidence intervals.
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
parser.add_argument('--n_tot_sub', default=10, type=int)
parser.add_argument('--n_boot_iter', default=10000, type=int)
parser.add_argument('--project_dir', default='/project/directory', type=str)
args = parser.parse_args()

print('\n\n\n>>> Training data amount stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
np.random.seed(seed=20200220)


# =============================================================================
# Loading the correlation results, averaged acorss DNNs
# =============================================================================
# Used image conditions, EEG repetitions and DNNs
used_img_cond = [4135, 8270, 12405, 16540]
used_eeg_rep = [1, 2, 3, 4]
dnns = ['alexnet', 'resnet50', 'cornet_s', 'moco']
correlation = []

# Loading the correlation results
for s in range(args.n_tot_sub):
	# Results matrix of shape: Used image condition × Used EEG repetitions
	corr_mat = np.zeros((len(used_img_cond),len(used_eeg_rep)))
	for c,img_cond in enumerate(used_img_cond):
		for r,eeg_rep in enumerate(used_eeg_rep):
			for d in dnns:
				corr_res = []
				data_dir = os.path.join('results', 'sub-'+format(s+1,'02'),
					'training_data_amount_analysis', 'dnn-'+d,
					'training_data_amount_n_img_cond-'+format(img_cond,'06')+
					'_n_eeg_rep-'+format(eeg_rep,'02')+'.npy')
				corr_res.append(np.load(os.path.join(args.project_dir,
					data_dir)))
			# Averaging the results across DNNs
			corr_res = np.mean(np.asarray(corr_res), 0)
			corr_mat[c,r] = corr_res
	correlation.append(corr_mat)

# Converting to numpy format
correlation = np.asarray(correlation)

# Selecting the correlation results of the encoding models built using varying
# amounts of training data (25%, 50%, 75%, 100%), while using either all the
# image conditions and a fractions of the EEG repetitions, or all of the EEG
# repetitions and a fraction of the image conditions
corr_res_all_img_cond = correlation[:,3,:]
corr_res_all_eeg_rep = correlation[:,:,3]


# =============================================================================
# Bootstrapping the confidence intervals (CIs)
# =============================================================================
# CI matrices of shape: Used image condition × Used EEG repetitions
ci_lower = np.zeros((correlation.shape[1],correlation.shape[2]))
ci_upper= np.zeros((correlation.shape[1],correlation.shape[2]))

# Calculating the CIs
for c in range(len(used_img_cond)):
	for r in range(len(used_eeg_rep)):
		sample_dist = np.zeros(args.n_boot_iter)
		for i in range(args.n_boot_iter):
			# Calculating the sample distribution
			sample_dist[i] = np.mean(resample(correlation[:,c,r]))
		# Calculating the confidence intervals
		ci_lower[c,r] = np.percentile(sample_dist, 2.5)
		ci_upper[c,r] = np.percentile(sample_dist, 97.5)


# =============================================================================
# Performing the two-way ANOVA
# =============================================================================
# Organizing the data for the two-way ANOVA
df = pd.DataFrame({
	'subs': np.repeat(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], 16),
	'img_cond': np.tile(np.repeat(['4135', '8270', '12405', '16540'], 4), 10),
	'eeg_rep': np.tile(np.tile(['1', '2', '3', '4'], 4), 10),
	'corr_res': list(np.reshape(correlation, -1))
	})

# Performing the two-way ANOVA
anova_summary = pg.rm_anova(data=df, dv='corr_res', within=['img_cond',
	'eeg_rep'], subject='subs')


# =============================================================================
# Performing the t-tests & multiple comparisons correction
# =============================================================================
# p-values matrices of shape: Training data amounts
p_values_ttest = np.ones((corr_res_all_img_cond.shape[1]))
for a in range(corr_res_all_img_cond.shape[1]):
	_, p_values_ttest[a] = ttest_rel(corr_res_all_img_cond[:,a],
		corr_res_all_eeg_rep[:,a], alternative='two-sided')

# Correcting for multiple comparisons
results = multipletests(p_values_ttest, 0.05, 'bonferroni')
significance_ttest = results[0]


# =============================================================================
# Saving the results
# =============================================================================
# Storing the results into a dictionary
stats_dict = {
	'correlation': correlation,
	'ci_lower': ci_lower,
	'ci_upper': ci_upper,
	'anova_summary': anova_summary,
	'corr_res_all_img_cond': corr_res_all_img_cond,
	'corr_res_all_eeg_rep': corr_res_all_eeg_rep,
	'significance_ttest': significance_ttest
}

# Saving directory
save_dir = os.path.join(args.project_dir, 'results', 'stats',
	'training_data_amount_analysis')
file_name = 'training_data_amount_analysis_stats.npy'

# Creating the directory if not existing and saving
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), stats_dict)
