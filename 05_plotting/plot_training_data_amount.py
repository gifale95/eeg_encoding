"""Plotting the training data amount analysis results.

Parameters
----------
n_tot_sub : int
	Number of total subjects used.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--n_tot_sub', default=10, type=int)
parser.add_argument('--project_dir', default='/project/directory', type=str)
args = parser.parse_args()


# =============================================================================
# Loading the correlation results and stats
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results', 'stats',
	'linearizing_encoding', 'training_data_amount_analysis',
	'training_data_amount_analysis_stats.npy')
# Loading the data
data_dict = np.load(data_dir, allow_pickle=True).item()
correlation = data_dict['correlation']
noise_ceiling = data_dict['noise_ceiling']
ci_lower = data_dict['ci_lower']
ci_upper = data_dict['ci_upper']
anova_summary = data_dict['anova_summary']
corr_res_all_img_cond = data_dict['corr_res_all_img_cond']
corr_res_all_eeg_rep = data_dict['corr_res_all_eeg_rep']
significance_ttest = data_dict['significance_ttest']

# Organizing the significance values for plotting
sig_ttest = np.zeros(significance_ttest.shape)
for a in range(len(sig_ttest)):
	if significance_ttest[a] == False:
		sig_ttest[a] = -100
	else:
		sig_ttest[a] = .48


# =============================================================================
# Plotting the results averaged across subjects
# =============================================================================
# Setting the plot parameters
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 30
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
matplotlib.rcParams['axes.linewidth'] = 3
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
colors = [(31/255, 119/255, 180/255), (255/255, 127/255, 14/255)]
color_noise_ceiling = (150/255, 150/255, 150/255)

# Plotting the the correlation results of the encoding models built using
# varying amounts (25%, 50%, 75%, 100%) of training image conditions and
# repetitions.
xlabels_1 = ['4,135', '8,270', '12,405', '16,540']
x = np.arange(len(xlabels_1))
width = 0.2
plt.figure(figsize=(9,6))
# Plotting the results
plt.bar(x - width*1.5, np.mean(correlation[:,0], 0), width=width)
plt.bar(x - width*.5, np.mean(correlation[:,1], 0), width=width)
plt.bar(x + width*.5, np.mean(correlation[:,2], 0), width=width)
plt.bar(x + width*1.5, np.mean(correlation[:,3], 0), width=width)
leg = ['1', '2', '3','4']
plt.legend(leg, title='Condition repetitions', ncol=4, fontsize=30,
	frameon=False, loc=2)
# Plotting the noise ceiling
plt.plot([x - width*2, x + width*2], [np.mean(noise_ceiling),
	np.mean(noise_ceiling)], '--', linewidth=4, color=color_noise_ceiling)
# Plotting the confidence intervals
ci_up = ci_upper - np.mean(correlation, 0)
ci_low = np.mean(correlation, 0) - ci_lower
conf_int = np.append(np.expand_dims(ci_low, 0), np.expand_dims(ci_up, 0), 0)
plt.errorbar(x - width*1.5, np.mean(correlation[:,0], 0), yerr=conf_int[:,:,0],
	fmt="none", ecolor="k", elinewidth=2, capsize=4)
plt.errorbar(x - width*.5, np.mean(correlation[:,1], 0), yerr=conf_int[:,:,1],
	fmt="none", ecolor="k", elinewidth=2, capsize=4)
plt.errorbar(x + width*.5, np.mean(correlation[:,2], 0), yerr=conf_int[:,:,2],
	fmt="none", ecolor="k", elinewidth=2, capsize=4)
plt.errorbar(x + width*1.5, np.mean(correlation[:,3], 0), yerr=conf_int[:,:,3],
	fmt="none", ecolor="k", elinewidth=2, capsize=4)
# Other plot parameters
plt.xlabel('Image conditions', fontsize=30)
plt.xticks(ticks=x, labels=xlabels_1)
plt.ylabel('Pearson\'s $r$', fontsize=30)
ylabels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
plt.yticks(ticks=np.arange(0,.51,0.1), labels=ylabels)
plt.ylim(bottom=0, top=.5)

# Plotting the the correlation results of the encoding models built using
# varying amounts of training data (25%, 50%, 75%, 100%), while using either all
# the image conditions and a fractions of the EEG repetitions, or all of the EEG
# repetitions and a fraction of the image conditions.
train_data_amounts = [25, 50, 75, 100]
ci_lower_cond = ci_lower[3,:]
ci_upper_cond = ci_upper[3,:]
ci_lower_rep = ci_lower[:,3]
ci_upper_rep = ci_upper[:,3]
plt.figure(figsize=(9,6))
# Plotting the results
plt.plot(train_data_amounts, np.mean(corr_res_all_img_cond, 0), color=colors[0],
	linewidth=4)
plt.plot(train_data_amounts, np.mean(corr_res_all_eeg_rep, 0), color=colors[1],
	linewidth=4)
# Plotting the confidence intervals
plt.fill_between(train_data_amounts, ci_upper_cond, ci_lower_cond,
	color=colors[0], alpha=.2)
plt.fill_between(train_data_amounts, ci_upper_rep, ci_lower_rep,
	color=colors[1], alpha=.2)
# Plotting the noise ceiling
plt.plot([0, 100], [np.mean(noise_ceiling), np.mean(noise_ceiling)], '--',
	linewidth=4,
	color=color_noise_ceiling)
# Plotting the significance markers
plt.plot(train_data_amounts, sig_ttest, 'ok', markersize=10)
# Other plot parameters
plt.xlabel('Amount of training data (%)', fontsize=30)
xlabels_2 = [25, 50, 75, 100]
plt.xticks(ticks=xlabels_2, labels=xlabels_2)
plt.xlim(left=20, right=100)
plt.ylabel('Pearson\'s $r$', fontsize=30)
ylabels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
plt.yticks(ticks=np.arange(0,.51,0.1), labels=ylabels)
plt.ylim(bottom=0, top=.5)
leg = ['All image conditions', 'All condition repetitions']
plt.legend(leg, fontsize=30, loc=4, frameon=False)


# =============================================================================
# Plotting the results for single subjects
# =============================================================================
# Plotting the the correlation results of the encoding models built using
# varying amounts (25%, 50%, 75%, 100%) of training image conditions and
# repetitions.
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(args.n_tot_sub):
	# Plotting the results
	axs[s].bar(x - width*1.5, correlation[s,0], width=width)
	axs[s].bar(x - width*.5, correlation[s,1], width=width)
	axs[s].bar(x + width*.5, correlation[s,2], width=width)
	axs[s].bar(x + width*1.5, correlation[s,3], width=width)
	# Plotting the noise ceiling
	axs[s].plot([x - width*2, x + width*2], [noise_ceiling[s],
		noise_ceiling[s]], '--', linewidth=4, color=color_noise_ceiling)
	# Other plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Image conditions (%)', fontsize=30)
		plt.xticks(ticks=x, labels=[25, 50, 75, 100])
	if s in [0, 4, 8]:
		axs[s].set_ylabel('Pearson\'s $r$', fontsize=30)
		plt.yticks(ticks=np.arange(0, .61, 0.3), labels=[0, 0.3, 0.6])
	axs[s].set_ylim(bottom=0, top=.6)
	tit = 'Participant ' + str(s+1)
	axs[s].set_title(tit, fontsize=30)
axs[10].set_xlabel('Image conditions (%)', fontsize=30)
axs[11].set_xlabel('Image conditions (%)', fontsize=30)

# Plotting the the correlation results of the encoding models built using
# varying amounts of training data (25%, 50%, 75%, 100%), while using either all
# the image conditions and a fractions of the EEG repetitions, or all of the EEG
# repetitions and a fraction of the image conditions.
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(args.n_tot_sub):
	# Plotting the noise ceiling
	axs[s].plot([0, 100], [noise_ceiling[s], noise_ceiling[s]],
		'--', color=color_noise_ceiling, linewidth=3)
	# Plotting the results
	axs[s].plot(train_data_amounts, corr_res_all_img_cond[s], color=colors[0],
		linewidth=3)
	axs[s].plot(train_data_amounts, corr_res_all_eeg_rep[s], color=colors[1],
		linewidth=3)
	# Other plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Amount of\ntraining data (%)', fontsize=30)
		plt.xticks(ticks=xlabels_2, labels=xlabels_2)
	if s in [0, 4, 8]:
		axs[s].set_ylabel('Pearson\'s $r$', fontsize=30)
		plt.yticks(ticks=np.arange(0, .61, 0.3), labels=[0, 0.3, 0.6])
	axs[s].set_xlim(left=20, right=100)
	axs[s].set_ylim(bottom=0, top=.6)
	tit = 'Participant ' + str(s+1)
	axs[s].set_title(tit, fontsize=30)
axs[10].set_xlabel('Amount of\ntraining data (%)', fontsize=30)
axs[11].set_xlabel('Amount of\ntraining data (%)', fontsize=30)
