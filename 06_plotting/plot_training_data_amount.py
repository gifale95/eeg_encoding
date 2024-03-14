"""Plot the training data amount analysis results.

Parameters
----------
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
parser.add_argument('--project_dir', default='../project/directory', type=str)
args = parser.parse_args()


# =============================================================================
# Set plot parameters
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
color_noise_ceiling = (150/255, 150/255, 150/255)
colors = [(31/255, 119/255, 180/255), (255/255, 127/255, 14/255),
	(44/255, 160/255, 44/255), (214/255, 39/255, 40/255)]


# =============================================================================
# Load the correlation results and stats
# =============================================================================
# Loading the results
pretrained = True
layers = 'all'
n_components = 1000
data_dir = os.path.join(args.project_dir, 'results', 'stats',
	'training_data_amount_analysis', 'pretrained-'+str(pretrained),
	'layers-'+layers, 'n_components-'+format(n_components,'05'),
	'training_data_amount_analysis_stats.npy')
results = np.load(data_dir, allow_pickle=True).item()

# Organizing the significance values for plotting
sig_ttest = np.zeros(len(results['significance']['all_layers']))
for a in range(len(sig_ttest)):
	if results['significance']['all_layers'][a] == False:
		sig_ttest[a] = -100
	else:
		sig_ttest[a] = .48


# =============================================================================
# Plot the the correlation results of the encoding models built using varying
# amounts (25%, 50%, 75%, 100%) of training image conditions and repetitions
# =============================================================================
# Subject-average results
xlabels_1 = ['4,135', '8,270', '12,405', '16,540']
x = np.arange(len(xlabels_1))
width = 0.2
plt.figure(figsize=(9,6))
# Plot the results
plt.bar(x - width*1.5, np.mean(results['correlation']['all_layers'][:,:,0], 0),
	width=width)
plt.bar(x - width*.5, np.mean(results['correlation']['all_layers'][:,:,1], 0),
	width=width)
plt.bar(x + width*.5, np.mean(results['correlation']['all_layers'][:,:,2], 0),
	width=width)
plt.bar(x + width*1.5, np.mean(results['correlation']['all_layers'][:,:,3], 0),
	width=width)
leg = ['1', '2', '3','4']
plt.legend(leg, title='Condition repetitions', title_fontsize=30, ncol=4,
	fontsize=30, frameon=False, loc=9)
# Plot the noise ceiling
plt.plot([x - width*2, x + width*2], [np.mean(results['noise_ceiling']),
	np.mean(results['noise_ceiling'])], '--', linewidth=4,
	color=color_noise_ceiling)
# Plot the confidence intervals
ci_up = results['ci_upper']['all_layers'] - \
	np.mean(results['correlation']['all_layers'], 0)
ci_low = np.mean(results['correlation']['all_layers'], 0) - \
	results['ci_lower']['all_layers']
conf_int = np.append(np.expand_dims(ci_low, 0), np.expand_dims(ci_up, 0), 0)
plt.errorbar(x - width*1.5, np.mean(results['correlation']['all_layers'][:,:,0],
	0), yerr=conf_int[:,:,0], fmt="none", ecolor="k", elinewidth=2, capsize=4)
plt.errorbar(x - width*.5, np.mean(results['correlation']['all_layers'][:,:,1],
	0), yerr=conf_int[:,:,1], fmt="none", ecolor="k", elinewidth=2, capsize=4)
plt.errorbar(x + width*.5, np.mean(results['correlation']['all_layers'][:,:,2],
	0), yerr=conf_int[:,:,2], fmt="none", ecolor="k", elinewidth=2, capsize=4)
plt.errorbar(x + width*1.5, np.mean(results['correlation']['all_layers'][:,:,3],
	0), yerr=conf_int[:,:,3], fmt="none", ecolor="k", elinewidth=2, capsize=4)
# Other plot parameters
plt.xlabel('Image conditions', fontsize=30)
plt.xticks(ticks=x, labels=xlabels_1)
plt.ylabel('Pearson\'s $r$', fontsize=30)
ylabels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
plt.yticks(ticks=np.arange(0,.51,0.1), labels=ylabels)
plt.ylim(bottom=0, top=.5)

# Single-subjects results
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(len(results['correlation']['all_layers'])):
	# Plot the results
	axs[s].bar(x - width*1.5, results['correlation']['all_layers'][s,:,0],
		width=width)
	axs[s].bar(x - width*.5, results['correlation']['all_layers'][s,:,1],
		width=width)
	axs[s].bar(x + width*.5, results['correlation']['all_layers'][s,:,2],
		width=width)
	axs[s].bar(x + width*1.5, results['correlation']['all_layers'][s,:,3],
		width=width)
	# Plot the noise ceiling
	axs[s].plot([x - width*2, x + width*2], [results['noise_ceiling'][s],
		results['noise_ceiling'][s]], '--', linewidth=4,
		color=color_noise_ceiling)
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


# =============================================================================
# Plot the the correlation results of the encoding models built using varying
# amounts of training data (25%, 50%, 75%, 100%), while using either all the
# image conditions and a fractions of the EEG repetitions, or all of the EEG
# repetitions and a fraction of the image conditions
# =============================================================================
# Subject-average results
train_data_amounts = [25, 50, 75, 100]
corr_res_all_img_cond = results['correlation']['all_layers'][:,3]
corr_res_all_eeg_rep = results['correlation']['all_layers'][:,:,3]
ci_lower_cond = results['ci_lower']['all_layers'][3]
ci_upper_cond = results['ci_upper']['all_layers'][3]
ci_lower_rep = results['ci_lower']['all_layers'][:,3]
ci_upper_rep = results['ci_upper']['all_layers'][:,3]
plt.figure(figsize=(9,6))
# Plot the results
plt.plot(train_data_amounts, np.mean(results['correlation']['all_layers'][:,3],
	0), color=colors[0], linewidth=4)
plt.plot(train_data_amounts, np.mean(
	results['correlation']['all_layers'][:,:,3], 0), color=colors[1],
	linewidth=4)
# Plot the confidence intervals
plt.fill_between(train_data_amounts, results['ci_upper']['all_layers'][3],
	results['ci_lower']['all_layers'][3], color=colors[0], alpha=.2)
plt.fill_between(train_data_amounts, results['ci_upper']['all_layers'][:,3],
	results['ci_lower']['all_layers'][:,3], color=colors[1], alpha=.2)
# Plot the noise ceiling
plt.plot([0, 100], [np.mean(results['noise_ceiling']), np.mean(
	results['noise_ceiling'])], '--', linewidth=4, color=color_noise_ceiling)
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

# Single-subject results
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(len(results['correlation']['all_layers'])):
	# Plotting the noise ceiling
	axs[s].plot([0, 100], [results['noise_ceiling'][s],
		results['noise_ceiling'][s]], '--', color=color_noise_ceiling,
		linewidth=3)
	# Plotting the results
	axs[s].plot(train_data_amounts, results['correlation']['all_layers'][s,3],
		color=colors[0], linewidth=3)
	axs[s].plot(train_data_amounts,
		results['correlation']['all_layers'][s,:,3], color=colors[1],
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
