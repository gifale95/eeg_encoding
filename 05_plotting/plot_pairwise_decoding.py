"""Plotting the pairwise decoding analysis results.

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
dnns = ['alexnet', 'resnet50', 'cornet_s', 'moco']
for d, dnn in enumerate(dnns):
	data_dir = os.path.join(args.project_dir, 'results', 'stats',
		'pairwise_decoding', 'dnn-'+dnn, 'pairwise_decoding_stats.npy')
	# Loading the data
	data_dict = np.load(data_dir, allow_pickle=True).item()
	if d == 0:
		pairwise_decoding_within = np.expand_dims(
			data_dict['pairwise_decoding_within'], 0)
		ci_lower_within = np.expand_dims(data_dict['ci_lower_within'], 0)
		ci_upper_within = np.expand_dims(data_dict['ci_upper_within'], 0)
		significance_within = np.expand_dims(data_dict['significance_within'],
			0)
		pairwise_decoding_between = np.expand_dims(
			data_dict['pairwise_decoding_between'], 0)
		ci_lower_between = np.expand_dims(data_dict['ci_lower_between'], 0)
		ci_upper_between = np.expand_dims(data_dict['ci_upper_between'], 0)
		significance_between = np.expand_dims(data_dict['significance_between'],
			0)
		noise_ceiling = np.expand_dims(data_dict['noise_ceiling'], 0)
		diff_noise_ceiling = np.expand_dims(data_dict['diff_noise_ceiling'], 0)
		ci_lower_diff_noise_ceiling = np.expand_dims(
			data_dict['ci_lower_diff_noise_ceiling'], 0)
		ci_upper_diff_noise_ceiling = np.expand_dims(
			data_dict['ci_upper_diff_noise_ceiling'], 0)
		significance_diff_noise_ceiling = np.expand_dims(
			data_dict['significance_diff_noise_ceiling'], 0)
		times = data_dict['times']
	else:
		pairwise_decoding_within = np.append(pairwise_decoding_within,
			np.expand_dims(data_dict['pairwise_decoding_within'], 0), 0)
		ci_lower_within = np.append(ci_lower_within, np.expand_dims(
			data_dict['ci_lower_within'], 0), 0)
		ci_upper_within = np.append(ci_upper_within, np.expand_dims(
			data_dict['ci_upper_within'], 0), 0)
		significance_within = np.append(significance_within, np.expand_dims(
			data_dict['significance_within'], 0), 0)
		pairwise_decoding_between = np.append(pairwise_decoding_between,
			np.expand_dims(data_dict['pairwise_decoding_between'], 0), 0)
		ci_lower_between = np.append(ci_lower_between, np.expand_dims(
			data_dict['ci_lower_between'], 0), 0)
		ci_upper_between = np.append(ci_upper_between, np.expand_dims(
			data_dict['ci_upper_between'], 0), 0)
		significance_between = np.append(significance_between, np.expand_dims(
			data_dict['significance_between'], 0), 0)
		noise_ceiling = np.append(noise_ceiling, np.expand_dims(
			data_dict['noise_ceiling'], 0), 0)
		diff_noise_ceiling = np.append(diff_noise_ceiling, np.expand_dims(
			data_dict['diff_noise_ceiling'], 0), 0)
		ci_lower_diff_noise_ceiling = np.append(ci_lower_diff_noise_ceiling,
			np.expand_dims(data_dict['ci_lower_diff_noise_ceiling'], 0), 0)
		ci_upper_diff_noise_ceiling = np.append(ci_upper_diff_noise_ceiling,
			np.expand_dims(data_dict['ci_upper_diff_noise_ceiling'], 0), 0)
		significance_diff_noise_ceiling = np.append(
			significance_diff_noise_ceiling, np.expand_dims(
			data_dict['significance_diff_noise_ceiling'], 0), 0)

# Averaging the noise ceiling across DNNs
noise_ceiling = np.mean(noise_ceiling, 0)

# Organizing the significance values for plotting
sig_within = np.zeros(significance_within.shape)
sig_between = np.zeros(significance_between.shape)
sig_diff_noise_ceiling = np.zeros(significance_diff_noise_ceiling.shape)
for d in range(len(dnns)):
	for t in range(significance_within.shape[1]):
		if significance_within[d,t] == False:
			sig_within[d,t] = -100
		else:
			sig_within[d,t] = 0.972 + (abs(d+1-len(dnns)) / 100 * .75)
		if significance_between[d,t] == False:
			sig_between[d,t] = -100
		else:
			sig_between[d,t] = 0.972 + (abs(d+1-len(dnns)) / 100 * .75)
		if significance_diff_noise_ceiling[d,t] == False:
			sig_diff_noise_ceiling[d,t] = -100
		else:
			sig_diff_noise_ceiling[d,t] = 0.275 + (abs(d+1-len(dnns)) /\
				100 * 0.6)


# =============================================================================
# Plotting the within-participant results averaged across subjects
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

plt.figure(figsize=(9,6))
for d in range(len(dnns)):
	# Plotting the results
	plt.plot(times, np.mean(pairwise_decoding_within[d], 0), color=colors[d],
		linewidth=4)
# Plotting the noise ceiling
plt.plot(times, np.mean(noise_ceiling, 0), '--', color=color_noise_ceiling,
	linewidth=4)
for d in range(len(dnns)):
	# Plotting the confidence intervals
	plt.fill_between(times, ci_upper_within[d], ci_lower_within[d],
		color=colors[d], alpha=.2)
	# Plotting the significance markers
	plt.plot(times, sig_within[d], 'o', color=colors[d], markersize=4)
# Plotting chance and stimulus onset dashed lines
plt.plot([-100, 100], [.5, .5], 'k--', [0, 0], [100, -100], 'k--', linewidth=4)
# Other plot parameters
plt.xlabel('Time (s)', fontsize=30)
xlabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
plt.xticks(ticks=[-.2, 0, .2, .4, .6, max(times)], labels=xlabels)
plt.xlim(left=min(times), right=max(times))
plt.ylabel('Decoding\naccuracy (%)', fontsize=30)
ylabels = [50, 60, 70, 80, 90, 100]
plt.yticks(ticks=np.arange(.5,1.01,.1), labels=ylabels)
plt.ylim(bottom=.47, top=1)
leg = ['AlexNet', 'ResNet-50', 'CORnet-S', 'MoCo']
plt.legend(leg, fontsize=30, ncol=2, frameon=False)


# =============================================================================
# Plotting the within-participant results for single subjects
# =============================================================================
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(args.n_tot_sub):
	# Plotting the noise ceiling
	axs[s].plot(times, noise_ceiling[s], '--', color=color_noise_ceiling,
		linewidth=3)
	for d in range(len(dnns)):
		# Plotting the results
		axs[s].plot(times, pairwise_decoding_within[d,s], color=colors[d],
			linewidth=3)
	# Plotting chance and stimulus onset dashed lines
	axs[s].plot([-100, 100], [.5, .5], 'k--', [0, 0], [100, -100], 'k--',
		linewidth=3)
	# Other plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Time (s)', fontsize=30)
		plt.xticks(ticks=[0, .4, max(times)], labels=[0, 0.4, 0.8])
	if s in [0, 4, 8]:
		axs[s].set_ylabel('Decoding\naccuracy (%)', fontsize=30)
		plt.yticks(ticks=np.arange(.5, 1.1, 0.25), labels=[50, 75, 100])
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[s].set_ylim(bottom=.47, top=1)
	tit = 'Participant ' + str(s+1)
	axs[s].set_title(tit, fontsize=30)
axs[10].set_xlabel('Time (s)', fontsize=30)
axs[11].set_xlabel('Time (s)', fontsize=30)


# =============================================================================
# Plotting the between-participant results averaged across subjects
# =============================================================================
plt.figure(figsize=(9,6))
for d in range(len(dnns)):
	# Plotting the results
	plt.plot(times, np.mean(pairwise_decoding_between[d], 0), color=colors[d],
		linewidth=4)
# Plotting the noise ceiling
plt.plot(times, np.mean(noise_ceiling, 0), '--', color=color_noise_ceiling,
	linewidth=4)
for d in range(len(dnns)):
	# Plotting the confidence intervals
	plt.fill_between(times, ci_upper_between[d], ci_lower_between[d],
		color=colors[d], alpha=.2)
	# Plotting the significance markers
	plt.plot(times, sig_between[d], 'o', color=colors[d], markersize=4)
# Plotting chance and stimulus onset dashed lines
plt.plot([-100, 100], [.5, .5], 'k--', [0, 0], [100, -100], 'k--', linewidth=4)
# Other plot parameters
plt.xlabel('Time (s)', fontsize=30)
xlabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
plt.xticks(ticks=[-.2, 0, .2, .4, .6, max(times)], labels=xlabels)
plt.xlim(left=min(times), right=max(times))
plt.ylabel('Decoding\naccuracy (%)', fontsize=30)
ylabels = [50, 60, 70, 80, 90, 100]
plt.yticks(ticks=np.arange(.5,1.01,.1), labels=ylabels)
plt.ylim(bottom=.47, top=1)
leg = ['AlexNet', 'ResNet-50', 'CORnet-S', 'MoCo']
plt.legend(leg, fontsize=30, ncol=2, frameon=False)


# =============================================================================
# Plotting the between-participant results for single subjects
# =============================================================================
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(args.n_tot_sub):
	for d in range(len(dnns)):
		# Plotting the results
		axs[s].plot(times, pairwise_decoding_between[d,s], color=colors[d],
			linewidth=3)
	# Plotting the noise ceiling
	axs[s].plot(times, noise_ceiling[s], '--', color=color_noise_ceiling,
		linewidth=3)
	# Plotting chance and stimulus onset dashed lines
	axs[s].plot([-100, 100], [.5, .5], 'k--', [0, 0], [100, -100], 'k--',
		linewidth=3)
	# Other plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Time (s)', fontsize=30)
		plt.xticks(ticks=[0, .4, max(times)], labels=[0, 0.4, 0.8])
	if s in [0, 4, 8]:
		axs[s].set_ylabel('Decoding\naccuracy (%)', fontsize=30)
		plt.yticks(ticks=np.arange(.5, 1.1, 0.25), labels=[50, 75, 100])
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[s].set_ylim(bottom=.47, top=1)
	tit = 'Participant ' + str(s+1)
	axs[s].set_title(tit, fontsize=30)
axs[10].set_xlabel('Time (s)', fontsize=30)
axs[11].set_xlabel('Time (s)', fontsize=30)


# =============================================================================
# Plotting the difference from the noise ceiling averaged across subjects
# =============================================================================
plt.figure(figsize=(9,6))
for d in range(len(dnns)):
	# Plotting the results
	plt.plot(times, np.mean(diff_noise_ceiling[d], 0), color=colors[d],
		linewidth=4)
for d in range(len(dnns)):
	# Plotting the confidence intervals
	plt.fill_between(times, ci_upper_diff_noise_ceiling[d],
		ci_lower_diff_noise_ceiling[d], color=colors[d], alpha=.2)
	# Plotting the significance markers
	plt.plot(times, sig_diff_noise_ceiling[d], 'o', color=colors[d],
		markersize=4)
# Plotting chance and stimulus onset dashed lines
plt.plot([-10, 10], [0, 0], 'k--', [0, 0], [10, -10], 'k--', linewidth=4)
# Other plot parameters
plt.xlabel('Time (s)', fontsize=30)
xlabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
plt.xticks(ticks=[-.2, 0, .2, .4, .6, max(times)], labels=xlabels)
plt.xlim(left=min(times), right=max(times))
plt.ylabel('$\Delta$ Decoding\naccuracy (%)', fontsize=30)
ylabels = [0, 10, 20, 30]
plt.yticks(ticks=np.arange(0, .31, .1), labels=ylabels)
plt.ylim(bottom=-.03, top=.3)
leg = ['AlexNet', 'ResNet-50', 'CORnet-S', 'MoCo']
plt.legend(leg, fontsize=30, ncol=2, frameon=False)


# =============================================================================
# Plotting the difference from the noise ceiling for single subjects
# =============================================================================
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(args.n_tot_sub):
	for d in range(len(dnns)):
		# Plotting the results
		axs[s].plot(times, diff_noise_ceiling[d,s], color=colors[d],
			linewidth=3)
	# Plotting chance and stimulus onset dashed lines
	axs[s].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
			linewidth=3)
	# Other plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Time (s)', fontsize=30)
		plt.xticks(ticks=[0, .4, max(times)], labels=[0, 0.4, 0.8])
	if s in [0, 4, 8]:
		axs[s].set_ylabel('$\Delta$ Decoding\naccuracy (%)', fontsize=30)
		plt.yticks(ticks=np.arange(0, .41, .2), labels=[0, 20, 40])
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[s].set_ylim(bottom=-.05, top=.4)
	tit = 'Participant ' + str(s+1)
	axs[s].set_title(tit, fontsize=30)
axs[10].set_xlabel('Time (s)', fontsize=30)
axs[11].set_xlabel('Time (s)', fontsize=30)
