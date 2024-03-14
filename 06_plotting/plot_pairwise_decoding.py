"""Plot the pairwise decoding analysis results.

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
# Plot the linearizing encoding pairwise decoding results
# =============================================================================
# Load the results
subjects = 'within'
pretrained = True
layers = 'all'
n_components = 1000
dnns = ['alexnet', 'resnet50', 'cornet_s', 'moco']
dnn_names = ['AlexNet', 'ResNet-50', 'CORnet-S', 'MoCo']
results = []
for d in dnns:
	data_dir = os.path.join(args.project_dir, 'results', 'stats',
		'pairwise_decoding', 'encoding-linearizing', 'subjects-'+subjects,
		'dnn-'+d, 'pretrained-'+str(pretrained), 'layers-'+layers,
		'n_components-'+format(n_components,'05'),
		'pairwise_decoding_stats.npy')
	results.append(np.load(data_dir, allow_pickle=True).item())
times = results[0]['times']
# Average the noise ceiling across DNNs
noise_ceiling_low = []
noise_ceiling_up = []
for d in range(len(dnns)):
	noise_ceiling_low.append(results[d]['noise_ceiling_low'])
	noise_ceiling_up.append(results[d]['noise_ceiling_up'])
noise_ceiling_low = np.mean(np.asarray(noise_ceiling_low), 0)
noise_ceiling_up = np.mean(np.asarray(noise_ceiling_up), 0)

# Organize the significance values for plotting
sig = np.zeros((len(dnns),len(results[0]['significance']['all_layers'])))
sig_diff_noise_ceiling = np.zeros(sig.shape)
for d in range(sig.shape[0]):
	for t in range(sig.shape[1]):
		if results[d]['significance']['all_layers'][t] == False:
			sig[d,t] = -100
		else:
			sig[d,t] = 0.46 + (abs(d+1-len(dnns)) / 100 * .85)
		if results[d]['significance_diff_noise_ceiling']['all_layers'][t] == False:
			sig_diff_noise_ceiling[d,t] = -100
		else:
			sig_diff_noise_ceiling[d,t] = 0.275 + (abs(d+1-len(dnns)) / \
				100 * 0.6)

# Plot the pairwise decoding results, averaged across subjects
plt.figure(figsize=(9,6))
# Plot the chance and stimulus onset dashed lines
plt.plot([-100, 100], [.5, .5], 'k--', [0, 0], [10, -10], 'k--',
	label='_nolegend_', linewidth=4)
for d in range(len(dnns)):
	# Plot the correlation results
	plt.plot(times, np.mean(results[d]['decoding']['all_layers'], 0),
		color=colors[d], linewidth=4)
for d in range(len(dnns)):
	# Plot the confidence intervals
	plt.fill_between(times, results[d]['ci_upper']['all_layers'],
		results[d]['ci_lower']['all_layers'], color=colors[d], alpha=.2)
	# Plot the significance markers
	plt.plot(times, sig[d], 'o', color=colors[d], markersize=4)
# Plot the noise ceiling
plt.fill_between(times, np.mean(noise_ceiling_low, 0), np.mean(
	noise_ceiling_up, 0), color=color_noise_ceiling, alpha=.3)
# Plot parameters
plt.xlabel('Time (s)', fontsize=30)
xticks = [-.2, 0, .2, .4, .6, max(times)]
xlabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))
plt.ylabel('Decoding\naccuracy (%)', fontsize=30)
yticks = np.arange(.5,1.01,.1)
ylabels = [50, 60, 70, 80, 90, 100]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=.445, top=1)
plt.legend(dnn_names, fontsize=30, ncol=2, frameon=False)

# Plot the single subjects pairwise decoding results
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(len(results[0]['decoding']['all_layers'])):
	# Plot the chance and stimulus onset dashed lines
	axs[s].plot([-10, 10], [.5, .5], 'k--', [0, 0], [100, -100], 'k--',
			linewidth=3)
	# Plot the noise ceiling
	axs[s].fill_between(times, noise_ceiling_low[s], noise_ceiling_up[s],
		color=color_noise_ceiling, alpha=.3)
	# Plot the correlation results
	for d in range(len(dnns)):
		axs[s].plot(times, results[d]['decoding']['all_layers'][s],
			color=colors[d], linewidth=3)
	# Plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Time (s)', fontsize=30)
		xticks = [0, .4, max(times)]
		xlabels = [0, 0.4, 0.8]
		plt.xticks(ticks=xticks, labels=xlabels)
	if s in [0, 4, 8]:
		axs[s].set_ylabel('Decoding\naccuracy (%)', fontsize=30)
		yticks = np.arange(.5, 1.1, 0.25)
		ylabels = [50, 75, 100]
		plt.yticks(ticks=yticks, labels=ylabels)
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[s].set_ylim(bottom=.47, top=1)
	tit = 'Participant ' + str(s+1)
	axs[s].set_title(tit, fontsize=30)
axs[10].set_xlabel('Time (s)', fontsize=30)
axs[11].set_xlabel('Time (s)', fontsize=30)

# Plot the difference from the noise ceiling, averaged across subjects
plt.figure(figsize=(9,6))
# Plot the chance and stimulus onset dashed lines
plt.plot([-10, 10], [0, 0], 'k--', [0, 0], [10, -10], 'k--',
	label='_nolegend_', linewidth=4)
for d in range(len(dnns)):
	# Plot the difference from the noise ceiling
	plt.plot(times, np.mean(
		results[d]['diff_noise_ceiling']['all_layers'], 0), color=colors[d],
		linewidth=4)
for d in range(len(dnns)):
	# Plot the confidence intervals
	plt.fill_between(times,
		results[d]['ci_upper_diff_noise_ceiling']['all_layers'],
		results[d]['ci_lower_diff_noise_ceiling']['all_layers'],
		color=colors[d], alpha=.2)
	# Plot the significance markers
	plt.plot(times, sig_diff_noise_ceiling[d], 'o', color=colors[d],
		markersize=4)
# Plot parameters
plt.xlabel('Time (s)', fontsize=30)
xticks = [-.2, 0, .2, .4, .6, max(times)]
xlabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))
plt.ylabel('$\Delta$ Decoding\naccuracy (%)', fontsize=30)
yticks = np.arange(0, .31, .1)
ylabels = [0, 10, 20, 30]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-.03, top=.3)
plt.legend(dnn_names, fontsize=30, ncol=2, frameon=False)

# Plot the single subjects difference from the noise ceiling
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(len(results[0]['decoding']['all_layers'])):
	# Plot the chance and stimulus onset dashed lines
	axs[s].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
			linewidth=3)
	# Plot the difference from the noise ceiling
	for d in range(len(dnns)):
		axs[s].plot(times, results[d]['diff_noise_ceiling']['all_layers'][s],
			color=colors[d], linewidth=3)
	# Plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Time (s)', fontsize=30)
		xticks = [0, .4, max(times)]
		xlabels = [0, 0.4, 0.8]
		plt.xticks(ticks=xticks, labels=xlabels)
	if s in [0, 4, 8]:
		axs[s].set_ylabel('$\Delta$ Decoding\naccuracy (%)', fontsize=30)
		yticks = np.arange(0, .41, .2)
		ylabels = [0, 20, 40]
		plt.yticks(ticks=yticks, labels=ylabels)
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[s].set_ylim(bottom=-.05, top=.4)
	tit = 'Participant ' + str(s+1)
	axs[s].set_title(tit, fontsize=30)
axs[10].set_xlabel('Time (s)', fontsize=30)
axs[11].set_xlabel('Time (s)', fontsize=30)


# =============================================================================
# Plot the between subjects linearizing encoding pairwise decoding results
# =============================================================================
# Load the results
subjects = 'between'
pretrained = True
layers = 'all'
n_components = 1000
results = []
for d in dnns:
	data_dir = os.path.join(args.project_dir, 'results', 'stats',
		'pairwise_decoding', 'encoding-linearizing', 'subjects-'+subjects,
		'dnn-'+d, 'pretrained-'+str(pretrained), 'layers-'+layers,
		'n_components-'+format(n_components,'05'),
		'pairwise_decoding_stats.npy')
	results.append(np.load(data_dir, allow_pickle=True).item())
# Organize the significance values for plotting
sig = np.zeros((len(dnns),len(results[0]['significance']['all_layers'])))
for d in range(sig.shape[0]):
	for t in range(sig.shape[1]):
		if results[d]['significance']['all_layers'][t] == False:
			sig[d,t] = -100
		else:
			sig[d,t] = 0.46 + (abs(d+1-len(dnns)) / 100 * .85)

# Plot the pairwise decoding results, averaged across subjects
plt.figure(figsize=(9,6))
# Plot the chance and stimulus onset dashed lines
plt.plot([-10, 10], [.5, .5], 'k--', [0, 0], [10, -10], 'k--',
	label='_nolegend_', linewidth=4)
for d in range(len(dnns)):
	# Plot the correlation results
	plt.plot(times, np.mean(results[d]['decoding']['all_layers'],
		0), color=colors[d], linewidth=4)
for d in range(len(dnns)):
	# Plot the confidence intervals
	plt.fill_between(times, results[d]['ci_upper']['all_layers'],
		results[d]['ci_lower']['all_layers'], color=colors[d], alpha=.2)
	# Plot the significance markers
	plt.plot(times, sig[d], 'o', color=colors[d], markersize=4)
# Plot the noise ceiling
plt.fill_between(times, np.mean(noise_ceiling_low, 0), np.mean(
	noise_ceiling_up, 0), color=color_noise_ceiling, alpha=.3)
# Plot parameters
plt.xlabel('Time (s)', fontsize=30)
xticks = [-.2, 0, .2, .4, .6, max(times)]
xlabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))
plt.ylabel('Decoding\naccuracy (%)', fontsize=30)
yticks = np.arange(.5,1.01,.1)
ylabels = [50, 60, 70, 80, 90, 100]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=.445, top=1)
plt.legend(dnn_names, fontsize=30, ncol=2, frameon=False)

# Plot the single subjects pairwise decoding results
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(len(results[0]['decoding']['all_layers'])):
	# Plot the chance and stimulus onset dashed lines
	axs[s].plot([-10, 10], [.5, .5], 'k--', [0, 0], [100, -100], 'k--',
			linewidth=3)
	# Plot the noise ceiling
	axs[s].fill_between(times, noise_ceiling_low[s], noise_ceiling_up[s],
		color=color_noise_ceiling, alpha=.3)
	# Plot the correlation results
	for d in range(len(dnns)):
		axs[s].plot(times, results[d]['decoding']['all_layers'][s],
			color=colors[d], linewidth=3)
	# Plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Time (s)', fontsize=30)
		xticks = [0, .4, max(times)]
		xlabels = [0, 0.4, 0.8]
		plt.xticks(ticks=xticks, labels=xlabels)
	if s in [0, 4, 8]:
		axs[s].set_ylabel('Decoding\naccuracy (%)', fontsize=30)
		yticks = np.arange(.5, 1.1, 0.25)
		ylabels = [50, 75, 100]
		plt.yticks(ticks=yticks, labels=ylabels)
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[s].set_ylim(bottom=.47, top=1)
	tit = 'Participant ' + str(s+1)
	axs[s].set_title(tit, fontsize=30)
axs[10].set_xlabel('Time (s)', fontsize=30)
axs[11].set_xlabel('Time (s)', fontsize=30)


# =============================================================================
# Plot the end-to-end encoding correlation results
# =============================================================================
# Load the results
modeled_time_points = ['all', 'single']
pretrained = False
lr = 1e-05
weight_decay = 0.
batch_size = 64
results = []
for m in modeled_time_points:
	data_dir = os.path.join(args.project_dir, 'results', 'stats',
		'pairwise_decoding', 'encoding-end_to_end', 'dnn-alexnet',
		'modeled_time_points-'+m, 'pretrained-'+
		str(pretrained), 'lr-{:.0e}'.format(lr)+
		'__wd-{:.0e}'.format(weight_decay)+'__bs-'+
		format(batch_size,'03'), 'pairwise_decoding_stats.npy')
	results.append(np.load(data_dir, allow_pickle=True).item())
# Organize the significance values for plotting
sig = np.zeros((len(modeled_time_points),
	len(results[0]['significance']['all_time_points'])))
for m, model in enumerate(modeled_time_points):
	for t in range(sig.shape[1]):
		if results[m]['significance'][model+'_time_points'][t] == False:
			sig[m,t] = -100
		else:
			sig[m,t] = 0.475 + (abs(m) / 100 * 1.1)

# Plot the pairwise decoding results, averaged across subjects
plt.figure(figsize=(9,6))
# Plot the chance and stimulus onset dashed lines
plt.plot([-10, 10], [.5, .5], 'k--', [0, 0], [10, -10], 'k--',
	label='_nolegend_', linewidth=4)
for m, model in enumerate(modeled_time_points):
	# Plot the correlation results
	plt.plot(times, np.mean(results[m]['decoding'][model+'_time_points'], 0),
		color=colors[m], linewidth=4)
for m, model in enumerate(modeled_time_points):
	# Plot the confidence intervals
	plt.fill_between(times, results[m]['ci_upper'][model+'_time_points'],
		results[m]['ci_lower'][model+'_time_points'], color=colors[m],
		alpha=.2)
	# Plot the significance markers
	plt.plot(times, sig[m], 'o', color=colors[m], markersize=4)
# Plot the noise ceiling
plt.fill_between(times, np.mean(noise_ceiling_low, 0), np.mean(
	noise_ceiling_up, 0), color=color_noise_ceiling, alpha=.3)
# Plot parameters
plt.xlabel('Time (s)', fontsize=30)
xticks = [-.2, 0, .2, .4, .6, max(times)]
xlabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))
plt.ylabel('Decoding\naccuracy (%)', fontsize=30)
yticks = np.arange(0,1.01,0.2)
ylabels = [0, 0.2, 0.4, 0.6, 0.8, 1]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=.46, top=1)
leg = ['All-time-points models', 'Single-time-points models']
plt.legend(leg, fontsize=30, ncol=2, frameon=False)

# Plot the single subjects correlation results
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(len(results[0]['decoding']['all_time_points'])):
	# Plot the chance and stimulus onset dashed lines
	axs[s].plot([-10, 10], [.5, .5], 'k--', [0, 0], [100, -100], 'k--',
			linewidth=3)
	# Plot the noise ceiling
	axs[s].fill_between(times, noise_ceiling_low[s], noise_ceiling_up[s],
		color=color_noise_ceiling, alpha=.3)
	# Plot the correlation results
	for m, model in enumerate(modeled_time_points):
		axs[s].plot(times, results[m]['decoding'][model+'_time_points'][s],
			color=colors[m], linewidth=3)
	# Plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Time (s)', fontsize=30)
		xticks = [0, .4, max(times)]
		xlabels = [0, 0.4, 0.8]
		plt.xticks(ticks=xticks, labels=xlabels)
	if s in [0, 4, 8]:
		axs[s].set_ylabel('Decoding\naccuracy (%)', fontsize=30)
		yticks = np.arange(0, 1.01, 0.5)
		ylabels = [0, 50, 100]
		plt.yticks(ticks=yticks, labels=ylabels)
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[s].set_ylim(bottom=.47, top=1)
	tit = 'Participant ' + str(s+1)
	axs[s].set_title(tit, fontsize=30)
axs[10].set_xlabel('Time (s)', fontsize=30)
axs[11].set_xlabel('Time (s)', fontsize=30)
