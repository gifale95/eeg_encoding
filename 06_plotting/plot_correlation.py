"""Plot the correlation analysis results.

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
	(44/255, 160/255, 44/255), (214/255, 39/255, 40/255),
	(148/255, 103/255, 189/255), (140/255, 86/255, 75/255),
	(227/255, 119/255, 194/255), (127/255, 127/255, 127/255)]


# =============================================================================
# Plot the linearizing encoding correlation results
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
	data_dir = os.path.join(args.project_dir, 'results', 'stats', 'correlation',
		'encoding-linearizing', 'subjects-'+subjects, 'dnn-'+d, 'pretrained-'+
		str(pretrained), 'layers-'+layers, 'n_components-'+
		format(n_components,'05'), 'correlation_stats.npy')
	results.append(np.load(data_dir, allow_pickle=True).item())
times = results[0]['times']
ch_names = results[0]['ch_names']
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
			sig[d,t] = -.085 + (abs(d+1-len(dnns)) / 100 * 1.75)
		if results[d]['significance_diff_noise_ceiling']['all_layers'][t] == False:
			sig_diff_noise_ceiling[d,t] = -100
		else:
			sig_diff_noise_ceiling[d,t] = 0.27 + (abs(d+1-len(dnns)) /\
				100 * 0.7)
results_baseline = results
sig_baseline = sig

# Plot the correlation results, averaged across subjects
plt.figure(figsize=(9,6))
# Plot the chance and stimulus onset dashed lines
plt.plot([-10, 10], [0, 0], 'k--', [0, 0], [10, -10], 'k--',
	label='_nolegend_', linewidth=4)
for d in range(len(dnns)):
	# Plot the correlation results
	plt.plot(times, np.mean(np.mean(results[d]['correlation']['all_layers'], 0),
		0), color=colors[d], linewidth=4)
for d in range(len(dnns)):
	# Plot the confidence intervals
	plt.fill_between(times, results[d]['ci_upper']['all_layers'],
		results[d]['ci_lower']['all_layers'], color=colors[d], alpha=.2)
	# Plot the significance markers
	plt.plot(times, sig[d], 'o', color=colors[d], markersize=4)
# Plot the noise ceiling
plt.fill_between(times, np.mean(np.mean(noise_ceiling_low, 0), 0), np.mean(
	np.mean(noise_ceiling_up, 0), 0), color=color_noise_ceiling, alpha=.3)
# Plot parameters
plt.xlabel('Time (s)', fontsize=30)
xticks = [-.2, 0, .2, .4, .6, max(times)]
xlabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))
plt.ylabel('Pearson\'s $r$', fontsize=30)
yticks = np.arange(0,1.01,0.2)
ylabels = [0, 0.2, 0.4, 0.6, 0.8, 1]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-.116, top=1)
plt.legend(dnn_names, fontsize=30, ncol=2, frameon=False)

# Plot the single subjects correlation results
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(len(results[0]['correlation']['all_layers'])):
	# Plot the chance and stimulus onset dashed lines
	axs[s].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
			linewidth=3)
	# Plot the noise ceiling
	axs[s].fill_between(times, np.mean(noise_ceiling_low[s], 0), np.mean(
		noise_ceiling_up[s], 0), color=color_noise_ceiling, alpha=.3)
	# Plot the correlation results
	for d in range(len(dnns)):
		axs[s].plot(times, np.mean(results[d]['correlation']['all_layers'][s],
			0), color=colors[d], linewidth=3)
	# Plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Time (s)', fontsize=30)
		xticks = [0, .4, max(times)]
		xlabels = [0, 0.4, 0.8]
		plt.xticks(ticks=xticks, labels=xlabels)
	if s in [0, 4, 8]:
		axs[s].set_ylabel('Pearson\'s $r$', fontsize=30)
		yticks = np.arange(0, 1.01, 0.5)
		ylabels = [0, 0.5, 1]
		plt.yticks(ticks=yticks, labels=ylabels)
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[s].set_ylim(bottom=-.05, top=1)
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
	plt.plot(times, np.mean(np.mean(
		results[d]['diff_noise_ceiling']['all_layers'], 0), 0), color=colors[d],
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
plt.ylabel('Pearson\'s $r$', fontsize=30)
yticks = np.arange(0, .31, .1)
ylabels = [0, 0.1, 0.2, 0.3]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-.1, top=.30)
plt.legend(dnn_names, fontsize=30, ncol=2, frameon=False)

# Plot the single subjects difference from the noise ceiling
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(len(results[0]['correlation']['all_layers'])):
	# Plot the chance and stimulus onset dashed lines
	axs[s].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
			linewidth=3)
	# Plot the difference from the noise ceiling
	for d in range(len(dnns)):
		axs[s].plot(times, np.mean(
			results[d]['diff_noise_ceiling']['all_layers'][s], 0),
			color=colors[d], linewidth=3)
	# Plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Time (s)', fontsize=30)
		xticks = [0, .4, max(times)]
		xlabels = [0, 0.4, 0.8]
		plt.xticks(ticks=xticks, labels=xlabels)
	if s in [0, 4, 8]:
		axs[s].set_ylabel('Pearson\'s $r$', fontsize=30)
		yticks = np.arange(0, .41, .2)
		ylabels = [0, 0.2, 0.4]
		plt.yticks(ticks=yticks, labels=ylabels)
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[s].set_ylim(bottom=-.15, top=.4)
	tit = 'Participant ' + str(s+1)
	axs[s].set_title(tit, fontsize=30)
axs[10].set_xlabel('Time (s)', fontsize=30)
axs[11].set_xlabel('Time (s)', fontsize=30)

# Plot the single-channel correlation results, averaged across subjects
fig, axs = plt.subplots(2, 2, 'all', 'all')
axs = np.reshape(axs, (-1))
for d, dnn in enumerate(dnn_names):
	img = axs[d].imshow(np.mean(results[d]['correlation']['all_layers'], 0),
		aspect='auto')
	# Plot parameters
	if d in [2, 3]:
		axs[d].set_xlabel('Time (s)', fontsize=30)
		xticks = [0, 20, 40, 60, 80, 99]
		xlabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
		plt.xticks(ticks=xticks, labels=xlabels)
	if d in [0, 2]:
		axs[d].set_ylabel('Channels', fontsize=30)
		yticks = np.arange(0, len(ch_names))
		plt.yticks(ticks=yticks, labels=ch_names)
	axs[d].set_title(dnn, fontsize=30)
plt.colorbar(img, label='Pearson\'s $r$', fraction=0.2, ax=axs[d])


# =============================================================================
# Plot the between subjects linearizing encoding correlation results
# =============================================================================
# Load the results
subjects = 'between'
pretrained = True
layers = 'all'
n_components = 1000
results = []
for d in dnns:
	data_dir = os.path.join(args.project_dir, 'results', 'stats', 'correlation',
		'encoding-linearizing', 'subjects-'+subjects, 'dnn-'+d, 'pretrained-'+
		str(pretrained), 'layers-'+layers, 'n_components-'+
		format(n_components,'05'), 'correlation_stats.npy')
	results.append(np.load(data_dir, allow_pickle=True).item())
# Organize the significance values for plotting
sig = np.zeros((len(dnns),len(results[0]['significance']['all_layers'])))
for d in range(sig.shape[0]):
	for t in range(sig.shape[1]):
		if results[d]['significance']['all_layers'][t] == False:
			sig[d,t] = -100
		else:
			sig[d,t] = -.085 + (abs(d+1-len(dnns)) / 100 * 1.75)

# Plot the correlation results, averaged across subjects
plt.figure(figsize=(9,6))
# Plot the chance and stimulus onset dashed lines
plt.plot([-10, 10], [0, 0], 'k--', [0, 0], [10, -10], 'k--',
	label='_nolegend_', linewidth=4)
for d in range(len(dnns)):
	# Plot the correlation results
	plt.plot(times, np.mean(np.mean(results[d]['correlation']['all_layers'], 0),
		0), color=colors[d], linewidth=4)
for d in range(len(dnns)):
	# Plot the confidence intervals
	plt.fill_between(times, results[d]['ci_upper']['all_layers'],
		results[d]['ci_lower']['all_layers'], color=colors[d], alpha=.2)
	# Plot the significance markers
	plt.plot(times, sig[d], 'o', color=colors[d], markersize=4)
# Plot the noise ceiling
plt.fill_between(times, np.mean(np.mean(noise_ceiling_low, 0), 0), np.mean(
	np.mean(noise_ceiling_up, 0), 0), color=color_noise_ceiling, alpha=.3)
# Plot parameters
plt.xlabel('Time (s)', fontsize=30)
xticks = [-.2, 0, .2, .4, .6, max(times)]
xlabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))
plt.ylabel('Pearson\'s $r$', fontsize=30)
yticks = np.arange(0,1.01,0.2)
ylabels = [0, 0.2, 0.4, 0.6, 0.8, 1]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-.116, top=1)
plt.legend(dnn_names, fontsize=30, ncol=2, frameon=False)

# Plot the single subjects correlation results
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(len(results[0]['correlation']['all_layers'])):
	# Plot the chance and stimulus onset dashed lines
	axs[s].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
			linewidth=3)
	# Plot the noise ceiling
	axs[s].fill_between(times, np.mean(noise_ceiling_low[s], 0), np.mean(
		noise_ceiling_up[s], 0), color=color_noise_ceiling, alpha=.3)
	# Plot the correlation results
	for d in range(len(dnns)):
		axs[s].plot(times, np.mean(results[d]['correlation']['all_layers'][s],
			0), color=colors[d], linewidth=3)
	# Plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Time (s)', fontsize=30)
		xticks = [0, .4, max(times)]
		xlabels = [0, 0.4, 0.8]
		plt.xticks(ticks=xticks, labels=xlabels)
	if s in [0, 4, 8]:
		axs[s].set_ylabel('Pearson\'s $r$', fontsize=30)
		yticks = np.arange(0, 1.01, 0.5)
		ylabels = [0, 0.5, 1]
		plt.yticks(ticks=yticks, labels=ylabels)
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[s].set_ylim(bottom=-.05, top=1)
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
		'correlation', 'encoding-end_to_end', 'dnn-alexnet',
		'modeled_time_points-'+m, 'pretrained-'+
		str(pretrained), 'lr-{:.0e}'.format(lr)+
		'__wd-{:.0e}'.format(weight_decay)+'__bs-'+
		format(batch_size,'03'), 'correlation_stats.npy')
	results.append(np.load(data_dir, allow_pickle=True).item())
# Organize the significance values for plotting
sig = np.zeros((len(modeled_time_points),
	len(results[0]['significance']['all_time_points'])))
for m, model in enumerate(modeled_time_points):
	for t in range(sig.shape[1]):
		if results[m]['significance'][model+'_time_points'][t] == False:
			sig[m,t] = -100
		else:
			sig[m,t] = -.085 + (abs(m+4.25-len(modeled_time_points)) / 100 * 1.75)

# Plot the correlation results, averaged across subjects
plt.figure(figsize=(9,6))
# Plot the chance and stimulus onset dashed lines
plt.plot([-10, 10], [0, 0], 'k--', [0, 0], [10, -10], 'k--',
	label='_nolegend_', linewidth=4)
for m, model in enumerate(modeled_time_points):
	# Plot the correlation results
	plt.plot(times, np.mean(np.mean(
		results[m]['correlation'][model+'_time_points'], 0), 0),
		color=colors[m], linewidth=4)
for m, model in enumerate(modeled_time_points):
	# Plot the confidence intervals
	plt.fill_between(times, results[m]['ci_upper'][model+'_time_points'],
		results[m]['ci_lower'][model+'_time_points'], color=colors[m],
		alpha=.2)
	# Plot the significance markers
	plt.plot(times, sig[m], 'o', color=colors[m], markersize=4)
# Plot the noise ceiling
plt.fill_between(times, np.mean(np.mean(noise_ceiling_low, 0), 0), np.mean(
	np.mean(noise_ceiling_up, 0), 0), color=color_noise_ceiling, alpha=.3)
# Plot parameters
plt.xlabel('Time (s)', fontsize=30)
xticks = [-.2, 0, .2, .4, .6, max(times)]
xlabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))
plt.ylabel('Pearson\'s $r$', fontsize=30)
yticks = np.arange(0,1.01,0.2)
ylabels = [0, 0.2, 0.4, 0.6, 0.8, 1]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-.075, top=1)
leg = ['All-time-points models', 'Single-time-points models']
plt.legend(leg, fontsize=30, ncol=1, frameon=False)

# Plot the single subjects correlation results
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(len(results[0]['correlation']['all_time_points'])):
	# Plot the noise ceiling
	axs[s].fill_between(times, np.mean(noise_ceiling_low[s], 0), np.mean(
		noise_ceiling_up[s], 0), color=color_noise_ceiling, alpha=.3)
	# Plot the correlation results
	for m, model in enumerate(modeled_time_points):
		axs[s].plot(times,
			np.mean(results[m]['correlation'][model+'_time_points'][s], 0),
			color=colors[m], linewidth=3)
	# Plot the chance and stimulus onset dashed lines
	axs[s].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
			linewidth=3)
	# Plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Time (s)', fontsize=30)
		xticks = [0, .4, max(times)]
		xlabels = [0, 0.4, 0.8]
		plt.xticks(ticks=xticks, labels=xlabels)
	if s in [0, 4, 8]:
		axs[s].set_ylabel('Pearson\'s $r$', fontsize=30)
		yticks = np.arange(0, 1.01, 0.5)
		ylabels = [0, 0.5, 1]
		plt.yticks(ticks=yticks, labels=ylabels)
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[s].set_ylim(bottom=-.05, top=1)
	tit = 'Participant ' + str(s+1)
	axs[s].set_title(tit, fontsize=30)
axs[10].set_xlabel('Time (s)', fontsize=30)
axs[11].set_xlabel('Time (s)', fontsize=30)

# Plot the single-channel correlation results, averaged across subjects
fig, axs = plt.subplots(1, 2, 'all', 'all')
axs = np.reshape(axs, (-1))
for m, model in enumerate(modeled_time_points):
	img = axs[m].imshow(np.mean(
		results[m]['correlation'][model+'_time_points'], 0), aspect='auto')
	# Plot parameters
	if m in [1]:
		axs[m].set_xlabel('Time (s)', fontsize=30)
		xticks = [0, 20, 40, 60, 80, 99]
		xlabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
		plt.xticks(ticks=xticks, labels=xlabels)
	if m in [0]:
		axs[m].set_ylabel('Channels', fontsize=30)
		yticks = np.arange(0, len(ch_names))
		plt.yticks(ticks=yticks, labels=ch_names)
	axs[m].set_title(leg[m], fontsize=30)
plt.colorbar(img, label='Pearson\'s $r$', fraction=0.2, ax=axs[m])


# =============================================================================
# Plot the linearizing encoding correlation results of single and appended DNN
# layers
# =============================================================================
# Load the single layer results
subjects = 'within'
pretrained = True
layers = 'single'
n_components = 1000
results = []
layer_names = []
for d, dnn in enumerate(dnns):
	data_dir = os.path.join(args.project_dir, 'results', 'stats', 'correlation',
		'encoding-linearizing', 'subjects-'+subjects, 'dnn-'+dnn, 'pretrained-'+
		str(pretrained), 'layers-'+layers, 'n_components-'+
		format(n_components,'05'), 'correlation_stats.npy')
	results.append(np.load(data_dir, allow_pickle=True).item())
	layer_names.append(results[d]['correlation'].keys())
# Organize the significance values for plotting
sig = []
for d in range(len(dnns)):
	sig.append({})
	for l, lay in enumerate(layer_names[d]):
		sig[d][lay] = np.zeros((len(results[d]['significance'][lay])))
		for t in range(len(sig[d][lay])):
			if results[d]['significance'][lay][t] == False:
				sig[d][lay][t] = -100
			else:
				sig[d][lay][t] = -.14 + (abs(l+1-len(layer_names[d])) / \
					100 * 1.75)

# Load the appended layers results
subjects = 'within'
pretrained = True
layers = 'appended'
n_components = 1000
results_app = []
for d in dnns:
	data_dir = os.path.join(args.project_dir, 'results', 'stats', 'correlation',
		'encoding-linearizing', 'subjects-'+subjects, 'dnn-'+d, 'pretrained-'+
		str(pretrained), 'layers-'+layers, 'n_components-'+
		format(n_components,'05'), 'correlation_stats.npy')
	results_app.append(np.load(data_dir, allow_pickle=True).item())
# Organize the significance values for plotting
sig_app = np.zeros((len(dnns),len(results_app[0]['significance']['appended_layers'])))
for d in range(sig_app.shape[0]):
	for t in range(sig_app.shape[1]):
		if results_app[d]['significance']['appended_layers'][t] == False:
			sig_app[d,t] = -100
		else:
			sig_app[d,t] = -.085 + (abs(1-len(dnns)) / 100 * 1.75)

# Plot the correlation results of single layers, averaged across subjects
fig, axs = plt.subplots(2, 2, 'all', 'all')
axs = np.reshape(axs, (-1))
for d, dnn in enumerate(dnn_names):
	# Plot the chance and stimulus onset dashed lines
	axs[d].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
			label='_nolegend_', linewidth=3)
	# Plot the baseline linearizing encoding results
	axs[d].plot(times, np.mean(np.mean(
		results_baseline[d]['correlation']['all_layers'], 0), 0), '--r',
		linewidth=3)
	axs[d].fill_between(times, results_baseline[d]['ci_upper']['all_layers'],
		results_baseline[d]['ci_lower']['all_layers'], color='r', alpha=.2)
	# Plot the correlation results
	for l, lay in enumerate(layer_names[d]):
		axs[d].plot(times, np.mean(np.mean(results[d]['correlation'][lay], 0),
			0), color=colors[l], linewidth=3)
	# Plot the stats
	for l, lay in enumerate(layer_names[d]):
		# Plot the confidence intervals
		axs[d].fill_between(times, results[d]['ci_upper'][lay],
			results[d]['ci_lower'][lay], color=colors[l], alpha=.2)
		# Plot the significance markers
		axs[d].plot(times, sig[d][lay], 'o', color=colors[l], markersize=4)
	# Plot the noise ceiling
	axs[d].fill_between(times, np.mean(np.mean(noise_ceiling_low, 0), 0),
		np.mean(np.mean(noise_ceiling_up, 0), 0), color=color_noise_ceiling,
		alpha=.3)
	# Plot parameters
	if d in [2, 3]:
		axs[d].set_xlabel('Time (s)', fontsize=30)
		xticks = [-.2, 0, .2, .4, .6, max(times)]
		xlabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
		plt.xticks(ticks=xticks, labels=xlabels)
	if d in [0, 2]:
		axs[d].set_ylabel('Pearson\'s $r$', fontsize=30)
		yticks = np.arange(0, 1.01, 0.25)
		ylabels = [0, 0.25, 0.5, 0.75, 1]
		plt.yticks(ticks=yticks, labels=ylabels)
	axs[d].set_xlim(left=min(times), right=max(times))
	axs[d].set_ylim(bottom=-.17, top=1)
	axs[d].set_title(dnn, fontsize=30)
	leg = layer_names[d]
	if dnn == 'AlexNet':
		ncol = 3
	else:
		ncol = 2
	leg = []
	for l in layer_names[d]:
		leg.append(l)
	leg.insert(0, 'All layers')
	axs[d].legend(leg, fontsize=25, ncol=ncol, frameon=True)

# Plot the correlation results of appended layers, averaged across subjects
fig, axs = plt.subplots(2, 2, 'all', 'all')
axs = np.reshape(axs, (-1))
for d, dnn in enumerate(dnn_names):
	# Plot the chance and stimulus onset dashed lines
	axs[d].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
			label='_nolegend_', linewidth=3)
	# Plot the baseline linearizing encoding results
	axs[d].plot(times, np.mean(np.mean(
		results_baseline[d]['correlation']['all_layers'], 0), 0), '--r',
		linewidth=3)
	axs[d].fill_between(times, results_baseline[d]['ci_upper']['all_layers'],
		results_baseline[d]['ci_lower']['all_layers'], color='r', alpha=.2)
	# Plot the appended layers linearizing encoding results
	axs[d].plot(times, np.mean(np.mean(
		results_app[d]['correlation']['appended_layers'], 0), 0),
		color=colors[0], linewidth=3)
	axs[d].fill_between(times, results_app[d]['ci_upper']['appended_layers'],
		results_app[d]['ci_lower']['appended_layers'], color=colors[0],
		alpha=.2)
	# Plot the significance markers
	axs[d].plot(times, sig_app[d], 'o', color=colors[0], markersize=4)
	# Plot the noise ceiling
	axs[d].fill_between(times, np.mean(np.mean(noise_ceiling_low, 0), 0),
		np.mean(np.mean(noise_ceiling_up, 0), 0), color=color_noise_ceiling,
		alpha=.3)
	# Plot parameters
	if d in [2, 3]:
		axs[d].set_xlabel('Time (s)', fontsize=30)
		xticks = [-.2, 0, .2, .4, .6, max(times)]
		xlabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
		plt.xticks(ticks=xticks, labels=xlabels)
	if d in [0, 2]:
		axs[d].set_ylabel('Pearson\'s $r$', fontsize=30)
		yticks = np.arange(0, 1.01, 0.25)
		ylabels = [0, 0.25, 0.5, 0.75, 1]
		plt.yticks(ticks=yticks, labels=ylabels)
	axs[d].set_xlim(left=min(times), right=max(times))
	axs[d].set_ylim(bottom=-.06, top=1)
	axs[d].set_title(dnn, fontsize=30)
	if d in [0]:
		leg = ['All layers', 'Appended layers']
		axs[d].legend(leg, fontsize=30, ncol=1, frameon=True)


# =============================================================================
# Plot the linearizing encoding correlation results of untrained DNNs
# =============================================================================
# Load the results
subjects = 'within'
pretrained = False
layers = 'all'
n_components = 1000
results = []
for d in dnns:
	data_dir = os.path.join(args.project_dir, 'results_debug_untrained', 'stats', 'correlation',
		'encoding-linearizing', 'subjects-'+subjects, 'dnn-'+d, 'pretrained-'+
		str(pretrained), 'layers-'+layers, 'n_components-'+
		format(n_components,'05'), 'correlation_stats.npy')
	results.append(np.load(data_dir, allow_pickle=True).item())
# Organize the significance values for plotting
sig = np.zeros((len(dnns),len(results[0]['significance']['all_layers'])))
for d in range(sig.shape[0]):
	for t in range(sig.shape[1]):
		if results[d]['significance']['all_layers'][t] == False:
			sig[d,t] = -100
		else:
			sig[d,t] = -.03

# Plot the correlation results of untrained DNNs, averaged across subjects
fig, axs = plt.subplots(2, 2, 'all', 'all')
axs = np.reshape(axs, (-1))
for d, dnn in enumerate(dnn_names):
	# Plot the chance and stimulus onset dashed lines
	axs[d].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
			label='_nolegend_', linewidth=3)
	# Plot the baseline linearizing encoding results
	axs[d].plot(times, np.mean(np.mean(
		results_baseline[d]['correlation']['all_layers'], 0), 0), '--r',
		linewidth=3)
	axs[d].fill_between(times, results_baseline[d]['ci_upper']['all_layers'],
		results_baseline[d]['ci_lower']['all_layers'], color='r', alpha=.2)
	# Plot the appended layers linearizing encoding results
	axs[d].plot(times, np.mean(np.mean(
		results[d]['correlation']['all_layers'], 0), 0),
		color=colors[0], linewidth=3)
	axs[d].fill_between(times, results[d]['ci_upper']['all_layers'],
		results[d]['ci_lower']['all_layers'], color=colors[0],
		alpha=.2)
	# Plot the significance markers
	axs[d].plot(times, sig[d], 'o', color=colors[0], markersize=4)
	# Plot the noise ceiling
	axs[d].fill_between(times, np.mean(np.mean(noise_ceiling_low, 0), 0),
		np.mean(np.mean(noise_ceiling_up, 0), 0), color=color_noise_ceiling,
		alpha=.3)
	# Plot parameters
	if d in [2, 3]:
		axs[d].set_xlabel('Time (s)', fontsize=30)
		xticks = [-.2, 0, .2, .4, .6, max(times)]
		xlabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
		plt.xticks(ticks=xticks, labels=xlabels)
	if d in [0, 2]:
		axs[d].set_ylabel('Pearson\'s $r$', fontsize=30)
		yticks = np.arange(0, 1.01, 0.25)
		ylabels = [0, 0.25, 0.5, 0.75, 1]
		plt.yticks(ticks=yticks, labels=ylabels)
	axs[d].set_xlim(left=min(times), right=max(times))
	axs[d].set_ylim(bottom=-.06, top=1)
	axs[d].set_title(dnn, fontsize=30)
	if d in [0]:
		leg = ['Trained DNNs', 'Untrained DNNs']
		axs[d].legend(leg, fontsize=30, ncol=1, frameon=True)


# =============================================================================
# Plot the linearizing encoding correlation results of different DNN components
# =============================================================================
# Load the results
subjects = 'within'
pretrained = True
layers = 'all'
n_components = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200,
	1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
results = []
for d in dnns:
	res_comp = []
	for c, comp in enumerate(n_components):
		data_dir = os.path.join(args.project_dir, 'results', 'stats',
			'correlation', 'encoding-linearizing', 'subjects-'+subjects,
			'dnn-'+d, 'pretrained-'+str(pretrained), 'layers-'+layers,
			'n_components-'+format(comp,'05'), 'correlation_stats.npy')
		res_comp.append(np.load(data_dir, allow_pickle=True).item())
	results.append(res_comp)
# Average the correlation results across subjects, EEG channels and EEG time
# points between 60-500ms
times = np.round(times, 2)
t_start = np.where(times == 0.06)[0][0]
t_end = np.where(times == 0.51)[0][0]
avg_res = []
avg_ci_low = []
avg_ci_up = []
for d in range(len(dnns)):
	res_comp = np.zeros(len(n_components))
	ci_low_comp = np.zeros(len(n_components))
	ci_up_comp = np.zeros(len(n_components))
	for c in range(len(n_components)):
		res_comp[c] = np.mean(
			results[d][c]['correlation']['all_layers'][:,:,t_start:t_end])
		ci_low_comp[c] = np.mean(
			results[d][c]['ci_lower']['all_layers'][t_start:t_end])
		ci_up_comp[c] = np.mean(
			results[d][c]['ci_upper']['all_layers'][t_start:t_end])
	avg_res.append(res_comp)
	avg_ci_low.append(ci_low_comp)
	avg_ci_up.append(ci_up_comp)
avg_nc_low = np.mean(noise_ceiling_low[:,:t,t_start:t_end])

# Plot the correlation results, averaged across subjects
x = np.arange(len(n_components))
width = .75
fig, axs = plt.subplots(2, 2, 'all', 'all')
axs = np.reshape(axs, (-1))
for d, dnn in enumerate(dnn_names):
	# Plot the correlation results
	axs[d].bar(x, avg_res[d], width)
	# Plot the confidence intervals
	low_ci = avg_res[d] - avg_ci_low[d]
	up_ci = avg_ci_up[d] - avg_res[d]
	conf_int = np.append(np.expand_dims(low_ci, 0), np.expand_dims(up_ci, 0), 0)
	axs[d].errorbar(x, avg_res[d], yerr=conf_int, fmt="none", ecolor="k",
		elinewidth=2, capsize=4)
	# Plot the noise ceiling
	axs[d].plot([min(x)-.35, max(x)+.35], [avg_nc_low, avg_nc_low], '--',
		linewidth=4, color=color_noise_ceiling)
	# Plot parameters
	axs[d].set_xlim(left=min(x)-.5, right=max(x)+.5)
	axs[d].set_ylim(bottom=0, top=.5)
	if d in [2, 3]:
		axs[d].set_xlabel('PCA components', fontsize=30)
		xticks = [4, 9, 14, 19]
		xlabels = [500, 1000, 1500, 2000]
		plt.xticks(ticks=xticks, labels=xlabels)
	if d in [0, 2]:
		axs[d].set_ylabel('Pearson\'s $r$', fontsize=30)
	axs[d].set_title(dnn, fontsize=30)
