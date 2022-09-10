"""Plot the zero-shot identification analysis results.

Parameters
----------
rank_correct : int
	Accepted correlation rank of the correct synthetic data image condition.
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
parser.add_argument('--rank_correct', default=1, type=int)
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
colors = [(31/255, 119/255, 180/255), (255/255, 127/255, 14/255),
	(44/255, 160/255, 44/255), (214/255, 39/255, 40/255)]


# =============================================================================
# Load the zero-shot identification results
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
		'zero_shot_identification', 'encoding-linearizing', 'subjects-'+
		subjects, 'dnn-'+d, 'pretrained-'+str(pretrained), 'layers-'+layers,
		'n_components-'+format(n_components,'05'), 'rank_correct-'+
		format(args.rank_correct,'02'), 'zero_shot_identification_stats.npy')
	results.append(np.load(data_dir, allow_pickle=True).item())
steps = results[0]['steps']
ch_names = results[0]['ch_names']

# Organize the significance values for plotting
sig = np.zeros((len(dnns),len(results[0]['significance']['all_layers'])))
for d in range(sig.shape[0]):
	for st in range(sig.shape[1]):
		if results[d]['significance']['all_layers'][st] == False:
			sig[d,st] = -100
		else:
			sig[d,st] = (94 + d * 1.75) / 100

# Create the chance variable
chance = np.zeros(len(steps))
for st in range(len(steps)):
	chance[st] = args.rank_correct / (200+steps[st])


# =============================================================================
# Plot the zero-shot identification results
# =============================================================================
# Plot the zero-shot identification results, averaged across subjects
plt.figure(figsize=(9,6))
for d in range(len(dnns)):
	# Plot chance dashed lines
	plt.plot(steps, chance, 'k--', label='_nolegend_', linewidth=4)
	# Plot the results
	plt.plot(steps, np.mean(results[d]['identification_accuracy']['all_layers'],
		0), color=colors[d], linewidth=4)
for d in range(len(dnns)):
	# Plot the confidence intervals
	plt.fill_between(steps, results[d]['ci_upper']['all_layers'],
		results[d]['ci_lower']['all_layers'], color=colors[d], alpha=.2)
	# Plot the significance markers
	plt.plot(steps, sig[d], 'o', color=colors[d], markersize=4)
# Other plot parameters
plt.xlabel('Image set size', fontsize=30)
xticks = np.arange(0, 150001, 30000)
xlabels = ['0', '30k', '60k', '90k', '120k', '150k']
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=0, right=150000)
plt.ylabel('Identification\naccuracy (%)', fontsize=30)
yticks = np.arange(0, 1.01, .2)
ylabels = [0, 20, 40, 60, 80, 100]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=0, top=1)
plt.legend(dnn_names, fontsize=30, ncol=2, frameon=False)

# Plot the single-subjects zero-shot identification results
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(len(results[0]['identification_accuracy']['all_layers'])):
	# Plot the chance dashed lines
	axs[s].plot(steps, chance, 'k--', linewidth=4)
	for d in range(len(dnns)):
		# Plot the results
		axs[s].plot(steps,
			results[d]['identification_accuracy']['all_layers'][s],
			color=colors[d], linewidth=3)
	# Other plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Image set size', fontsize=30)
		xticks = [0, 75000, 150000]
		xlabels = [0, '75k', '150k']
		plt.xticks(ticks=xticks, labels=xlabels)
	if s in [0, 4, 8]:
		axs[s].set_ylabel('Identification\naccuracy (%)', fontsize=30)
		yticks = np.arange(0, 1.01, .5)
		ylabels = [0, 50, 100]
		plt.yticks(ticks=yticks, labels=ylabels)
	axs[s].set_xlim(left=min(steps), right=max(steps))
	axs[s].set_ylim(bottom=0, top=1)
	tit = 'Participant ' + str(s+1)
	axs[s].set_title(tit, fontsize=30)
axs[10].set_xlabel('Image set size', fontsize=30)
axs[11].set_xlabel('Image set size', fontsize=30)


# =============================================================================
# Plot the 300 best EEG features chosen for the identification
# =============================================================================
# Plot the best features, summed over subjects
fig, axs = plt.subplots(2, 2, 'all', 'all')
axs = np.reshape(axs, (-1))
for d, dnn in enumerate(dnn_names):
	img = axs[d].imshow(np.sum(
		results[d]['best_features_masks']['all_layers']*10, 0), aspect='auto')
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
plt.colorbar(img, label='% of participants', fraction=0.2, ax=axs[d])


# =============================================================================
# Plot the extrapolation results
# =============================================================================
# Plot the correlation results, averaged across subjects
x = np.arange(len(results[0]['extr_10_percent']['all_layers']))
avg_x = x[-1] + 1
width = .2
fig, axs = plt.subplots(2, 1, 'all')
axs = np.reshape(axs, (-1))
for s in range(len(axs)):
	# Plot the single-subject extrapolations
	if s == 0:
		axs[s].bar(x-width*1.5, results[0]['extr_10_percent']['all_layers'],
			width, label=dnn_names[0], color=colors[0])
		axs[s].bar(x-width*.5, results[1]['extr_10_percent']['all_layers'],
			width, label=dnn_names[1], color=colors[1])
		axs[s].bar(x+width*.5, results[2]['extr_10_percent']['all_layers'],
			width, label=dnn_names[2], color=colors[2])
		axs[s].bar(x+width*1.5, results[3]['extr_10_percent']['all_layers'],
			width, label=dnn_names[3], color=colors[3])
	elif s == 1:
		axs[s].bar(x-width*1.5,
			results[0]['extr_0point5_percent']['all_layers'], width,
			label=dnn_names[0], color=colors[0])
		axs[s].bar(x-width*.5,
			results[1]['extr_0point5_percent']['all_layers'], width,
			label=dnn_names[1], color=colors[1])
		axs[s].bar(x+width*.5,
			results[2]['extr_0point5_percent']['all_layers'], width,
			label=dnn_names[2], color=colors[2])
		axs[s].bar(x+width*1.5,
			results[3]['extr_0point5_percent']['all_layers'], width,
			label=dnn_names[3], color=colors[3])
	if s == 0:
		# Plot the extrapolations averaged across subjects
		axs[s].bar(avg_x-width*1.5, np.mean(
			results[0]['extr_10_percent']['all_layers']), width,
			label='_nolegend_', color=colors[0])
		axs[s].bar(avg_x-width*.5, np.mean(
			results[1]['extr_10_percent']['all_layers']), width,
			label='_nolegend_', color=colors[1])
		axs[s].bar(avg_x+width*.5, np.mean(
			results[2]['extr_10_percent']['all_layers']), width,
			label='_nolegend_', color=colors[2])
		axs[s].bar(avg_x+width*1.5, np.mean(
			results[3]['extr_10_percent']['all_layers']), width,
			label='_nolegend_', color=colors[3])
		# Plot the confidence intervals
		conf_int_0 = np.reshape(np.append(np.expand_dims(np.mean(
			results[0]['extr_10_percent']['all_layers']) -
			results[0]['ci_lower_extr_10']['all_layers'], 0),
			np.expand_dims(results[0]['ci_upper_extr_10']['all_layers'] -
			np.mean(results[0]['extr_10_percent']['all_layers']), 0)), (-1,1))
		conf_int_1 = np.reshape(np.append(np.expand_dims(np.mean(
			results[1]['extr_10_percent']['all_layers']) -
			results[1]['ci_lower_extr_10']['all_layers'], 0),
			np.expand_dims(results[1]['ci_upper_extr_10']['all_layers'] -
			np.mean(results[1]['extr_10_percent']['all_layers']), 0)), (-1,1))
		conf_int_2 = np.reshape(np.append(np.expand_dims(np.mean(
			results[2]['extr_10_percent']['all_layers']) -
			results[2]['ci_lower_extr_10']['all_layers'], 0),
			np.expand_dims(results[2]['ci_upper_extr_10']['all_layers'] -
			np.mean(results[2]['extr_10_percent']['all_layers']), 0)), (-1,1))
		conf_int_3 = np.reshape(np.append(np.expand_dims(np.mean(
			results[3]['extr_10_percent']['all_layers']) -
			results[3]['ci_lower_extr_10']['all_layers'], 0),
			np.expand_dims(results[3]['ci_upper_extr_10']['all_layers'] -
			np.mean(results[3]['extr_10_percent']['all_layers']), 0)), (-1,1))
		axs[s].errorbar(avg_x-width*1.5, np.mean(
			results[0]['extr_10_percent']['all_layers']), yerr=conf_int_0,
			fmt="none", ecolor="k", elinewidth=2, capsize=4)
		axs[s].errorbar(avg_x-width*.5, np.mean(
			results[1]['extr_10_percent']['all_layers']), yerr=conf_int_1,
			fmt="none", ecolor="k", elinewidth=2, capsize=4)
		axs[s].errorbar(avg_x+width*.5, np.mean(
			results[2]['extr_10_percent']['all_layers']), yerr=conf_int_2,
			fmt="none", ecolor="k", elinewidth=2, capsize=4)
		axs[s].errorbar(avg_x+width*1.5, np.mean(
			results[3]['extr_10_percent']['all_layers']), yerr=conf_int_3,
			fmt="none", ecolor="k", elinewidth=2, capsize=4)
	elif s == 1:
		# Plot the extrapolations averaged across subjects
		axs[s].bar(avg_x-width*1.5,
			np.mean(results[0]['extr_0point5_percent']['all_layers']), width,
			label='_nolegend_', color=colors[0])
		axs[s].bar(avg_x-width*.5,
			np.mean(results[1]['extr_0point5_percent']['all_layers']), width,
			label='_nolegend_', color=colors[1])
		axs[s].bar(avg_x+width*.5,
			np.mean(results[2]['extr_0point5_percent']['all_layers']), width,
			label='_nolegend_', color=colors[2])
		axs[s].bar(avg_x+width*1.5,
			np.mean(results[3]['extr_0point5_percent']['all_layers']), width,
			label='_nolegend_', color=colors[3])
		# Plot the confidence intervals
		conf_int_0 = np.reshape(np.append(np.expand_dims(np.mean(
			results[0]['extr_0point5_percent']['all_layers']) -
			results[0]['ci_lower_extr_0point5']['all_layers'], 0),
			np.expand_dims(results[0]['ci_upper_extr_0point5']['all_layers'] -
			np.mean(results[0]['extr_0point5_percent']['all_layers']), 0)), (-1,1))
		conf_int_1 = np.reshape(np.append(np.expand_dims(np.mean(
			results[1]['extr_0point5_percent']['all_layers']) -
			results[1]['ci_lower_extr_0point5']['all_layers'], 0),
			np.expand_dims(results[1]['ci_upper_extr_0point5']['all_layers'] -
			np.mean(results[1]['extr_0point5_percent']['all_layers']), 0)), (-1,1))
		conf_int_2 = np.reshape(np.append(np.expand_dims(np.mean(
			results[2]['extr_0point5_percent']['all_layers']) -
			results[2]['ci_lower_extr_0point5']['all_layers'], 0),
			np.expand_dims(results[2]['ci_upper_extr_0point5']['all_layers'] -
			np.mean(results[2]['extr_0point5_percent']['all_layers']), 0)), (-1,1))
		conf_int_3 = np.reshape(np.append(np.expand_dims(np.mean(
			results[3]['extr_0point5_percent']['all_layers']) -
			results[3]['ci_lower_extr_0point5']['all_layers'], 0),
			np.expand_dims(results[3]['ci_upper_extr_0point5']['all_layers'] -
			np.mean(results[3]['extr_0point5_percent']['all_layers']), 0)), (-1,1))
		axs[s].errorbar(avg_x-width*1.5, np.mean(
			results[0]['extr_0point5_percent']['all_layers']), yerr=conf_int_0,
			fmt="none", ecolor="k", elinewidth=2, capsize=4)
		axs[s].errorbar(avg_x-width*.5, np.mean(
			results[1]['extr_0point5_percent']['all_layers']), yerr=conf_int_1,
			fmt="none", ecolor="k", elinewidth=2, capsize=4)
		axs[s].errorbar(avg_x+width*.5, np.mean(
			results[2]['extr_0point5_percent']['all_layers']), yerr=conf_int_2,
			fmt="none", ecolor="k", elinewidth=2, capsize=4)
		axs[s].errorbar(avg_x+width*1.5, np.mean(
			results[3]['extr_0point5_percent']['all_layers']), yerr=conf_int_3,
			fmt="none", ecolor="k", elinewidth=2, capsize=4)
	# Plot parameters
	axs[s].set_yscale('log')
	axs[s].set_ylabel('Image set size', fontsize=30)
	if s == 0:
		yticks = [10**5, 10**6, 10**7, 10**8]
		plt.yticks(ticks=yticks)
		axs[s].set_ylim(top=10**8)
		axs[s].legend(fontsize=30, frameon=False, loc=9, ncol=4)
	elif s == 1:
		yticks = [10**8, 10**10, 10**12, 10**14]
		plt.yticks(ticks=yticks)
		axs[s].set_ylim(top=10**14)
		axs[s].set_xlabel('Participants', fontsize=30)
		xticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		xlabels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'Average']
		plt.xticks(ticks=xticks, labels=xlabels)
		axs[s].set_xlim(left=min(x)-.5, right=avg_x+.5)
