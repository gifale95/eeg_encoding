"""Plotting the zero-shot identification analysis results.

Parameters
----------
n_tot_sub : int
	Number of total subjects used.
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
parser.add_argument('--n_tot_sub', default=10, type=int)
parser.add_argument('--rank_correct', default=1, type=int)
parser.add_argument('--project_dir', default='/project/directory', type=str)
args = parser.parse_args()


# =============================================================================
# Loading the correlation results and stats
# =============================================================================
dnns = ['alexnet', 'resnet50', 'cornet_s', 'moco']
for d, dnn in enumerate(dnns):
	data_dir = os.path.join(args.project_dir, 'results', 'stats',
		'zero_shot_identification', 'dnn-'+dnn, 'rank_correct-'+
		format(args.rank_correct,'02'), 'zero_shot_identification_stats.npy')
	# Loading the data
	data_dict = np.load(data_dir, allow_pickle=True).item()
	if d == 0:
		identification_accuracy = np.expand_dims(
			data_dict['identification_accuracy'], 0)
		ci_lower = np.expand_dims(data_dict['ci_lower'], 0)
		ci_upper = np.expand_dims(data_dict['ci_upper'], 0)
		significance = np.expand_dims(data_dict['significance'],
			0)
		extr_10_percent = np.expand_dims(
			data_dict['extr_10_percent'], 0)
		extr_0point5_percent = np.expand_dims(
			data_dict['extr_0point5_percent'], 0)
		steps = data_dict['steps']
	else:
		identification_accuracy = np.append(identification_accuracy,
			np.expand_dims(data_dict['identification_accuracy'], 0), 0)
		ci_lower = np.append(ci_lower, np.expand_dims(
			data_dict['ci_lower'], 0), 0)
		ci_upper = np.append(ci_upper, np.expand_dims(
			data_dict['ci_upper'], 0), 0)
		significance = np.append(significance, np.expand_dims(
			data_dict['significance'], 0), 0)
		extr_10_percent = np.append(extr_10_percent,
			np.expand_dims(data_dict['extr_10_percent'], 0), 0)
		extr_0point5_percent = np.append(extr_0point5_percent,
			np.expand_dims(data_dict['extr_0point5_percent'], 0), 0)

# Organizing the significance values for plotting
sig = np.zeros(significance.shape)
for d in range(len(dnns)):
	for st in range(significance.shape[1]):
		if significance[d,st] == False:
			sig[d,st] = -100
		else:
			sig[d,st] = 94 + d * 1.75

# Creating the chance variable
chance = np.zeros(len(steps))
for st in range(len(steps)):
	chance[st] = args.rank_correct / (200+steps[st]) * 100


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
color_noise_ceiling = (150/255, 150/255, 150/255)
colors = [(31/255, 119/255, 180/255), (255/255, 127/255, 14/255),
	(44/255, 160/255, 44/255), (214/255, 39/255, 40/255)]

plt.figure(figsize=(9,6))
for d in range(len(dnns)):
	# Plotting the results
	plt.plot(steps, np.mean(identification_accuracy[d], 0), color=colors[d],
		linewidth=4)
for d in range(len(dnns)):
	# Plotting the confidence intervals
	plt.fill_between(steps, ci_upper[d], ci_lower[d], color=colors[d], alpha=.2)
	# Plotting the significance markers
	plt.plot(steps, sig[d], 'o', color=colors[d], markersize=4)
# Plotting chance and stimulus onset dashed lines
plt.plot(steps, chance, 'k--', linewidth=4)
# Other plot parameters
plt.xlabel('Image set size', fontsize=30)
xlabels = ['0', '30k', '60k', '90k', '120k', '150k']
plt.xticks(ticks=np.arange(0,150001,30000), labels=xlabels)
plt.xlim(left=0, right=150000)
plt.ylabel('Decoding\naccuracy (%)', fontsize=30)
ylabels = [0, 20, 40, 60, 80, 100]
plt.yticks(ticks=np.arange(0,101,20), labels=ylabels)
plt.ylim(bottom=0, top=100)
leg = ['AlexNet', 'ResNet-50', 'CORnet-S', 'MoCo']
plt.legend(leg, fontsize=30, ncol=2, frameon=False)


# =============================================================================
# Plotting the results for single subjects
# =============================================================================
fig, axs = plt.subplots(3, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for s in range(args.n_tot_sub):
	for d in range(len(dnns)):
		# Plotting the results
		axs[s].plot(steps, identification_accuracy[d,s], color=colors[d],
			linewidth=3)
	# Plotting chance and stimulus onset dashed lines
	axs[s].plot(steps, chance, 'k--', linewidth=4)
	# Other plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Image set size', fontsize=30)
		plt.xticks(ticks=[0, 75000, 150000], labels=[0, '75k', '150k'])
	if s in [0, 4, 8]:
		axs[s].set_ylabel('Decoding\naccuracy (%)', fontsize=30)
		plt.yticks(ticks=np.arange(0, 101, 50), labels=[0, 50, 100])
	axs[s].set_xlim(left=min(steps), right=max(steps))
	axs[s].set_ylim(bottom=0, top=100)
	tit = 'Participant ' + str(s+1)
	axs[s].set_title(tit, fontsize=30)
axs[10].set_xlabel('Image set size', fontsize=30)
axs[11].set_xlabel('Image set size', fontsize=30)
