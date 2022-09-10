"""Visualize the ERPs of the biological and synthetic test EEG data.

Parameters
----------
used_subs : list
	List of subjects data to plot.
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
parser.add_argument('--used_subs', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
	type=list)
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/'
	'projects/eeg_encoding/paradigm_3', type=str)
#parser.add_argument('--project_dir', default='../project/directory', type=str)
args = parser.parse_args()


# =============================================================================
# Load the biological EEG training and test data
# =============================================================================
bio_erps = []
for s in args.used_subs:
	data_dir = os.path.join('eeg_dataset', 'preprocessed_data', 'sub-'+
		format(s,'02'), 'preprocessed_eeg_test.npy')
	bio_data_test = np.load(os.path.join(args.project_dir, data_dir),
		allow_pickle=True).item()
	# Append the training and test data and average across image conditions
	# and repetitions
	data_test = bio_data_test['preprocessed_eeg_data']
	data_test = np.reshape(data_test, (-1,data_test.shape[2],
		data_test.shape[3]))
	# Averaged the EEG test data across image conditions and repetitions
	bio_erps.append(np.mean(data_test, 0))
	times = bio_data_test['times']
	ch_names = bio_data_test['ch_names']
	del bio_data_test, data_test
bio_erps = np.asarray(bio_erps)


# =============================================================================
# Load the linearizing and end-to-end AlexNet synthetic EEG test data
# =============================================================================
lin_synt_erps = []
end_synt_erps = []
for s in args.used_subs:
	data_dir_lin = os.path.join(args.project_dir, 'results', 'sub-'+
		format(s,'02'), 'synthetic_eeg_data', 'encoding-linearizing',
		'subjects-within', 'dnn-alexnet', 'pretrained-True', 'layers-all',
		'n_components-01000', 'synthetic_eeg_test.npy')
	data_dir_end = os.path.join(args.project_dir, 'results', 'sub-'+
		format(s,'02'), 'synthetic_eeg_data', 'encoding-end_to_end',
		'dnn-alexnet', 'modeled_time_points-all', 'pretrained-False',
		'lr-{:.0e}'.format(1e-5)+'__wd-{:.0e}'.format(0.)+'__bs-'+
		format(64,'03'), 'synthetic_eeg_test.npy')
	lin_synt_data = np.load(os.path.join(args.project_dir, data_dir_lin),
		allow_pickle=True).item()
	end_synt_data = np.load(os.path.join(args.project_dir, data_dir_end),
		allow_pickle=True).item()
	# Averaged the EEG data across image conditions
	lin_synt_erps.append(np.mean(
		lin_synt_data['synthetic_data']['all_layers'], 0))
	end_synt_erps.append(np.mean(
		end_synt_data['synthetic_data']['all_time_points'], 0))
	del lin_synt_data, end_synt_data
lin_synt_erps = np.asarray(lin_synt_erps)
end_synt_erps = np.asarray(end_synt_erps)


# =============================================================================
# Plot parameters
# =============================================================================
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
colors = [(57/255, 158/255, 52/255), (42/255, 121/255, 178/255),
	(252/255, 129/255, 37/255)]


# =============================================================================
# Plot biological and synthetic data ERPs over time
# =============================================================================
# The biological/synthetic data ERPs are created by averaging across the
# biological/synthetic test data images conditions and repetitions.

# Plot the ERPs of the first subject
fig, axs = plt.subplots(2, 1, 'all')
axs = np.reshape(axs, (-1))
for s in range(2):
	# Plot baseline and stimulus onset dashed lines
	axs[s].plot([-10, 10], [0, 0], 'k--', [0, 0],
		[100, -100], 'k--', label='_nolegend_', linewidth=3)
	# Plot the single channels ERPs
	if s == 0:
		axs[s].plot(times, np.transpose(end_synt_erps[0]), color=colors[2],
			linewidth=2)
		axs[s].plot(times, np.transpose(lin_synt_erps[0]), color=colors[1],
			linewidth=2)
		axs[s].plot(times, np.transpose(bio_erps[0]), color=colors[0],
			linewidth=2)
		axs[s].set_ylim(bottom=-1.1, top=1)
		yticks = [-1, -0.5, 0, 0.5, 1, 1.5]
		axs[s].set_yticks(ticks=yticks)
	# Plot the ERPs averaged across channels
	if s == 1:
		axs[s].plot(times, np.mean(end_synt_erps[0], 0), color=colors[2],
			linewidth=4)
		axs[s].plot(times, np.mean(lin_synt_erps[0], 0), color=colors[1],
			linewidth=4)
		axs[s].plot(times, np.mean(bio_erps[0], 0), color=colors[0],
			linewidth=4)
		axs[s].set_ylim(bottom=-.125, top=.075)
		yticks = [-.1, -.05, 0, .05, .1]
		axs[s].set_yticks(ticks=yticks)
	# Other plot parameters
	if s in [1]:
		axs[s].set_xlabel('Time (s)', fontsize=30)
		xticks = [-.2, -.1, 0, .1, .2, .3, .4, .5, .6, .7, max(times)]
		xlabels = [-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
		plt.xticks(ticks=xticks, labels=xlabels)
		leg = ['End-to-end SynTest', 'Linearizing SynTest', 'BioTest']
		axs[s].legend(leg, fontsize=30, ncol=3, frameon=False)
	axs[s].set_ylabel('Voltage', fontsize=30)
	axs[s].set_xlim(left=min(times), right=max(times))

# Plot the single channel ERPs of all subjects
fig, axs = plt.subplots(5, 2, 'all', 'all')
axs = np.reshape(axs, (-1))
for s, sub in enumerate(args.used_subs):
	# Plot zero and stimulus onset dashed lines
	axs[s].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
			label='_nolegend_', linewidth=3)
	axs[s].plot(times, np.transpose(end_synt_erps[s]), color=colors[2],
		linewidth=1)
	axs[s].plot(times, np.transpose(lin_synt_erps[s]), color=colors[1],
		linewidth=1)
	axs[s].plot(times, np.transpose(bio_erps[s]), color=colors[0],
		linewidth=1)
	# Other plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Time (s)', fontsize=30)
		plt.xticks(ticks=[0, .2, .4, .6, max(times)],
			labels=[0, 0.2, 0.4, 0.6, 0.8])
	if s in [0, 2, 4, 6, 8]:
		axs[s].set_ylabel('Voltage', fontsize=30)
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[s].set_ylim(bottom=-1.25, top=1.75)
	tit = 'Participant ' + str(sub)
	axs[s].set_title(tit, fontsize=30)

# Plot the averaged channel ERPs of all subjects
fig, axs = plt.subplots(5, 2, 'all', 'all')
axs = np.reshape(axs, (-1))
for s, sub in enumerate(args.used_subs):
	# Plot zero and stimulus onset dashed lines
	axs[s].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
			label='_nolegend_', linewidth=3)
	axs[s].plot(times, np.mean(end_synt_erps[s], 0), color=colors[2],
		linewidth=3)
	axs[s].plot(times, np.mean(lin_synt_erps[s], 0), color=colors[1],
		linewidth=3)
	axs[s].plot(times, np.mean(bio_erps[s], 0), color=colors[0],
		linewidth=3)
	# Other plot parameters
	if s in [8, 9]:
		axs[s].set_xlabel('Time (s)', fontsize=30)
		plt.xticks(ticks=[0, .2, .4, .6, max(times)],
			labels=[0, 0.2, 0.4, 0.6, 0.8])
	if s in [0, 2, 4, 6, 8]:
		axs[s].set_ylabel('Voltage', fontsize=30)
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[s].set_ylim(bottom=-.26, top=.125)
	tit = 'Participant ' + str(sub)
	axs[s].set_title(tit, fontsize=30)

#plt.savefig('erp_sub_1', dpi=600)
#plt.savefig('erp_all_sub_sing_chan', dpi=600)
#plt.savefig('erp_all_sub_avg_chan', dpi=600)
#plt.savefig('legend', dpi=600)