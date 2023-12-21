"""Convert the source data into raw data.

Parameters
----------
sub : int
	Used subject.
session : int
	Used sessions.
partition : str
	Used data partition ['test' or 'train'].
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
from scipy import io
import numpy as np
import mne


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--session', default=1, type=int)
parser.add_argument('--partition', default='train', type=str)
parser.add_argument('--project_dir', default='../project_directory', type=str)
args = parser.parse_args()

# Printing the arguments
print('\n\n\n>>> EEG source to raw <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Loading the behavioral and EEG data
# =============================================================================
if args.partition == 'train':
	n_parts = 5
elif args.partition == 'test':
	n_parts = 1
data_dir = os.path.join(args.project_dir, 'eeg_dataset', 'source_data',
	'sub-'+format(args.sub,'02'), 'ses-'+format(args.session,'02'))

for p in range(n_parts):
	if args.partition == 'train':
		beh_dir = os.path.join(data_dir, 'beh', 'sub-'+format(args.sub,'02')+
			'_ses-'+format(args.session,'02')+'_task-'+args.partition+'_part-'+
			format(p+1,'02')+'_beh.mat')
		eeg_dir = os.path.join(data_dir, 'eeg', 'sub-'+format(args.sub,'02')+
			'_ses-'+format(args.session,'02')+'_task-'+args.partition+'_part-'+
			format(p+1,'02')+'_eeg.vhdr')
	elif args.partition == 'test':
		beh_dir = os.path.join(data_dir, 'beh', 'sub-'+format(args.sub,'02')+
			'_ses-'+format(args.session,'02')+'_task-'+args.partition+
			'_beh.mat')
		eeg_dir = os.path.join(data_dir, 'eeg', 'sub-'+format(args.sub,'02')+
			'_ses-'+format(args.session,'02')+'_task-'+args.partition+
			'_eeg.vhdr')

	beh_data = io.loadmat(beh_dir)['data']
	eeg_data = mne.io.read_raw_brainvision(eeg_dir, preload=True)


# =============================================================================
# Extracting the EEG data and creating the events channel
# =============================================================================
	# EEG data and recording info
	provv_eeg_mat = eeg_data.get_data()
	info = eeg_data.info

	# Change the target events from 0 to 9999
	events_beh = np.asarray(beh_data[0][0][2]['tot_img_number'][0], dtype=int)
	idx_targ = np.where(events_beh == 0)[0]
	events_beh[idx_targ] = 99999
	del beh_data

	# Getting the event samples info from the EEG data
	events_samples, _ = mne.events_from_annotations(eeg_data)
	events_samples = events_samples[1:,0]
	del eeg_data

	# Creating the events channel, and appending it to the EEG data
	events_channel = np.zeros((1,provv_eeg_mat.shape[1]))
	idx = 0
	for s in range(events_channel.shape[1]):
		if idx < len(events_beh):
			if events_samples[idx] == s:
				events_channel[0,s] = events_beh[idx]
				idx += 1
	provv_eeg_mat = np.append(provv_eeg_mat, events_channel, 0)
	del events_channel, events_samples

	# Appending the data of the different training parts
	if p == 0:
		eeg_data_matrix = provv_eeg_mat
	else:
		eeg_data_matrix = np.append(eeg_data_matrix, provv_eeg_mat, 1)
	del provv_eeg_mat


# =============================================================================
# Prepare the data info
# =============================================================================
ch_names = info.ch_names
ch_names.append('stim')
ch_types = []
for c in range(len(ch_names)-1):
	ch_types.append('eeg')
ch_types.append('stim')


# =============================================================================
# Saving the EEG data and the data info
# =============================================================================
data = {
	'raw_eeg_data': eeg_data_matrix,
	'ch_names': ch_names,
	'ch_types': ch_types,
	'sfreq': 1000,
	'highpass': 0.01,
	'lowpass': 100
}
del eeg_data_matrix

if args.partition == 'train':
	file_name = 'raw_eeg_training.npy'
elif args.partition == 'test':
	file_name = 'raw_eeg_test.npy'
save_dir = os.path.join(args.project_dir, 'eeg_dataset', 'raw_data', 'sub-'+
	format(args.sub,'02'), 'ses-'+format(args.session,'02'))

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), data)
