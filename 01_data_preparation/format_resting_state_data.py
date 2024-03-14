"""Reformatting and saving the resting state EEG data.

Parameters
----------
sub : int
	Used subject.
session : int
	Used sessions.
run : int
	If '1', sort the resting state at the beginning of the session. If '2', sort
	the resting state at the end of the session.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import mne


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--session', default=1, type=int)
parser.add_argument('--run', default=1, type=int)
parser.add_argument('--project_dir', default='../project_directory', type=str)
args = parser.parse_args()

# Printing the arguments
print('\n\n\n>>> Sorting raw resting state EEG data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Loading the EEG data
# =============================================================================
eeg_dir = os.path.join(args.project_dir, 'eeg_dataset', 'source_data',
	'sub-'+format(args.sub,'02'), 'ses-'+format(args.session,'02'), 'eeg',
	'sub-'+format(args.sub,'02')+'_ses-'+format(args.session,'02')+'_task-rest'+
	format(args.run, '01')+'_eeg.vhdr')

eeg_data = mne.io.read_raw_brainvision(eeg_dir, preload=True)


# =============================================================================
# Extracting the EEG data and creating the events channel
# =============================================================================
# EEG data and recording info
eeg_data_matrix = eeg_data.get_data()
info = eeg_data.info

# Getting the event samples info from the EEG data
events_samples, _ = mne.events_from_annotations(eeg_data)
events_samples = events_samples[1:]
del eeg_data

# Creating the events channel, and appending it to the EEG data
events_channel = np.zeros((1,eeg_data_matrix.shape[1]))
events_channel[0,events_samples[0,0]] = 1
events_channel[0,events_samples[1,0]] = 1
eeg_data_matrix = np.append(eeg_data_matrix, events_channel, 0)
del events_channel, events_samples


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

file_name = 'eeg_resting_state_'+str(args.run)+'.npy'
save_dir = os.path.join(args.project_dir, 'eeg_dataset', 'resting_state_data',
	'sub-'+format(args.sub,'02'), 'ses-'+format(args.session,'02'))

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), data)
