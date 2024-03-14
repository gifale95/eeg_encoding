"""Extratact and save the behavioral results.

Parameters
----------
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from scipy import io
from tqdm import tqdm


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
args = parser.parse_args()

# Printing the arguments
print('\n\n\n>>> Extract behavioral data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Get behavioral data
# =============================================================================
subjects = 10
sessions = 4
train_partitions = 5

for sub in tqdm(range(subjects)):
	for ses in range(sessions):
		beh_dir = os.path.join(args.project_dir, 'source_data', 'sub-'+
			format(sub+1,'02'), 'ses-'+format(ses+1, '02'), 'beh')
		# Test runs
		test_file = 'sub-'+format(sub+1,'02')+'_ses-'+format(ses+1, '02')+\
			'_task-test_beh.mat'
		behav = io.loadmat(os.path.join(beh_dir, test_file))['data']
		target_present = np.asarray(behav[0][0][3]['target'][0], dtype=int)
		response = np.asarray(behav[0][0][3]['response'][0], dtype=int)
		# Training runs
		for part in range(train_partitions):
			train_file = 'sub-'+format(sub+1,'02')+'_ses-'+format(ses+1, '02')+\
				'_task-train_part-'+format(part+1, '02')+'_beh.mat'
			behav = io.loadmat(os.path.join(beh_dir, train_file))['data']
			target_present = np.append(target_present, np.asarray(
				behav[0][0][3]['target'][0], dtype=int))
			response = np.append(response, np.asarray(
				behav[0][0][3]['target'][0], dtype=int))
		# Check that all sequences are there
		if len(target_present) != 1044:
			raise Exception('Some sequences results are missing!')

		# Save the data
		behav_results = {
			'target_present': target_present,
			'response': response
			}
		save_dir = os.path.join(args.project_dir, 'behavioral_data',
			'sub-'+format(sub+1,'02'), 'ses-'+format(ses+1, '02'))
		file_name = 'behavioral_data'
		if os.path.isdir(save_dir) == False:
			os.makedirs(save_dir)
		np.save(os.path.join(save_dir, file_name), behav_results)
