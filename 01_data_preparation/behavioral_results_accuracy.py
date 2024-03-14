"""Calculate the behavioral results accuracy.

Parameters
----------
project_dir : str
		Directory of the project folder.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_1samp


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
args = parser.parse_args()

# Printing the arguments
print('\n\n\n>>> Behavioral data accuracy <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Calculate accuracy
# =============================================================================
subjects = 10
sessions = 4
accuracy = np.zeros((subjects,sessions))

for sub in range(subjects):
	for ses in range(sessions):
		# Load the data
		file_dir = os.path.join(args.project_dir, 'sub-'+format(sub+1,'02'),
				'ses-'+format(ses+1, '02'), 'behavioral_data.npy')
		beh = np.load(os.path.join(args.project_dir, file_dir),
			allow_pickle=True).item()
		target_present = beh['target_present']
		response = beh['response']
		# Calculate accuracy
		accuracy[sub,ses] = sum(target_present == response) / len(target_present) * 100


# =============================================================================
# Compute stats
# =============================================================================
accuracy = np.mean(accuracy, 1)
_, p_value = ttest_1samp(accuracy, 50, alternative='greater')
sig = False
if p_value < 0.05:
	sig = True

print('\n\nMean accuracy: ' + str(np.mean(accuracy)))
print('SD: ' + str(np.std(accuracy)))
print('Significant: ' + str(sig))
