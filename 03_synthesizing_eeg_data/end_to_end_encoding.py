"""Training end-to-end DNN models which predict the EEG responses to the test
images. The EEG responses to each test image condition are synthesized in a
cross-validated fashion, by using the training epoch weights which yielded the
highest correlation prediction score for all the other N-1 test image
conditions.

Parameters
----------
sub : int
	Used subject.
tot_eeg_chan : int
	Total amount of EEG channels.
tot_eeg_time : int
	Total amount of EEG time points.
dnn : str
	DNN model used.
pretrained : bool
	Whether to use or not pretrained DNN models.
epochs : int
	Number of training epochs.
lr : float
	Learning rate.
weight_decay : float
	Weight decay coefficient.
batch_size : int
	Batch size for weight update.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import random
import torch
from end_to_end_encoding_utils import load_images
from end_to_end_encoding_utils import load_eeg_data
from end_to_end_encoding_utils import create_dataloader
from end_to_end_encoding_utils import get_model
from scipy.stats import pearsonr as corr

parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default=1)
parser.add_argument('--tot_eeg_chan', type=int, default=17)
parser.add_argument('--tot_eeg_time', type=int, default=100)
parser.add_argument('--dnn', type=str, default='alexnet')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--project_dir', default='/project/directory', type=str)
args = parser.parse_args()

print('>>> End-to-end encoding <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Set random seeds to make results reproducible and GPU
# =============================================================================
# Random seeds
seed = 20200220
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Checking for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Loading the images (X) and the EEG data (y)
# =============================================================================
X_train, X_test = load_images(args)

# Snthetic EEG data matrix of shape:
# (Test image conditions × EEG channels × EEG time points)
synthetic_data = np.zeros((len(X_test),args.tot_eeg_chan,args.tot_eeg_time))
# Loop across EEG time points
for t in range(args.tot_eeg_time):
	print(f'\nTime: [{t+1}/{args.tot_eeg_time}]')
	y_train, y_test, ch_names, times = load_eeg_data(args, t)


# =============================================================================
# Creating PyTorch-compatible training and test Dataloaders
# =============================================================================
	train_dl, test_dl = create_dataloader(args, X_train, y_train, X_test,
		y_test)
	del y_train, y_test


# =============================================================================
# Loading the DNN model and changing the last layer to 17 features
# =============================================================================
	model = get_model(args)
	model.to(device)


# =============================================================================
# Defining loss function, optimizer, training and test loops
# =============================================================================
	# Loss function
	loss_fn = torch.nn.MSELoss(reduction='sum')

	# Optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
		weight_decay=args.weight_decay)

	# Training loop
	def train_loop(train_dl, model, loss_fn, optimizer):
		model.train()
		for batch, (X, y) in enumerate(train_dl):
			X, y = X.to(device), y.to(device)
			# Compute prediction and loss
			pred = model(X).squeeze()
			loss = loss_fn(pred, y)
			# Backpropagation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	# Test loop
	def test_loop(test_dl, model, corr_results):
		model.eval()
		with torch.no_grad():
			for X, y in test_dl:
				X, y = X.to(device), y.to(device)
				pred = model(X).squeeze()
				# Performing the correlation in a leave-one-image-condition-out
				# fashion for cross-validation
				corr_mean = []
				for i in range(pred.shape[0]):
					pred_cond = np.delete(pred.cpu().numpy(), i, 0)
					y_cond = np.delete(y.cpu().numpy(), i, 0)
					correlation = []
					for c in range(pred_cond.shape[1]):
						correlation.append(corr(pred_cond[:,c], y_cond[:,c])[0])
					corr_mean.append(np.mean(correlation))
		corr_results.append(corr_mean)


# =============================================================================
# Training the model
# =============================================================================
	model_epochs_weights = []
	corr_results = []
	for e in range(args.epochs):
		model.to(device)
		train_loop(train_dl, model, loss_fn, optimizer)
		test_loop(test_dl, model, corr_results)
		# Saving the model's weights at each training epoch
		model_epochs_weights.append(model.cpu())
	del model


# =============================================================================
# Synthesizing the EEG data
# =============================================================================
	# Finding the best epoch for each test image condition
	corr_results = np.asarray(corr_results)
	best_epoch_idx = np.argsort(corr_results, 0)[::-1][0]

	# Synthesizing the EEG data response to each test image codition using the
	# epoch weights which best synthesized all other test image conditions, as
	# measured through the correlation score
	with torch.no_grad():
		for X, y in test_dl:
			for i in range(len(best_epoch_idx)):
				X_cond = X[i:i+1]
				X_cond = X_cond.to(device)
				model = model_epochs_weights[best_epoch_idx[i]]
				model.to(device)
				model.eval()
				synthetic_data[i,:,t] = model(X_cond).squeeze().cpu().numpy()
				del model
	del model_epochs_weights


# =============================================================================
# Saving the synthesized EEG data
# =============================================================================
data_dict_test = {
	'synthetic_data' : synthetic_data,
	'ch_names' : ch_names,
	'times': times
}

save_dir = os.path.join(args.project_dir, 'results', 'sub-'+
	format(args.sub,'02'), 'synthetic_eeg_data', 'end_to_end_encoding',
	'dnn-'+args.dnn)
file_name_test = 'synthetic_eeg_test'

if not os.path.exists(save_dir):
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name_test), data_dict_test)
