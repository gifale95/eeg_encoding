"""Train DNN models end-to-end to predict the EEG responses to arbitrary images.

Parameters
----------
sub : int
	Used subject.
modeled_time_points : str
	Whether each DNN should model 'single' or 'all' EEG time points.
dnn : str
	DNN model used.
pretrained : bool
	If True use a pretrained network, if False use a randomly initialized one.
epochs : int
	Number of training epochs.
lr : float
	Learning rate.
weight_decay : float
	Weight decay coefficient.
batch_size : int
	Batch size for weight update.
save_trained_models : bool
	If True, the weights of the trained models are saved.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import random
import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from end_to_end_encoding_utils import load_images
from end_to_end_encoding_utils import load_eeg_data
from end_to_end_encoding_utils import create_dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default=1)
parser.add_argument('--modeled_time_points', type=str, default='single')
parser.add_argument('--dnn', type=str, default='alexnet')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-7)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--save_trained_models', type=bool, default=False)
parser.add_argument('--project_dir', default='../project_directory', type=str)
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
# Load the images (X) and the EEG data (y)
# =============================================================================
# One image exemplar for each of the 1,654 training object concepts is used for
# model validation, for a total of 1,654 validation images.
# The remaining 14,886 images (1,654 training object concepts × 9 image
# exemplars per concept) are used for training the encoding models.
# Finally, the 200 test object concepts are used for testing the models.

# Load the images
X_train, X_val, X_test = load_images(args)

# Load the EEG data
y_train, y_val, y_test, ch_names, times = load_eeg_data(args)


# =============================================================================
# Models loop
# =============================================================================
# Loop across the EEG temporal dimension if the encoding models are trained
# independently on each time point.
# If the encoding models are predicting all time points at once, then the loop
# will only run for one iteration.

# Define the amount of models which will be trained and the amount of their EEG
# output features
if args.modeled_time_points == 'single':
	num_models = y_test.shape[2]
	out_features = y_test.shape[1]
elif args.modeled_time_points == 'all':
	num_models = 1
	out_features = y_test.shape[1] * y_test.shape[2]

# Snthetic EEG data matrix of shape:
# (Test image conditions × EEG channels × EEG time points)
synthetic_data = np.zeros((y_test.shape))
best_epochs = np.zeros((num_models))

# Loop across independent models
for m in range(num_models):
	print(f'\nModel: [{m+1}/{num_models}]')


# =============================================================================
# Create PyTorch-compatible Dataloaders
# =============================================================================
	train_dl, val_dl, test_dl = create_dataloader(args, m, X_train, X_val,
		X_test, y_train, y_val, y_test)


# =============================================================================
# Load the DNN model and change the last layer to the amount of EEG features
# =============================================================================
	# AlexNet
	if args.dnn == 'alexnet':
		model = torchvision.models.alexnet(pretrained=args.pretrained)
		model.classifier[6] = nn.Linear(in_features=4096,
			out_features=out_features)
	# ResNet-50
	if args.dnn == 'resnet50':
		model = torchvision.models.resnet50(pretrained=args.pretrained)
		model.fc = nn.Linear(in_features=2048, out_features=out_features)
	model.to(device)


# =============================================================================
# TensorBoard, loss function and optimizer
# =============================================================================
	# TensorBoard
	save_dir = os.path.join(args.project_dir, 'results', 'sub-'+
		format(args.sub,'02'), 'synthetic_eeg_data', 'end_to_end_encoding',
		'dnn-'+args.dnn, 'modeled_time_points-'+args.modeled_time_points,
		'pretrained-'+str(args.pretrained), 'lr-{:.0e}'.format(args.lr)+
		'__wd-{:.0e}'.format(args.weight_decay)+'__bs-'+
		format(args.batch_size,'03'))
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)
	writer = SummaryWriter(save_dir)

	# Loss function
	loss_fn = torch.nn.MSELoss().to(device)

	# Optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
		weight_decay=args.weight_decay)


# =============================================================================
# Training and evaluation loops
# =============================================================================
	def train_loop(train_dl, model, loss_fn, optimizer):
		tot_loss = 0
		model.train()
		for X, y in train_dl:
			# Prediction and loss
			X = X.to(device, non_blocking=False)
			y = y.to(device, non_blocking=False)
			y_pred = model(X).squeeze()
			loss = loss_fn(y_pred, y)
			# Backpropagation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# Total loss
			tot_loss += loss.item() * len(X)
		tot_loss /= len(train_dl.dataset)
		return tot_loss

	def eval_loop(eval_dl, model, loss_fn):
		tot_loss = 0
		model.eval()
		with torch.no_grad():
			for X, y in eval_dl:
				# Prediction and loss
				X = X.to(device, non_blocking=False)
				y = y.to(device, non_blocking=False)
				y_pred = model(X).squeeze()
				loss = loss_fn(y_pred, y)
				tot_loss += loss.item() * len(X)
			tot_loss /= len(eval_dl.dataset)
		return tot_loss


# =============================================================================
# Train the model and log the training statistics to TensorBoard
# =============================================================================
	best_val_loss = 5000

	for e in tqdm(range(args.epochs)):
		# Train the model and train loss
		train_loss = train_loop(train_dl, model, loss_fn, optimizer)
		# Validation and test losses
		val_loss = eval_loop(val_dl, model, loss_fn)
		test_loss = eval_loop(test_dl, model, loss_fn)
		# Add the training stats to TensorBoard
		writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss,
			'test': test_loss,}, e+1)
		writer.flush()
		# Retain the best epoch model according to the validation loss
		if val_loss < best_val_loss:
			best_model = deepcopy(model)
			best_epochs[m] = e + 1
			best_val_loss = val_loss

	# Delete the model from GPU memory
	del model
	if device == 'cuda':
		torch.cuda.empty_cache()


# =============================================================================
# Synthesize the EEG data using the best model
# =============================================================================
	best_model.to(device)
	best_model.eval()
	with torch.no_grad():
		for X, y in test_dl:
			X = X.to(device, non_blocking=False)
			y = y.to(device, non_blocking=False)
			if args.modeled_time_points == 'single':
				synthetic_data[:,:,m] = best_model(X).detach().cpu().numpy()
			elif args.modeled_time_points == 'all':
				synthetic_data = np.reshape(
					best_model(X).detach().cpu().numpy(),
					(synthetic_data.shape))


# =============================================================================
# Save the best model
# =============================================================================
	if args.save_trained_models == True:
		file_name = 'model_state_dict.pt'
		best_model = best_model.to('cpu')
		torch.save({
			'args': args,
			'best_model': best_model.state_dict(),
			'out_features': out_features,
			'best_epochs': best_epochs
			}, os.path.join(save_dir, file_name))

	# Delete the model from GPU memory
	del best_model
	if device == 'cuda':
		torch.cuda.empty_cache()


# =============================================================================
# Save the modeling results
# =============================================================================
modeling_results = {
	'synthetic_data': synthetic_data,
	'ch_names': ch_names,
	'times': times,
	'best_epochs': best_epochs
}
file_name = 'synthetic_eeg_test'
np.save(os.path.join(save_dir, file_name), modeling_results)
