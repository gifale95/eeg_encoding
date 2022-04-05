def load_images(args):
	"""Loading and preprocessing the training and test images.

	Parameters
	----------
	args : Namespace
		Input arguments.

	Returns
	-------
	X_train : list of tensor
		Training images.
	X_test : list of tensor
		Test images.

	"""

	import os
	from torchvision import transforms
	from tqdm import tqdm
	from PIL import Image

	### Defining the image preprocesing ###
	preprocess = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	### Loading and preprocessing the training images ###
	img_dirs = os.path.join(args.project_dir, 'image_set',
		'training_images')
	image_list = []
	for root, dirs, files in os.walk(img_dirs):
		for file in files:
			if file.endswith(".jpg"):
				image_list.append(os.path.join(root,file))
	image_list.sort()
	X_train = []
	for image in tqdm(image_list):
		img = Image.open(image).convert('RGB')
		img = preprocess(img)
		X_train.append(img)

	### Loading and preprocessing the test images ###
	img_dirs = os.path.join(args.project_dir, 'image_set',
		'test_images')
	image_list = []
	for root, dirs, files in os.walk(img_dirs):
		for file in files:
			if file.endswith(".jpg"):
				image_list.append(os.path.join(root,file))
	image_list.sort()
	X_test = []
	for image in tqdm(image_list):
		img = Image.open(image).convert('RGB')
		img = preprocess(img)
		X_test.append(img)

	### Output ###
	return X_train, X_test


def load_eeg_data(args, eeg_time):
	"""Loading the EEG training and test data, and selecting only the desired
	channel and time point.

	Parameters
	----------
	args : Namespace
		Input arguments.
	eeg_time : int
		EEG time point used.

	Returns
	-------
	y_train : float
		Training EEG data.
	y_test : float
		Test EEG data.
	ch_names : list of str
		EEG channel names.
	times : float
		EEG time points.

	"""

	import os
	import numpy as np

	### Loading the EEG training data ###
	data_dir = os.path.join('eeg_dataset', 'preprocessed_data', 'sub-'+
		format(args.sub,'02'))
	training_file = 'preprocessed_eeg_training.npy'
	data = np.load(os.path.join(args.project_dir, data_dir, training_file),
		allow_pickle=True).item()
	y_train = data['preprocessed_eeg_data']
	ch_names = data['ch_names']
	times = data['times']
	# Selecting the EEG time point
	y_train = y_train[:,:,:,eeg_time]
	# Averaging across repetitions
	y_train = np.mean(y_train, 1)
	# Converting to float32 (for DNN training with Pytorch)
	y_train = np.float32(y_train)

	### Loading the EEG test data ###
	test_file = 'preprocessed_eeg_test.npy'
	data = np.load(os.path.join(args.project_dir, data_dir, test_file),
		allow_pickle=True).item()
	y_test = data['preprocessed_eeg_data']
	# Selecting the EEG time point
	y_test = y_test[:,:,:,eeg_time]
	# Averaging across repetitions
	y_test = np.mean(y_test, 1)
	# Converting to float32 (for DNN training with Pytorch)
	y_test = np.float32(y_test)

	### Output ###
	return y_train, y_test, ch_names, times


def create_dataloader(args, X_train, y_train, X_test, y_test):
	"""Putting the training and test data into a PyTorch-compatible Dataloader
	format.

	Parameters
	----------
	args : Namespace
		Input arguments.
	X_train : list of tensor
		Training images.
	y_train : float
		Training EEG data.
	X_test : list of tensor
		Test images.
	y_test : float
		Test EEG data.

	Returns
	----------
	train_dl : Dataloader
		Training Dataloader.
	test_dl : Dataloader
		Test Dataloader.

	"""

	from torch.utils.data import Dataset
	from torch.utils.data import DataLoader

	### Dataset class ###
	class EegDataset(Dataset):
		def __init__(self, X_train, y_train, transform=None,
				target_transform=None):
			self.X_train = X_train
			self.y_train = y_train
			self.transform = transform
			self.target_transform = target_transform

		def __len__(self):
			return len(self.y_train)

		def __getitem__(self, idx):
			image = self.X_train[idx]
			target = self.y_train[idx]
			if self.transform:
				image = self.transform(image)
			if self.target_transform:
				target = self.target_transform(target)
			return image, target

	### Converting the data to PyTorch's Dataset format ###
	train_ds = EegDataset(X_train, y_train)
	test_ds = EegDataset(X_test, y_test, target_transform=None)

	### Converting the Datasets to PyTorch's Dataloader format ###
	train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
	test_dl = DataLoader(test_ds, batch_size=test_ds.__len__(), shuffle=False)

	### Output ###
	return train_dl, test_dl


def get_model(args):
	"""Loading the DNN model, and changing the last layer to 17 features, one
	for each EEG channel.

	Parameters
	----------
	args : Namespace
		Input arguments.

	Returns
	----------
	model : PytorchModel
		DNN model.

	"""

	import torchvision
	import torch.nn as nn

	### Alexnet ###
	if args.dnn == 'alexnet':
		model = torchvision.models.alexnet(pretrained=args.pretrained)
		model.classifier[6] = nn.Linear(in_features=4096, out_features=17)

	### ResNet-50 ###
	if args.dnn == 'resnet50':
		model = torchvision.models.resnet50(pretrained=args.pretrained)
		model.fc = nn.Linear(in_features=2048, out_features=17)

	### Output ###
	return model
