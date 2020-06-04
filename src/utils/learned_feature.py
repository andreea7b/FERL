import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm, trange
import itertools
import random
import pandas as pd
from transform_input import transform_input, get_subranges
from networks import DNN, DNNconvex, masked_DNN
from torch.utils.data import Dataset, DataLoader


class LearnedFeature(object):
	"""
	Params:
	nb_layers	number of hidden layers
	nb_units	number of hidden units per layer

	LF_dict		dict containing all settings for the learned feature, example see below

	LF_dict = {'bet_data':5, 'sin':False, 'cos':False, 'rpy':False, 'lowdim':False, 'norot':False,
           'noangles':True, '6D_laptop':False, '6D_human':False, '9D_coffee':False, 'EErot':True,
           'noxyz':False}

	bet_data [int]: Number of copies of each Start-Goal pair to be added to the data set.
	"""
	def __init__(self, nb_layers, nb_units, LF_dict):
		# ---- Initialize stored training data ---- #
		# [trajectory] then np.array inside with [n_timesteps] x [input dim]

		self.trajectory_list = []
		self.full_data_array = np.empty((0, 5), float)
		self.start_labels = []
		self.end_labels = []
		self.subspaces_list = get_subranges(LF_dict)
		self.max_labels = [1 for _ in range(len(self.subspaces_list))]
		self.min_labels = [0 for _ in range(len(self.subspaces_list))]
		self.LF_dict = LF_dict
		self.models = []

		if len(self.subspaces_list) == 1:
			self.final_model = 0
		else:
			self.final_model = None

		# ---- Initialize Function approximators for each subspace ---- #
		for sub_range in self.subspaces_list:
			self.models.append(DNN(nb_layers, nb_units, sub_range[1] - sub_range[0]))

	def function(self, x, model=None, torchify=False, norm=False):

		if model is None: # then called from TrajOpt so return normalized final model
			model = self.final_model
			norm = True

		# Torchify the input
		x = self.input_torchify(x)

		# Transform the input
		x = transform_input(x, self.LF_dict)

		# transform to the model specific subspace input
		range = self.subspaces_list[model]
		x = x[:, range[0]:range[1]]
		y = self.models[model](x)

		if norm:
			y = (y - self.min_labels[model]) / (self.max_labels[model] - self.min_labels[model])
			if torchify:
				return y
			else:
				return np.array(y.detach())
		else:
			return y

	def input_torchify(self, x):
		if not torch.is_tensor(x):
			x = torch.Tensor(x)
		if len(x.shape) == 1:
			x = torch.unsqueeze(x, axis=0)
		return x

	def add_data(self, trajectory, start_label=0.0, end_label=1.0):
		# trajectory is Tx7 np.array
		self.trajectory_list.append(trajectory)
		self.start_labels.append(start_label)
		self.end_labels.append(end_label)

	def get_train_test_arrays(self, train_idx, test_idx):
		"""Input: the idx of the trajectories for train & test"""
		full_data_array = np.empty((0, 5), float)
		ordered_list = train_idx + test_idx
		test_set_idx = None

		for idx in ordered_list:
			# check if already test set
			if idx == test_idx[0]:
				# log where that is so we can split the full array later
				test_set_idx = full_data_array.shape[0]
			data_tuples_to_append = []
			for combi in list(itertools.combinations(range(self.trajectory_list[idx].shape[0]), 2)):
				# Sample two points on that trajectory trace.
				idx_s0, idx_s1 = combi

				# Create label differentials if necessary.
				s0_delta = 0
				s1_delta = 0
				if idx_s0 == 0:
					s0_delta = -self.start_labels[idx]
				if idx_s1 == self.trajectory_list[idx].shape[0] - 1:
					s1_delta = 1. - self.end_labels[idx]

				data_tuples_to_append.append((
											 self.trajectory_list[idx][idx_s0, :], self.trajectory_list[idx][idx_s1, :],
											 idx_s0 < idx_s1, s0_delta, s1_delta))
			full_data_array = np.vstack((full_data_array, np.array(data_tuples_to_append)))

			# Add between traces tuples
			if ordered_list.index(idx) > 0:
				tuples_to_append = []
				for other_traj_idx in ordered_list[:ordered_list.index(idx)]:
					S_tuple = [(self.trajectory_list[other_traj_idx][0, :], self.trajectory_list[idx][0, :], 0.5,
								-self.start_labels[other_traj_idx], -self.start_labels[idx])] * self.LF_dict['bet_data']
					G_tuple = [(self.trajectory_list[other_traj_idx][-1, :], self.trajectory_list[idx][-1, :], 0.5,
								1 - self.end_labels[other_traj_idx], 1 - self.end_labels[idx])] * self.LF_dict['bet_data']
					tuples_to_append = tuples_to_append + S_tuple + G_tuple
				full_data_array = np.vstack((full_data_array, np.array(tuples_to_append)))

		# split in train & test tuples
		self.full_data_array = full_data_array
		train_data_array = full_data_array[:test_set_idx]
		test_data_array = full_data_array[test_set_idx:]

		return test_data_array, train_data_array

	def select_subspace(self, epochs, batch_size, learning_rate, weight_decay, s_g_weight):
		n_test = int(math.floor(len(self.trajectory_list) * 0.5))
		print("Select subspace training, testing on " + str(n_test) + " unseen Trajectory")
		# split trajectory list
		test_idx = np.random.choice(np.arange(len(self.trajectory_list)), size=n_test, replace=False)
		train_idx = np.setdiff1d(np.arange(len(self.trajectory_list)), test_idx)

		test_data_array, train_data_array = self.get_train_test_arrays(train_idx.tolist(), test_idx.tolist())

		train_dataset = FeatureLearningDataset(train_data_array)
		print("len train: ", train_data_array.shape[0])
		test_dataset = FeatureLearningDataset(test_data_array)
		print("len test: ", test_data_array.shape[0])
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

		optimizers = []
		for i in range(len(self.models)):
			# initialize optimizers
			optimizers.append(optim.Adam(self.models[i].parameters(), lr=learning_rate, weight_decay=weight_decay))

		# unnecessary if only one subspace
		if len(self.subspaces_list) ==1:
			print("Only one subspace.")
			return optimizers[0]

		in_train_losses = [[] for _ in range(len(self.subspaces_list))]
		in_test_losses = [[] for _ in range(len(self.subspaces_list))]

		with trange(epochs) as T:
			for t in T:
				# Description will be displayed on the left
				T.set_description('epoch %i' % i)

				# Needed if non-standard labeling is used
				# for idx in range(self.LF_dict['n_ensamble']):
				# 	self.update_normalizer(idx)

				for i, model in enumerate(self.models):
					avg_in_loss = []
					for batch in train_loader:
						optimizers[i].zero_grad()
						loss = self.in_traj_loss(batch, model_idx=i, s_g_weight=s_g_weight)
						loss.backward()
						optimizers[i].step()
						avg_in_loss.append(loss.item())
					# self.update_normalizer(i) # technically correct but costs a lot of compute

					in_train_losses[i].append(sum(avg_in_loss) / len(avg_in_loss))

				# calculate test loss
				for i, model in enumerate(self.models):
					avg_in_loss = []
					for batch in test_loader:
						loss = self.in_traj_loss(batch, model_idx=i, s_g_weight=s_g_weight)
						avg_in_loss.append(loss.item())
					# log over training
					in_test_losses[i].append(sum(avg_in_loss) / len(avg_in_loss))

				T.set_postfix(in_test_loss=[loss[-1] for loss in in_test_losses])

		for idx in range(len(self.models)):
			self.update_normalizer(idx)

		# select final model
		# Method 1: Take lowest last test loss
		final_test_losses = [loss[-1] for loss in in_test_losses]
		val, last_loss_idx = min((val, idx) for (idx, val) in enumerate(final_test_losses))

		print("Model of subspace " + str(last_loss_idx) + "selected as final model")
		self.final_model = last_loss_idx

		return optimizers[last_loss_idx]

	def update_normalizer(self, model_idx):
		# log min/max on all data until now
		s_0s_array = np.array([tup[0] for tup in self.full_data_array]).squeeze()
		s_1s_array = np.array([tup[1] for tup in self.full_data_array]).squeeze()
		s0_logits = self.function(s_0s_array, model=model_idx).view(-1).detach()
		s1_logits = self.function(s_1s_array, model=model_idx).view(-1).detach()
		all_logits = np.vstack((s0_logits, s1_logits))
		self.max_labels[model_idx] = np.amax(all_logits)
		self.min_labels[model_idx] = np.amin(all_logits)
	
	def in_traj_loss(self, batch, model_idx, s_g_weight):
		# arrays of the states
		s_1s_array = batch['s1']
		s_2s_array = batch['s2']

		# arrays of the start & end labels
		delta_1s_array = batch['l1']
		delta_2s_array = batch['l2']

		# label for classifiers
		labels = batch['label']

		weights = torch.ones(labels.shape)
		weights = weights + (labels == 0.5)*torch.full(labels.shape, s_g_weight)

		s1_adds = (delta_1s_array * (self.max_labels[model_idx] - self.min_labels[model_idx])).reshape(-1,1)
		s2_adds = (delta_2s_array * (self.max_labels[model_idx] - self.min_labels[model_idx])).reshape(-1, 1)

		# calculate test_loss  (with additive thing)
		s1_logits = self.function(s_1s_array, model=model_idx) + s1_adds
		s2_logits = self.function(s_2s_array, model=model_idx) + s2_adds

		# final loss
		loss = nn.BCEWithLogitsLoss(weight=weights)((s2_logits - s1_logits).view(-1), labels)
		return loss

	def train(self, epochs=100, batch_size=32, learning_rate=1e-3, weight_decay=0.001, s_g_weight=10.):
		# Heuristic to select subspace by training for 10 epochs a NN on each of them
		final_mod_optimizer = self.select_subspace(10, batch_size, learning_rate, weight_decay, s_g_weight)
		# Train on full dataset
		print("Train subspace model " + str(self.final_model) + " on all " + str(
			len(self.trajectory_list)) + " Trajectories")
		train_dataset = FeatureLearningDataset(self.full_data_array)
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

		# check if any non-standard label is used
		if sum([l != 0 for l in self.start_labels]) > 0 or sum([l != 1 for l in self.end_labels]) > 0:
			norm_per_epoch = True
		else:
			norm_per_epoch = False

		in_train_losses = []
		with trange(epochs) as T:
			for t in T:
				# Description will be displayed on the left
				T.set_description('epoch %i' % t)

				# update normalizer labels
				if norm_per_epoch:
					self.update_normalizer(self.final_model)

				avg_in_loss = []
				for batch in train_loader:
					final_mod_optimizer.zero_grad()
					loss = self.in_traj_loss(batch, model_idx=self.final_model, s_g_weight=s_g_weight)
					loss.backward()
					final_mod_optimizer.step()
					avg_in_loss.append(loss.item())

					in_train_losses.append(sum(avg_in_loss) / len(avg_in_loss))

				T.set_postfix(train_loss=in_train_losses[-1])

		self.update_normalizer(self.final_model)

		print("Final model trained!")
		return in_train_losses


class FeatureLearningDataset(Dataset):
	"""Feature Learning dataset."""

	def __init__(self, array_of_tuples):
		self.array_of_tuples = array_of_tuples

	def __len__(self):
		return self.array_of_tuples.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample = {'s1': self.array_of_tuples[idx][0].astype(np.float32),
				  's2': self.array_of_tuples[idx][1].astype(np.float32),
				  'label': np.array(self.array_of_tuples[idx][2]).astype(np.float32),
				  'l1': np.array(self.array_of_tuples[idx][3]).astype(np.float32),
				  'l2': np.array(self.array_of_tuples[idx][4]).astype(np.float32)
				  }

		return sample


class RegressionFeature(object):
	"""
	tbd.
	"""
	def __init__(self, nb_layers, nb_units, input_dim, output_dim=1, activation="softplus"):
		# ---- Initialize Function approximator ---- #
		self.torch_function = DNN(nb_layers, nb_units, input_dim, activation)

		# ---- Initialize stored training data ---- #
		self.feature_vector = np.empty((0, input_dim), float)
		self.regression_labels = np.empty((0, output_dim), float)

	def function(self, x, torchify=False):
		x = torch.Tensor(x)
		y = self.torch_function(x)
		if torchify:
			return y
		return y.detach()

	def add_data(self, trajectory, regression_label):
		# trajectory is Tx7 np.array, lable is Tx1
		self.feature_vector = np.vstack((self.feature_vector, trajectory))
		self.regression_labels = np.vstack((self.regression_labels, regression_label))

	def train(self, p_train=0.8, epochs=100, batch_size=64, learning_rate=1e-3):
		# trains with current dataset & sends back test error
		# split into train & test
		n_samples = self.feature_vector.shape[0]
		indices = np.arange(n_samples)
		np.random.shuffle(indices)

		train_inds = indices[:int(n_samples*p_train)]
		test_inds = indices[int(n_samples*p_train):]

		train = torch.utils.data.TensorDataset(torch.Tensor(self.feature_vector[train_inds]), torch.Tensor(self.regression_labels[train_inds]))
		train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
		# initialize optimizer
		criterion = nn.MSELoss()
		optimizer = optim.SGD(self.torch_function.parameters(), lr=learning_rate, momentum=0.9)

		train_losses = []
		test_losses = []
		for epoch in range(epochs):
			# SGD loop
			for i, data in enumerate(train_loader, 0):
				inputs, labels = data
				# zero the parameter gradients
				optimizer.zero_grad()
				# forward + backward + optimize
				outputs = self.torch_function(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

			# calculate test_loss
			train_loss = criterion(self.torch_function(torch.Tensor(self.feature_vector[train_inds])),
								   torch.Tensor(self.regression_labels[train_inds]))
			test_loss = criterion(self.torch_function(torch.Tensor(self.feature_vector[test_inds])),
								  torch.Tensor(self.regression_labels[test_inds]))
			train_losses.append(train_loss.item())
			test_losses.append(test_loss.item())
			if epoch % 100 == 99:
				print('Epoch {}, Test loss:'.format(epoch), test_loss.item(), 'Train loss:', train_loss.item())
		print('Finished Training')
		return np.array(train_losses), np.array(test_losses)
