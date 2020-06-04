import torch
import torch.nn as nn
import torch.nn.functional as F


class masked_DNN(nn.Module):
	def __init__(self, n_h_layers, nb_units, input_dim, subspace_ranges):
		super(masked_DNN, self).__init__()

		self.n_subspaces = len(subspace_ranges)
		layers = []
		masked_dim_list = [input_dim] + [nb_units] * n_h_layers + [self.n_subspaces]
		bet_hid_ranges = [[i * nb_units / (self.n_subspaces), (i + 1) * nb_units / (self.n_subspaces)] for i in
						  range(self.n_subspaces)]
		subspace_ranges = [subspace_ranges] + [bet_hid_ranges] * n_h_layers

		for i in range(len(masked_dim_list) - 1):
			layers.append(MaskedLinear(masked_dim_list[i], masked_dim_list[i + 1]))
			layers[-1].set_mask(subspace_ranges[i])

		# add last fc layer
		layers.append(nn.Linear(masked_dim_list[-1], 1, bias=True))

		self.fc = nn.ModuleList(layers)

		# initialize weights
		def weights_init(m):
			if isinstance(m, nn.Linear):
				torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
				if m.bias is not None:
					torch.nn.init.zeros_(m.bias)

		self.apply(weights_init)

	def forward(self, x):
		for layer in self.fc[:-1]:
			x = F.leaky_relu(layer(x))
		#print("before merging", x)
		x = F.softplus(self.fc[-1](x))
		return x


class MaskedLinear(nn.Linear):
	def __init__(self, *args, **kwargs):
		super(MaskedLinear, self).__init__(*args, **kwargs)

	def set_mask(self, subspace_ranges):
		self.mask = get_three_subspace_mask(self.weight.shape[1], self.weight.shape[0], subspace_ranges)
		self.weight.data = self.weight.data * self.mask

	def forward(self, x):
		self.weight.data = self.weight.data * self.mask
		return super(MaskedLinear, self).forward(x)


def get_three_subspace_mask(in_features, out_features, subspace_ranges):
	mask = torch.zeros((out_features, in_features))

	n_sub = len(subspace_ranges)

	if out_features % n_sub != 0 or subspace_ranges[-1][1] != in_features:
		print(
			"number of hidden units must be divisible by len(subspace_ranges) and ranges should cover full input space")
		assert False
	n_out = int(out_features / n_sub)

	for i, tup in enumerate(subspace_ranges):
		mask[i * n_out:(i + 1) * n_out, tup[0]:tup[1]] = 1.0
	return mask


class DNN(nn.Module):
	def __init__(self, nb_layers, nb_units, input_dim):
		super(DNN, self).__init__()
		self.nb_layers = nb_layers

		layers = []
		dim_list = [input_dim] + [nb_units] * nb_layers + [1]

		for i in range(len(dim_list) - 1):
			layers.append(nn.Linear(dim_list[i], dim_list[i+1]))

		self.fc = nn.ModuleList(layers)

		# initialize weights
		def weights_init(m):
			if isinstance(m, nn.Linear):
				torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
				torch.nn.init.zeros_(m.bias)

		self.apply(weights_init)

	def forward(self, x):
		for layer in self.fc[:-1]:
			x = F.leaky_relu(layer(x))
		x = F.softplus(self.fc[-1](x))
		return x


class DNNconvex(nn.Module):
	def __init__(self, nb_layers, nb_units, input_dim, activation="softplus"):
		super(DNNconvex, self).__init__()
		fc = []
		dc = []
		self.nb_layers = nb_layers
		if activation not in ["relu", "sigmoid", "softplus", "softmax"]:
			print("activation not implemented")
			assert False
		self.activation = activation

		for i in range(nb_layers):
			if i == 0: # first layer
				fc.append(nn.Linear(input_dim, nb_units))
			elif i == nb_layers-1: # last layer
				fc.append(nn.Linear(nb_units, 1))
				dc.append(nn.Linear(input_dim, 1, bias=False))
			else: # other layers
				fc.append(nn.Linear(nb_units, nb_units))
				dc.append(nn.Linear(input_dim, nb_units, bias=False))
		self.fc = nn.ModuleList(fc)
		self.dc = nn.ModuleList(dc)

		# initialize weights
		def weights_init(m):
			if hasattr(m, 'weight'):
				torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
				if m.bias is not None:
					torch.nn.init.zeros_(m.bias)

		self.apply(weights_init)

	def forward(self, x):
		x_0 = x
		# for all layers until last: leaky ReLu
		for i in range(self.nb_layers-1):
			if i == 0: # first layer
				x = F.leaky_relu(self.fc[i](x))
			else: # other layers
				x = F.leaky_relu(self.fc[i](x) + self.dc[i-1](x_0))
		# for last layer see activationn input
		if self.activation == "relu":
			x = F.relu(self.fc[-1](x) + self.dc[-1](x_0))
		elif self.activation == "sigmoid":
			x = F.sigmoid(self.fc[-1](x) + self.dc[-1](x_0))
		elif self.activation == "softplus":
			x = F.softplus(self.fc[-1](x) + self.dc[-1](x_0))
		elif self.activation == "softmax":
			x = F.softmax(self.fc[-1](x) + self.dc[-1](x_0))
		return x


