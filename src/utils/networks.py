import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
	"""
	Creates a NN with leaky ReLu non-linearity.
	---
	input nb_layers, nb_units, input_dim
	output scalar
	"""
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
