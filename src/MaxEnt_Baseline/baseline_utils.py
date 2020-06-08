import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# NN stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.networks import masked_DNN

# stuff for TarjOpt
from src.planners.trajopt_planner import TrajoptPlanner
from src.utils.traces_plot_utils import *
from src.utils.environment import Environment


def map_to_raw_dim(env, expert_demos):
	out_demos = []
	for j, traj in enumerate(expert_demos):
		raw_feature_traj = []
		temp_traj = traj.copy()
		for i in range(temp_traj.shape[0]):
			out = env.raw_features(temp_traj[i, :])
			raw_feature_traj.append(out)
		out_demos.append(np.array(raw_feature_traj))

	return out_demos


def generate_cost_perturb_trajs(planner, env, scale, n_traj, start, goal, goal_pose, T, timestep):
	# Idea: let's perturb the weights of the feature slightly to get different trajectories
	gt_weights = env.weights
	# Step 1: generate nominal optimal plan
	opt_traj = planner.replan(start, goal, goal_pose, T, timestep, seed=None)

	# Step 2: generate n_demos-1 soft-optimal trajectories
	expert_demos = [opt_traj.waypts]

	for _ in range(n_traj - 1):
		# perturb weights in env
		env.weights = gt_weights + np.random.normal(loc=0, scale=scale, size=env.weights.shape[0])
		# perturb start & goal slightly
		cur_start = start + np.random.normal(loc=0, scale=scale, size=7)
		cur_goal = goal + np.random.normal(loc=0, scale=scale, size=7)
		# plan with perturbed weights
		traj = planner.replan(cur_start, cur_goal, goal_pose, T, timestep, seed=None).waypts
		# append
		expert_demos.append(np.array(traj))

	# reset env weights to gt
	env.weights = gt_weights
	return expert_demos


def generate_Gaus_MaxEnt_trajs(planner, scale, n_traj, start, goal, goal_pose, T, timestep):
	# Idea: let's perturb the 7D angle waypoints with a normal to get soft-optimal trajectories
	# Step 1: generate nominal optimal plan
	opt_traj = planner.replan(start, goal, goal_pose, T, timestep, seed=None)

	# Step 2: generate n_demos-1 soft-optimal trajectories
	expert_demos = [opt_traj.waypts]

	for _ in range(n_traj - 1):
		cur_traj = []
		for i in range(opt_traj.waypts.shape[0]):
			cur_traj.append(opt_traj.waypts[i, :] + np.random.normal(loc=0, scale=scale, size=7))
		expert_demos.append(np.array(cur_traj))

	return expert_demos


def init_env(feat_list, weights, env_only=False,
			 object_centers={'HUMAN_CENTER': [-0.6, -0.55, 0.0], 'LAPTOP_CENTER': [-0.8, 0.0, 0.0]},
			FEAT_RANGE = {'table': 0.98, 'coffee': 1.0, 'laptop': 0.3, 'human': 0.3, 'efficiency': 0.22, 'proxemics': 0.3, 'betweenobjects': 0.2}
			 ):
	model_filename = "jaco_dynamics"
	feat_range = [FEAT_RANGE[feat_list[feat]] for feat in range(len(feat_list))]

	# Planner
	max_iter = 50
	num_waypts = 5

	environment = Environment(model_filename, object_centers, feat_list, feat_range, np.array(weights), viewer=False)
	if env_only:
		return environment
	else:
		planner = TrajoptPlanner(max_iter, num_waypts, environment)
		return environment, planner


class ReLuNet(nn.Module):
	def __init__(self, nb_layers, nb_units, raw_input_dim, input_residuals=0):
		super(ReLuNet, self).__init__()

		self.nb_layers = nb_layers
		self.input_residuals = input_residuals

		layers = []
		dim_list = [raw_input_dim] + [nb_units] * nb_layers + [1]

		for i in range(len(dim_list) - 1):
			layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))

		self.fc = nn.ModuleList(layers)

		# initialize weights
		def weights_init(m):
			if isinstance(m, nn.Linear):
				torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
				torch.nn.init.zeros_(m.bias)
		self.apply(weights_init)

		# last layer combining from masked_DNN & known features
		self.weighting = nn.Linear(1 + input_residuals, 1, bias=True)

	def forward(self, x):
		if self.input_residuals > 0:
			x_residuals = x[:, -self.input_residuals:]
			# calculate normal path
			x = F.leaky_relu(self.fc[0](x[:, :-self.input_residuals]))
			for layer in self.fc[1:]:
				x = F.leaky_relu(layer(x))
			# combine normal path & residuals
			x = F.softplus(self.weighting(torch.cat((x, x_residuals), dim=1)))
		else:
			for layer in self.fc[:-1]:
				x = F.leaky_relu(layer(x))
			x = F.softplus(self.fc[-1](x))
		return x


class masked_Net(nn.Module):
	def __init__(self, n_h_layers, nb_units, raw_input_dim, subspace_ranges, input_residuals=0):
		super(masked_Net, self).__init__()

		self.masked_DNN = masked_DNN(n_h_layers, nb_units, raw_input_dim, subspace_ranges)
		self.input_residuals = input_residuals

		# last layer combining from masked_DNN & known features
		self.weighting = nn.Linear(1 + input_residuals, 1, bias=True)

	def forward(self, x):
		if self.input_residuals > 0:
			x_residuals = x[:, -self.input_residuals:]
			x = self.masked_DNN(x[:, :-self.input_residuals])
			x = self.weighting(torch.cat((x, x_residuals), dim=1))
		else:
			x = self.masked_DNN(x)
		return x


def calculate_MEIRL_trajs_distance(IRL):
    #  extract from all demos distances
    MSE_distances =  []
    if IRL.goal_poses is None:
        g_poses = [None for _ in range(len(IRL.starts))]
    else:
        g_poses = IRL.goal_poses

    for exp_traj, start, goal, goal_pose in zip(IRL.s_g_exp_trajs, IRL.starts, IRL.goals, g_poses):
        # get induced traj
        ind_traj = IRL.get_trajs_with_cur_reward(1, 0.01, start, goal, goal_pose)[0]
        ind_euclidean = ind_traj[:, 88:91]
        exp_euclidean = exp_traj[0][:, 88:91]
        MSE = np.linalg.norm(ind_euclidean-exp_euclidean, axis=1)
        MSE_distances.append(MSE.mean())

	return sum(MSE_distances)/len(MSE_distances)

def plot_IRL_comparison(IRL):
	# get laptop position
	laptop = IRL.env.object_centers['LAPTOP_CENTER']
	# Experts
	to_plot = np.empty((0, 4))
	for s_g_trajs in IRL.s_g_exp_trajs:
		traj = s_g_trajs[0]
		labels = IRL.function(traj)
		euclidean = traj[:, 88:91]
		to_plot = np.vstack((to_plot, np.hstack((euclidean, labels))))
	df = pd.DataFrame(to_plot)
	fig = go.Figure(data=go.Scatter3d(x=df.iloc[:, 0], y=df.iloc[:, 1], z=df.iloc[:, 2], mode='markers',
									  marker=dict(color=df.iloc[:, 3], showscale=True), showlegend=False))

	fig.add_scatter3d(x=[laptop[0]], y=[laptop[1]], z=[laptop[2]], mode='markers',
					  marker=dict(size=10, color='black'), showlegend=False)
	fig.update_layout(title='Expert Trajectory with learned reward')
	fig.show()

	to_plot = np.empty((0, 4))
	if IRL.goal_poses is None:
		g_poses = [None for _ in range(len(IRL.starts))]
	else:
		g_poses = IRL.goal_poses

	for start, goal, goal_pose in zip(IRL.starts, IRL.goals, g_poses):
		traj = IRL.get_trajs_with_cur_reward(1, 0.01, start, goal, goal_pose)[0]
		labels = IRL.function(traj)
		euclidean = traj[:, 88:91]
		to_plot = np.vstack((to_plot, np.hstack((euclidean, labels))))
	df = pd.DataFrame(to_plot)
	fig = go.Figure(data=go.Scatter3d(x=df.iloc[:, 0], y=df.iloc[:, 1], z=df.iloc[:, 2], mode='markers',
									  marker=dict(color=df.iloc[:, 3], showscale=True), showlegend=False))

	fig.add_scatter3d(x=[laptop[0]], y=[laptop[1]], z=[laptop[2]], mode='markers',
					  marker=dict(size=10, color='black'), showlegend=False)
	fig.update_layout(title='Current Trajectory with learned reward')
	fig.show()


def plot_trajs(demos, object_centers, title='some_title', func=None):

	# get laptop position
	laptop = object_centers['LAPTOP_CENTER']
	human = object_centers['HUMAN_CENTER']
	# Experts
	points = np.empty((0, 4))
	for traj in demos:
		if func is not None:
			labels = func(traj)
		else:
			labels = traj[:, 90].reshape((-1, 1))
		euclidean = traj[:, 88:91]
		points = np.vstack((points, np.hstack((euclidean, labels))))
	df = pd.DataFrame(points)
	fig = go.Figure(data=go.Scatter3d(x=df.iloc[:, 0], y=df.iloc[:, 1], z=df.iloc[:, 2], mode='markers',
									  marker=dict(color=df.iloc[:, 3], showscale=True), showlegend=False))
	fig.data[0]['text'] = ['color: ' + str(round(i,2)) for i in fig.data[0]['marker']['color']]

	fig.add_scatter3d(x=[laptop[0]], y=[laptop[1]], z=[laptop[2]], mode='markers',
					  marker=dict(size=10, color='black'), showlegend=False, hovertext=['Laptop'])

	fig.add_scatter3d(x=[human[0]], y=[human[1]], z=[human[2]], mode='markers',
					  marker=dict(size=10, color='red'), showlegend=False, hovertext=['Human'])
	fig.update_layout(title=title)
	fig.show()


class TorchFeatureTransform(object):
	def __init__(self, object_centers, feat_list, feat_range_dict):
		self.object_centers = object_centers
		self.feature_list = feat_list
		self.num_features = len(self.feature_list)
		feat_range = [feat_range_dict[feat_list[feat]] for feat in range(len(feat_list))]
		self.feat_range = feat_range

		self.feature_func_list = []

		for feat in self.feature_list:
			if feat == 'table':
				self.feature_func_list.append(self.table_features)
			elif feat == 'coffee':
				self.feature_func_list.append(self.coffee_features)
			elif feat == 'human':
				self.feature_func_list.append(self.human_features)
			elif feat == 'laptop':
				self.feature_func_list.append(self.laptop_features)

	def featurize(self, high_dim_waypt):
		"""
		Input: 97D raw torch Tensor
		Output: #known_features torch Tensor
		"""
		features = torch.empty((high_dim_waypt.shape[0], 0), requires_grad=True)
		for feat_func in self.feature_func_list:
			features = torch.cat((features, feat_func(high_dim_waypt).unsqueeze(1)), dim=1)

		return features

	def table_features(self, high_dim_waypt):
		"""
		Input: 97D raw torch Tensor
		Output: torch scalar
		"""
		return high_dim_waypt[:, 90] / self.feat_range[self.feature_list.index("table")]

	def coffee_features(self, high_dim_waypt):
		"""
		Computes the coffee orientation feature value for waypoint
		by checking if the EE is oriented vertically.
		---
		input waypoint, output scalar feature
		"""
		featval = 1 - high_dim_waypt[:, 67]

		return featval / self.feat_range[self.feature_list.index("coffee")]

	def laptop_features(self, high_dim_waypt):
		"""
		Computes distance from end-effector to laptop in xy coords
		input trajectory, output scalar distance where
			0: EE is at more than 0.4 meters away from laptop
			+: EE is closer than 0.4 meters to laptop
		"""
		EE_coord_xy = high_dim_waypt[:, 88:90]
		laptop_xy = torch.Tensor(self.object_centers['LAPTOP_CENTER'][0:2])
		dist = torch.norm(EE_coord_xy - laptop_xy, dim=1) - 0.3

		return -((dist < 0) * dist) / self.feat_range[self.feature_list.index('laptop')]

	def human_features(self, high_dim_waypt):
		"""
		Computes distance from end-effector to laptop in xy coords
		input trajectory, output scalar distance where
			0: EE is at more than 0.4 meters away from laptop
			+: EE is closer than 0.4 meters to laptop
		"""
		EE_coord_xy = high_dim_waypt[:, 88:90]
		human_xy = torch.Tensor(self.object_centers['HUMAN_CENTER'][0:2])
		dist = torch.norm(EE_coord_xy - human_xy, dim=1) - 0.3
		return -((dist < 0) * dist) / self.feat_range[self.feature_list.index('human')]


def get_coords_gt_cost(gen, expert_env, parent_dir, n_waypoints=1000):
	# Step 1: Generate ground truth data, sampling uniformly from 7D angle space
	if gen == True:
		waypts = np.random.uniform(size=(n_waypoints, 7), low=0, high=np.pi*2)
		# Transform to 97D
		raw_waypts = []
		for waypt in waypts:
			raw_waypts.append(expert_env.raw_features(waypt))
		raw_waypts = np.array(raw_waypts)

	else:
		# load coordinates above the table
		data_file = parent_dir + '/data/gtdata/data_table.npz'
		npzfile = np.load(data_file)
		raw_waypts = npzfile['x']

	# generate gt_labels
	feat_idx = list(np.arange(expert_env.num_features))
	features = [[0.0 for _ in range(len(raw_waypts))] for _ in range(0, len(expert_env.feature_list))]
	for index in range(len(raw_waypts)):
		for feat in range(len(feat_idx)):
			features[feat][index] = expert_env.featurize_single(raw_waypts[index,:7], feat_idx[feat])

	features = np.array(features).T
	gt_cost = np.matmul(features, np.array(expert_env.weights).reshape(-1,1))

	return raw_waypts, gt_cost