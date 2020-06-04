from baseline_utils import *
from src.utils.transform_input import transform_input, get_subranges
import random
import torch.optim as optim
from tqdm import tqdm, trange


class DeepMaxEntIRL:

	def __init__(self, s_g_exp_trajs, goal_poses, known_feat_list, NN_dict, gen, T=20., timestep=0.5,
				 obj_center_dict = {'HUMAN_CENTER': [-0.6, -0.55, 0.0], 'LAPTOP_CENTER': [-0.8, 0., 0.]},
				 feat_range_dict={'table': 0.98, 'coffee': 1.0, 'laptop': 0.3, 'human': 0.3, 'efficiency': 0.22,'proxemics': 0.3, 'betweenobjects': 0.2}):

		# Create an Env with only ReLuNet as learned FT, weight 1
		env, planner = init_env(feat_list=[], weights=[], object_centers=obj_center_dict, FEAT_RANGE=feat_range_dict)
		self.planner = planner
		# planner settings
		self.T = T
		self.timestep = timestep

		# Make Env IRL ready
		env.learned_features.append(self)
		env.feature_list.append('learned_feature')
		env.num_features += 1
		env.weights = np.array([1.])
		env.feature_func_list.append(self.function)

		self.env = env
		self.gen = gen

		# care about known features
		self.known_feat_list = known_feat_list
		self.known_feat_transformer = TorchFeatureTransform(obj_center_dict, known_feat_list, feat_range_dict)

		# get some derivative data from the s_g_exp_trajs
		self.s_g_exp_trajs = []
		self.starts = []
		self.goals = []
		self.goal_poses = goal_poses
		# full data for normalization
		self.full_exp_data = np.empty((0, 97), float)

		for s_g_trajs in s_g_exp_trajs:
			self.starts.append(s_g_trajs[0][0, :7])
			self.goals.append(s_g_trajs[0][-1, :7])
			full_dim_trajs = map_to_raw_dim(self.env, s_g_trajs)
			self.s_g_exp_trajs.append(full_dim_trajs)
			for traj in full_dim_trajs:
				self.full_exp_data = np.vstack((self.full_exp_data, traj))
		self.NN_dict = NN_dict
		self.max_label = 1.
		self.min_label = 0.

		# get the input dim
		self.raw_input_dim = transform_input(torch.ones(97), NN_dict).shape[1]
		if NN_dict['masked']:
			self.cost_nn = masked_Net(NN_dict['n_layers'], NN_dict['n_units'], self.raw_input_dim, subspace_ranges=get_subranges(NN_dict),
									  input_residuals=len(known_feat_list))
		else:
			self.cost_nn = ReLuNet(NN_dict['n_layers'], NN_dict['n_units'], self.raw_input_dim, input_residuals=len(known_feat_list))

	def function(self, x, torchify=False, unnorm=False):
		# used for Trajopt that optimizes it
		x_raw = self.torchify(x)

		# transform 97D input
		x_trans = transform_input(x_raw, self.NN_dict)

		# add known feature values
		if len(self.known_feat_list) > 0:
			known_features = self.known_feat_transformer.featurize(x_raw)
			# add inputs together
			x = torch.cat((x_trans, known_features), dim=1)
		else:
			x = x_trans

		y = self.cost_nn(x)
		if unnorm:
			if torchify:
				return y
			else:
				return y.detach()
		else:
			y = (y - self.min_label) / (self.max_label - self.min_label)
			if torchify:
				return y
			else:
				return y.detach()

	def torchify(self, x):
		x = torch.Tensor(x)
		if len(x.shape) == 1:
			x = x.unsqueeze(0)
		return x

	def update_normalizer(self):
		# log max of expert demonstration data
		# Note: if there are few expert demo, it might lead to too low max_label (feature values get high)
		all_logits = self.function(self.full_exp_data, unnorm=True).view(-1).detach()
		self.max_label = np.amax(all_logits.numpy())
		self.min_label = np.amin(all_logits.numpy())

	def get_trajs_with_cur_reward(self, n_traj, std, start, goal, pose):
		# note the hack that they did in AIRL:
		# mix in rew_trajs of previous iterations because otherwise reward
		# can overfit to the most recent policy and forget from earlier

		if self.gen == 'waypt':
			cur_rew_traj = generate_Gaus_MaxEnt_trajs(self.planner, std,
													  n_traj, start, goal, pose, self.T, self.timestep)
		elif self.gen == 'cost':
			cur_rew_traj = generate_cost_perturb_trajs(self.planner, self.env, std,
													   n_traj, start, goal, pose, self.T, self.timestep)
		else:
			print("gen has to be either waypt or cost")
			assert False
		return map_to_raw_dim(self.env, cur_rew_traj)

	def get_total_cost(self, waypt_array):
		# Input: right now only as a torch.Tensor
		# note if a list of trajectories, maybe flatten?
		waypt_rewards = self.function(waypt_array, torchify=True, unnorm=True)
		return torch.mean(waypt_rewards, 0)

	def deep_max_ent_irl(self, n_iters, n_cur_rew_traj, lr=1e-3, weight_decay=0.01, n_traj_per_batch=1, std=0.01):
		# TODO: how to minibatch? Maybe two data loaders
		loss_log = []
		optimizer = optim.Adam(self.cost_nn.parameters(), lr=lr, weight_decay=weight_decay)
		with trange(n_iters) as T:
			for it in T:
				T.set_description('Iteration %i' % it)
				# Step 0: generate traj under current reward
				s_g_specific_trajs = []
				if self.goal_poses is None:
					g_poses = [None for _ in range(len(self.starts))]
				else:
					g_poses = self.goal_poses

				for start, goal, goal_pose in zip(self.starts, self.goals, g_poses):
					s_g_specific_trajs.append(self.get_trajs_with_cur_reward(n_cur_rew_traj, std, start, goal, goal_pose))

				s_g_indices = np.arange(len(self.starts)).tolist()
				random.shuffle(s_g_indices)

				loss_tracker = []
				# iterate over start_goal configurations
				for j in s_g_indices:
					# self.update_normalizer() # if in the cost calculation unnorm=True, we don't need this)
					indices = np.arange(min(len(s_g_specific_trajs[j]), len(self.s_g_exp_trajs[j]))).tolist()
					random.shuffle(indices)

					# Subloop for the batches within one S-G Configuration
					for i in indices:
						exp_traj = self.s_g_exp_trajs[j][i % len(self.s_g_exp_trajs[j])]
						cur_rew_traj = s_g_specific_trajs[j][i % n_cur_rew_traj]

						# Step 1: calculate the cost for expert & current optimal
						exp_rew = self.get_total_cost(exp_traj)
						cur_opt_rew = self.get_total_cost(cur_rew_traj)

						# Step 2: calculate loss & backpropagate
						loss = (exp_rew - cur_opt_rew)
						optimizer.zero_grad()
						loss.backward()
						optimizer.step()
						loss_tracker.append(loss.item())

				loss_log.append(sum(loss_tracker) / len(loss_tracker))

				T.set_postfix(avg_loss=loss_log[-1])
		self.update_normalizer()
