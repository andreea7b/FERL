import math
import matplotlib.pyplot as plt

from baseline_utils import *
import glob
import pickle


class GT_Reward_Expert:

	def __init__(self, feat_list, gt_weights, gen, starts, goals, goal_poses=None, combi=False, T=20., timestep=0.5,
				 obj_center_dict = {'HUMAN_CENTER': [-0.6, -0.55, 0.0], 'LAPTOP_CENTER': [-0.8, 0.0, 0.0]},
				 feat_range_dict = {'table': 0.98, 'coffee': 1.0, 'laptop': 0.3, 'human': 0.3, 'efficiency': 0.22, 'proxemics': 0.3, 'betweenobjects': 0.2}):

		# needs access to the environment & trajOpt...
		env, planner = init_env(feat_list, gt_weights, object_centers=obj_center_dict, FEAT_RANGE=feat_range_dict)
		self.env = env
		self.planner = planner
		self.s_g_exp_trajs = []
		self.gen = gen

		if goal_poses is not None and len(goals) != len(goal_poses):
			print("Goal pose needs to be either None or same length as len(goals)")
			assert False

		if combi:
			combis = [(x, y) for x in range(len(starts)) for y in range(len(goals))]
			self.starts = [starts[tup[0]] for tup in combis]
			self.goals = [goals[tup[1]] for tup in combis]
			if goal_poses is not None:
				self.goal_poses = [goal_poses[tup[1]] for tup in combis]
		else:
			self.starts = starts[:min(len(starts), len(goals))]
			self.goals = goals[:min(len(starts), len(goals))]
			if goal_poses is not None:
				self.goal_poses = goal_poses

		if goal_poses is None:
			self.goal_poses = [None for _ in range(len(self.starts))]

		self.T = T
		self.timestep = timestep

	def generate_expert_demos(self, n_per_s_g, scale=0.01):
		# TODO: Try different soft-optimality methods

		for start, goal, goal_pose in zip(self.starts, self.goals, self.goal_poses):
			if self.gen == 'waypt':
				expert_demos = generate_Gaus_MaxEnt_trajs(self.planner, self.env, scale,
														  n_per_s_g, start, goal, goal_pose, self.T, self.timestep)
			elif self.gen == 'cost':
				expert_demos = generate_cost_perturb_trajs(self.planner, self.env, scale,
														   n_per_s_g, start, goal, goal_pose, self.T, self.timestep)
			# add for that s_g configuration
			self.s_g_exp_trajs.append(expert_demos)

	def plot_expert_rew_dist(self):
		# TODO: efficiency feature is missing
		def calc_gt_reward(env, waypts):
			per_waypt = np.matmul(np.array(env.featurize(waypts[:, :7])).T, np.array(env.weights))
			return per_waypt.sum(), per_waypt

		demo_rews = []
		for s_g_demos in self.s_g_exp_trajs:
			for demo in s_g_demos:
				traj_rew, _ = calc_gt_reward(self.env, demo)
				demo_rews.append(traj_rew)

		#         print("GT reward is: ", demo_rews[0])
		n_bins = 60
		plt.hist(demo_rews, n_bins)
		plt.show()

	def cut_off_below_gt(self):
		# TODO: efficiency feature is missing
		def calc_gt_reward(env, waypts):
			per_waypt = np.matmul(np.array(env.featurize(waypts[:, :7])).T, np.array(env.weights))
			return per_waypt.sum(), per_waypt

		for j, s_g_demos in enumerate(self.s_g_exp_trajs):
			demo_rews = []
			for demo in s_g_demos:
				traj_rew, _ = calc_gt_reward(self.env, demo)
				demo_rews.append(traj_rew)

			self.s_g_exp_trajs[j] = [self.s_g_exp_trajs[j][i] for i in
									 np.arange(len(self.s_g_exp_trajs[j]))[np.array(demo_rews) >= demo_rews[0]]]

	def generate_rand_start_goal(self, n_trajs, min_dist=0.7):
		trajs = []
		starts = []
		goals = []
		while len(trajs) < n_trajs:
			# sample
			start_sample = np.random.uniform(low=0, high=2 * math.pi, size=7)
			goal_sample = np.random.uniform(low=0, high=2 * math.pi, size=7)
			# plan
			opt_traj = self.planner.replan(start_sample, goal_sample, None, self.T, self.timestep, seed=None)
			# get raw and x,y,z of start and end of the trajectory
			raw = map_to_raw_dim(self.env, [opt_traj.waypts])
			distance = np.linalg.norm(raw[0][0][88:91] - raw[0][-1][88:91])
			if distance > min_dist:
				trajs.append(raw[0])
				starts.append(start_sample)
				goals.append(goal_sample)
		self.starts = self.starts + starts
		self.goals = self.goals + goals
		self.goal_poses = self.goal_poses + [None]*len(starts)

	def return_trajs(self):
		return self.s_g_exp_trajs

	def load_trajs(self, trajectories):
		self.s_g_exp_trajs = trajectories

	def plot_trajs(self):
		all_trajs = []
		for s_g_demos in self.s_g_exp_trajs:
			high_dim_demos = []
			for angle_traj in s_g_demos:
				high_dim_demos.append(map_to_raw_dim(self.env, [angle_traj])[0])
			all_trajs = all_trajs + high_dim_demos
		plot_trajs(all_trajs, object_centers=self.env.object_centers, title='Expert Trajectories')