import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

os.chdir('../..')
sys.path.append(os.path.abspath(os.getcwd()))
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pickle
import math
import pandas as pd
import numpy as np
import random

from MaxEnt_Baseline.baseline_utils import get_coords_gt_cost
from MaxEnt_Baseline.Reward_Expert import GT_Reward_Expert
from MaxEnt_Baseline.MaxEntBaseline import DeepMaxEntIRL
from utils.trajectory import Trajectory

# Settings
n_traj_max = 10
n_samples_per_setting = 10
trajfeat = "tableproxemics_case3"
file_name = 'MEIRLFalse_{}.p'.format(trajfeat)

# define the ground truth function to learn
feat_list = ['coffee', "table", "proxemics"]
weights = [0.0, 10.0, 10.0]
known_feat_list = ['coffee', 'table']
object_centers = {'HUMAN_CENTER': [-0.2,-0.5,0.6], 'LAPTOP_CENTER': [-0.8, 0.0, 0.0]}
FEAT_RANGE = {'table':0.98, 'coffee':1.0, 'laptop':0.3, 'human':0.3, 'efficiency':0.22, 'proxemics': 0.3, 'betweenobjects': 0.2}

# planner settings for expert demonstrations & eval of current reward
T = 20.0
timestep = 0.5

# IRL Network
NN_dict = {'n_layers': 2, 'n_units':128, 'masked':False, 'sin':False, 'cos':False, 
		   'noangles':True, 'norot':True, 'noxyz':False, 'rpy':False, 'lowdim':False,
           '6D_laptop':False, '6D_human':False, '9D_coffee':False, 'EErot':False}
IRL_dict = {'n_iters': 50, 'n_cur_rew_traj': 1, 'lr':1e-3, 'weight_decay':0.001, 'n_traj_per_batch':1, 'std':0.01}

# Types of trajectories for sampling.
#traj_types = {"known": [3, 4, 13, 14], "unknown": [0, 1, 2, 10, 11, 12], "both": [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]} # case1
#traj_types = {"known": [4, 5, 14, 15], "unknown": [0, 1, 2, 3, 10, 11, 12, 13], "both": [6, 7, 8, 9, 16, 17, 18, 19]} # case2
traj_types = {"known": [6, 7, 13, 14], "unknown": [0, 1, 2, 10, 11, 12], "both": [3, 4, 5, 8, 9, 15, 16, 17, 18, 19]} # case3
#traj_types = {"known": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]}

############ Functions ##############

def structured_sampling(n_traj):
	types = len(traj_types.keys())
	traj_idxes = []
	for key in traj_types.keys():
		traj_idxes.extend(random.sample(traj_types[key], n_traj/types))
	remaining_keys = random.sample(traj_types.keys(), n_traj - len(traj_idxes))
	for key in remaining_keys:
		while True:
			idx = random.sample(traj_types[key], 1)[0]
			if idx not in traj_idxes:
				traj_idxes.append(idx)
				break
	return traj_idxes

# function to test generalization
def calculate_generalization_distribution(n_traj, seed, src_dir):
	# Step 0: set_up S_G pairs or trajectories
	data_file = src_dir + '/data/demonstrations/demos/demos_{}.p'.format(trajfeat)
	trajectory_list = pickle.load(open( data_file, "rb" ) )
	traj_idxes = structured_sampling(n_traj)

	# Step 1: generate Expert demonstratons
	Expert = GT_Reward_Expert(feat_list, weights, gen='cost', starts=[], goals=[], goal_poses=None,
							  obj_center_dict=object_centers, feat_range_dict=FEAT_RANGE, combi=False)
	for i in traj_idxes:
		waypts = trajectory_list[i]
		waypts_time = np.linspace(0.0, T, waypts.shape[0])
		traj = Trajectory(waypts, waypts_time)

		# Downsample/Upsample trajectory to fit desired timestep and T.
		num_waypts = int(T / timestep) + 1
		if num_waypts < len(traj.waypts):
			demo = traj.downsample(int(T / timestep) + 1)
		else:
			demo = traj.upsample(int(T / timestep) + 1)
		Expert.s_g_exp_trajs.append([demo.waypts])

	# Step 2: Train IRL Function
	IRL = DeepMaxEntIRL(s_g_exp_trajs=Expert.return_trajs(), goal_poses=None,
						known_feat_list=known_feat_list, NN_dict=NN_dict, gen='waypt',
						obj_center_dict=object_centers, feat_range_dict=FEAT_RANGE)
	filepath = src_dir+'/data/model_checkpoints/'+'MEIRL_{}_{}_{}.pt'.format(trajfeat, n_traj, seed)
	if os.path.exists(filepath):
		print "File exists! Loading model."
		IRL.cost_nn = torch.load(filepath)
	else:
		IRL.deep_max_ent_irl(n_iters=IRL_dict['n_iters'], n_cur_rew_traj=IRL_dict['n_cur_rew_traj'],
							 lr=IRL_dict['lr'], weight_decay=IRL_dict['weight_decay'],
							 n_traj_per_batch=IRL_dict['n_traj_per_batch'], std=IRL_dict['std'])

	# Step 3: compare expert with gt
	raw_waypts, gt_cost = get_coords_gt_cost(False, Expert.env, src_dir)

	# Step 3.1: get labels from trained function
	learned_costs = IRL.function(raw_waypts).detach().numpy()

	# Step 3.2: normalize to 0-1 and calculate delta distribution amongst them
	learned_norm = (learned_costs - np.amin(learned_costs)) / (np.amax(learned_costs) - np.amin(learned_costs))
	gt_norm = (gt_cost - np.amin(gt_cost)) / (np.amax(gt_cost) - np.amin(gt_cost))

	delta_list = learned_norm - gt_norm
	delta_list = delta_list.squeeze().tolist()

	# Compute total "amount" of data used.
	waypt_data = 0
	length_data = 0
	for i in traj_idxes:
		traj = trajectory_list[i]
		dist = [np.linalg.norm(traj[j] - traj[j+1]) for j in range(traj.shape[0]-1)]
		dist.insert(0, 0)
		length_data += sum(dist)
		waypt_data += traj.shape[0]

	# track
	return delta_list, waypt_data, length_data, IRL.cost_nn


def main():
	feature_learning_pwd = os.path.abspath(os.path.join(__file__, "../"))

	delta_lists = []
	waypt_list = []
	length_list = []
	n_traj_list = []

	# run a for loop over it
	for n_traj in range(1, n_traj_max+1):
		for i in range(n_samples_per_setting):
			cur_delta_dist, cur_waypts, cur_length, model = calculate_generalization_distribution(n_traj, i, feature_learning_pwd)
			delta_lists.append(np.array(cur_delta_dist))
			waypt_list.append(cur_waypts)
			length_list.append(cur_length)
			n_traj_list.append(n_traj)
			#torch.save(model, feature_learning_pwd+'/data/model_checkpoints/'+'MEIRL_{}_{}_{}.pt'.format(trajfeat, n_traj, i))
			#with open(feature_learning_pwd+'/data/model_checkpoints/'+'MEIRL_{}_{}_{}.p'.format(trajfeat, n_traj, i), "wb") as fp:
			#	pickle.dump({"waypt_list": cur_waypts, "length_list": cur_length}, fp)

	# build a df
	to_df = {'n_traj': n_traj_list, 'delta_dist': delta_lists, 'waypt_list': waypt_list, 'length_list': length_list}
	df = pd.DataFrame(to_df)
	with open(feature_learning_pwd + '/data/generalization_tests/' + file_name, "wb") as fp:
		pickle.dump(df[['n_traj', 'delta_dist', 'waypt_list', 'length_list']], fp)


if __name__ == '__main__':
	main()