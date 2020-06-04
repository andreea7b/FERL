import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

os.chdir('../..')
sys.path.append(os.path.abspath(os.getcwd()))
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pickle
import math
import pandas as pd
import numpy as np
from ray import tune

from MaxEnt_Baseline.baseline_utils import get_coords_gt_cost
from MaxEnt_Baseline.Reward_Expert import GT_Reward_Expert
from MaxEnt_Baseline.MaxEntBaseline import DeepMaxEntIRL



# Settings
n_traj_max = 10
n_samples_per_setting = 3
file_name = 'MaxEnt_IRL_tab_lap_human_demos.p'
trajfeat = "tablelaptop"

# define the ground truth function to learn
feat_list = ['coffee', "table", "laptop"]
weights = [0.0, 1.0, 1.0]
known_feat_list = ['coffee', 'table']
object_centers = {'HUMAN_CENTER': [-0.6, -0.55, 0.0], 'LAPTOP_CENTER': [-0.8, 0.0, 0.0]}
FEAT_RANGE = {'table':0.98, 'coffee':1.0, 'laptop':0.4, 'human':0.4, 'efficiency':0.22}

# planner settings for expert demonstrations & eval of current reward
T = 20.0
timestep = 0.5

# IRL Network
NN_dict = {'n_layers': 2, 'n_units':128, 'masked':False, 'sin':False, 'cos':False, 
		   'noangles':True, 'norot':True, 'rpy':False, 'lowdim':False,
           '6D_laptop':False, '6D_human':False, '9D_coffee':False}
IRL_dict = {'n_iters': 50, 'n_cur_rew_traj': 1, 'lr':1e-3, 'weight_decay':0.001, 'n_traj_per_batch':1, 'std':0.01}

############ Functions ##############

# function to test generalization
def calculate_generalization_distribution(config):
	# Step 0: set_up S_G pairs or trajectories
	traj_idxes = np.random.choice(a=np.arange(n_traj_max), size=config['n_traj'], replace=False)
	data_file = config['src_dir'] + '/data/demonstrations/demos/demos_{}.p'.format(trajfeat)
	trajectory_list = pickle.load(open( data_file, "rb" ) )	

	# Step 1: generate Expert demonstratons
	Expert = GT_Reward_Expert(feat_list, weights, gen='cost', starts=[], goals=[], goal_poses=None,
							  obj_center_dict=object_centers, feat_range_dict=FEAT_RANGE, combi=True)
	Expert.s_g_exp_trajs = [[trajectory_list[i]] for i in traj_idxes]

	# Step 2: Train IRL Function
	IRL = DeepMaxEntIRL(s_g_exp_trajs=Expert.return_trajs(), goal_poses=None,
						known_feat_list=known_feat_list, NN_dict=NN_dict, gen='waypt',
						obj_center_dict=object_centers, feat_range_dict=FEAT_RANGE)
	IRL.deep_max_ent_irl(n_iters=IRL_dict['n_iters'], n_cur_rew_traj=IRL_dict['n_cur_rew_traj'],
						 lr=IRL_dict['lr'], weight_decay=IRL_dict['weight_decay'],
						 n_traj_per_batch=IRL_dict['n_traj_per_batch'], std=IRL_dict['std'])

	# Step 3: compare expert with gt
	raw_waypts, gt_cost = get_coords_gt_cost(False, Expert.env, config["src_dir"])

	# Step 3.1: get labels from trained function
	learned_costs = IRL.function(raw_waypts).detach().numpy()

	# Step 3.2: normalize to 0-1 and calculate delta distribution amongst them
	learned_norm = (learned_costs - np.amin(learned_costs)) / (np.amax(learned_costs) - np.amin(learned_costs))
	gt_norm = (gt_cost - np.amin(gt_cost)) / (np.amax(gt_cost) - np.amin(gt_cost))

	delta_list = learned_norm - gt_norm
	delta_list = delta_list.squeeze().tolist()

	# track
	tune.track.log(delta_list=delta_list)

def main():
	feature_learning_pwd = os.path.abspath(os.path.join(__file__, "../"))

	# Set up the hypertune experiment
	config_dic = {"n_traj": tune.grid_search(range(1, n_traj_max+1)),
				  "src_dir": feature_learning_pwd}

	analysis = tune.run(calculate_generalization_distribution, config=config_dic, num_samples=n_samples_per_setting)

	# Get a dataframe for analyzing trial results.
	df = analysis.dataframe()

	# safe the results
	delta_list = []
	for exp in df['delta_list']:
		cleaned_list = exp.strip('[]').split(',')
		delta_list.append(np.array([float(i) for i in cleaned_list]))
	df.insert(1, "delta_dist", delta_list, True)

	with open(feature_learning_pwd + '/data/generalization_tests/' + file_name, "wb") as fp:
		pickle.dump(df[['config/n_traj', 'delta_dist']], fp)

if __name__ == '__main__':
	main()