import sys
import os
import torch

os.chdir('../..')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.getcwd()) + '/src')

# os.chdir('..')
# sys.path.append(os.path.abspath(os.getcwd()))
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# sys.path.append(os.path.join(os.path.abspath(os.getcwd()),".."))

import pickle
import math
import numpy as np
import pickle
import random
import glob

from MaxEnt_Baseline.baseline_utils import *

# Setup for the expert.
gt_feat = ["coffee", "table", "proxemics"]
gt_weights = [0.0, 10.0, 10.0]

# known
object_centers = {'HUMAN_CENTER': [-0.2,-0.5,0.6], 'LAPTOP_CENTER': [-0.8, 0.0, 0.0]}
known_features = ['coffee', 'table']
known_weights = [0.0, 0.0]
FEAT_RANGE = {'table':0.98, 'coffee':1.0, 'laptop':0.3, 'human':0.3, 'efficiency':0.22, 'proxemics': 0.3, 'betweenobjects': 0.2, 'learned_feature': 1.0}
feat_range = [FEAT_RANGE[known_features[feat]] for feat in range(len(known_features))]

# learned via pushes
#learned_weights_from_pushes = np.array([0.0, 4.03901256, 5.51417794]) # case1
#learned_weights_from_pushes = np.array([0.0, 4.35964768, 4.88110989]) # case2
learned_weights_from_pushes = np.array([0.0, 3.09983027, 5.1572305]) # case3

def main():
	featstr = gt_feat[-1]
	parent_dir = os.path.abspath(os.getcwd())

	# Setup.
	n_train_traj = range(2, 11)
	delta_lists = []
	n_traj_list = []

	for seed in range(10):
		for ntrajs in n_train_traj:
			data_file = 'Models/FAIRLfinal_traces_{}_{}_{}.pt'.format(featstr, ntrajs, seed)
			# Step 1: Create new environment with all features.
			known_features = ['coffee', 'table']
			env = Environment("jaco_dynamics", object_centers, known_features, feat_range, known_weights, viewer=False)
			env.new_learned_feature(nb_layers=2, nb_units=64, checkpoint_name=data_file)

			# Step 2: set the learned weights & calculate overall cost function
			env.weights = learned_weights_from_pushes

			# Create gt environment.
			gt_feat_range = [FEAT_RANGE[gt_feat[feat]] for feat in range(len(gt_feat))]
			gt_env = Environment("jaco_dynamics", object_centers, gt_feat, gt_feat_range, gt_weights, viewer=False)
			raw_features, gt_cost = get_coords_gt_cost(False, gt_env, parent_dir)
			angle_waypts = raw_features[:, :7]

			# get the number of learned features
			n_learned = env.feature_list.count('learned_feature')

			# calculate costs
			feat_idx = list(np.arange(env.num_features))
			features = [[0.0 for _ in range(len(angle_waypts))] for _ in range(0, len(feat_idx))]
			for index in range(len(angle_waypts)):
			    for feat in feat_idx:
			        features[feat][index] = env.featurize_single(angle_waypts[index], feat)
			            
			learned_costs = np.matmul(np.array(features).T, np.array(env.weights)).reshape(-1,1)

			# Step 3.2: normalize to 0-1 and calculate delta distribution amongst them
			learned_norm = (learned_costs - np.amin(learned_costs)) / (np.amax(learned_costs) - np.amin(learned_costs))
			gt_norm = (gt_cost - np.amin(gt_cost)) / (np.amax(gt_cost) - np.amin(gt_cost))

			delta_list = learned_norm - gt_norm
			delta_list = delta_list.squeeze()

			delta_lists.append(np.array(delta_list))
			n_traj_list.append(ntrajs)
			env.kill_environment()
			gt_env.kill_environment()
			env = None
			gt_env = None

	# build a df
	to_df = {'n_traj': n_traj_list, 'delta_dist': delta_lists}
	df = pd.DataFrame(to_df)
	with open(parent_dir + '/data/generalization_tests/' + "FERL_cost_{}.p".format(featstr), "wb") as fp:
		pickle.dump(df[['n_traj', 'delta_dist']], fp)

if __name__ == '__main__':
	main()
