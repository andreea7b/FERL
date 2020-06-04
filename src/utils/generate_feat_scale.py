import math
import numpy as np
import pickle
import sys, os
import ast

from environment import Environment

def generate_feat_scale(feat_list, trajs_path="/data/traj_sets/traj_rand_merged_H.p"):
	# Before calling this function, you need to decide what features you care
	# about, from a choice of table, coffee, human, origin, and laptop.
	pick = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 225.0]
	place = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)
	goal_pose = None
	T = 20.0
	timestep = 0.5

	# Openrave parameters for the environment.
	model_filename = "jaco_dynamics"
	object_centers = {'HUMAN_CENTER': [-0.6,-0.55,0.0], 'LAPTOP_CENTER': [-0.7929,-0.1,0.0]}
	feat_list = [x.strip() for x in feat_list.split(',')]
	num_features = len(feat_list)
	feat_range = [1.0] * len(feat_list)
	weights = [0.0] * len(feat_list)
	environment = Environment(model_filename, object_centers, feat_list, feat_range, np.array(weights))

	here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../'))
	trajs = pickle.load(open(here + trajs_path, "rb" ))
	feat_range = np.array(np.zeros(environment.weights.shape))
	feat_min = np.array(np.zeros(environment.weights.shape))
	for rand_i, traj_str in enumerate(trajs.keys()):
		curr_traj = np.array(ast.literal_eval(traj_str))
		features = environment.featurize(curr_traj)
		feat_range = np.maximum(feat_range, np.max(features, axis=1))
		feat_min = np.minimum(feat_min, np.min(features, axis=1))
	import pdb;pdb.set_trace()
	return feat_range

if __name__ == '__main__':
	feat_list = sys.argv[1]
	print(generate_feat_scale(feat_list))



