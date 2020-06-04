import sys
sys.path.append('../')
import numpy as np
import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir('../..')
from ray import tune

import pickle
import random

def calculate_generalization_distribution(config):
	sys.path.append(config["current_dir"] + '/src')
	from utils.learned_feature import LearnedFeature
	
	# settings
	data_traj = config["traj_file"]
	data_gt = config["gt_file"]

	# load trajectories
	data_file = config["current_dir"] + '/data/traces/' + data_traj
	trajectory_list = pickle.load(open(data_file, "rb"))	
	
	# shuffle trajectory list because some might be more informative than others
	traj_idxes = range(len(trajectory_list))
	random.shuffle(traj_idxes)

	# load ground truth data
	test_idx = None
	if "laptopmoving" in data_file:
		test_idx = traj_idxes[config["n_train_trajs"]]
		data_file = config["current_dir"] + '/data/gtdata/' + data_gt
		data_file = data_file[:-4] + "L{}.npz".format(str(test_idx+1)) 
	else:
		data_file = config["current_dir"] + '/data/gtdata/' + data_gt
	npzfile = np.load(data_file)
	coordinates = npzfile['x']
	gt_labels = npzfile['y']
	gt_labels = gt_labels.reshape(len(gt_labels), 1)

	# Initialize NN feature
	if "coffee" in data_gt:
		LF_dict = {'convex':False, 'L_proj':None, 'n_ensamble':1, 'bet_loss':"in_loss", 'bet_data':5, 'activation':config["activation"],
			   'sin':False, 'cos':False, 'rpy':False, 'lowdim':False, 'noangles':True, 'norot':False, 'noxyz':True, 
			   '9D_coffee':True, '6D_laptop':False, '6D_human':False, 'masked':False, 'EErot':False}
	else:
		LF_dict = {'convex':False, 'L_proj':None, 'n_ensamble':1, 'bet_loss':"in_loss", 'bet_data':5, 'activation':config["activation"],
			   'sin':False, 'cos':False, 'rpy':False, 'lowdim':False, 'noangles':True, 'norot':True, 'noxyz':False, 
			   '9D_coffee':False, '6D_laptop':False, '6D_human':False, 'masked':False, 'EErot':False}

	feature = LearnedFeature(config["nb_layers"], config["nb_units"], None, LF_dict)
	
	waypt_data = 0
	length_data = 0
	for i in traj_idxes[:config["n_train_trajs"]]:
		traj = trajectory_list[i]
		if "betweenobjects" in data_traj:
			label = 0.5
			if i < 24:
				feature_raw.add_data(traj)
			elif i == 24 or i >=39:
				feature_raw.add_data(traj, start_label=label, end_label=label)
			elif i > 24:
				feature_raw.add_data(traj, end_label=label)
		else:
			feature.add_data(traj)
		dist = [np.linalg.norm(traj[j] - traj[j+1]) for j in range(traj.shape[0]-1)]
		length_data += sum(dist)
		waypt_data += traj.shape[0]

	# calculate pointwise differences between randomly initialized feature and ground truth
	feature.function(coordinates)

	# get costs and normalize them
	learned_costs = feature.function(coordinates)
	learned_norm = (learned_costs - np.amin(learned_costs)) / (np.amax(learned_costs) - np.amin(learned_costs))
	delta_random_list = learned_norm - gt_labels
	delta_random_list = delta_random_list.squeeze().tolist()

	feature.train(epochs=config["epochs"], learning_rate=config["learning_rate"], weight_decay=config["weight_decay"], s_g_weight=10.)

	# evaluate feature on whole state space
	# calculate pointwise differences between learned_feature values and ground truth
	learned_costs = feature.function(coordinates)
	learned_norm = (learned_costs - np.amin(learned_costs)) / (np.amax(learned_costs) - np.amin(learned_costs))
	delta_list = learned_norm - gt_labels
	delta_list = delta_list.squeeze().tolist()
	
	# track
	tune.track.log(delta_list=delta_list, delta_random_list=delta_random_list, length=length_data, numwaypt=waypt_data, test=test_idx)

	# Save models and data in case of failure.
	dirpath = config["current_dir"]+'/data/model_checkpoints/'
	filepath = dirpath + 'FAIRLfinal_{}_{}_{}.pt'.format(config["traj_file"][:-2], config["n_train_trajs"], 0)
	i = 1
	while os.path.exists(filepath):
		filepath = dirpath + 'FAIRLfinal_{}_{}_{}.pt'.format(config["traj_file"][:-2], config["n_train_trajs"], str(i))
		i+=1
	torch.save(feature, filepath)


def main():
	ground_truth_data_files = ['data_table.npz', 'data_coffee.npz', 'data_laptop.npz', 'data_proxemics.npz']#, 'data_laptopmoving.npz', 'data_betweenobjects.npz']
	trajectory_files = ['traces_table.p', 'traces_coffee.p', 'traces_laptop.p', 'traces_proxemics.p']#, 'traces_laptopmoving', 'traces_betweenobjects.p']
	ground_truth_data_files = ['data_laptopmoving.npz', 'data_betweenobjects.npz']
	trajectory_files = ['traces_laptopmoving.p', 'traces_betweenobjects.p']

	parent_dir = os.path.abspath(os.getcwd())

	N_traj_max = 10
	for gt_file, traj_file in zip(ground_truth_data_files, trajectory_files):
		if "laptopmoving" in traj_file:
			n_train_traj = range(2, 2*N_traj_max, 2)
		elif "betweenobjects" in traj_file:
			n_train_traj = range(2, 3*N_traj_max, 3)
		else:
			n_train_traj = range(2, N_traj_max)

		# Set up the hypertune experiment
		config_dic = {"nb_units": tune.grid_search([64]),
					  "nb_layers": tune.grid_search([2]),
					  "epochs": tune.grid_search([100]),
					  "learning_rate": tune.grid_search([1e-3]),
					  "weight_decay": tune.grid_search([0.001]),
					  "activation": tune.grid_search(["softplus"]),
					  "n_train_trajs": tune.grid_search(n_train_traj),
					  "gt_file": gt_file,
					  "traj_file": traj_file,
					  "current_dir": parent_dir,
					  }

		analysis = tune.run(calculate_generalization_distribution, config=config_dic, num_samples=10)

		# Get a dataframe for analyzing trial results.
		df = analysis.dataframe()

		# Save the results.
		delta_list = []
		for exp in df['delta_list']:
			cleaned_list = exp.strip('[]').split(',')
			delta_list.append(np.array([float(i) for i in cleaned_list]))
		df.insert(1, "delta_dist", delta_list, True)

		delta_list = []
		for exp in df['delta_random_list']:
			cleaned_list = exp.strip('[]').split(',')
			delta_list.append(np.array([float(i) for i in cleaned_list]))
		df.insert(2, "delta_random_dist", delta_list, True)

		with open(parent_dir + '/data/generalization_tests/FERL_' + traj_file[:-2] + '_gen_data.p', "wb") as fp:
			pickle.dump(df[['config/n_train_trajs', 'delta_dist', 'delta_random_dist', 'length', 'numwaypt', 'test']], fp)


if __name__ == '__main__':
	main()
