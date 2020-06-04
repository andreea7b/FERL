import sys
sys.path.append('../')
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from ray import tune

import pickle
import random


# function to test generalization
def calculate_generalization_distribution(config):
	import sys
	sys.path.append(config["current_dir"] + '/src')
	from utils.learned_feature import LearnedFeature
	# settings
	data_traj = config["traj_data"]
	data_gt = config["gt_file"]

	# load ground truth data
	data_file = config["current_dir"] + '/data/model_checkpoints/' + data_gt
	npzfile = np.load(data_file)
	coordinates = npzfile['x']
	gt_labels = npzfile['y']
	gt_labels = gt_labels.reshape(len(gt_labels), 1)

	# load trajectories
	data_file = config["current_dir"] + '/data/model_checkpoints/' + data_traj
	trajectory_list = pickle.load(open(data_file, "rb"))	

	# shuffle trajectory list because some might be more informative than others
	random.shuffle(trajectory_list)

	# train NN
	if config["input_dim"] == 3:
		raw_idx = [25, 26, 27]
	else:
		raw_idx = range(97)
	feature = LearnedFeature(config["nb_layers"], config["nb_units"], config["input_dim"],
							 bet_loss=config["bet_loss"], bet_data=config["bet_data"])
	for traj in trajectory_list:
		feature.add_data(traj[:, raw_idx])

	feature.train(epochs=config["epochs"], learning_rate=config["learning_rate"],
				  weight_decay=config["weight_decay"], l_reg=config["l_reg"])

	# evaluate feature on whole state space
	# calculate pointwise differences between learned_feature values and ground truth
	delta_list = feature.function(coordinates[:, raw_idx]) - gt_labels
	delta_list = delta_list.squeeze().tolist()
	# track
	tune.track.log(delta_list=delta_list)

def main():
	ground_truth_data_files =['data_proxemics.npz'] #'data_coffee.npz', 'data_human.npz', 'data_laptop.npz', 'data_table.npz']
	trajectory_files = ['trajs_proxemics3.p'] #'trajs_coffee.p', 'trajs_human.p', 'trajs_laptop.p', 'trajs_table.p']

	os.chdir('../..')
	parent_dir = os.path.abspath(os.getcwd())

	for gt_data, traj_file in zip(ground_truth_data_files, trajectory_files):
		# set up experiment
		# load the trajectories to check their length
		with open(parent_dir + '/data/model_checkpoints/' + traj_file, 'rb') as f:
			u = pickle.Unpickler(f)
			u.encoding = 'latin1'
			trajectory_list = u.load()

		# Set up the hypertune experiment
		config_dic = {"nb_units": tune.grid_search([128]),
					  "nb_layers": tune.grid_search([3]),
					  "epochs": tune.grid_search([100]),
					  "learning_rate": tune.grid_search([1e-3]),
					  "weight_decay": tune.grid_search([0.0, 0.01, 0.1, 1.0]),
					  "l_reg": tune.grid_search([0.1, 1.0, 10.0]),
					  "bet_loss": tune.grid_search(["in_loss", "SG_overlap", "minmax_overlap", "SG"]),
					  "bet_data": tune.grid_search([0, 1, 5]),
					  "input_dim": tune.grid_search([3, 97]),
					  "gt_file": gt_data,
					  "traj_data": traj_file,
					  "current_dir": parent_dir,
					  }

		analysis = tune.run(calculate_generalization_distribution, config=config_dic, num_samples=1)

		# Get a dataframe for analyzing trial results.
		df = analysis.dataframe()

		# safe the results
		delta_list = []
		mean_delta = []
		for exp in df['delta_list']:
			cleaned_list = exp.strip('[]').split(',')
			delta_list.append(np.array([float(i) for i in cleaned_list]))
			mean_delta.append(delta_list[-1].mean())
		df.insert(1, "mean_delta", mean_delta, True)
		df.insert(2, "delta_dist", delta_list, True)

		with open(parent_dir + '/data/generalization_tests/' + traj_file[:-2] + '_gen_data.p', "wb") as fp:
			pickle.dump(df, fp)


if __name__ == '__main__':
	main()
