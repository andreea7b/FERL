import numpy as np
import math
from src.utils.openrave_utils import *
from src.utils.environment import Environment
from src.planners.trajopt_planner import TrajoptPlanner
import torch
import os, sys

nb_layers = 3
nb_units = 32
n_samples = 10000

# Maybe use stuff like: Label perturbation or something that makes the assumption that labeled data is not perfect..


def generate_gt_data(n_samples, environment, feature):
	n_per_dim = math.ceil(n_samples ** (1 / 7))
	#  we are in 7D radian space
	dim_vector = np.linspace(0, 2 * np.pi, n_per_dim)
	train_points = []
	regression_labels = []

	for i in range(n_samples):
		sample = np.random.uniform(0, 2 * np.pi, 7)
		if feature == "table":
			rl = table_features(environment, sample)
		elif feature == "coffee":
			rl = coffee_features(environment, sample)
		elif feature == "human":
			rl = human_features(environment, sample)
		elif feature == "laptop":
			rl = laptop_features(environment, sample)
		train_points.append(sample)
		regression_labels.append(rl)

	# normalize
	regression_labels = np.array(regression_labels) / max(regression_labels)
	return np.array(train_points), regression_labels

# -- Distance to Table -- #

def table_features(environment, waypt):
	"""
	Computes the total feature value over waypoints based on 
	z-axis distance to table.
	---
	input waypoint, output scalar feature
	"""
	if len(waypt) < 10:
		waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
		waypt[2] += math.pi

	environment.robot.SetDOFValues(waypt)
	coords = robotToCartesian(environment.robot)
	EEcoord_z = coords[6][2]
	return EEcoord_z

# -- Coffee (or z-orientation of end-effector) -- #

def coffee_features(environment, waypt):
	"""
	Computes the coffee orientation feature value for waypoint
	by checking if the EE is oriented vertically.
	---
	input waypoint, output scalar feature
	"""
	if len(waypt) < 10:
		waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
		waypt[2] += math.pi

	environment.robot.SetDOFValues(waypt)
	EE_link = environment.robot.GetLinks()[7]
	Rx = EE_link.GetTransform()[:3,0]
	return 1 - EE_link.GetTransform()[:3,0].dot([0,0,1])

# -- Distance to Laptop -- #

def laptop_features(environment, waypt):
	"""
	Computes distance from end-effector to laptop in xy coords
	input trajectory, output scalar distance where 
		0: EE is at more than 0.4 meters away from laptop
		+: EE is closer than 0.4 meters to laptop
	"""
	if len(waypt) < 10:
		waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
		waypt[2] += math.pi

	environment.robot.SetDOFValues(waypt)
	coords = robotToCartesian(environment.robot)
	EE_coord_xy = coords[6][0:2]
	laptop_xy = np.array(environment.object_centers['LAPTOP_CENTER'][0:2])
	dist = 1.5 - np.linalg.norm(EE_coord_xy - laptop_xy)
	return dist

# -- Distance to Human -- #

def human_features(environment, waypt):
	"""
	Computes distance from end-effector to human in xy coords
	input trajectory, output scalar distance where 
		0: EE is at more than 0.4 meters away from human
		+: EE is closer than 0.4 meters to human
	"""
	if len(waypt) < 10:
		waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
		waypt[2] += math.pi
	environment.robot.SetDOFValues(waypt)
	coords = robotToCartesian(environment.robot)
	EE_coord_xy = coords[6][0:2]
	human_xy = np.array(environment.object_centers['HUMAN_CENTER'][0:2])
	dist = 1.5 - np.linalg.norm(EE_coord_xy - human_xy)
	return dist



def main(feature):
	# create environment instance
	print "Creating environment"
	environment = Environment("jaco_dynamics",
							  {'HUMAN_CENTER': [-0.6, -0.55, 0.0], 'LAPTOP_CENTER': [-0.7929, -0.1, 0.0]}
							  , [feature], [1.0], np.array([0.0]))
	print "Finished environment"
	# create Learned_Feature
	environment.new_learned_feature(nb_layers, nb_units, 7)
	print "Generating data..."
	# generate training data
	train, labels = generate_gt_data(n_samples, environment, feature)
	train, labels = train[labels >= 0], labels[labels >= 0]
	# Create raw features
	train_raw = np.empty((0, 97), float)
	for dp in train:
		train_raw = np.vstack((train_raw, environment.raw_features(dp)))
	here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../'))
	np.savez(here+'/data/model_checkpoints/data_{}.npz'.format(feature), x=train_raw, y=labels)
	import pdb;pdb.set_trace()
	print "Finished generating data."
	# feed perfect data to learned_feature & train NN
	labels = labels.reshape(len(labels),1)
	environment.learned_features[-1].add_data(train, labels)
	environment.learned_features[-1].train(p_train=0.8, epochs=1000, batch_size=16, learning_rate=1e-2)

	# safe parameter values in checkpoint
	here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../'))
	torch.save(environment.learned_features[-1].torch_function.state_dict(), here+'/data/model_checkpoints/check.pt')
	print("Model parameters saved")

	# extend it to show mean & variance of error with increasing perfect samples


if __name__ == '__main__':
	feat = sys.argv[1]
	main(feat)

