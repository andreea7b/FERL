import numpy as np
import math
from openrave_utils import *
from environment import Environment
from planners.trajopt_planner import TrajoptPlanner
import torch
import os, sys

nb_layers = 3
nb_units = 32
n_samples = 10000

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
		elif feature == "EEcoord":
			rl = waypt_to_EEcoords(environment, sample)
		elif feature == "EEorientation":
			rl = waypt_to_EEorientation(environment, sample)
		elif feature == "proxemics":
			rl = proxemics_features(environment, sample)
		elif feature == "betweenobjects":
			rl = betweenobjects_features(environment, sample)
		train_points.append(sample)
		regression_labels.append(rl)

	# normalize
	regression_labels = np.array(regression_labels) / max(regression_labels)
	return np.array(train_points), regression_labels

def waypt_to_EEcoords(environment, waypt):
	if len(waypt) < 10:
		waypt_openrave = np.append(waypt.reshape(7), np.array([0, 0, 0]))
		waypt_openrave[2] += math.pi

	environment.robot.SetDOFValues(waypt_openrave)
	coords = np.array(robotToCartesian(environment.robot))
	return coords[6][:3]

def waypt_to_EEorientation(environment, waypt):
	if len(waypt) < 10:
		waypt_openrave = np.append(waypt.reshape(7), np.array([0, 0, 0]))
		waypt_openrave[2] += math.pi

	environment.robot.SetDOFValues(waypt_openrave)
	orientations = np.array(robotToOrientation(environment.robot))
	return orientations[6].flatten()

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
	dist = np.linalg.norm(EE_coord_xy - laptop_xy) - 0.3
	if dist > 0:
		return 0
	return -dist

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
	dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.3
	if dist > 0:
		return 0
	return -dist

# -- Proxemics -- #

def proxemics_features(environment, waypt):
	"""
	Computes distance from end-effector to human proxemics in xy coords
	input trajectory, output scalar distance where 
		0: EE is at more than 0.3 meters away from human
		+: EE is closer than 0.3 meters to human
	"""
	if len(waypt) < 10:
		waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
		waypt[2] += math.pi
	environment.robot.SetDOFValues(waypt)
	coords = robotToCartesian(environment.robot)
	EE_coord_xy = coords[6][0:2]
	human_xy = np.array(environment.object_centers['HUMAN_CENTER'][0:2])
	# Modify ellipsis distance.
	EE_coord_xy[1] /= 3
	human_xy[1] /= 3
	dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.3
	if dist > 0:
		return 0
	return -dist

def betweenobjects_features(environment, waypt):
	"""
	Computes distance from end-effector to 2 objects.
	"""
	if len(waypt) < 10:
		waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
		waypt[2] += math.pi
	environment.robot.SetDOFValues(waypt)
	coords = robotToCartesian(environment.robot)
	EE_coord_xy = coords[6][0:2]
	object1_xy = np.array(environment.object_centers['OBJECT1'][0:2])
	object2_xy = np.array(environment.object_centers['OBJECT2'][0:2])

	# Determine where the point lies with respect to the segment between the two objects.
	o1EE = np.linalg.norm(object1_xy - EE_coord_xy)
	o2EE = np.linalg.norm(object2_xy - EE_coord_xy)
	o1o2 = np.linalg.norm(object1_xy - object2_xy)
	o1angle = np.arccos((o1EE**2 + o1o2**2 - o2EE**2) / (2*o1o2*o1EE))
	o2angle = np.arccos((o2EE**2 + o1o2**2 - o1EE**2) / (2*o1o2*o2EE))

	dist1 = 0
	if o1angle < np.pi/2 and o2angle < np.pi/2:
		dist1 = np.linalg.norm(np.cross(object2_xy - object1_xy, object1_xy - EE_coord_xy)) / o1o2 - 0.2
	dist1 = dist1*0.5 # control how much less it is to go between the objects versus on top of them
	dist2 = min(np.linalg.norm(object1_xy - EE_coord_xy), np.linalg.norm(object2_xy - EE_coord_xy)) - 0.2

	if dist1 > 0 and dist2 > 0:
		return 0
	elif dist2 > 0:
		return -dist1
	elif dist1 > 0:
		return -dist2
	return -min(dist1, dist2)

def main(feature):
	# create environment instance
	print "Creating environment"
	environment = Environment("jaco_dynamics",
							  {'OBJECT1': [-0.6,-0.2,0.0], 'OBJECT2': [-0.2,0.0,0.0]}
							  #{'HUMAN_CENTER': [-0.2,-0.5,0.6], 'LAPTOP_CENTER': [-0.8, 0.0, 0.0]}
							  , [feature], [1.0], np.array([0.0]), viewer=False)
	print "Finished environment"
	# create Learned_Feature
	environment.new_learned_feature(nb_layers, nb_units)
	print "Generating data..."
	# generate training data
	if feature == "laptopmoving":
		positions = {"L1": [-0.8, 0.0, 0.0], "L2": [-0.6, 0.0, 0.0], "L3": [-0.4, 0.0, 0.0], "L4": [-0.8, 0.2, 0.0],
		 			 "L5": [-0.6, 0.2, 0.0], "L6": [-0.4, 0.2, 0.0], "L7": [-0.8, -0.2, 0.0], "L8": [-0.6, -0.2, 0.0],
					 "L9": [-0.4, -0.2, 0.0], "L10": [-0.5, -0.1, 0.0], "L11": [-0.7, -0.1, 0.0], "L12": [-0.3, -0.1, 0.0],
					 "L13": [-0.3, 0.1, 0.0], "L14": [-0.5, 0.1, 0.0], "L15": [-0.7, 0.1, 0.0], "L16": [-0.3, 0.3, 0.0],
					 "L17": [-0.5, 0.3, 0.0], "L18": [-0.7, 0.3, 0.0], "L19": [-0.5, -0.3, 0.0], "L20": [-0.7, -0.3, 0.0],
					 "L21": [-0.3, -0.3, 0.0], "L22": [-0.6, -0.3, 0.0], "L23": [-0.5, -0.2, 0.0], "L24": [-0.7, -0.2, 0.0],
		 			 "L25": [-0.3, 0.0, 0.0], "L26": [-0.5, 0.0, 0.0], "L27": [-0.6, 0.1, 0.0], "L28": [-0.8, 0.1, 0.0],
					 "L29": [-0.4, 0.3, 0.0], "L30": [-0.6, 0.3, 0.0]}
		for lidx in positions.keys():
			environment.object_centers["LAPTOP_CENTER"] = positions[lidx]
			train, labels = generate_gt_data(n_samples, environment, "laptop")
			#train, labels = train[labels >= 0], labels[labels >= 0]
			# Create raw features
			train_raw = np.empty((0, 97), float)
			for dp in train:
				train_raw = np.vstack((train_raw, environment.raw_features(dp)))
			here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../'))
			np.savez(here+'/data/gtdata/data_{}{}.npz'.format(feature, lidx), x=train_raw, y=labels)
	else:
		train, labels = generate_gt_data(n_samples, environment, feature)
		train, labels = train[labels >= -0.1016], labels[labels >= -0.1016]
		# Create raw features
		train_raw = np.empty((0, 97), float)
		for dp in train:
			train_raw = np.vstack((train_raw, environment.raw_features(dp)))
		here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../'))
		np.savez(here+'/data/gtdata/data_{}.npz'.format(feature), x=train_raw, y=labels)
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


if __name__ == '__main__':
	feat = sys.argv[1]
	main(feat)

