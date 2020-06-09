import numpy as np
import math
from openrave_utils import *
from environment import Environment
from planners.trajopt_planner import TrajoptPlanner
import torch
import os, sys

def sample_data(environment, feature, n_samples=10000):
    """
    Generates feature data with the ground truth feature function.
    Params:
        environment -- environment where the feature lives
        feature -- string representing the feature to sample
        n_samples -- number of samples (default: 10000)
    Returns:
        train_points -- configuration space waypoint samples
        regression_labels -- feature labels for train_points
    """
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
		elif feature == "proxemics":
			rl = proxemics_features(environment, sample)
		elif feature == "betweenobjects":
			rl = betweenobjects_features(environment, sample)
		train_points.append(sample)
		regression_labels.append(rl)

	# normalize
	regression_labels = np.array(regression_labels) / max(regression_labels)
	return np.array(train_points), regression_labels

# -- Distance to Table -- #

def table_features(environment, waypt):
    """
    Computes the total feature value over waypoints based on z-axis distance to table.
    ---
    Params:
        environment -- environment where the feature lives
        waypt -- single waypoint
    Returns:
        dist -- scalar feature
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
    Params:
        environment -- environment where the feature lives
        waypt -- single waypoint
    Returns:
        dist -- scalar feature
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
    Params:
        environment -- environment where the feature lives
        waypt -- single waypoint
    Returns:
        dist -- scalar distance where
            0: EE is at more than 0.3 meters away from laptop
            +: EE is closer than 0.3 meters to laptop
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
    Params:
        environment -- environment where the feature lives
        waypt -- single waypoint
    Returns:
        dist -- scalar distance where
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
    Params:
        environment -- environment where the feature lives
        waypt -- single waypoint
    Returns:
        dist -- scalar distance where
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
    Computes distance from end-effector to 2 objects in xy coords.
    Params:
        environment -- environment where the feature lives
        waypt -- single waypoint
    Returns:
        dist -- scalar distance where
            0: EE is at more than 0.2 meters away from the objects and between
            +: EE is closer than 0.2 meters to the objects and between
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

def generate_gt_data(feature):
	# create environment instance
	print "Creating environment"
    if feature == "between_objects":
        objects = {'OBJECT1': [-0.6,-0.2,0.0], 'OBJECT2': [-0.2,0.0,0.0]}
    else:
        objects = {'HUMAN_CENTER': [-0.2,-0.5,0.6], 'LAPTOP_CENTER': [-0.8, 0.0, 0.0]}
	environment = Environment("jaco_dynamics", objects, [feature], [1.0], np.array([0.0]), viewer=False)
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
			train, labels = sample_data(environment, "laptop")
			# Create raw features
			train_raw = np.empty((0, 97), float)
			for dp in train:
				train_raw = np.vstack((train_raw, environment.raw_features(dp)))
			here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../'))
			np.savez(here+'/data/gtdata/data_{}{}.npz'.format(feature, lidx), x=train_raw, y=labels)
	else:
		train, labels = sample_data(environment, feature)
		# Create raw features
		train_raw = np.empty((0, 97), float)
		for dp in train:
			train_raw = np.vstack((train_raw, environment.raw_features(dp)))
		here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../'))
		np.savez(here+'/data/gtdata/data_{}.npz'.format(feature), x=train_raw, y=labels)
	print "Finished generating data."

if __name__ == '__main__':
	feat = sys.argv[1]
	generate_gt_data(feat)

