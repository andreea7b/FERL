import numpy as np
import math
import json
import copy
import torch

import trajoptpy

from utils.openrave_utils import *
from utils.trajectory import Trajectory


class TrajoptPlanner(object):
	"""
	This class plans a trajectory from start to goal with TrajOpt, given
	features and feature weights (optionally).
	"""
	def __init__(self, max_iter, num_waypts, environment):

		# These variables are trajopt parameters.
		self.MAX_ITER = max_iter
		self.num_waypts = num_waypts

		# Set OpenRAVE environment.
		self.environment = environment

	# -- Interpolate feature value between neighboring waypoints to help planner optimization. -- #

	def interpolate_features(self, waypt, prev_waypt, feat_idx, NUM_STEPS=4):
		"""
		Computes feature value over waypoints, interpolating and
		sampling between each pair to check for intermediate collisions.
		---
		input neighboring waypoints and feature function, output scalar feature
		"""
		feat_val = 0.0
		for step in range(NUM_STEPS):
			inter_waypt = prev_waypt + ((1.0 + step)/NUM_STEPS)*(waypt - prev_waypt)
			feat_val += self.environment.featurize_single(inter_waypt, feat_idx)
		return feat_val / NUM_STEPS

	# ---- Costs ---- #

	def efficiency_cost(self, waypt):
		"""
		Computes the total efficiency cost
		---
		input waypoint, output scalar cost
		"""
		feature_idx = self.environment.feature_list.index('efficiency')
		feature = self.interpolate_features(waypt, waypt, feature_idx, NUM_STEPS=1)
		return feature*self.environment.weights[feature_idx]

	def origin_cost(self, waypt):
		"""
		Computes the total distance from EE to base of robot cost.
		---

		input waypoint, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature_idx = self.environment.feature_list.index('origin')
		feature = self.interpolate_features(curr_waypt, prev_waypt, feature_idx)
		return feature*self.environment.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

	def table_cost(self, waypt):
		"""
		Computes the total distance to table cost.
		---
		input waypoint, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature_idx = self.environment.feature_list.index('table')
		feature = self.interpolate_features(curr_waypt, prev_waypt, feature_idx)
		return feature*self.environment.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

	def coffee_cost(self, waypt):
		"""
		Computes the total coffee (EE orientation) cost.
		---
		input waypoint, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature_idx = self.environment.feature_list.index('coffee')
		feature = self.interpolate_features(curr_waypt, prev_waypt, feature_idx)
		return feature*self.environment.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

	def laptop_cost(self, waypt):
		"""
		Computes the total distance to laptop cost
		---
		input waypoint, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature_idx = self.environment.feature_list.index('laptop')
		feature = self.interpolate_features(curr_waypt, prev_waypt, feature_idx)
		return feature*self.environment.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

	def human_cost(self, waypt):
		"""
		Computes the total distance to human cost.
		---
		input waypoint, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature_idx = self.environment.feature_list.index('human')
		feature = self.interpolate_features(curr_waypt, prev_waypt, feature_idx)
		return feature*self.environment.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

	def learned_feature_costs(self, waypt):
		"""
		Computes the cost for all the learned features.
		---
		input waypoint, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		# get the number of learned features
		n_learned = self.environment.feature_list.count('learned_feature')
	
		feature_values = []
		for i, feature in enumerate(self.environment.learned_features):
			# get the value of the feature
			feat_idx = self.environment.num_features - n_learned + i
			feature_values.append(self.interpolate_features(curr_waypt, prev_waypt, feat_idx))
		# calculate the cost
		return np.matmul(self.environment.weights[-n_learned:], np.array(feature_values))*np.linalg.norm(curr_waypt - prev_waypt)
	
	def learned_feature_cost_derivatives(self, waypt):
		"""
		Computes the cost derivatives for all the learned features.
		---
		input waypoint, output scalar cost
		"""
		# get the number of learned features
		n_learned = self.environment.feature_list.count('learned_feature')
	
		J = []
		sols = []
		for i, feature in enumerate(self.environment.learned_features):
			# Setup for computing Jacobian.
			x = torch.tensor(waypt, requires_grad=True)
	
			# Get the value of the feature
			feat_idx = self.environment.num_features - n_learned + i
			feat_val = torch.tensor(0.0, requires_grad=True)
			NUM_STEPS = 4
			for step in range(NUM_STEPS):
				delta = torch.tensor((1.0 + step)/NUM_STEPS, requires_grad=True)
				inter_waypt = x[:7] + delta * (x[7:] - x[:7])
				# Compute feature value.
				z = self.environment.feature_func_list[feat_idx](self.environment.raw_features(inter_waypt).float(), torchify=True)
				feat_val = feat_val + z
			y = feat_val / torch.tensor(float(NUM_STEPS), requires_grad=True)
			y = y * torch.tensor(self.environment.weights[-n_learned+i:], requires_grad=True) * torch.norm(x[7:] - x[:7])
			y.backward()
			J.append(x.grad.data.numpy())
		return np.sum(np.array(J), axis = 0).reshape((1,-1))

	# ---- Here's TrajOpt --- #

	def trajOpt(self, start, goal, goal_pose, traj_seed=None):
		"""
		Computes a plan from start to goal using trajectory optimizer.
		Reference: http://joschu.net/docs/trajopt-paper.pdf
		---
		Paramters:
			start -- The start position.
			goal -- The goal position.
			goal_pose -- The goal pose (optional: can be None).
			traj_seed [optiona] -- An optional initial trajectory seed.

		Returns:
			waypts_plan -- A downsampled trajectory resulted from the TrajOpt
			optimization problem solution.
		"""

		# --- Initialization --- #
		if len(start) < 10:
			aug_start = np.append(start.reshape(7), np.array([0, 0, 0]))
		self.environment.robot.SetDOFValues(aug_start)

		# --- Linear interpolation seed --- #
		if traj_seed is None:
			# print("Using straight line initialization!")
			init_waypts = np.zeros((self.num_waypts, 7))
			for count in range(self.num_waypts):
				init_waypts[count, :] = start + count/(self.num_waypts - 1.0)*(goal - start)
		else:
			print("Using trajectory seed initialization!")
			init_waypts = traj_seed

		# --- Request construction --- #
		# If pose is given, must include pose constraint.
		if goal_pose is not None:
			# print("Using goal pose for trajopt computation.")
			xyz_target = goal_pose
			quat_target = [1, 0, 0, 0]  #wxyz
			constraint = [
				{
					"type": "pose",
					"params": {"xyz" : xyz_target,
								"wxyz" : quat_target,
								"link": "j2s7s300_link_7",
								"rot_coeffs" : [0, 0, 0],
								"pos_coeffs" : [35, 35, 35],
								}
				}
			]
		else:
			# print("Using goal for trajopt computation.")
			constraint = [
				{
					"type": "joint",
					"params": {"vals": goal.tolist()}
				}
			]

		request = {
			"basic_info": {
				"n_steps": self.num_waypts,
				"manip" : "j2s7s300",
				"start_fixed" : True,
				"max_iter" : self.MAX_ITER
			},
			"costs": [
			{
				"type": "joint_vel",
				"params": {"coeffs": [0.22]}
			}
			],
			"constraints": constraint,
			"init_info": {
				"type": "given_traj",
				"data": init_waypts.tolist()
			}
		}

		s = json.dumps(request)
		prob = trajoptpy.ConstructProblem(s, self.environment.env)
		for t in range(1, self.num_waypts):
			if 'coffee' in self.environment.feature_list:
				prob.AddCost(self.coffee_cost, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "coffee%i"%t)
			if 'table' in self.environment.feature_list:
				prob.AddCost(self.table_cost, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "table%i"%t)
			if 'laptop' in self.environment.feature_list:
				prob.AddCost(self.laptop_cost, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "laptop%i"%t)
			if 'origin' in self.environment.feature_list:
				prob.AddCost(self.origin_cost, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "origin%i"%t)
			if 'human' in self.environment.feature_list:
				prob.AddCost(self.human_cost, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "human%i"%t)
			if 'efficiency' in self.environment.feature_list:
				prob.AddCost(self.efficiency_cost, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "efficiency%i"%t)
			if 'learned_feature' in self.environment.feature_list:
				prob.AddErrorCost(self.learned_feature_costs, self.learned_feature_cost_derivatives, [(t-1, j) for j in range(7)]+[(t, j) for j in range(7)], "ABS", "learned_features%i"%t)
				# [(t-1, j) for j in range(7)]+
		for t in range(1, self.num_waypts - 1):
			prob.AddConstraint(self.environment.table_constraint, [(t, j) for j in range(7)], "INEQ", "up%i"%t)

		result = trajoptpy.OptimizeProblem(prob)
		return result.GetTraj()

	def replan(self, start, goal, goal_pose, T, timestep, seed=None):
		"""
		Replan the trajectory from start to goal given weights.
		---
		Parameters:
			start -- Start position
			goal -- Goal position.
			goal_pose -- Goal pose (optional: can be None).
			T [float] -- Time horizon for the desired trajectory.
			timestep [float] -- Frequency of waypoints in desired trajectory.
		Returns:
			traj [Trajectory] -- The optimal trajectory satisfying the arguments.
		"""
		waypts = self.trajOpt(start, goal, goal_pose, traj_seed=seed)
		waypts_time = np.linspace(0.0, T, self.num_waypts)
		traj = Trajectory(waypts, waypts_time)
		return traj.upsample(int(T/timestep) + 1)

