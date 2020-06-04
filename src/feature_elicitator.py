#! /usr/bin/env python
"""
This node demonstrates velocity-based PID control by moving the Jaco so that it
maintains a fixed distance to a target. Additionally, it supports human-robot
interaction in the form of online physical corrections.
The novelty is that it contains a protocol for elicitating new features from
human input provided to the robot.

Author: Andreea Bobu (abobu@eecs.berkeley.edu)
"""
import torch
import roslib; roslib.load_manifest('kinova_demo')

import rospy
import math
import sys, select, os
import time

import kinova_msgs.msg
from kinova_msgs.srv import *

from controllers.pid_controller import PIDController
from planners.trajopt_planner import TrajoptPlanner
from learners.phri_learner import PHRILearner
from utils import ros_utils, openrave_utils, experiment_utils
from utils.environment import Environment
from utils.trajectory import Trajectory

import numpy as np
import pickle

class FeatureElicitator():
	"""
	This class represents a node that moves the Jaco with PID control AND supports receiving human corrections online.
	Additionally, it implements a protocol for elicitating novel features from human input.

	Subscribes to:
		/$prefix$/out/joint_angles	- Jaco sensed joint angles
		/$prefix$/out/joint_torques - Jaco sensed joint torques

	Publishes to:
		/$prefix$/in/joint_velocity	- Jaco commanded joint velocities
	"""

	def __init__(self):
		# Create ROS node.
		rospy.init_node("feature_elicitator")

		# Load parameters and set up subscribers/publishers.
		self.load_parameters()
		self.register_callbacks()

		# Start admittance control mode.
		ros_utils.start_admittance_mode(self.prefix)

		# Publish to ROS at 100hz.
		r = rospy.Rate(100)

		print "----------------------------------"
		print "Moving robot, type Q to quit:"

		while not rospy.is_shutdown():

			if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
				line = raw_input()
				if line == "q" or line == "Q" or line == "quit":
					break

			self.vel_pub.publish(ros_utils.cmd_to_JointVelocityMsg(self.cmd))
			r.sleep()

		print "----------------------------------"

		# Ask whether to save experimental data for pHRI corrections.
		print "Type [yes/y/Y] if you'd like to save experimental data."
		line = raw_input()
		if (line is not "yes") and (line is not "Y") and (line is not "y"):
			print "Not happy with recording. Terminating experiment."
		else:
			print "Please type in the ID number (e.g. [0/1/2/...])."
			ID = raw_input()
			print "Please type in the task number."
			task = raw_input()
			print "Saving experimental data to file..."
			settings_string = "ID" + ID + "_" + self.feat_method + "_" + "_".join(self.environment.feature_list) + "_task" + task
			weights_filename = "weights_" + settings_string
			betas_filename = "betas_" + settings_string
			force_filename = "force_" + settings_string
			interaction_pts_filename = "interaction_pts_" + settings_string
			tracked_filename = "tracked_" + settings_string
			deformed_filename = "deformed_" + settings_string
			deformed_waypts_filename = "deformed_waypts_" + settings_string
			replanned_filename = "replanned_" + settings_string
			replanned_waypts_filename = "replanned_waypts_" + settings_string

			self.expUtil.pickle_weights(weights_filename)
			self.expUtil.pickle_betas(betas_filename)
			self.expUtil.pickle_force(force_filename)
			self.expUtil.pickle_interaction_pts(interaction_pts_filename)
			self.expUtil.pickle_tracked_traj(tracked_filename)
			self.expUtil.pickle_deformed_trajList(deformed_filename)
			self.expUtil.pickle_deformed_wayptsList(deformed_waypts_filename)
			self.expUtil.pickle_replanned_trajList(replanned_filename)
			self.expUtil.pickle_replanned_wayptsList(replanned_waypts_filename)

		ros_utils.stop_admittance_mode(self.prefix)

	def load_parameters(self):
		"""
		Loading parameters and setting up variables from the ROS environment.
		"""
		# ----- General Setup ----- #
		self.prefix = rospy.get_param("setup/prefix")
		pick = rospy.get_param("setup/start")
		place = rospy.get_param("setup/goal")
		self.start = np.array(pick)*(math.pi/180.0)
		self.goal = np.array(place)*(math.pi/180.0)
		self.goal_pose = None if rospy.get_param("setup/goal_pose") == "None" else rospy.get_param("setup/goal_pose")
		self.T = rospy.get_param("setup/T")
		self.timestep = rospy.get_param("setup/timestep")
		self.save_dir = rospy.get_param("setup/save_dir")
		self.INTERACTION_TORQUE_THRESHOLD = rospy.get_param("setup/INTERACTION_TORQUE_THRESHOLD")
		self.INTERACTION_TORQUE_EPSILON = rospy.get_param("setup/INTERACTION_TORQUE_EPSILON")
		self.CONFIDENCE_THRESHOLD = rospy.get_param("setup/CONFIDENCE_THRESHOLD")
		self.N_QUERIES = rospy.get_param("setup/N_QUERIES")
		self.nb_layers = rospy.get_param("setup/nb_layers")
		self.nb_units = rospy.get_param("setup/nb_units")

		# Openrave parameters for the environment.
		model_filename = rospy.get_param("setup/model_filename")
		object_centers = rospy.get_param("setup/object_centers")
		feat_list = rospy.get_param("setup/feat_list")
		weights = rospy.get_param("setup/feat_weights")
		FEAT_RANGE = rospy.get_param("setup/FEAT_RANGE")
		feat_range = [FEAT_RANGE[feat_list[feat]] for feat in range(len(feat_list))]
		self.environment = Environment(model_filename, object_centers, feat_list, feat_range, np.array(weights))

		# ----- Planner Setup ----- #
		# Retrieve the planner specific parameters.
		planner_type = rospy.get_param("planner/type")
		if planner_type == "trajopt":
			max_iter = rospy.get_param("planner/max_iter")
			num_waypts = rospy.get_param("planner/num_waypts")

			# Initialize planner and compute trajectory to track.
			self.planner = TrajoptPlanner(max_iter, num_waypts, self.environment)
		else:
			raise Exception('Planner {} not implemented.'.format(planner_type))

		self.traj = self.planner.replan(self.start, self.goal, self.goal_pose, self.T, self.timestep)
		self.traj_plan = self.traj.downsample(self.planner.num_waypts)

		# Track if you have reached the start/goal of the path.
		self.reached_start = False
		self.reached_goal = False
		self.feature_learning_mode = False
		self.interaction_mode = False

		# Save the intermediate target configuration. 
		self.curr_pos = None

		# Track data and keep stored.
		self.interaction_data = []
		self.interaction_time = []
		self.feature_data = []
		self.track_data = False

		# ----- Controller Setup ----- #
		# Retrieve controller specific parameters.
		controller_type = rospy.get_param("controller/type")
		if controller_type == "pid":
			# P, I, D gains.
			P = rospy.get_param("controller/p_gain") * np.eye(7)
			I = rospy.get_param("controller/i_gain") * np.eye(7)
			D = rospy.get_param("controller/d_gain") * np.eye(7)

			# Stores proximity threshold.
			epsilon = rospy.get_param("controller/epsilon")

			# Stores maximum COMMANDED joint torques.
			MAX_CMD = rospy.get_param("controller/max_cmd") * np.eye(7)

			self.controller = PIDController(P, I, D, epsilon, MAX_CMD)
		else:
			raise Exception('Controller {} not implemented.'.format(controller_type))

		# Planner tells controller what plan to follow.
		self.controller.set_trajectory(self.traj)

		# Stores current COMMANDED joint torques.
		self.cmd = np.eye(7)

		# ----- Learner Setup ----- #
		constants = {}
		constants["step_size"] = rospy.get_param("learner/step_size")
		constants["P_beta"] = rospy.get_param("learner/P_beta")
		constants["alpha"] = rospy.get_param("learner/alpha")
		constants["n"] = rospy.get_param("learner/n")
		self.feat_method = rospy.get_param("learner/type")
		self.learner = PHRILearner(self.feat_method, self.environment, constants)

		# ---- Experimental Utils ---- #
		self.expUtil = experiment_utils.ExperimentUtils(self.save_dir)
		# Update the list of replanned plans with new trajectory plan.
		self.expUtil.update_replanned_trajList(0.0, self.traj_plan.waypts)
		# Update the list of replanned waypoints with new waypoints.
		self.expUtil.update_replanned_wayptsList(0.0, self.traj.waypts)

		# Add learned feature artificially.
		#self.environment.new_learned_feature(self.nb_layers, self.nb_units, checkpoint_name="model_laptop6D.pt")
		

	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""

		# Create joint-velocity publisher.
		self.vel_pub = rospy.Publisher(self.prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)

		# Create subscriber to joint_angles.
		rospy.Subscriber(self.prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)
		# Create subscriber to joint torques
		rospy.Subscriber(self.prefix + '/out/joint_torques', kinova_msgs.msg.JointTorque, self.joint_torques_callback, queue_size=1)

	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate torque command to move the robot to the target.
		"""
		# Read the current joint angles from the robot.
		curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# Convert to radians.
		curr_pos = curr_pos*(math.pi/180.0)

		# Check if we are in feature learning mode.
		if self.feature_learning_mode:
			# Allow the person to move the end effector with no control resistance.
			self.cmd = np.zeros((7,7))

			# If we are tracking feature data, update raw features and time.
			if self.track_data == True:
				# Add recording to feature data.
				self.feature_data.append(self.environment.raw_features(curr_pos))
			return

		# When not in feature learning stage, update position.
		self.curr_pos = curr_pos

		# Update cmd from PID based on current position.
		self.cmd = self.controller.get_command(self.curr_pos)

		# Check is start/goal has been reached.
		if self.controller.path_start_T is not None:
			self.reached_start = True
			self.expUtil.set_startT(self.controller.path_start_T)
		if self.controller.path_end_T is not None:
			self.reached_goal = True
			self.expUtil.set_endT(self.controller.path_end_T)

		# Update the experiment utils executed trajectory tracker.
		if self.reached_start and not self.reached_goal:
			timestamp = time.time() - self.controller.path_start_T
			self.expUtil.update_tracked_traj(timestamp, self.curr_pos)

	def joint_torques_callback(self, msg):
		"""
		Reads the latest torque sensed by the robot and records it for
		plotting & analysis
		"""
		# Read the current joint torques from the robot.
		torque_curr = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))
		interaction = False
		for i in range(7):
			# Center torques around zero.
			torque_curr[i][0] -= self.INTERACTION_TORQUE_THRESHOLD[i]
			# Check if interaction was not noise.
			if np.fabs(torque_curr[i][0]) > self.INTERACTION_TORQUE_EPSILON[i] and self.reached_start:
				interaction = True

		if interaction:
			if self.reached_start and not self.reached_goal:
				timestamp = time.time() - self.controller.path_start_T
				self.interaction_data.append(torque_curr)
				self.interaction_time.append(timestamp)
				if self.interaction_mode == False:
					self.interaction_mode = True
				self.expUtil.update_tauH(timestamp, torque_curr)
				self.expUtil.update_interaction_point(timestamp, self.curr_pos)
		else:
			if self.interaction_mode == True:
				# Check if betas are above CONFIDENCE_THRESHOLD
				betas = self.learner.learn_betas(self.traj, self.interaction_data[0], self.interaction_time[0])
				if max(betas) < self.CONFIDENCE_THRESHOLD:
					# We must learn a new feature that passes CONFIDENCE_THRESHOLD before resuming.
					print "The robot does not understand the input!"
					self.feature_learning_mode = True
					feature_learning_timestamp = time.time()
					input_size = len(self.environment.raw_features(torque_curr))
					self.environment.new_learned_feature(self.nb_layers, self.nb_units)#, checkpoint_name="model_laptop6D.pt")
					while True:
						# Keep asking for input until we're happy.
						#here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../'))
						#with open(here+'/data/model_checkpoints/trajs_table_diverse.p', "rb") as fp:
						#	feature_data = pickle.load(fp)
						for i in range(self.N_QUERIES):
							print "Need more data to learn the feature!"
							self.feature_data = []

							# Request the person to place the robot in a low feature value state.
							print "Place the robot in a low feature value state and press ENTER when ready."
							line = raw_input()
							self.track_data = True

							# Request the person to move the robot to a high feature value state.
							print "Move the robot in a high feature value state and press ENTER when ready."
							line = raw_input()
							self.track_data = False

							# Pre-process the recorded data before training.
							feature_data = np.squeeze(np.array(self.feature_data))
							lo = 0
							hi = feature_data.shape[0] - 1
							while np.linalg.norm(feature_data[lo] - feature_data[lo + 1]) < 0.01 and lo < hi:
								lo += 1
							while np.linalg.norm(feature_data[hi] - feature_data[hi - 1]) < 0.01 and hi > 0:
								hi -= 1
							feature_data = feature_data[lo:hi+1, :]
							print "Collected {} samples out of {}.".format(feature_data.shape[0], len(self.feature_data))

							# Provide optional start and end labels.
							start_label = 0.0
							end_label = 1.0
							print("Would you like to label your start? Press ENTER if not or enter a number from 0-10")
							line = raw_input()
							if line in [str(i) for i in range(11)]:
								start_label = int(i) / 10.0

							print("Would you like to label your goal? Press ENTER if not or enter a number from 0-10")
							line = raw_input()
							if line in [str(i) for i in range(11)]:
								end_label = float(i) / 10.0

							# Add the newly collected data.
							self.environment.learned_features[-1].add_data(feature_data, start_label, end_label)
						import pdb;pdb.set_trace()
						# Train new feature with data of increasing "goodness".
						self.environment.learned_features[-1].train()

						# Check if we're happy with the input.
						print "Are you happy with the training? (yes/no)"
						line = raw_input()
						if line == "yes" or line == "Y" or line == "y":
							break

					# Plot trajectory gradient
					for w in np.linspace(0.0, 10.0, 21):
						self.environment.weights[-1] = w
						c = np.array([1, w/10, 0])
						self.traj = self.planner.replan(self.start, self.goal, self.goal_pose, self.T, self.timestep)
						openrave_utils.plotTraj(self.environment.env, self.environment.robot, self.environment.bodies, self.traj.waypts, size=0.015, color=c.tolist())

					# Compute new beta for the new feature.
					beta_new = self.learner.learn_betas(self.traj, torque_curr, timestamp, [self.environment.num_features - 1])[0]
					betas.append(beta_new)

					# Move time forward to return to interaction position.
					self.controller.path_start_T += (time.time() - feature_learning_timestamp)

				# We do not have misspecification now, so resume reward learning.
				self.feature_learning_mode = False

				# Learn reward.
				for i in range(len(self.interaction_data)):
					self.learner.learn_weights(self.traj, self.interaction_data[i], self.interaction_time[i], betas)
				self.traj = self.planner.replan(self.start, self.goal, self.goal_pose, self.T, self.timestep, seed=self.traj_plan.waypts)
				self.traj_plan = self.traj.downsample(self.planner.num_waypts)
				self.controller.set_trajectory(self.traj)

				# Update the experimental data with new weights and new betas.
				timestamp = time.time() - self.controller.path_start_T
				self.expUtil.update_weights(timestamp, self.environment.weights)
				self.expUtil.update_betas(timestamp, betas)

				# Update the list of replanned plans with new trajectory plan.
				self.expUtil.update_replanned_trajList(timestamp, self.traj_plan.waypts)

				# Update the list of replanned trajectory waypts with new trajectory.
				self.expUtil.update_replanned_wayptsList(timestamp, self.traj.waypts)

				# Store deformed trajectory plan.
				traj_deform = self.traj.deform(torque_curr, timestamp, self.learner.alpha, self.learner.n)
				deformed_traj = traj_deform.downsample(self.planner.num_waypts)
				self.expUtil.update_deformed_trajList(timestamp, deformed_traj.waypts)

				# Store deformed trajectory waypoints.
				self.expUtil.update_deformed_wayptsList(timestamp, traj_deform.waypts)

				# Turn off interaction mode.
				self.interaction_mode = False
				self.interaction_data = []
				self.interaction_time = []

if __name__ == '__main__':
	FeatureElicitator()




