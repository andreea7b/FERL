import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from openrave_utils import robotToCartesian
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Check Generalization
def mean_delta_over_epochs(df, title):
	fig, ax = plt.subplots()
	ax.scatter(df['epochs'], df['means'])
	# ax.set(ylim=(0, 2))
	ax.set_title('Generalization ' + title)
	ax.set_xlabel('epochs')
	ax.set_ylabel('mean_delta')
	plt.show()


def consolidate_data(df, filter_outliers, abs_threshold=1e3, z_threshold=3):
	# create vector of all n_trajs
	list_epochs = df['epochs']

	# create a list of delta_dist arrays
	list_of_dist_arrays = []
	n_outliers = []
	for i in df['delta_dist']:
		temp_delta_array = np.array(i)
		# filter outliers
		if filter_outliers == 1:  # using z-score
			n_outliers.append(
				(np.abs((temp_delta_array - temp_delta_array.mean()) / temp_delta_array.std()) > z_threshold).sum())
			temp_delta_array = temp_delta_array[
				(np.abs((temp_delta_array - temp_delta_array.mean()) / temp_delta_array.std()) < z_threshold)]
		if filter_outliers == 2:  # absolute value
			n_outliers.append((np.abs((temp_delta_array - temp_delta_array.mean())) > abs_threshold).sum())
			temp_delta_array = temp_delta_array[(np.abs(temp_delta_array - temp_delta_array.mean()) < abs_threshold)]
		list_of_dist_arrays.append(temp_delta_array)

	return list_of_dist_arrays, list_epochs, n_outliers


def variances(df, title, filter_outliers, y_lim=None, abs_threshold=1e3, z_threshold=3):
	list_of_dist_arrays, list_epochs, _ = consolidate_data(df, filter_outliers, abs_threshold, z_threshold)

	# calculate variances
	variances = []
	for delta_dis in list_of_dist_arrays:
		variances.append(delta_dis.var())

	fig, ax = plt.subplots()
	ax.scatter(list_epochs, variances)
	# ax.set_title('Variances over n_trajectories trained on of 10 samples (zero-meaned)')
	ax.set_xlabel('epochs')
	ax.set_title('Variances ' + title)
	ax.set_ylabel('Variance of deltas across full state space')
	if y_lim is not None:
		ax.set_ylim(y_lim)
	plt.show()


def boxplots(df, title, filter_outliers, y_lim=None, abs_threshold=1e3, z_threshold=3):
	list_of_dist_arrays, list_epochs, _ = consolidate_data(df, filter_outliers, abs_threshold, z_threshold)

	# Boxplot stuff
	fig, ax = plt.subplots()
	ax.set_title('Generalization Boxplots ' + title)
	ax.boxplot(list_of_dist_arrays, positions=list_epochs)
	ax.set_xlabel('epochs')
	ax.set_ylabel('delta_distribution')
	if y_lim is not None:
		ax.set_ylim(y_lim)
	plt.show()


def plot_n_outliers(df, title, filter_outliers, abs_threshold=1e3, z_threshold=3):
	if filter_outliers not in [1, 2]:
		print("not filtering outliers")
		return

	list_of_dist_arrays, list_epochs, n_outliers = consolidate_data(df, filter_outliers, abs_threshold, z_threshold)
	# plot stuff
	fig, ax = plt.subplots()
	ax.scatter(list_epochs, n_outliers)
	# ax.set_title('n_outliers over n_trajectories trained on of 10 samples (zero-meaned)')
	ax.set_title('Outliers ' + title)
	ax.set_xlabel('epochs')
	ax.set_ylabel('n_outliers across full state space')
	plt.show()


# Plot stuff
def angles_to_coords(data, feat, env):
    coords_list = np.empty((0, 3), float)
    for i in range(data.shape[0]):
        waypt = data[i]
        if len(waypt) < 10:
            waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
            waypt[2] += math.pi
        env.robot.SetDOFValues(waypt)
        if feat == "coffee":
            EE_link = env.robot.GetLinks()[7]
            coords_list = np.vstack((coords_list, EE_link.GetTransform()[:3,0]))
        else:
            coords = robotToCartesian(env.robot)
            coords_list = np.vstack((coords_list, coords[6]))
    return coords_list


def plot_gt3D(parent_dir, feat, env):
    data_file = parent_dir + '/data/gtdata/data_{}.npz'.format(feat)
    npzfile = np.load(data_file)
    train = npzfile['x'][:,:7]
    labels = npzfile['y']
    labels = labels.reshape(len(labels), 1)
    euclidean = angles_to_coords(train, feat, env)
    df = pd.DataFrame(np.hstack((euclidean, labels)))
    fig = px.scatter_3d(df, x=0, y=1, z=2, color=3)
    fig.show()


def plot_learned_traj(feature_function, train_data, feat, env):
    output = feature_function(train_data)
    euclidean = angles_to_coords(train_data[:, :7], feat, env)
    df = pd.DataFrame(np.hstack((euclidean, output)))
    fig = px.scatter_3d(df, x=0, y=1, z=2, color=3)
    fig.show()


def plot_learned3D(parent_dir, feature_function, feat, env, raw_dim=97):
    data_file = parent_dir + '/data/gtdata/data_{}.npz'.format(feat)
    npzfile = np.load(data_file)
    train = npzfile['x'][:,:7]
    train_raw = np.empty((0, raw_dim), float)
    for dp in train:
        train_raw = np.vstack((train_raw, env.raw_features(dp)[:raw_dim]))
    labels = feature_function(train_raw)
    euclidean = angles_to_coords(train, feat, env)
    df = pd.DataFrame(np.hstack((euclidean, labels)))
    fig = px.scatter_3d(df, x=0, y=1, z=2, color=3)
    fig.show()


# compute L
def compare_empirical_L(feature_list, titles, parent_dir, feat='table'):
	# compute the empirical L's for all of them
	feature_L_list = []
	for feature in feature_list:
		L_list = compute_L(feat, feature, parent_dir)
		feature_L_list.append(L_list)

	# plot the histograms
	for i, L_list in enumerate(feature_L_list):
		n_bins = 100
		plt.hist(L_list, n_bins)
		plt.title(titles[i] + ': empirical 3D L distribution. MaxL = ' + str(max(L_list)))
		plt.xlabel('counts of occurance')
		plt.ylabel('Empirical estimate of 3D L')
		plt.show()


def compare_L(feature_data_list, titles):
	length = min([len(feat_list[1]) for feat_list in feature_data_list])
	fig = go.Figure()
	for i in range(len(titles)):
		fig.add_trace(go.Scatter(x=np.arange(length), y=feature_data_list[i][1],
								 mode='lines',
								 name=titles[i]))
	fig.update_layout(
		title="Analytical L comparison",
		xaxis_title="training time in epochs",
		yaxis_title="Estimate of Upper Bound on L")
	fig.show()


def compare_losses(feature_data_list, titles):
	length = min([len(feat_list[1]) for feat_list in feature_data_list])
	fig = go.Figure()
	for i in range(len(titles)):
		fig.add_trace(go.Scatter(x=np.arange(length),
								 y=0.1 * np.array(feature_data_list[i][3]) + np.array(feature_data_list[i][2]),
								 mode='lines',
								 name='Test loss ' + titles[i]))
		fig.add_trace(go.Scatter(x=np.arange(length),
								 y=0.1 * np.array(feature_data_list[i][5]) + np.array(feature_data_list[i][4]),
								 mode='lines',
								 name='Train loss ' + titles[i]))

	fig.update_layout(
		title="Losses over time",
		xaxis_title="training time in epochs",
		yaxis_title="Loss values")
	fig.show()


def compute_L(feat, feature, parent_dir):
	data_file = parent_dir + '/data/gtdata/data_{}.npz'.format(feat)
	data = np.load(data_file)['x']
	data_3D = data[:, 88:91]

	N = 10
	nbrs = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(data_3D)
	distances, indices = nbrs.kneighbors(data_3D)

	Ls = np.empty((0, N), float)
	for i in range(len(data)):
		Ls = np.vstack((Ls, np.zeros((1, N))))
		for idx in range(N):
			j = indices[i][idx]
			if not np.array_equal(data_3D[i], data_3D[j]):
				Ls[i][idx] = np.abs(feature.function(data[i]) - feature.function(data[j])) / np.linalg.norm(
					data_3D[i] - data_3D[j])
	return np.max(Ls, axis=1).tolist()