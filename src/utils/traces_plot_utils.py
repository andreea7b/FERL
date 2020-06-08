import numpy as np
import plotly.express as px
import pandas as pd
import os
import math
from openrave_utils import robotToCartesian
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Plot stuff
def angles_to_coords(data, feat, env):
    """
    	Transforms an array of 7D Angle coordinates to xyz coordinates expect for coffee (to x vector of EE rotation vec.)
    """
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
    """
        Plot the ground truth 3D Half-Sphere for a specific feature.
    """
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
    """
        Plot the traces labled with the function values of feature_function.
    """
    output = feature_function(train_data)
    euclidean = angles_to_coords(train_data[:, :7], feat, env)
    df = pd.DataFrame(np.hstack((euclidean, output)))
    fig = px.scatter_3d(df, x=0, y=1, z=2, color=3)
    fig.show()


def plot_learned3D(parent_dir, feature_function, feat, env, raw_dim=97):
    """
        Plot the learned 3D ball over the 10.000 points Test Set in the gt_data
    """
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