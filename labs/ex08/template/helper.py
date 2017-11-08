# -*- coding: utf-8 -*-
"""Some helper functions."""
import os
import shutil
import numpy as np
from scipy import misc


def load_data():
    """Load data and convert it to the metrics system."""
    path_dataset = "faithful.csv"
    data = np.loadtxt(path_dataset, delimiter=" ", skiprows=0)
    return data


def normalize_data(data):
    """normalize the data by (x - mean(x)) / std(x)."""
    mean_data = np.mean(data, axis=0)
    data = data - mean_data
    std_data = np.std(data)
    data = data / std_data
    return data


def build_dir(dir):
    """build a new dir. if it exists, remove it and build a new one."""
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def load_image(path):
    """use the scipy.misc to load the image."""
    return misc.imread(path)


#def build_distance_matrix(data, mu):
#    """build a distance matrix.
#
#    row of the matrix represents the data point,
#    column of the matrix represents the k-th cluster.
#    """
#    distance_list = []
#    num_cluster, _ = mu.shape
#    for k_th in range(num_cluster):
#        sum_squares = np.sum(np.square(data - mu[k_th, :]), axis=1)
#        distance_list.append(sum_squares)
#    return np.matrix(distance_list).T


def build_distance_matrix(data, mu):
    """build a distance matrix.
    return
        distance matrix:
            row of the matrix represents the data point,
            column of the matrix represents the k-th cluster.
    """
    
    number_of_datapoints = data.shape[0]
    number_of_k = mu.shape[0]
    
    distance_matrix = np.empty([number_of_datapoints,number_of_k])
    
    for datapoint_index in range(number_of_datapoints):
        for k_index in range(number_of_k):
            
            datapoint = data[datapoint_index]
            mu_val = mu[k_index]
            distance = np.linalg.norm(datapoint - mu_val)
            distance_matrix[datapoint_index][k_index] = distance
            
    return np.matrix(distance_matrix)