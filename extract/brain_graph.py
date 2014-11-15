import pdb
"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp

Functions to extract graphs from Allen mouse connectivity.
"""

LINEAR_MODEL_DIRECTORY = '../../friday-harbor/linear_model'
CENTROID_DIRECTORY = '../../mouse_connectivity_data'

import auxiliary as aux
import graph_tools.auxiliary as aux_tools
import networkx as nx
import numpy as np


def binary_undirected(p_th=.01, w_th=0, data_dir=LINEAR_MODEL_DIRECTORY):
    """Load brain as binary undirected graph.
    
    Returns:
        NetworkX graph, adjacency matrix, row labels & column labels"""
    # Load weights & p-values
    W, P, row_labels, col_labels = aux.load_W_and_P(data_dir)
    # Threshold weights via weights & p-values
    W[(W < w_th)] = 0.
    W[(P > p_th)] = 0.
    # Symmetrize W by summing reciprocal weights
    W = W + W.T
    # Set self-weights to zero
    np.fill_diagonal(W,0.)
    # Create adjacency matrix
    A = (W > 0).astype(int)
    
    # Create graph from adjacency matrix
    G = nx.from_numpy_matrix(A)
    
    return G, A, row_labels, col_labels
    

def weighted_undirected(p_th=.01, w_th=0, data_dir=LINEAR_MODEL_DIRECTORY):
    """Load brain as binary undirected graph.
    
    Returns:
        NetworkX graph, weight matrix, row labels & column labels"""
    # Load weights & p-values
    W, P, row_labels, col_labels = aux.load_W_and_P(data_dir=data_dir)

    # Threshold weights via weights & p-values
    W[(W < w_th)] = 0.
    W[(P > p_th)] = 0.
    # Symmetrize W by summing reciprocal weights
    W = W + W.T
    
    # Set self-weights to zero
    np.fill_diagonal(W,0.)
    
    # Create graph from weight matrix
    G = nx.from_numpy_matrix(W)
    
    return G, W, row_labels, col_labels


def distance_matrix(lm_dir=LINEAR_MODEL_DIRECTORY, cent_dir=CENTROID_DIRECTORY):
    """Compute distance matrix from centroid data.
    
    Returns:
        distance matrix, centroid matrix"""
    # Get labels
    _, _, row_labels, _ = aux.load_W_and_P(data_dir=lm_dir)
    # Load centroids
    centroids = aux.load_centroids(row_labels,data_dir=cent_dir)
    # Compute distance matrix
    dist_mat = aux_tools.dist_mat(centroids, in_100um=True)
    
    return dist_mat, centroids