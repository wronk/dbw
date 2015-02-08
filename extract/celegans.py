"""
Created on Sat Feb 7 16:23:43 2014

@author: rkp

Functions to extract celegans connectome from mat file.
"""

import networkx as nx
import scipy.io as sio

CELEGANS_FILE = '../data/celegans277.mat'

def binary_directed():
    """Make directed graph from C. elegans adjacency matrix."""
    
    # load all data
    D = sio.loadmat(CELEGANS_FILE)
    
    # get adjacency matrix
    A = D['celegans277matrix']
    
    # create networkx graph
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    
    return G