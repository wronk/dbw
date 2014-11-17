"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp

Functions to extract graphs from Allen mouse connectivity.
"""

LINEAR_MODEL_DIRECTORY = '../../friday-harbor/linear_model'
CENTROID_DIRECTORY = '../../mouse_connectivity_data'

import numpy as np
import scipy.io as sio
from friday_harbor.structure import Ontology

def load_W_and_P(data_dir=LINEAR_MODEL_DIRECTORY):
    """Load weight and p-value matrices."""

    # Load files
    D_W_ipsi = sio.loadmat(data_dir + '/W_ipsi.mat')
    D_W_contra = sio.loadmat(data_dir + '/W_contra.mat')
    D_PValue_ipsi = sio.loadmat(data_dir + '/PValue_ipsi.mat')
    D_PValue_contra = sio.loadmat(data_dir + '/PValue_contra.mat')

    # Make weight matrix for each side, then concatenate them
    W_L = np.concatenate([D_W_ipsi['data'], D_W_contra['data']], 1)
    W_R = np.concatenate([D_W_contra['data'], D_W_ipsi['data']], 1)
    W = np.concatenate([W_L, W_R], 0)
    # Make p_value matrix in the same manner
    P_L = np.concatenate([D_PValue_ipsi['data'], D_PValue_contra['data']], 1)
    P_R = np.concatenate([D_PValue_contra['data'], D_PValue_ipsi['data']], 1)
    P = np.concatenate([P_L, P_R], 0)

    col_labels = D_W_ipsi['col_labels']
    # Add ipsi & contra to col_labels
    col_labels_L = [label.split(' ')[0] + '_L' for label in col_labels]
    col_labels_R = [label.split(' ')[0] + '_R' for label in col_labels]
    col_labels_full = col_labels_L + col_labels_R
    row_labels_full = col_labels_full[:]

    return W, P, row_labels_full, col_labels_full
    
def load_centroids(labels, data_dir=CENTROID_DIRECTORY, in_mm=True):
    """Load centroids."""
    
    onto = Ontology(data_dir=data_dir)
    centroids = np.zeros((len(labels),3),dtype=float)
    for a_idx,area in enumerate(labels):
        s_id = onto.structure_by_acronym(area[:-2]).structure_id
        if area[-1] == 'L':
            mask = onto.get_mask_from_id_left_hemisphere_nonzero(s_id)
        elif area[-1] == 'R':
            mask = onto.get_mask_from_id_right_hemisphere_nonzero(s_id)
        centroids[a_idx,:] = mask.centroid
    if in_mm:
        centroids /= 10.
        
    return centroids