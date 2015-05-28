"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp

Functions to extract graphs from Allen mouse connectivity.
"""

LINEAR_MODEL_DIRECTORY = '../../friday-harbor/linear_model'
STRUCTURE_DIRECTORY = '../../mouse_connectivity_data'

import numpy as np
import scipy.io as sio
import aux_random_graphs
#from friday_harbor.structure import Ontology

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
    labels = col_labels_L + col_labels_R

    return W, P, labels


def load_centroids(labels, data_dir=STRUCTURE_DIRECTORY, in_mm=True):
    """Load centroids."""
    centroids = aux_random_graphs.get_coords()
    centroidsMat = np.zeros([len(labels),3])
    for i,node in enumerate(labels):
        centroidsMat[i,:] = centroids[node]

    if in_mm:
        centroidsMat /= 10.

    return centroidsMat

'''
# This function doesn't work for me because I don't have
# friday_harbor.Ontology...
def mask_specific_structures(structure_list, parent_structures=['CTX'],
                             data_dir=STRUCTURE_DIRECTORY):
    """Return mask for specific structures in super structure.

    Returns boolean mask where True values are structures in structure_list that
    are substructures of parent_structure.

    Args:
        structure_list: list of structure names (with _L or _R appended)
        parent_structure: parent structure
    Returns:
        boolean mask of same length as structure_list"""
    onto = Ontology(data_dir=data_dir)

    # Make sure parent_structures is list
    if not isinstance(parent_structures,list):
        parent_structures = [parent_structures]

    # Get ids of parent structures
    parent_ids = [onto.structure_by_acronym(structure).structure_id \
    for structure in parent_structures]

    # Get ancestors of each structure in structure_list
    ancestors_list = [onto.structure_by_acronym(structure[:-2]).path_to_root \
    for structure in structure_list]

    # Get boolean mask of which structures have ancestors in parent_ids
    mask = [bool(set(parent_ids) & set(ancestors)) \
    for ancestors in ancestors_list]

    return np.array(mask)
'''
