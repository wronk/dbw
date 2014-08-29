import pdb
"""
Created on Fri Aug 29 00:31:34 2014

@author: rkp

Functions for a 2D visualization of a network.
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import tsne

from friday_harbor.structure import Ontology
# Get ontological dictionary with acronyms as keys
DATA_DIR = '../data'
ONTO = Ontology(data_dir=DATA_DIR)
# Load linear model to get labels
LM_DIR = '../friday-harbor/linear_model'
DW_IPSI = sio.loadmat(LM_DIR + '/W_ipsi.mat')
AREAS = [label.split(' ')[0] for label in DW_IPSI['col_labels']]

def get_area_centroids_2D():
    """Return a dictionary of tsne-determined 2D brain area centroids."""
    # Get all structure ids
    s_ids = [ONTO.structure_by_acronym(area).structure_id \
             for area in AREAS]
    # Get centroids
    ctrds_L = [ONTO.get_mask_from_id_left_hemisphere_nonzero(s_id).centroid
               for s_id in s_ids]
    ctrds_R = [ONTO.get_mask_from_id_right_hemisphere_nonzero(s_id).centroid
               for s_id in s_ids]
    centroids = np.concatenate([np.array(ctrds_L),np.array(ctrds_R)],0)
    
    # Get lateralized names
    areas_L = [area + '_L' for area in AREAS]
    areas_R = [area + '_R' for area in AREAS]
    areas_LR = areas_L + areas_R
    
    # Run tsne
    centroids_2D = tsne.tsne(centroids,2,max_iter=1000)
    
    # Align symmetrically
    centroids_2D = sym_align(centroids_2D,areas_LR)
    return areas_LR, centroids_2D
    
def sym_align(centroids_2D,areas_LR,num_angles=3500):
    """Align a set of 2D points so they are symmetric about the y-axis."""
    # Center around center of mass
    centroids_2D_aligned = centroids_2D - centroids_2D.mean()
    # Attempt 360 rotations
    rot_angs = np.linspace(0,2*np.pi,num_angles+1)[:-1]
    # Calculate symmetry distance for each rotation
    sym_dist = np.zeros((len(rot_angs),))
    for rot_idx,ang in enumerate(rot_angs):
        rot_mat = np.array([[np.cos(ang),-np.sin(ang)],
                            [np.sin(ang),np.cos(ang)]])
        rotated_centroids = centroids_2D_aligned.dot(rot_mat.T)
        sym_dist[rot_idx] = sym(rotated_centroids,1)
    # Find best rotation
    best_rot_ang = rot_angs[sym_dist.argmin()]
    # Rotate centroids
    best_rot_mat = np.array([[np.cos(best_rot_ang),-np.sin(best_rot_ang)],
                             [np.sin(best_rot_ang),np.cos(best_rot_ang)]])
    centroids_2D_aligned_rotated = centroids_2D_aligned.dot(best_rot_mat.T)
    return centroids_2D_aligned_rotated
    
def sym(xy,axis=0):
    """Return symmetry of 2D data set about certain axis."""
    # Generate flipped data
    flipped = xy.copy()
    if axis == 0:
        flipped[:,1] *= -1.
    elif axis == 1:
        flipped[:,0] *= -1.
    # Calculate euclidian distance between original and flipped data
    dist = np.sqrt(((xy - flipped)**2).sum())
    return dist
    
if __name__ == '__main__':
#    areas_LR, centroids_2D = get_area_centroids_2D()
    plt.scatter(centroids_2D[:213,0],centroids_2D[:213,1],c='r')
    plt.scatter(centroids_2D[213:,0],centroids_2D[213:,1],c='k')