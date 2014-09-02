import pdb
"""
Created on Mon Sep  1 17:14:07 2014

@author: rkp

Calculate bidirectionality coefficients of node pairs.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stats

import network_gen
import network_compute
import DataBrowser

plt.close('all')

from friday_harbor.structure import Ontology

# Get ontological dictionary with acronyms as keys
DATA_DIR = '../data'
ONTO = Ontology(data_dir=DATA_DIR)

## Make network
# Set parameters
p_th = .01 # P-value threshold
w_th = 0 # Weight-value threshold

# Set relative directory path
dir_name = '../friday-harbor/linear_model'

# Load weights & p-values
W,P,row_labels,col_labels = network_gen.load_weights(dir_name)
# Threshold weights according to weights & p-values
W_net,mask = network_gen.threshold(W,P,p_th=p_th,w_th=w_th)
# Set weights to zero if they don't satisfy threshold criteria
W_net[W_net==-1] = 0.
# Set diagonal weights to zero
np.fill_diagonal(W_net,0)

## Calculate coordinates of all areas
centroids = np.zeros((len(row_labels),3))
for a_idx,area in enumerate(row_labels):
    # Get structure for this area
    s = ONTO.structure_by_acronym(area[:-2])
    s_id = s.structure_id # id
    if area[-1] == 'L':
            mask = ONTO.get_mask_from_id_left_hemisphere_nonzero(s_id)
    elif area[-1] == 'R':
        mask = ONTO.get_mask_from_id_right_hemisphere_nonzero(s_id)
    centroids[a_idx,:] = mask.centroid
    
## Calculate bidirectional metrics
edges,bd_metrics = network_compute.bidirectional_metrics(W_net,centroids,row_labels)
# Remove unidirectional nodes
ud_idxs = bd_metrics[:,1] == 0
f0 = bd_metrics[~ud_idxs,0]
f1 = bd_metrics[~ud_idxs,1]
edges = [edges[idx] for idx in range(len(edges)) if not ud_idxs[idx]]
names = [[ONTO.structure_by_acronym(edge[0][:-2]).name,
          ONTO.structure_by_acronym(edge[1][:-2]).name]
          for edge in edges]

data_dict = {idx:{'acronym0':edges[idx][0],
                  'acronym1':edges[idx][1],
                  'name0':names[idx][0],
                  'name1':names[idx][1]} for idx in range(len(edges))}

# Build data browser
db = DataBrowser.DataBrowser(f0,f1,data_dict,data_type='edge')
fig,axs = plt.subplots(1,2,facecolor='w')
line = axs[0].plot(f0,f1,'o',picker=5)
axs[0].set_xlim(-1,110)
axs[0].set_ylim(-.1,1.1)
axs[0].set_xlabel('distance')
axs[0].set_ylabel('BDC')
# Turn on GUI
db.set_GUI(fig,axs,line)
fig.canvas.mpl_connect('pick_event', db.onclick)
plt.show()