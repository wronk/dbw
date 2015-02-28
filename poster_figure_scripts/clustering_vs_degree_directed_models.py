"""
Created on Mon Nov 24 09:17:11 2014

@author: rkp

Plot clustering vs. degree for mouse connectome and standard random graphs.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random_graph.binary_directed import biophysical_reverse_outdegree as biophysical_model

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

import brain_constants as bc

# IMPORT PLOT PARAMETERS
import undirected_plot_config as cf

# PLOTTING PARAMETERS
FACECOLOR = 'black'
COLOR = 'c'

Gnonspatial, _, _ = biophysical_model(N=bc.num_brain_nodes,
                            N_edges=bc.num_brain_edges_directed,
                            L=np.inf,
                            gamma=1.)
                            
Gspatial, _, _ = biophysical_model(N=bc.num_brain_nodes,
                                   N_edges=bc.num_brain_edges_directed,
                                   L=.75,
                                   gamma=1.)

Gs = [Gnonspatial, Gspatial]
colors = [COLOR, COLOR]

fig, axs = plt.subplots(1, 2, figsize=cf.FIGSIZE_DIRECTED_MODELS,
                        facecolor=FACECOLOR, tight_layout=True)

# Plot clustering vs. degree scatter plots
for ctr, ax in enumerate(axs.flatten()):
    deg = nx.degree(Gs[ctr].to_undirected()).values()
    cc = nx.clustering(Gs[ctr].to_undirected()).values()
    
    ax.scatter(deg, cc, s=cf.MARKERSIZE, c=colors[ctr], lw=cf.LW)
    
# Set labels and limits
for ax in axs:
    ax.set_xlabel('Degree')
    ax.set_xlim(cf.XLIMITS)
    ax.set_ylim(cf.YLIMITS)
    ax.set_xticks([0,125,250])
axs[0].set_ylabel('Clustering\ncoefficient')

axs[0].set_title('Random\nattachment')
axs[1].set_title('Proximal\nattachment')

# Set all fontsizes and axis colors
for ax in axs.flatten():
    set_all_text_fontsizes(ax, cf.FONTSIZE)
    set_all_colors(ax, 'w')

plt.draw()

fig.savefig('/Users/rkp/Desktop/directed_clustering_vs_degree.png',
            transparent=True)