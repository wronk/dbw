"""
Created on Sun Apr 19 15:59:40 2015

@author: wronk

Do progressive percolation of brain and plot k-Core at each step.
"""

import networkx as nx
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import network_plot.network_viz as n_viz
import area_compute

import extract.brain_graph

###############################
# Graphing params
###############################
MAX_MARKER_SIZE = 20.

###############################
# Create graph/ compute metrics
###############################

# Extract brain graph
G_brain, _, G_brain_labels = extract.brain_graph.binary_undirected(p_th=0.1)
G_positions = area_compute.get_centroids(G_brain_labels)

###############################
# Percolation parameters
###############################

percentiles = range(10, 35)
deg_tiers = np.round(np.percentile(G_brain.degree().values(), percentiles))

core_nums = nx.core_number(G_brain)
kCores = [nx.k_core(G_brain, k=k) for k in range(10, 35)]

#core_vals = core_nums.values() - (np.min(core_nums.values()) - 1)
node_sizes = MAX_MARKER_SIZE * np.array(core_nums.values()) / np.max(core_nums.values())
node_colors = ['blue'] * G_brain.number_of_nodes()
edges = None

#TODO: Also check this method when using precomputed core_numbers as it may be
# more efficient

###############################
# Plot series of k-core graphs
###############################
plt.ion()
figsize = (6, 4)

fig = plt.figure(figsize=figsize)
ax1 = fig.add_subplot('111', projection='3d')

for ni, node_pt in enumerate(G_positions):
    ax1.scatter(node_pt[0], node_pt[1], node_pt[2],
                s=node_sizes[ni])

n_viz.plot_3D_network(ax1,
                      node_names=G_brain_labels,
                      node_positions=G_positions,
                      node_label_set=[False] * G_brain.number_of_nodes(),
                      node_sizes=node_sizes,
                      node_colors=node_colors,
                      edges=[])

#fig2 = plt.figure(figsize=(12, 4))
#for core_G in enumerate(kCores):

plt.show()
