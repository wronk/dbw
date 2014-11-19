"""
Created on Wed Nov 19 13:04:50 2014

@author: rkp

Compare degree, clustering, & degree vs. clustering in either brain or cortex.
"""

# PLOT PARAMETERS
FACECOLOR = 'white'

import extract.brain_graph
import extract.brain_subgraph

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G_brain, A_brain, _ = extract.brain_graph.binary_undirected()
D_brain, _ = extract.brain_graph.distance_matrix()
G_cortex, A_cortex, _ = extract.brain_subgraph.cortex_binary_undirected()
D_cortex, _ = extract.brain_subgraph.cortex_distance_matrix()

# Get edge distance distribution
edge_dist_brain = D_brain[np.triu(A_brain, k=1) > 0]
edge_dist_cortex = D_cortex[np.triu(A_cortex, k=1) > 0]

# Get degree & clustering coefficients
degree_brain = np.array(nx.degree(G_brain).values())
clustering_brain = np.array(nx.clustering(G_brain).values())
degree_cortex = np.array(nx.degree(G_cortex).values())
clustering_cortex = np.array(nx.clustering(G_cortex).values())

# Plot edge distance distribution
fig, ax = plt.subplots(1, 1, facecolor=FACECOLOR)
ax.hist([edge_dist_brain, edge_dist_cortex], bins=20, normed=True)
ax.set_xlabel('Edge distance')
ax.set_ylabel('Probability')

# Plot degree distributions
fig, ax = plt.subplots(1, 1, facecolor=FACECOLOR)
ax.hist([degree_brain, degree_cortex], bins=20, normed=True)
ax.set_xlabel('Degree')
ax.set_ylabel('Probability')

# Plot clustering coefficients
fig, ax = plt.subplots(1, 1, facecolor=FACECOLOR)
ax.hist([clustering_brain, clustering_cortex], bins=20, normed=True)
ax.set_xlabel('Clustering')
ax.set_ylabel('Probability')

# Plot degree vs. clustering
fig, axs = plt.subplots(1, 2, facecolor=FACECOLOR)
axs[0].scatter(degree_brain, clustering_brain)
axs[0].set_title('Brain')
axs[1].scatter(degree_cortex, clustering_cortex)
axs[1].set_title('Cortex')
for ax in axs:
    ax.set_xlim(0,150)
    ax.set_ylim(0,1)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Clustering')