"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp

Compare mouse connectivity network to standard random graphs.
"""

# PLOTTING PARAMETERS
FACECOLOR = 'white'
BIO_COLOR = 'k'
DEGREE_BINS = 20
CLUSTERING_BINS = 20

# ANALYSIS PARAMETERS
N_BIO_GRAPHS = 1
L = 1. # Length scale parameter
GAMMA = 1.5 # Preferential attachment parameter

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import extract.brain_graph
import random_graph.binary_undirected as rg

# Load mouse connectivity graph
G_brain, W_brain, _, _ = extract.brain_graph.binary_undirected()
n_nodes = len(G_brain.nodes())
n_edges = len(G_brain.edges())

# Calculate degree & clustering coefficient distribution
degree = nx.degree(G_brain).values()
clustering = nx.clustering(G_brain).values()

# Create several ER graphs and get their degree & clustering coefficients
print 'Generating biophysical graphs...'
BIO_degree_hist = np.zeros((N_BIO_GRAPHS, DEGREE_BINS))
BIO_degree_bins = DEGREE_BINS
BIO_clustering_hist = np.zeros((N_BIO_GRAPHS, CLUSTERING_BINS))
BIO_clustering_bins = CLUSTERING_BINS
for graph_idx in range(N_BIO_GRAPHS):
    G_BIO, A_BIO, D_BIO = rg.biophysical(n_nodes, n_edges, L=L, gamma=GAMMA)
    BIO_degree = nx.degree(G_BIO).values()
    BIO_degree_hist[graph_idx,:], BIO_degree_bins = \
    np.histogram(BIO_degree, BIO_degree_bins, normed=True)
    BIO_clustering = nx.clustering(G_BIO).values()
    BIO_clustering_hist[graph_idx,:], BIO_clustering_bins = \
    np.histogram(BIO_clustering, BIO_clustering_bins, normed=True)
# Take averages
BIO_degree_hist = BIO_degree_hist.mean(axis=0)
BIO_clustering_hist = BIO_clustering_hist.mean(axis=0)


# Plot mouse degree & clustering histograms, overlaid with averaged equivalent
# ER & BA histograms
fig_degree, ax_degree = plt.subplots(1, 1, facecolor=FACECOLOR)
ax_degree.hist(degree, bins=DEGREE_BINS, normed=True)
BIO_degree_bins = .5 * (BIO_degree_bins[:-1] + BIO_degree_bins[1:])
ax_degree.plot(BIO_degree_bins, BIO_degree_hist, color=BIO_COLOR, lw=3)

fig_clustering, ax_clustering = plt.subplots(1, 1, facecolor=FACECOLOR)
BIO_clustering_bins = .5 * (BIO_clustering_bins[:-1] + BIO_clustering_bins[1:])
ax_clustering.hist(clustering, bins=CLUSTERING_BINS, normed=True)
ax_clustering.plot(BIO_clustering_bins, BIO_clustering_hist, color=BIO_COLOR, 
                   lw=3)

# Set labels
ax_degree.set_xlabel('Degree')
ax_degree.set_ylabel('Probability')

ax_clustering.set_xlabel('Clustering coefficient')
ax_clustering.set_ylabel('Probability')

# Plot mouse degree vs. clustering & last ER & BA degree vs. clustering
fig_degvcc, axs_degvcc = plt.subplots(1, 2, facecolor=FACECOLOR)
axs_degvcc[0].scatter(degree, clustering)
axs_degvcc[1].scatter(BIO_degree, BIO_clustering)

# Set axes
for ax_idx, ax in enumerate(axs_degvcc):
    ax.set_xlim(0,170)
    ax.set_ylim(0,1)
    if ax_idx == 0:
        ax.set_ylabel('Clustering coefficient')
    ax.set_xlabel('Degree')
        
# Set labels
axs_degvcc[0].set_title('Mouse brain')
axs_degvcc[1].set_title('Biophysical')