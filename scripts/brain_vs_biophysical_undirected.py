"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp

Compare mouse connectivity network to standard random graphs.
"""

# PLOTTING PARAMETERS
FACECOLOR = 'white'
DEGREE_BINS = 20
CLUSTERING_BINS = 20
COLORS = ['k','r','g','c','m','b']

# ANALYSIS PARAMETERS
N_MODEL_GRAPHS = 10
L = 1.3 # Length scale parameter
GAMMAS = [1., 1.5, 1.7, 1.9, 2.0] # Preferential attachment parameter
BRAIN_SIZE = [10., 10, 10] # Size of volume in which brain regions are distributed

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import extract.brain_graph
import random_graph.binary_undirected as rg

# Load mouse connectivity graph
G_brain, W_brain, _ = extract.brain_graph.binary_undirected()
n_nodes = len(G_brain.nodes())
n_edges = len(G_brain.edges())

# Calculate degree & clustering coefficient distribution
degree = nx.degree(G_brain).values()
clustering = nx.clustering(G_brain).values()

model_degree_bins_gammas = [None for gamma in GAMMAS]
model_degree_hist_gammas = [None for gamma in GAMMAS]
model_clustering_bins_gammas = [None for gamma in GAMMAS]
model_clustering_hist_gammas = [None for gamma in GAMMAS]
model_degree_example_gammas = [None for gamma in GAMMAS]
model_clustering_example_gammas = [None for gamma in GAMMAS]

# Loop through choices of gamma
for gamma_idx, gamma in enumerate(GAMMAS):
    # Create several ER graphs and get their degree & clustering coefficients
    print 'Generating model graphs for gamma = %.2f' % gamma
    model_degree_hist = np.zeros((N_MODEL_GRAPHS, DEGREE_BINS))
    model_degree_bins = DEGREE_BINS
    model_clustering_hist = np.zeros((N_MODEL_GRAPHS, CLUSTERING_BINS))
    model_clustering_bins = CLUSTERING_BINS
    for graph_idx in range(N_MODEL_GRAPHS):
        G_model, A_model, D_model = rg.biophysical(n_nodes, n_edges, L, gamma,
                                                   BRAIN_SIZE)
        model_degree = nx.degree(G_model).values()
        model_degree_hist[graph_idx,:], model_degree_bins = \
        np.histogram(model_degree, model_degree_bins, normed=True)
        model_clustering = nx.clustering(G_model).values()
        model_clustering_hist[graph_idx,:], model_clustering_bins = \
        np.histogram(model_clustering, model_clustering_bins, normed=True)
        
    # Take averages
    model_degree_hist = model_degree_hist.mean(axis=0)
    model_clustering_hist = model_clustering_hist.mean(axis=0)
    model_degree_hist_gammas[gamma_idx] = model_degree_hist
    model_degree_bins_gammas[gamma_idx] = model_degree_bins
    model_clustering_hist_gammas[gamma_idx] = model_clustering_hist
    model_clustering_bins_gammas[gamma_idx] = model_clustering_bins
    # Store examples
    model_degree_example_gammas[gamma_idx] = model_degree
    model_clustering_example_gammas[gamma_idx] = model_clustering

# Plot mouse degree & clustering histograms, overlaid with averaged equivalent
# ER & BA histograms
fig_degree, ax_degree = plt.subplots(1, 1, facecolor=FACECOLOR)
ax_degree.hist(degree, bins=DEGREE_BINS, normed=True)
fig_clustering, ax_clustering = plt.subplots(1, 1, facecolor=FACECOLOR)
ax_clustering.hist(clustering, bins=CLUSTERING_BINS, normed=True)
# Plot mouse degree vs. clustering & last ER & BA degree vs. clustering
fig_degvcc, axs_degvcc = plt.subplots(2, 3, facecolor=FACECOLOR)
axs_degvcc[0,0].scatter(degree, clustering)

for gamma_idx, gamma in enumerate(GAMMAS):
    color = COLORS[gamma_idx]
    # Plot degree
    model_degree_hist_gamma = model_degree_hist_gammas[gamma_idx]
    model_degree_bins_gamma = model_degree_bins_gammas[gamma_idx]
    model_degree_centers = .5 * (model_degree_bins_gamma[:-1] + \
    model_degree_bins_gamma[1:])
    ax_degree.plot(model_degree_centers, model_degree_hist_gamma, color=color,
                   lw=3)
    # Plot clustring
    model_clustering_hist_gamma = model_clustering_hist_gammas[gamma_idx]
    model_clustering_bins_gamma = model_clustering_bins_gammas[gamma_idx]
    model_clustering_centers = .5 * (model_clustering_bins_gamma[:-1] + \
    model_clustering_bins_gamma[1:])
    ax_clustering.plot(model_clustering_centers, model_clustering_hist, 
                       color=color, lw=3)
    # Plot degree vs. clustering   
    model_degree_example = model_degree_example_gammas[gamma_idx]
    model_clustering_example = model_clustering_example_gammas[gamma_idx]
    ax_degvcc_idx = np.unravel_index(gamma_idx + 1, (2,3))
    axs_degvcc[ax_degvcc_idx].scatter(model_degree_example,
                                      model_clustering_example, c=color)

# Set labels
ax_degree.set_xlabel('Degree')
ax_degree.set_ylabel('Probability')
ax_clustering.set_xlabel('Clustering coefficient')
ax_clustering.set_ylabel('Probability')

# Set axes
for row_idx, row in enumerate(axs_degvcc):
    for col_idx, ax in enumerate(row):
        ax.set_xlim(0,250)
        ax.set_ylim(0,1)
        if row_idx == 1:
            ax.set_xlabel('Degree')
        if col_idx == 0:
            ax.set_ylabel('Clustering coefficient')