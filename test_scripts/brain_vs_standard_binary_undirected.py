"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp

Compare mouse connectivity network to standard random graphs.
"""

# PLOTTING PARAMETERS
FACECOLOR = 'white'
FONTSIZE = 20
ER_COLOR = 'r'
WS_COLOR = 'g'
BA_COLOR = 'k'
DEGREE_BINS = 20
CLUSTERING_BINS = 20

# ANALYSIS PARAMETERS
N_ER_GRAPHS = 10
N_WS_GRAPHS = 10
N_BA_GRAPHS = 10

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import extract.brain_graph

# Load mouse connectivity graph
G_brain, A_brain, _ = extract.brain_graph.binary_undirected()
n_nodes = len(G_brain.nodes())
n_edges = len(G_brain.edges())
p_edge = float(n_edges) / ((n_nodes * (n_nodes - 1)) / 2)

# Calculate degree & clustering coefficient distribution
degree = nx.degree(G_brain).values()
clustering = nx.clustering(G_brain).values()

degree_mean = np.mean(degree)

# Create several ER graphs and get their degree & clustering coefficients
print 'Generating ER graphs...'
ER_degree_hist = np.zeros((N_ER_GRAPHS, DEGREE_BINS))
ER_degree_bins = DEGREE_BINS
ER_clustering_hist = np.zeros((N_ER_GRAPHS, CLUSTERING_BINS))
ER_clustering_bins = CLUSTERING_BINS
for graph_idx in range(N_ER_GRAPHS):
    G_ER = nx.erdos_renyi_graph(n_nodes, p_edge)
    ER_degree = nx.degree(G_ER).values()
    ER_degree_hist[graph_idx,:], ER_degree_bins = \
    np.histogram(ER_degree, ER_degree_bins, normed=True)
    ER_clustering = nx.clustering(G_ER).values()
    ER_clustering_hist[graph_idx,:], ER_clustering_bins = \
    np.histogram(ER_clustering, ER_clustering_bins, normed=True)
# Take averages
ER_degree_hist = ER_degree_hist.mean(axis=0)
ER_clustering_hist = ER_clustering_hist.mean(axis=0)

# Create several WS graphs and get their degree & clustering coefficients
print 'Generating WS graphs...'
WS_degree_hist = np.zeros((N_WS_GRAPHS, DEGREE_BINS))
WS_degree_bins = DEGREE_BINS
WS_clustering_hist = np.zeros((N_WS_GRAPHS, CLUSTERING_BINS))
WS_clustering_bins = CLUSTERING_BINS
for graph_idx in range(N_WS_GRAPHS):
    G_WS = nx.watts_strogatz_graph(n_nodes,int(round(degree_mean)),0.159)
    WS_degree = nx.degree(G_WS).values()
    WS_degree_hist[graph_idx,:], WS_degree_bins = \
    np.histogram(WS_degree, WS_degree_bins, normed=True)
    WS_clustering = nx.clustering(G_WS).values()
    WS_clustering_hist[graph_idx,:], WS_clustering_bins = \
    np.histogram(WS_clustering, WS_clustering_bins, normed=True)
# Take averages
WS_degree_hist = WS_degree_hist.mean(axis=0)
WS_clustering_hist = WS_clustering_hist.mean(axis=0)

# Create several BA (scale-free) graphs and get their degree & clustering 
# coefficients
print 'Generating BA graphs...'
BA_degree_hist = np.zeros((N_ER_GRAPHS, DEGREE_BINS))
BA_degree_bins = DEGREE_BINS
BA_clustering_hist = np.zeros((N_ER_GRAPHS, CLUSTERING_BINS))
BA_clustering_bins = CLUSTERING_BINS
for graph_idx in range(N_ER_GRAPHS):
    G_BA = nx.barabasi_albert_graph(n_nodes, int(round(degree_mean/2.)))
    BA_degree = nx.degree(G_BA).values()
    BA_degree_hist[graph_idx,:], BA_degree_bins = \
    np.histogram(BA_degree, BA_degree_bins, normed=True)
    BA_clustering = nx.clustering(G_BA).values()
    BA_clustering_hist[graph_idx,:], BA_clustering_bins = \
    np.histogram(BA_clustering, BA_clustering_bins, normed=True)
# Take averages
BA_degree_hist = BA_degree_hist.mean(axis=0)
BA_clustering_hist = BA_clustering_hist.mean(axis=0)

# Plot mouse degree & clustering histograms, overlaid with averaged equivalent
# ER & BA histograms
fig_degree, ax_degree = plt.subplots(1, 1, facecolor=FACECOLOR)
ax_degree.hist(degree, bins=DEGREE_BINS, normed=True)
ER_degree_bins = .5 * (ER_degree_bins[:-1] + ER_degree_bins[1:])
WS_degree_bins = .5 * (WS_degree_bins[:-1] + WS_degree_bins[1:])
BA_degree_bins = .5 * (BA_degree_bins[:-1] + BA_degree_bins[1:])
ax_degree.plot(ER_degree_bins, ER_degree_hist, color=ER_COLOR, lw=3)
ax_degree.plot(WS_degree_bins, WS_degree_hist, color=WS_COLOR, lw=3)
ax_degree.plot(BA_degree_bins, BA_degree_hist, color=BA_COLOR, lw=3)

fig_clustering, ax_clustering = plt.subplots(1, 1, facecolor=FACECOLOR)
ER_clustering_bins = .5 * (ER_clustering_bins[:-1] + ER_clustering_bins[1:])
WS_clustering_bins = .5 * (WS_clustering_bins[:-1] + WS_clustering_bins[1:])
BA_clustering_bins = .5 * (BA_clustering_bins[:-1] + BA_clustering_bins[1:])
ax_clustering.hist(clustering, bins=CLUSTERING_BINS, normed=True)
ax_clustering.plot(ER_clustering_bins, ER_clustering_hist, color=ER_COLOR, lw=3)
ax_clustering.plot(WS_clustering_bins, WS_clustering_hist, color=WS_COLOR, lw=3)
ax_clustering.plot(BA_clustering_bins, BA_clustering_hist, color=BA_COLOR, lw=3)

# Set labels
ax_degree.set_xlabel('Degree', fontsize=FONTSIZE)
ax_degree.set_ylabel('Probability', fontsize=FONTSIZE)

plt.draw()
plt.tight_layout()

ax_clustering.set_xlabel('Clustering coefficient', fontsize=FONTSIZE)
ax_clustering.set_ylabel('Probability', fontsize=FONTSIZE)

plt.draw()
plt.tight_layout()

# Plot mouse degree vs. clustering & last ER & BA degree vs. clustering
fig_degvcc, axs_degvcc = plt.subplots(2, 2, facecolor=FACECOLOR)
axs_degvcc[0,0].scatter(degree, clustering)
axs_degvcc[0,1].scatter(ER_degree, ER_clustering)
axs_degvcc[1,0].scatter(WS_degree, WS_clustering)
axs_degvcc[1,1].scatter(BA_degree, BA_clustering)

# Set axes
for ax in [ax_degree, ax_clustering]:
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(FONTSIZE)
for row_idx, ax_row in enumerate(axs_degvcc):
    for col_idx, ax in enumerate(ax_row):
        ax.set_xlim(0,170)
        ax.set_ylim(0,1)
        if col_idx == 0:
            ax.set_ylabel('Clustering coefficient', fontsize=FONTSIZE)
        if row_idx == 1:
            ax.set_xlabel('Degree', fontsize=FONTSIZE)
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontsize(FONTSIZE)
        
# Set labels
axs_degvcc[0,0].set_title('Mouse brain', fontsize=FONTSIZE)
axs_degvcc[0,1].set_title('Erdos Renyi', fontsize=FONTSIZE)
axs_degvcc[1,0].set_title('Watts Strogatz', fontsize=FONTSIZE)
axs_degvcc[1,1].set_title('Barabasi Albert', fontsize=FONTSIZE)

plt.draw()
plt.tight_layout()