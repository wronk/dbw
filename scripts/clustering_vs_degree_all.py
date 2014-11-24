"""
Created on Mon Nov 24 09:17:11 2014

@author: rkp

Plot the clustering vs. degree for mouse connectome, standard random graphs,
and binary undirected biophysical model at multiple gammas.
"""

# PLOTTING PARAMETERS
FACECOLOR = 'white'
FONTSIZE = 24
BRAIN_COLOR = 'k'
ER_COLOR = 'r'
WS_COLOR = 'g'
BA_COLOR = 'b'
MODEL_COLOR = 'c'
DEG_MAX = 250
DEG_TICKS = [0, 125, 250]
CC_TICKS = [0, .2, .4, .6, .8, 1.0]

BRAIN_POS = (0,0)
ER_POS = (0,1)
WS_POS = (1,0)
BA_POS = (1,1)

MODEL_POS = [(0,2), (0,3), (1,2), (1,3)]

# ANALYSIS PARAMETERS
N_MODEL_GRAPHS = 10
LS = [2.2, 2.2, 2.2, 1.7] # Length scale parameter
GAMMAS = [1., 1.333, 1.667, 2.0] # Preferential attachment parameters
BRAIN_SIZE = [7., 7, 7] # Size of volume in which brain regions are distributed

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import extract.brain_graph
import random_graph.binary_undirected as rg

# Load mouse connectivity graph
G_brain, W_brain, _ = extract.brain_graph.binary_undirected()
n_nodes = len(G_brain.nodes())
n_edges = len(G_brain.edges())
p_edge = float(n_edges) / ((n_nodes * (n_nodes - 1)) / 2)

# Calculate degree & clustering coefficient distribution
brain_degree = nx.degree(G_brain).values()
brain_clustering = nx.clustering(G_brain).values()
brain_degree_mean = np.mean(brain_degree)

# Build standard graphs & get their degree & clustering coefficient
# Erdos-Renyi
G_ER = nx.erdos_renyi_graph(n_nodes, p_edge)
ER_degree = nx.degree(G_ER).values()
ER_clustering = nx.clustering(G_ER).values()

# Watts-Strogatz
G_WS = nx.watts_strogatz_graph(n_nodes,int(round(brain_degree_mean)),0.159)
WS_degree = nx.degree(G_WS).values()
WS_clustering = nx.clustering(G_WS).values()

# Barabasi-Albert
G_BA = nx.barabasi_albert_graph(n_nodes, int(round(brain_degree_mean/2.)))
BA_degree = nx.degree(G_BA).values()
BA_clustering = nx.clustering(G_BA).values()


# Loop through model graphs with different gamma
model_degrees = [None for gamma in GAMMAS]
model_clusterings = [None for gamma in GAMMAS]

for gamma_idx, gamma in enumerate(GAMMAS):
    L = LS[gamma_idx]
    print 'Generating model graph for gamma = %.2f' % gamma
    G_model, A_model, D_model = rg.biophysical(n_nodes, n_edges, L, gamma,
                                               BRAIN_SIZE)
    model_degree = nx.degree(G_model).values()
    model_clustering = nx.clustering(G_model).values()
    
    # Store examples
    model_degrees[gamma_idx] = model_degree
    model_clusterings[gamma_idx] = model_clustering
    
# Make 8 clustering vs. degree plots
fig, axs = plt.subplots(2, 4, facecolor=FACECOLOR)

# Brain
axs[BRAIN_POS].scatter(brain_degree, brain_clustering, color=BRAIN_COLOR)

# Standard random graphs
axs[ER_POS].scatter(ER_degree, ER_clustering, color=ER_COLOR)
axs[WS_POS].scatter(WS_degree, WS_clustering, color=WS_COLOR)
axs[BA_POS].scatter(BA_degree, BA_clustering, color=BA_COLOR)

# Model graphs w/ different gammas
for gamma_idx, gamma in enumerate(GAMMAS):
    degree = model_degrees[gamma_idx]
    clustering = model_clusterings[gamma_idx]
    axs[MODEL_POS[gamma_idx]].scatter(degree, clustering, color=MODEL_COLOR)
    
# Set axis limits and ticks, and label subplots
labels = ('a', 'c', 'e', 'g', 'b', 'd', 'f', 'h')
for ax_idx, ax in enumerate(axs.flat):
    ax.set_xlim(0, DEG_MAX)
    ax.set_ylim(0, 1)
    ax.set_xticks(DEG_TICKS)
    ax.set_yticks(CC_TICKS)
    ax.text(10, .88, labels[ax_idx], fontsize=FONTSIZE, fontweight='bold')
    
# Hide x ticklabels in top row & y ticklabels in right columns
for ax in axs[0,:]:
    ax.set_xticklabels('')
for row in axs[:,1:]:
    for ax in row:
        ax.set_yticklabels('')
        
# Set titles
axs[BRAIN_POS].set_title('Mouse', fontsize=FONTSIZE)
axs[ER_POS].set_title('Erdos-Renyi', fontsize=FONTSIZE)
axs[WS_POS].set_title('Small-World', fontsize=FONTSIZE)
axs[BA_POS].set_title('Scale-Free', fontsize=FONTSIZE)
for gamma_idx, gamma in enumerate(GAMMAS):
    title = r'$\gamma$ = %.2f' % gamma
    axs[MODEL_POS[gamma_idx]].set_title(title, fontsize=FONTSIZE)
    
# Set xlabels
for ax in axs[1,:]:
    ax.set_xlabel('Degree', fontsize=FONTSIZE)
for ax in axs[:,0]:
    ax.set_ylabel('Clustering\ncoefficient', fontsize=FONTSIZE)
    
# Adjust fontsizes of ticklabels
for ax in axs.flat:
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(FONTSIZE)
        
plt.draw()