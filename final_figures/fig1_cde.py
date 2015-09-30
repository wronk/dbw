"""
Created on Mon Nov 24 09:17:11 2014

@author: rkp, wronk
clustering_vs_degree_brain_standard_graph.py

Plot clustering vs. degree for mouse connectome and standard random graphs.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import extract.brain_graph
import config
import scipy.stats as stats

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

# PLOTTING PARAMETERS
FACECOLOR = config.FACE_COLOR
FIGSIZE = (12, 3.5)
FONT_SIZE = config.FONT_SIZE
BRAIN_COLOR = config.COLORS['brain']
RAND_COLOR = config.COLORS['configuration']
WS_COLOR = config.COLORS['small-world']
BA_COLOR = config.COLORS['scale-free']
DEG_MAX = 150
DEG_TICKS = [0, 50, 100, 150]
CC_TICKS = [0, .2, .4, .6, .8, 1.0]
graph_names = ['Connectome', 'Random', 'Small-world', 'Scale-free']
labels = ['c', 'd', 'e', 'f']  # Upper corner labels for each plot
plt.ion()
plt.close('all')

########################################################

# Load mouse connectivity graph
G_brain, W_brain, _ = extract.brain_graph.binary_undirected()
n_nodes = len(G_brain.nodes())
n_edges = len(G_brain.edges())
p_edge = float(n_edges) / ((n_nodes * (n_nodes - 1)) / 2.)

# Calculate degree & clustering coefficient distribution
brain_degree = nx.degree(G_brain).values()
brain_clustering = nx.clustering(G_brain).values()
brain_degree_mean = np.mean(brain_degree)

# Build standard graphs & get their degree & clustering coefficient
# Configuration model (random with fixed degree sequence)
#G_CM = nx.random_degree_sequence_graph(brain_degree, tries=100)
CM_degree = nx.degree(G_CM).values()
CM_clustering = nx.clustering(G_CM).values()

# Watts-Strogatz
G_WS = nx.watts_strogatz_graph(n_nodes, int(round(brain_degree_mean)), 0.23)
WS_degree = nx.degree(G_WS).values()
WS_clustering = nx.clustering(G_WS).values()

# Barabasi-Albert
G_BA = nx.barabasi_albert_graph(n_nodes, int(round(brain_degree_mean / 2.)))
BA_degree = nx.degree(G_BA).values()
BA_clustering = nx.clustering(G_BA).values()
##################
# Powerlaw fitting
##################

# TODO: Insert power law fitting here and calculate R^2 vals
reg_brain = stats.linregress(np.log(brain_degree),np.log(brain_clustering))
r_brain = stats.pearsonr(brain_clustering,np.exp(reg_brain[1])*brain_degree**reg_brain[0])
reg_CM = stats.linregress(np.log(CM_degree),np.log(CM_clustering))
r_CM = stats.pearsonr(CM_clustering,np.exp(reg_CM[1])*CM_degree**reg_CM[0])
reg_WS = stats.linregress(np.log(WS_degree),np.log(WS_clustering))
r_WS = stats.pearsonr(WS_clustering,np.exp(reg_WS[1])*WS_degree**reg_WS[0])
reg_BA = stats.linregress(np.log(BA_degree),np.log(BA_clustering))
r_BA = stats.pearsonr(BA_clustering,np.exp(reg_BA[1])*BA_degree**reg_BA[0])



r_squared_vals = [r_brain[0]**2, r_CM[0]**2, r_WS[0]**2, r_BA[0]**2]

############
# Plot
############
# Make clustering vs. degree plots
fig, axs = plt.subplots(1, 4, facecolor=FACECOLOR, figsize=FIGSIZE,
                        tight_layout=True)

# Brain
x = np.linspace(0.01,160,501)
axs[0].scatter(brain_degree, brain_clustering, color=BRAIN_COLOR)
axs[0].plot(x,np.exp(reg_brain[1])*x**reg_brain[0],'k',linestyle='--',lw=3)


# Standard random graphs
axs[1].scatter(CM_degree, CM_clustering, color=RAND_COLOR)
axs[1].plot(x,np.exp(reg_CM[1])*x**reg_CM[0],'k',linestyle='--',lw=3)


axs[2].scatter(WS_degree, WS_clustering, color=WS_COLOR)
axs[2].plot(x,np.exp(reg_WS[1])*x**reg_WS[0],'k',linestyle='--',lw=3)

axs[3].scatter(BA_degree, BA_clustering, color=BA_COLOR)
axs[3].plot(x,np.exp(reg_BA[1])*x**reg_BA[0],'k',linestyle='--',lw=3)


# Set axis limits and ticks, and label subplots
for ax_idx, ax in enumerate(axs.flatten()):
    ax.set_xlim(0, DEG_MAX)
    ax.set_ylim(0, 1)
    ax.set_xticks(DEG_TICKS)
    ax.set_yticks(CC_TICKS)
    ax.text(.08, .87, labels[ax_idx], color='k', fontsize=FONT_SIZE,
            fontweight='bold', transform=ax.transAxes)

    ax.set_xlabel('Degree')
    set_all_text_fontsizes(ax, FONT_SIZE)
    set_all_colors(ax, 'k')

    # Set titles
    ax.set_title(graph_names[ax_idx], fontsize=FONT_SIZE)

    # Hide x ticklabels in top row & y ticklabels in right columns

    if ax_idx == 0:
        ax.set_ylabel('Clustering\ncoefficient')
        ax.text(0.5, 0.85, r'$\mathrm{R^{2}}$ = %.2f' % r_squared_vals[ax_idx],
                color='k', fontsize=FONT_SIZE - 2, transform=ax.transAxes)

    else:

        ax.text(0.725, 0.85, '%0.2f' % r_squared_vals[ax_idx],
                color='k', fontsize=FONT_SIZE - 2, transform=ax.transAxes)

        ax.set_yticklabels('')

fig.subplots_adjust(wspace=0.18)

plt.show()
