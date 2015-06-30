"""
Calculate the node betweenness and node-averaged inverse shortest path length distributions for the brain and the ER and PGPA models (the latter two averaged over several instantiations).
"""
from __future__ import print_function, division

import os
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
plt.ion()

from extract import brain_graph
from random_graph.binary_directed import biophysical_reverse_outdegree as pgpa
from random_graph.binary_directed import biophysical_indegree as pa
from metrics import binary_directed as metrics_bd
from network_plot import change_settings

import brain_constants as bc
from config.graph_parameters import LENGTH_SCALE
from config import COLORS, FACE_COLOR, AX_COLOR, FONT_SIZE

# parameters for this particular plot
FIG_SIZE = (15, 6)
N_GRAPH_SAMPLES = 100
BINS_BTWN = np.linspace(0, .03, 25)
BINCS_BTWN = 0.5 * (BINS_BTWN[:-1] + BINS_BTWN[1:])

BINS_NAISPL = np.linspace(0, 1, 25)
BINCS_NAISPL = 0.5 * (BINS_NAISPL[:-1] + BINS_NAISPL[1:])

SAVE_FILE_NAME = 'er_and_pgpa_graphs_with_betweenness_and_efficiency.pickle'

# look for file containing graphs
data_dir = os.getenv('DBW_DATA_DIRECTORY')
save_file_path = os.path.join(data_dir, SAVE_FILE_NAME)

if os.path.isfile(save_file_path):
    print('File "{}" found.'.format(SAVE_FILE_NAME))
    print('Attempting to open...')
    try:
        with open(save_file_path, 'rb') as f:
            data = pickle.load(f)
            graphs_er = data['graphs_er']
            graphs_pgpa = data['graphs_pgpa']
            graphs_pg = data['graphs_pg']
            graphs_pa = data['graphs_pa']
    except Exception, e:
        raise IOError('Error loading data from file "{}"'.format(SAVE_FILE_NAME))
    else:
        print('File loaded successfully!')
else:
    print('looping over {} graph instantiations...'.format(N_GRAPH_SAMPLES))

    graphs_er = []
    graphs_pgpa = []
    graphs_pg = []
    graphs_pa = []

    for g_ctr in range(N_GRAPH_SAMPLES):

        # create directed ER graph
        G_er = nx.erdos_renyi_graph(bc.num_brain_nodes,
                                    bc.p_brain_edge_directed,
                                    directed=True)

        # create pgpa graph
        G_pgpa, _, _ = pgpa(N=bc.num_brain_nodes, N_edges=bc.num_brain_edges_directed, L=LENGTH_SCALE)

        # create preferential growth only graph
        G_pg, _, _ = pgpa(N=bc.num_brain_nodes, N_edges=bc.num_brain_edges_directed, L=np.inf)

        # create preferential attachment graph
        G_pa, _, _ = pa(N=bc.num_brain_nodes, N_edges=bc.num_brain_edges_directed, L=np.inf)

        # calculate things for both types of graph
        for label, G in zip(['er', 'pgpa', 'pref-growth', 'pref-attachment'], [G_er, G_pgpa, G_pg, G_pa]):
            G.betweenness = nx.betweenness_centrality(G)
            G.counts_betweenness = np.histogram(G.betweenness.values(), bins=BINS_BTWN)[0]

            G.efficiency_matrix = metrics_bd.efficiency_matrix(G)
            # naispl stands for node-averaged inverse shortest path length
            naispl = np.sum(G.efficiency_matrix, axis=1) / (len(G.nodes()) - 1)
            G.counts_naispl = np.histogram(naispl, bins=BINS_NAISPL)[0]

            if label == 'er':
                graphs_er += [G]
            elif label == 'pgpa':
                graphs_pgpa += [G]
            elif label == 'pref-growth':
                graphs_pg += [G]
            elif label == 'pref-attachment':
                graphs_pa += [G]

        if (g_ctr + 1) % 1 == 0:
            print('{} of {} samples completed.'.format(g_ctr + 1, N_GRAPH_SAMPLES))

    # save file so that we don't have to remake everything
    print('Saving file to disk...')
    with open(save_file_path, 'wb') as f:
        save_dict = {'graphs_er': graphs_er,
                     'graphs_pgpa': graphs_pgpa,
                     'graphs_pg': graphs_pg,
                     'graphs_pa': graphs_pa}
        pickle.dump(save_dict, f)
    print('File "{}" saved successfully in directory "{}"'.format(SAVE_FILE_NAME, data_dir))

print('Taking averages and generating plots...')

# calculate mean and std of betweenness counts for er and pgpa
counts_betweenness_mean_er = np.array([G.counts_betweenness for G in graphs_er]).mean(axis=0)
counts_betweenness_std_er = np.array([G.counts_betweenness for G in graphs_er]).std(axis=0)
counts_betweenness_mean_pgpa = np.array([G.counts_betweenness for G in graphs_pgpa]).mean(axis=0)
counts_betweenness_std_pgpa = np.array([G.counts_betweenness for G in graphs_pgpa]).std(axis=0)
counts_betweenness_mean_pg = np.array([G.counts_betweenness for G in graphs_pg]).mean(axis=0)
counts_betweenness_std_pg = np.array([G.counts_betweenness for G in graphs_pg]).std(axis=0)
counts_betweenness_mean_pa = np.array([G.counts_betweenness for G in graphs_pa]).mean(axis=0)
counts_betweenness_std_pa = np.array([G.counts_betweenness for G in graphs_pa]).std(axis=0)

# calculate mean and std of naispl
counts_naispl_mean_er = np.array([G.counts_naispl for G in graphs_er]).mean(axis=0)
counts_naispl_std_er = np.array([G.counts_naispl for G in graphs_er]).std(axis=0)
counts_naispl_mean_pgpa = np.array([G.counts_naispl for G in graphs_pgpa]).mean(axis=0)
counts_naispl_std_pgpa = np.array([G.counts_naispl for G in graphs_pgpa]).std(axis=0)
counts_naispl_mean_pg = np.array([G.counts_naispl for G in graphs_pg]).mean(axis=0)
counts_naispl_std_pg = np.array([G.counts_naispl for G in graphs_pg]).std(axis=0)
counts_naispl_mean_pa = np.array([G.counts_naispl for G in graphs_pa]).mean(axis=0)
counts_naispl_std_pa = np.array([G.counts_naispl for G in graphs_pa]).std(axis=0)

# load brain graph
G_brain, _, _ = brain_graph.binary_directed()

# calculate betweenness and naispl for brain
G_brain.betweenness = nx.betweenness_centrality(G_brain)
G_brain.efficiency_matrix = metrics_bd.efficiency_matrix(G_brain)
G_brain.naispl = np.sum(G_brain.efficiency_matrix, axis=1) / (len(G_brain.nodes()) - 1)

# plot histograms of all three betweenness and naispl distributions
fig, axs = plt.subplots(1, 2, facecolor=FACE_COLOR, figsize=FIG_SIZE, tight_layout=True)

for a_ctr, ax in enumerate(axs):
    # brain
    if a_ctr == 0:
        ax.hist(G_brain.betweenness.values(), bins=BINS_BTWN, color=COLORS['brain'], lw=0)
    elif a_ctr == 1:
        ax.hist(G_brain.naispl, bins=BINS_NAISPL, color=COLORS['brain'], lw=0)

    if a_ctr == 0:
        # er
        ax.plot(BINCS_BTWN, counts_betweenness_mean_er, color=COLORS['er'], lw=3)
        ax.fill_between(BINCS_BTWN, counts_betweenness_mean_er - counts_betweenness_std_er,
                        counts_betweenness_mean_er + counts_betweenness_std_er,
                        color=COLORS['er'], alpha=0.5)

        # pgpa
        ax.plot(BINCS_BTWN, counts_betweenness_mean_pgpa, color=COLORS['pgpa'], lw=3)
        ax.fill_between(BINCS_BTWN, counts_betweenness_mean_pgpa - counts_betweenness_std_pgpa,
                        counts_betweenness_mean_pgpa + counts_betweenness_std_pgpa,
                        color=COLORS['pgpa'], alpha=0.5)

        # pref-growth
        ax.plot(BINCS_BTWN, counts_betweenness_mean_pg, color=COLORS['pref-growth'], lw=3)
        ax.fill_between(BINCS_BTWN, counts_betweenness_mean_pg - counts_betweenness_std_pg,
                        counts_betweenness_mean_pg + counts_betweenness_std_pg,
                        color=COLORS['pref-growth'], alpha=0.5)

        # pref-attach
        ax.plot(BINCS_BTWN, counts_betweenness_mean_pa, color=COLORS['pref-attachment'], lw=3)
        ax.fill_between(BINCS_BTWN, counts_betweenness_mean_pa - counts_betweenness_std_pa,
                        counts_betweenness_mean_pa + counts_betweenness_std_pa,
                        color=COLORS['pref-attachment'], alpha=0.5)

        ax.set_xlim(0, 0.03)
        ax.set_ylim(0, 270)

        ax.set_xlabel('node betweenness')
        ax.set_ylabel('counts')

        ax.legend(['Directed ER', 'PGPA', 'Pref-growth', 'Pref-attach', 'Connectome'], fontsize=FONT_SIZE)

    elif a_ctr == 1:
        # er
        ax.plot(BINCS_NAISPL, counts_naispl_mean_er, color=COLORS['er'], lw=3)
        ax.fill_between(BINCS_NAISPL, counts_naispl_mean_er - counts_naispl_std_er,
                        counts_naispl_mean_er + counts_naispl_std_er,
                        color=COLORS['er'], alpha=0.5)

        # pgpa
        ax.plot(BINCS_NAISPL, counts_naispl_mean_pgpa, color=COLORS['pgpa'], lw=3)
        ax.fill_between(BINCS_NAISPL, counts_naispl_mean_pgpa - counts_naispl_std_pgpa,
                        counts_naispl_mean_pgpa + counts_naispl_std_pgpa,
                        color=COLORS['pgpa'], alpha=0.5)

        # pref-growth
        ax.plot(BINCS_NAISPL, counts_naispl_mean_pg, color=COLORS['pref-growth'], lw=3)
        ax.fill_between(BINCS_NAISPL, counts_naispl_mean_pg - counts_naispl_std_pg,
                        counts_naispl_mean_pg + counts_naispl_std_pg,
                        color=COLORS['pref-growth'], alpha=0.5)

        # pref-attach
        ax.plot(BINCS_NAISPL, counts_naispl_mean_pa, color=COLORS['pref-attachment'], lw=3)
        ax.fill_between(BINCS_NAISPL, counts_naispl_mean_pa - counts_naispl_std_pa,
                        counts_naispl_mean_pa + counts_naispl_std_pa,
                        color=COLORS['pref-attachment'], alpha=0.5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 270)

        ax.set_xlabel('nodal efficiency')
        ax.set_ylabel('counts')

        ax.legend()

for ax in axs:
    change_settings.set_all_colors(ax, AX_COLOR)
    change_settings.set_all_text_fontsizes(ax, FONT_SIZE)

plt.draw()
plt.show(block=True)