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
from random_graph.binary_directed import random_directed_deg_seq
from metrics import binary_directed as metrics_bd
from network_plot import change_settings

import brain_constants as bc
from config.graph_parameters import LENGTH_SCALE
from config import COLORS, FACE_COLOR, AX_COLOR, FONT_SIZE

# parameters for this particular plot
FIG_SIZE = (15, 6)
INSET_COORDINATES = (0.27, 0.6, 0.2, 0.3)
X_LIM_EFFICIENCY = (0, 1.)
Y_LIM_EFFICIENCY = (0, 130)
ALPHA_DEG_VS_CC = 0.7
N_GRAPH_SAMPLES = 100
DEGREE_VS_CLUSTERING_GRAPH_IDX = 0

BINS_NODAL_EFFICIENCY = np.linspace(0, 1, 25)
BINCS_NODAL_EFFICIENCY = 0.5 * (BINS_NODAL_EFFICIENCY[:-1] + BINS_NODAL_EFFICIENCY[1:])

SAVE_FILE_NAME = 'er_and_pgpa_graphs_with_efficiency.pickle'


# load brain graph
G_brain, _, _ = brain_graph.binary_directed()

# look for file containing graphs
data_dir = os.path.join(os.getenv('DBW_DATA_DIRECTORY'), 'graphs')
save_file_path = os.path.join(data_dir, SAVE_FILE_NAME)

if os.path.isfile(save_file_path):
    print('File "{}" found.'.format(SAVE_FILE_NAME))
    print('Attempting to open...')
    try:
        with open(save_file_path, 'rb') as f:
            data = pickle.load(f)
            graphs_er = data['graphs_er']
            graphs_pgpa = data['graphs_pgpa']
            graphs_pa = data['graphs_pa']
            graphs_pg = data['graphs_pg']
            graphs_rand = data['graphs_rand']
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
    graphs_rand = []  # approximate degree-controlled random graphs

    in_degree_brain = G_brain.in_degree().values()
    out_degree_brain = G_brain.out_degree().values()

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

        # create directed degree-controlled random graph
        G_rand, _, _ = random_directed_deg_seq(in_sequence=in_degree_brain,
                                               out_sequence=out_degree_brain,
                                               simplify=True)

        # calculate things for both types of graph
        labels = ['er', 'pgpa', 'pref-growth', 'pref-attachment', 'random']
        Gs = [G_er, G_pgpa, G_pg, G_pa, G_rand]
        for label, G in zip(labels, Gs):

            G.efficiency_matrix = metrics_bd.efficiency_matrix(G)
            # calculate nodal efficiency
            nodal_efficiency = np.sum(G.efficiency_matrix, axis=1) / (len(G.nodes()) - 1)
            G.counts_nodal_efficiency = np.histogram(nodal_efficiency, bins=BINS_NODAL_EFFICIENCY)[0]

            if label == 'er':
                graphs_er += [G]
            elif label == 'pgpa':
                graphs_pgpa += [G]
            elif label == 'pref-growth':
                graphs_pg += [G]
            elif label == 'pref-attachment':
                graphs_pa += [G]
            elif label == 'random':
                graphs_rand += [G]

        if (g_ctr + 1) % 1 == 0:
            print('{} of {} samples completed.'.format(g_ctr + 1, N_GRAPH_SAMPLES))

    # save file so that we don't have to remake everything
    print('Saving file to disk...')
    with open(save_file_path, 'wb') as f:
        save_dict = {'graphs_er': graphs_er,
                     'graphs_pgpa': graphs_pgpa,
                     'graphs_pg': graphs_pg,
                     'graphs_pa': graphs_pa,
                     'graphs_rand': graphs_rand}
        pickle.dump(save_dict, f)
    print('File "{}" saved successfully in directory "{}"'.format(SAVE_FILE_NAME, data_dir))

names = ('er', 'pgpa', 'pref-growth', 'pref-attachment', 'random')
graphss = (graphs_er, graphs_pgpa, graphs_pg, graphs_pa, graphs_rand)

print('Taking averages and generating plots...')

# calculate mean and std of nodal efficiency
counts_nodal_efficiency_mean_pgpa = np.array([G.counts_nodal_efficiency for G in graphs_pgpa]).mean(axis=0)
counts_nodal_efficiency_std_pgpa = np.array([G.counts_nodal_efficiency for G in graphs_pgpa]).std(axis=0)
counts_nodal_efficiency_mean_pg = np.array([G.counts_nodal_efficiency for G in graphs_pg]).mean(axis=0)
counts_nodal_efficiency_std_pg = np.array([G.counts_nodal_efficiency for G in graphs_pg]).std(axis=0)
counts_nodal_efficiency_mean_rand = np.array([G.counts_nodal_efficiency for G in graphs_rand]).mean(axis=0)
counts_nodal_efficiency_std_rand = np.array([G.counts_nodal_efficiency for G in graphs_rand]).std(axis=0)

# calculate nodal efficiency for brain
G_brain.efficiency_matrix = metrics_bd.efficiency_matrix(G_brain)
G_brain.nodal_efficiency = np.sum(G_brain.efficiency_matrix, axis=1) / (len(G_brain.nodes()) - 1)

# calculate power-law fits for each graph type and brain
power_law_fits = {}
fits_r_squared = {}
for name, graphs in zip(names, graphss):
    gammas = []
    r_squareds = []
    for graph in graphs:
        fit = metrics_bd.power_law_fit_deg_cc(graph)
        gammas.append(fit[0])
        r_squareds.append(fit[2] ** 2)

    power_law_fits[name] = np.array(gammas)
    fits_r_squared[name] = np.array(r_squareds)

power_law_fits['brain'] = metrics_bd.power_law_fit_deg_cc(G_brain)[0]

# plot clustering vs. degree and nodal_efficiencies for brain and three models
fig, axs = plt.subplots(1, 2, facecolor=FACE_COLOR, figsize=FIG_SIZE, tight_layout=True)

for a_ctr, ax in enumerate(axs):
    # brain
    if a_ctr == 0:
        cc = nx.clustering(G_brain.to_undirected()).values()
        deg = nx.degree(G_brain.to_undirected()).values()
        ax.scatter(deg, cc, lw=0, alpha=ALPHA_DEG_VS_CC, c=COLORS['brain'])

    elif a_ctr == 1:
        hist_connectome = ax.hist(G_brain.nodal_efficiency, bins=BINS_NODAL_EFFICIENCY, color=COLORS['brain'], lw=0)

    if a_ctr == 0:
        Gs = [graphs_pgpa[DEGREE_VS_CLUSTERING_GRAPH_IDX],
              graphs_pg[DEGREE_VS_CLUSTERING_GRAPH_IDX],
              graphs_rand[DEGREE_VS_CLUSTERING_GRAPH_IDX]]
        labels = ['pgpa', 'pref-growth', 'random']

        for G, label in zip(Gs, labels):
            cc = nx.clustering(G.to_undirected()).values()
            deg = nx.degree(G.to_undirected()).values()

            ax.scatter(deg, cc, lw=0, alpha=ALPHA_DEG_VS_CC, c=COLORS[label])

        ax.set_xlim(0, 150)
        ax.set_xticks((0, 50, 100, 150))
        ax.set_ylim(0, 1)
        ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))

        ax.set_xlabel('Degree')
        ax.set_ylabel('Clustering coefficient')

    elif a_ctr == 1:

        # pgpa
        line_pgpa = ax.plot(BINCS_NODAL_EFFICIENCY, counts_nodal_efficiency_mean_pgpa, color=COLORS['pgpa'], lw=3, zorder=100)
        ax.fill_between(BINCS_NODAL_EFFICIENCY,
                        counts_nodal_efficiency_mean_pgpa - counts_nodal_efficiency_std_pgpa,
                        counts_nodal_efficiency_mean_pgpa + counts_nodal_efficiency_std_pgpa,
                        color=COLORS['pgpa'], alpha=0.5, zorder=100)

        # pref-growth
        line_pg = ax.plot(BINCS_NODAL_EFFICIENCY, counts_nodal_efficiency_mean_pg, color=COLORS['pref-growth'], lw=3)
        ax.fill_between(BINCS_NODAL_EFFICIENCY,
                        counts_nodal_efficiency_mean_pg - counts_nodal_efficiency_std_pg,
                        counts_nodal_efficiency_mean_pg + counts_nodal_efficiency_std_pg,
                        color=COLORS['pref-growth'], alpha=0.5, zorder=100)

        # deg-controlled rand
        line_rand = ax.plot(BINCS_NODAL_EFFICIENCY, counts_nodal_efficiency_mean_rand, color=COLORS['random'], lw=3)
        ax.fill_between(BINCS_NODAL_EFFICIENCY,
                        counts_nodal_efficiency_mean_rand - counts_nodal_efficiency_std_rand,
                        counts_nodal_efficiency_mean_rand + counts_nodal_efficiency_std_rand,
                        color=COLORS['random'], alpha=0.5, zorder=100)

        ax.set_xlim(X_LIM_EFFICIENCY)
        ax.set_ylim(Y_LIM_EFFICIENCY)

        ax.set_xlabel('Nodal efficiency')
        ax.set_ylabel('Number of nodes')

lines = [line_rand[0], line_pg[0], line_pgpa[0], hist_connectome[-1][0]]
labels = ['Random', 'PG', 'PGPA', 'Connectome']
axs[1].legend(lines, labels, fontsize=FONT_SIZE)

labels = ('a', 'b')
for ax, label in zip(axs, labels):
    change_settings.set_all_colors(ax, AX_COLOR)
    change_settings.set_all_text_fontsizes(ax, FONT_SIZE)
    ax.text(0.05, 0.95, label, fontsize=20, fontweight='bold',
            transform=ax.transAxes,  ha='center', va='center')

# add inset with power-law fit bar plot
ax_inset = fig.add_axes(INSET_COORDINATES)
bar_names = ('random', 'pref-growth', 'pgpa')
gamma_means = [power_law_fits[name].mean() for name in bar_names]
gamma_stds = [power_law_fits[name].std() for name in bar_names]
gamma_median_r_squareds = [np.median(fits_r_squared[name]) for name in names]
colors = [COLORS[name] for name in bar_names]
bar_width = .8
x_pos = np.arange(len(bar_names)) - bar_width/2
error_kw = {'ecolor': 'k', 'elinewidth': 2, 'markeredgewidth': 2, 'capsize': 6}

ax_inset.bar(x_pos, gamma_means, width=bar_width, color=colors, yerr=gamma_stds, error_kw=error_kw)
ax_inset.bar([-bar_width/2 + 3], power_law_fits['brain'], width=bar_width, color=COLORS['brain'])
ax_inset.set_xticks(np.arange(len(bar_names) + 1))
ax_inset.set_xticklabels(['Random', 'PG', 'PGPA', 'Connectome'], rotation='vertical')

ax_inset.set_yticks([0, -0.2, -0.4, -0.6])
ax_inset.set_ylabel(r'$\gamma$')

change_settings.set_all_colors(ax_inset, AX_COLOR)
change_settings.set_all_text_fontsizes(ax_inset, FONT_SIZE)

print(zip(names, gamma_median_r_squareds))

plt.draw()
plt.show(block=True)