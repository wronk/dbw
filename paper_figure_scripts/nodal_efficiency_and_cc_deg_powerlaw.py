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

import scipy.stats

plt.rc('text',usetex=True);# plt.rc('font',family='serif')
# parameters for this particular plot
FIG_SIZE = (15, 6)
X_LIM_EFFICIENCY = (0, 0.8)
Y_LIM_EFFICIENCY = (0, 300)
ALPHA_DEG_VS_CC = 0.7
N_GRAPH_SAMPLES = 1
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
            graphs_pg = data['graphs_pg']
            graphs_pa = data['graphs_pa']
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
        labels = ['er', 'pgpa', 'pref-growth', 'pref-attachment', 'rand']
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
            elif label == 'rand':
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

print('Taking averages and generating plots...')

# calculate mean and std of nodal efficiency
counts_nodal_efficiency_mean_er = np.array([G.counts_nodal_efficiency for G in graphs_er]).mean(axis=0)
counts_nodal_efficiency_std_er = np.array([G.counts_nodal_efficiency for G in graphs_er]).std(axis=0)
counts_nodal_efficiency_mean_pgpa = np.array([G.counts_nodal_efficiency for G in graphs_pgpa]).mean(axis=0)
counts_nodal_efficiency_std_pgpa = np.array([G.counts_nodal_efficiency for G in graphs_pgpa]).std(axis=0)
counts_nodal_efficiency_mean_pg = np.array([G.counts_nodal_efficiency for G in graphs_pg]).mean(axis=0)
counts_nodal_efficiency_std_pg = np.array([G.counts_nodal_efficiency for G in graphs_pg]).std(axis=0)
counts_nodal_efficiency_mean_pa = np.array([G.counts_nodal_efficiency for G in graphs_pa]).mean(axis=0)
counts_nodal_efficiency_std_pa = np.array([G.counts_nodal_efficiency for G in graphs_pa]).std(axis=0)
counts_nodal_efficiency_mean_rand = np.array([G.counts_nodal_efficiency for G in graphs_rand]).mean(axis=0)
counts_nodal_efficiency_std_rand = np.array([G.counts_nodal_efficiency for G in graphs_rand]).std(axis=0)

# calculate nodal efficiency for brain
G_brain.efficiency_matrix = metrics_bd.efficiency_matrix(G_brain)
G_brain.nodal_efficiency = np.sum(G_brain.efficiency_matrix, axis=1) / (len(G_brain.nodes()) - 1)

# plot histograms of all three betweenness and naispl distributions
fig, axs = plt.subplots(1, 2, facecolor=FACE_COLOR, figsize=FIG_SIZE, tight_layout=True)

for a_ctr, ax in enumerate(axs):
    # brain
    if a_ctr == 1:
        x = np.linspace(0,150,501)
        gammas = []
        rs = []
        cc = nx.clustering(G_brain.to_undirected()).values()
        deg = nx.degree(G_brain.to_undirected()).values()
        reg = scipy.stats.linregress(np.log(deg),np.log(cc))
        gammas.append(reg[0])
        rs.append(reg[2])

        #axs[1].plot(x,np.exp(reg[1])*x**reg[0],lw=3,c=COLORS['brain'])


    elif a_ctr == 0:
        hist_connectome = ax.hist(G_brain.nodal_efficiency, bins=BINS_NODAL_EFFICIENCY, color=COLORS['brain'], lw=0)

    if a_ctr == 1:
        Gs = [graphs_er[DEGREE_VS_CLUSTERING_GRAPH_IDX],
              graphs_pgpa[DEGREE_VS_CLUSTERING_GRAPH_IDX],
              graphs_pg[DEGREE_VS_CLUSTERING_GRAPH_IDX],
              graphs_pa[DEGREE_VS_CLUSTERING_GRAPH_IDX],
              graphs_rand[DEGREE_VS_CLUSTERING_GRAPH_IDX]]
        labels = ['er', 'pgpa', 'pref-growth', 'pref-attachment', 'configuration']


        for G, label in zip(Gs, labels):
            cc = np.array(nx.clustering(G.to_undirected()).values())
            cc = cc[~np.isnan(cc)]; cc = cc[cc>0]
            deg = np.array(nx.degree(G.to_undirected()).values())
            deg = deg[~np.isnan(cc)]; deg = deg[cc>0] # discard nans
            reg = scipy.stats.linregress(np.log(deg),np.log(cc))
            #axs[1].plot(x,np.exp(reg[1])*x**reg[0],lw=3,c=COLORS[label])
            gammas.append(reg[0])
            rs.append(reg[2])


        
        xwidth = 5
        labels.insert(0,'brain') # prepend brain label
        x = range(0,len(labels)*xwidth,xwidth)
        ax.bar(x,gammas,color=[COLORS[label] for label in labels],width=4)
        ax.set_xlim(-1, (len(Gs)+1)*xwidth)
        ax.set_xticks(np.array(x)+xwidth/2.)
        ax.set_xticklabels(['brain','ER','PGPA','PG','PA','config'])
        ax.set_ylim(-.6, 0.1)
        ax.set_yticks([-.6,-0.4,-0.2,0])

#        ax.set_xlabel('Network')
        ax.set_ylabel('$\gamma$')

    elif a_ctr == 0:
        # er
        line_er = ax.plot(BINCS_NODAL_EFFICIENCY, counts_nodal_efficiency_mean_er, color=COLORS['er'], lw=3)
        ax.fill_between(BINCS_NODAL_EFFICIENCY,
                        counts_nodal_efficiency_mean_er - counts_nodal_efficiency_std_er,
                        counts_nodal_efficiency_mean_er + counts_nodal_efficiency_std_er,
                        color=COLORS['er'], alpha=0.5)

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
                        color=COLORS['pref-growth'], alpha=0.5)

        # pref-attach
        line_pa = ax.plot(BINCS_NODAL_EFFICIENCY, counts_nodal_efficiency_mean_pa, color=COLORS['pref-attachment'], lw=3)
        ax.fill_between(BINCS_NODAL_EFFICIENCY,
                        counts_nodal_efficiency_mean_pa - counts_nodal_efficiency_std_pa,
                        counts_nodal_efficiency_mean_pa + counts_nodal_efficiency_std_pa,
                        color=COLORS['pref-attachment'], alpha=0.5)

        # deg-controlled rand
        line_rand = ax.plot(BINCS_NODAL_EFFICIENCY, counts_nodal_efficiency_mean_rand, color=COLORS['configuration'], lw=3)
        ax.fill_between(BINCS_NODAL_EFFICIENCY,
                        counts_nodal_efficiency_mean_rand - counts_nodal_efficiency_std_rand,
                        counts_nodal_efficiency_mean_rand + counts_nodal_efficiency_std_rand,
                        color=COLORS['configuration'], alpha=0.5)

        ax.set_xlim(X_LIM_EFFICIENCY)
        ax.set_ylim(Y_LIM_EFFICIENCY)

        ax.set_xlabel('Nodal efficiency')
        ax.set_ylabel('Counts')

lines = [hist_connectome[-1][0], line_er[0], line_pgpa[0], line_pg[0], line_pa[0], line_rand[0]]
labels = ['Connectome', 'Directed ER', 'PGPA', 'Pref-growth',
           'Pref-attach', 'Configuration']
#leg = axs[1].legend(lines, labels, fontsize=FONT_SIZE)

labels = ('a', 'b')
for ax, label in zip(axs, labels):
    change_settings.set_all_colors(ax, AX_COLOR)
    change_settings.set_all_text_fontsizes(ax, FONT_SIZE)
    ax.text(0.05, 0.95, label, fontsize=20, fontweight='bold',
            transform=ax.transAxes,  ha='center', va='center')

plt.draw()
plt.show(block=True)
