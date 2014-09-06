import pdb
"""
Created on Wed Aug 27 23:17:01 2014

@author: wronk

lesion specific nodes and analyze network properties
"""

import numpy as np
import matplotlib.pyplot as plt

import collect_areas
import network_gen
import area_compute
import network_viz
import area_plot
import plot_net
import network_compute
from copy import deepcopy
import networkx as nx
import aux_random_graphs
import scipy.io as sio
from os import path as op
import pickle

# Network generation parameters
p_th = .01  # P-value threshold
w_th = 0  # Weight-value threshold

# Set relative directory path to linear model & ontology
dir_LM = '../friday-harbor/linear_model'
movie_save_path = './movie/images'
stat_save_path = './cache'

calc_features = True
show_example_plots = False
show_whole_stats = False
make_movie = True
show_area_stats = False

network_type = 'allen'
###################################
### Create network
if network_type is 'allen':
    # Load weights & p-values
    W, P, row_labels, col_labels = network_gen.load_weights(dir_LM)
    # Threshold weights according to weights & p-values
    W_net, mask = network_gen.threshold(W, P, p_th=p_th, w_th=w_th)

    # Set weights to zero if they don't satisfy threshold criteria
    W_net[W_net == -1] = 0.
    # Set diagonal weights to zero
    np.fill_diagonal(W_net, 0)

    # Put everything in a dictionary
    net_dict = {'row_labels': row_labels, 'col_labels': col_labels,
                'data': W_net}
    # Create networkx graph
    G = network_gen.import_weights_to_graph(net_dict)

    W_net = nx.adjacency_matrix(G, nodelist=row_labels).toarray()
    net_dict['data'] = W_net

elif network_type == 'biophysical':
    n = 426
    row_labels = range(n)
    col_labels = range(n)
    G, W_net, _ = aux_random_graphs.biophysical_graph(n, N_edges=7804,
                                                      L=1, power=1.5, mode=0)

    # Put everything in a dictionary
    net_dict = {'row_labels': row_labels, 'col_labels': col_labels,
                'data': W_net}
else:
    n = 426
    row_labels = range(n)
    col_labels = range(n)

    # Create networkx graph
    if network_type == 'powerlaw_cluster':
        temp_G = nx.powerlaw_cluster_graph(n=n, m=19, p=1)
    elif network_type == 'scale_free':
        temp_G = nx.barabasi_albert_graph(n=n, m=19)
    elif network_type == 'random':
        temp_G = nx.erdos_renyi_graph(n, 0.123)
    elif network_type == 'small_world':  # small_world
        temp_G = nx.watts_strogatz_graph(n, 36, 0.159)
    else:
        print 'Network type not recognized'

    # Set weights for all the egdes
    wts = {}
    for e in temp_G.edges():
        wts[e] = 1.
    nx.set_edge_attributes(temp_G, 'weight', wts)

    W_net = nx.adjacency_matrix(temp_G, nodelist=row_labels).toarray()

    # Put everything in a dictionary and convert back to graph
    net_dict = {'row_labels': row_labels, 'col_labels': col_labels,
                'data': W_net}
    G = network_gen.import_weights_to_graph(net_dict)

###################################

# Collect & sort areas & edges according to various attributes
sorted_areas = collect_areas.collect_and_sort(G, W_net, labels=row_labels,
                                              print_out=False)
###################################
### Lesion areas
# Set number of lesions
lesion_is_node = True  # Set if node or edge lesion

targeted_attack = True
# Find areas to lesion. node_btwn, ccoeff, degree (append with _labels)
lesion_attr = 'degree_labels'
if not targeted_attack:
    lesion_attr = 'random'
bilateral = False
num_lesions = 30

###################################

# Record pre-lesioned network statistics
#lesion_results = [area_compute.get_feature_dicts(G.nodes(), G, W_net,
#                                                 row_labels)]

graph_list = [deepcopy(G)]
net_dict_list = [deepcopy(net_dict)]
lesion_list_labels = []
graph_stats = [network_compute.whole_graph_metrics(G)]

print 'Model: ' + network_type + '\nTargeted: ' + str(targeted_attack) + \
    '\nby: ' + lesion_attr + ' x' + str(num_lesions)
# Lesion areas
for i in range(num_lesions):
    if lesion_is_node:
        # Find target indices (relative to weight matrix)
        # Unilateral  0:1, 1:2, 2:3
        # Bilateral   0:2, 2:4, 4:6
        if targeted_attack:
            targets = [l for l in
                       sorted_areas[lesion_attr][i * (bilateral + 1):
                                                 (i + 1) * (bilateral + 1)]]
        else:
            targets = np.random.choice(sorted_areas['degree_labels'],
                                       size=(1 + bilateral), replace=False)
            for t in targets:
                sorted_areas['degree_labels'].remove(t)

        # Call lesion function, update weight mat
        W_lesion_dict = network_gen.lesion_node(net_dict_list[-1], targets)
        lesion_list_labels.extend(targets)
        print 'Removed ' + str(targets) + ', Weight matrix size: ' + \
            str(W_lesion_dict['data'].shape)

    else:
        # TODO: Edge attack untested
        # Find names of nodes between target edges
        target_edges = [[n_from, n_to] for n_from, n_to in
                        sorted_areas[lesion_attr][0: num_lesions *
                                                  (1 + bilateral)]]
        # Find target indices (relative to weight matrix)
        target_edge_inds = [[row_labels.index(n_from), col_labels.index(n_to)]
                            for n_from, n_to in target_edges]
        # Call lesion function, get copy of updated weight mat
        W_lesion, cxns = network_gen.lesion_edge(net_dict_list[-1]['data'],
                                                 targets)

        lesion_list_labels.extend(target_edges)
    # Convert to networkx graph object
    graph_list.append(network_gen.import_weights_to_graph(W_lesion_dict,
                                                          directed=False))
    net_dict_list.append(deepcopy(W_lesion_dict))

    '''
    # Compute statistics for all areas
    lesion_results.append(area_compute.get_feature_dicts(
        graph_list[-1].nodes(), graph_list[-1], net_dict_list[-1]['data'],
        net_dict_list[-1]['row_labels']))
    '''
    graph_stats.append(network_compute.whole_graph_metrics(graph_list[-1],
                                                           weighted=False))


stats_to_graph = ['avg_shortest_path', 'avg_eccentricity', 'avg_ccoeff',
                  'avg_node_btwn', 'avg_edge_btwn', 'isolates']
f_name = network_type + '_lesionBy_' + lesion_attr + 'x' + \
    str(num_lesions) + '_stats.pkl'

pickle.dump({'stats': graph_stats, 'stat_names': stats_to_graph,
             'graph': network_type, 'targeted': targeted_attack,
             'target': lesion_attr, 'num_lesions': num_lesions,
             'bilateral': bilateral}, open(op.join(stat_save_path,
                                                   f_name), 'wb'))

if show_whole_stats:

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True)
    # Possible measures 'avg_shortest_path', 'avg_eccentricity', 'avg_ccoeff',
    #    'avg_node_btwn', 'avg_edge_btwn', 'isolates'
    stats_to_graph = ['avg_shortest_path', 'avg_eccentricity', 'avg_ccoeff',
                      'avg_node_btwn', 'avg_edge_btwn', 'isolates']

    # Construct matrix out of stats
    stat_mat = np.zeros((len(net_dict_list), len(stats_to_graph)))

    for gi in range(len(graph_stats)):
        for si, stat in enumerate(stats_to_graph):
            stat_mat[gi, si] = graph_stats[gi][stats_to_graph[si]]

    for ai in range(len(stats_to_graph)):
        axes[ai / 3, ai % 3].scatter(range(len(graph_stats)), stat_mat[:, ai])
        axes[ai / 3, ai % 3].set_ylabel(stats_to_graph[ai])
        axes[ai / 3, ai % 3].set_xlim([0, num_lesions + 1])
        if(ai / 3 == 0 and ai % 3 == 1):
            axes[ai / 3, ai % 3].set_title('Graph: ' + network_type +
                                           ' Lesion by ' + lesion_attr)
        if(ai / 3 == 1):
            axes[ai / 3, ai % 3].set_xlabel('Number of Lesions')
    #plt.show()

if make_movie:

    target_node_inds = [range(0, 5), range(5, 10), range(10, 15),
                        range(15, 20), range(20, 25), range(25, 30)]
    angles = [range(-270, -90, 3), range(-90, 90, 3)]

    # Construct matrix out of stats
    stat_mat = np.zeros((len(net_dict_list)))
    selected_stat = 'avg_shortest_path'
    for gi in range(len(graph_stats)):
        stat_mat[gi] = graph_stats[gi][selected_stat]

    # Png params
    elev = 20.  # elevation angle for movie

    for ti, targets in enumerate(target_node_inds):
        print 'Targeting: ' + str(targets)
        # Make stat graph on right hand side
        fig = plt.figure(figsize=(16, 6))

        ax1 = fig.add_subplot('121', projection='3d', axisbg='black')
        ax2 = fig.add_subplot('122')

        #
        #Change animation on right
        #
        ax2.scatter(range(1, len(graph_stats) + 1), stat_mat[:], c='b', s=40)
        midY = (np.max(stat_mat) + np.min(stat_mat)) / 2.
        ax2.annotate('N = ' + str(1 + targets[-1]), xy=(targets[-1], midY),
                     xycoords='data', xytext=(.66, .33),
                     textcoords='axes fraction', size=25,
                     color='DarkRed', arrowprops=None)
        ax2.set_title('Target High Degree Nodes', fontsize=22)
        ax2.set_xlabel('# Nodes Lesioned', fontsize=18)
        ax2.set_ylabel('Avg Shortest Path', fontsize=18)
        ax2.tick_params(labelsize=14)
        ax2.grid()

        _, ymax = ax2.get_ylim()
        #ax2.set_ylim((0, ymax))
        ymin, ymax = ax2.get_ylim()
        ax2.vlines(targets[-1] + 1, ymin, ymax, colors='r', lw=2, linestyle='--')
        ax2.set_xlim((0, ax2.get_xlim()[1]))

        #
        #Change animation on left
        #
        # Compute feature dictionary for all areas
        G_to_plot = graph_list[targets[0]]
        G_dict_to_plot = net_dict_list[targets[0]]
        area_dict = area_compute.get_feature_dicts(G_to_plot.nodes(),
                                                   G_to_plot,
                                                   G_dict_to_plot['data'],
                                                   G_dict_to_plot['row_labels'])

        # Collect & sort areas & edges according to various attributes
        sorted_areas_temp = \
            collect_areas.collect_and_sort(G_to_plot, G_dict_to_plot['data'],
                                           labels=G_dict_to_plot['row_labels'],
                                           print_out=False)
        # Get pair of neighbors for each area
        area0 = [sorted_areas_temp[lesion_attr][i] for i in targets]
        neighbors0 = []
        for l in [area_dict[a]['neighbors'] for a in area0]:
            neighbors0.append(l)

        edges0 = []
        # Get edges for each area
        for ai, a1 in enumerate(area0):
            temp_list = []
            temp_list.append([(a1, areaX) for areaX in neighbors0[ai]])
            edges0 = edges0 + temp_list

        # Put areas and neighbors together & remove duplicates
        nodes = area0 + [item for sublist in neighbors0 for item in sublist]
        edges = [item for sublist in edges0 for item in sublist]
        nodes = list(np.unique(nodes))
        edges = list(np.unique(edges))
        # Get remaining nodes
        rem_nodes = [area for area in sorted_areas_temp[lesion_attr]
                     if area not in nodes]
        # Make combined list
        all_nodes = nodes + rem_nodes
        # Get volumes
        all_vols = [area_dict[node]['volume'] for node in all_nodes]
        all_vols = np.array(all_vols)
        all_vols *= (1000. / all_vols.max())
        # Get centroids
        all_centroids = [area_dict[node]['centroid'] for node in all_nodes]
        all_centroids = np.array(all_centroids)
        # Swap columns so that S <-> I is on z axis
        all_centroids = all_centroids.take([0, 2, 1], 1)
        all_centroids[:, 2] *= -1
        # Get logical indices of area nodes
        neighbor_idxs = np.array([name in nodes for name in all_nodes])
        area_idxs = np.array([name in area0 for name in all_nodes])

        # Compute feature dictionary for all areas
        area_dict = area_compute.get_feature_dicts(G.nodes(), G,
                                                   W_net, row_labels)
        # Set sizes, colors, and alphas
        node_sizes = all_vols
        edge_sizes = 2 * np.ones((len(edges),))

        node_colors = np.array(['WhiteSmoke' for node_idx in
                                range(len(all_nodes))])
        #node_colors[neighbor_idxs] = '#00B200'
        node_colors[area_idxs] = 'Magenta'
        edge_colors = np.array(['#1565B2' for edge_idx in range(len(edges))])

        node_alphas = .4 * np.ones((len(all_nodes),), dtype=float)
        node_alphas[neighbor_idxs] = .4
        node_alphas[area_idxs] = .8
        edge_alphas = .2 * np.ones((len(edges),), dtype=float)

        # Call visualization to plot 3D nodes
        network_viz.plot_3D_network(ax1, node_names=nodes,
                                    node_positions=all_centroids,
                                    node_label_set=[False] * len(all_nodes),
                                    node_sizes=node_sizes,
                                    node_colors=node_colors,
                                    node_alpha=node_alphas,
                                    edges=edges,
                                    edge_label_set=[False] * len(edges),
                                    edge_colors=edge_colors,
                                    edge_alpha=edge_alphas,
                                    edge_sizes=edge_sizes)
        ax1.set_xlim((25., 115.))
        ax1.set_ylim((10., 110.))
        ax1.set_zlim((-70., -10.))

        # Rotate through angle on left and save images
        # flip between the two 180 deg angle sets for each set of targets
        for ai, ang in enumerate(angles[ti % 2]):
            ax1.view_init(elev=elev, azim=ang)
            fig.savefig(op.join(movie_save_path, 'mov_%03i_%03i.png' % (ti, ai)),
                        ec='black', fc='black', bbox_inches='tight',
                        pad_inches=0.)


'''
if show_area_stats:
    feats_lists = [[['degree', 'node_btwn'], ['degree', 'ccoeff']]]
    #[['inj_volume', 'degree'], ['inj_volume', 'out_deg']]
    for gi, g in enumerate(graph_list):
        for feats in feats_lists:
            fig, axs = plt.subplots(1, len(feats))
            for ax_idx, ax in enumerate(axs):
                area_plot.scatter_2D(ax, lesion_results[gi], feats[ax_idx][0],
                                     feats[ax_idx][1], s=50, c='r')

        fig2, axs2 = plt.subplots(1, 3)

        plot_net.plot_degree_distribution(axs2[0], g)
        plot_net.plot_shortest_path_distribution(axs2[1], g)
        plot_net.plot_clustering_coeff_pdf(axs2[2], g, np.linspace(0, 2, 100))
'''

'''
if show_example_plots:
    # Visualize individual areas & their cxns
    num_nets_to_plot = 1
    for net_dict in lesion_results[0:num_nets_to_plot]

        # Get pair of areas
        area0 = sorted_areas['ccoeff_labels'][2 * top_deg_idx]
        # Get neighbors for each area
        area1 = sorted_areas['ccoeff_labels'][2 * top_deg_idx + 1]
        neighbors0 = area_dict[area0]['neighbors']
        neighbors1 = area_dict[area1]['neighbors']
        # Get edges for each area
        edges0 = [(area0, areaX) for areaX in neighbors0]
        edges1 = [(area1, areaX) for areaX in neighbors1]

        # Put areas and neighbors together & remove duplicates
        nodes = [area0, area1] + neighbors0 + neighbors1
        edges = edges0 + edges1
        nodes = list(np.unique(nodes))
        edges = list(np.unique(edges))

        # Get remaining nodes
        rem_nodes = [area for area in sorted_areas['degree_labels']
                     if area not in nodes]
        # Make combined list
        all_nodes = nodes + rem_nodes

        # Get volumes and normalize by maximum area
        all_vols = [area_dict[node]['volume'] for node in all_nodes]
        all_vols = np.array(all_vols)
        all_vols *= (1000. / all_vols.max())
        # Get centroids
        all_centroids = [area_dict[node]['centroid'] for node in all_nodes]
        all_centroids = np.array(all_centroids)

        # Swap columns so that S <-> I is on z axis
        all_centroids = all_centroids.take([0, 2, 1], 1)
        all_centroids[:, 2] *= -1

        # Get logical indices of area nodes
        neighbor_idxs = np.array([name in nodes for name in all_nodes])
        area_idxs = np.array([name in [area0, area1] for name in all_nodes])

        # Set sizes & alphas
        node_sizes = all_vols
        node_alphas = .25 * np.ones((len(all_nodes),),
                                    dtype=float)  # Whole brain
        node_alphas[neighbor_idxs] = .5
        node_alphas[area_idxs] = .8
        edge_sizes = 2 * np.ones((len(edges),))
        edge_alphas = .5 * np.ones((len(edges),), dtype=float)
        # Specify colors
        node_colors = np.array(['k' for node_idx in range(len(all_nodes))])
        node_colors[neighbor_idxs] = 'r'
        node_colors[area_idxs] = 'b'
        edge_colors = np.array(['b' for edge_idx in range(len(edges))])

        # Plot 3D nodes
        network_viz.plot_3D_network(node_names=nodes,
                                    node_positions=all_centroids,
                                    node_label_set=[False] * len(all_nodes),
                                    node_sizes=node_sizes,
                                    node_colors=node_colors,
                                    node_alpha=node_alphas,
                                    edges=edges,
                                    edge_label_set=[False] * len(edges),
                                    edge_colors=edge_colors,
                                    edge_alpha=edge_alphas,
                                    edge_sizes=edge_sizes,
                                    save_movie=True)
'''
