import pdb
"""
Created on Wed Aug 27 23:17:01 2014

@author: rkp

Analyze properties of specific brain areas with extreme ranks according to
specific criteria.
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

# Network generation parameters
p_th = .01  # P-value threshold
w_th = 0  # Weight-value threshold

# Set relative directory path to linear model & ontology
dir_LM = '../friday-harbor/linear_model'

calc_features = True
show_example_plots = True
show_whole_stats = True
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

elif network_type == 'powerlaw_cluster':
    n = 426
    row_labels = range(n)
    col_labels = range(n)

    # Create networkx graph
    temp_G = nx.powerlaw_cluster_graph(n=n, m=20, p=.33)

    # Set weights for all the egdes
    wts = {}
    for e in temp_G.edges():
        wts[e] = 1.
    nx.set_edge_attributes(temp_G, 'weight', wts)

    W_net = nx.adjacency_matrix(temp_G, nodelist=row_labels).toarray()

    # Put everything in a dictionary
    net_dict = {'row_labels': row_labels, 'col_labels': col_labels,
                'data': W_net}
    G = network_gen.import_weights_to_graph(net_dict)

'''
elif network_type == 'small_world':
    # Load weights & p-values

elif network_type == 'random':
    # Load weights & p-values

elif network_type == 'scale_free':
    # Load weights & p-values


'''

###################################

# Collect & sort areas & edges according to various attributes
sorted_areas = collect_areas.collect_and_sort(G, W_net, labels=row_labels,
                                              print_out=False)

###################################
### Lesion areas
# Set number of lesions
lesion_is_node = True  # Set if node or edge lesion

targeted_attack = False
# Find areas to lesion. node_btwn, ccoeff, degree (append with _labels)
lesion_attr = 'node_btwn_labels'
bilateral = False
num_lesions = 150

###################################

# Record pre-lesioned network statistics
#lesion_results = [area_compute.get_feature_dicts(G.nodes(), G, W_net,
#                                                 row_labels)]

graph_list = [deepcopy(G)]
net_dict_list = [deepcopy(net_dict)]
graph_stats = [network_compute.whole_graph_metrics(G)]

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
            targets = np.random.choice(sorted_areas[lesion_attr],
                                       size=(1 + bilateral), replace=False)
            for t in targets:
                sorted_areas[lesion_attr].remove(t)

        # Call lesion function, update weight mat
        W_lesion_dict = network_gen.lesion_node(net_dict_list[-1], targets)
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

if show_whole_stats:

    stats_to_graph = ['avg_shortest_path', 'avg_eccentricity', 'avg_ccoeff',
                      'avg_node_btwn', 'avg_edge_btwn', 'isolates']

    # Construct matrix out of stats
    stat_mat = np.zeros((len(net_dict_list), len(stats_to_graph)))

    for gi in range(len(graph_stats)):
        for si, stat in enumerate(stats_to_graph):
            stat_mat[gi, si] = graph_stats[gi][stats_to_graph[si]]

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True)
    for ai in range(len(stats_to_graph)):
        axes[ai / 3, ai % 3].scatter(range(len(graph_stats)), stat_mat[:, ai])
        axes[ai / 3, ai % 3].set_ylabel(stats_to_graph[ai])
        axes[ai / 3, ai % 3].set_xlim([0, num_lesions + 1])
        if(ai / 3 == 0 and ai % 3 == 1):
            axes[ai / 3, ai % 3].set_title('Lesion by ' + lesion_attr)
        if(ai / 3 == 1):
            axes[ai / 3, ai % 3].set_xlabel('Number of Lesions')
    plt.show()

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
