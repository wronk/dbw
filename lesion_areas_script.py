import pdb
"""
Created on Wed Aug 27 23:17:01 2014

@author: rkp

Analyze properties of specific brain areas with extreme ranks according to
specific criteria.
"""

import pprint as pp
import numpy as np
import matplotlib.pyplot as plt

import collect_areas
import network_gen
import area_compute
import network_viz
import area_plot
import plot_net
from copy import deepcopy

# Network generation parameters
p_th = .01  # P-value threshold
w_th = 0  # Weight-value threshold

calc_features = True
show_example_plots = True
show_stat_plots = True

###################################
### Create network
# Set relative directory path to linear model & ontology
dir_LM = '../friday-harbor/linear_model'

# Load weights & p-values
W, P, row_labels, col_labels = network_gen.load_weights(dir_LM)
# Threshold weights according to weights & p-values
W_net, mask = network_gen.threshold(W, P, p_th=p_th, w_th=w_th)

# Set weights to zero if they don't satisfy threshold criteria
W_net[W_net == -1] = 0.
# Set diagonal weights to zero
np.fill_diagonal(W_net, 0)

# Put everything in a dictionary
W_net_dict = {'row_labels': row_labels, 'col_labels': col_labels,
              'data': W_net}

# Create networkx graph
G = network_gen.import_weights_to_graph(W_net_dict)

# Collect & sort areas & edges according to various attributes
sorted_areas = collect_areas.collect_and_sort(G, W_net, labels=row_labels,
                                              print_out=True)

###################################
### Lesion areas num_lesions = 1  # Set number of lesions
lesion_is_node = True  # Set if node or edge lesion

# Find areas to lesion. node_btwn, ccoeff, degree, edge_btwn
# out, in, in, out_in
lesion_attr = 'degree_labels'
bilateral = True
num_lesions = 1

# Record pre-lesioned network statistics
lesion_results = [area_compute.get_feature_dicts(G.nodes(), G, W_net,
                                                 row_labels)]
graph_list = [deepcopy(G)]
weight_list = [deepcopy(W_net)]

# Lesion areas
for i in range(num_lesions):
    if lesion_is_node:
        # Find target indices (relative to weight matrix)
        # Unilateral  0:1, 1:2, 2:3
        # Bilateral   0:2, 2:4, 4:6
        target_inds = [row_labels.index(t) for t in
                       sorted_areas[lesion_attr][i * (bilateral + 1):
                                                 (i + 1) * (bilateral + 1)]]
        # Call lesion function, update weight mat
        W_lesion, cxns = network_gen.lesion_node(weight_list[-1], target_inds)

    else:
        # Find names of nodes between target edges
        target_edges = [[n_from, n_to] for n_from, n_to in
                        sorted_areas[lesion_attr][0: num_lesions *
                                                  (1 + bilateral)]]
        # Find target indices (relative to weight matrix)
        target_edge_inds = [[row_labels.index(n_from), col_labels.index(n_to)]
                            for n_from, n_to in target_edges]
        # Call lesion function, get copy of updated weight mat
        W_lesion, cxns = network_gen.lesion_edge(graph_list[-1]['data'],
                                                 target_inds)

    # TODO: modify to delete the labels as well
    # Convert to networkx graph object
    W_lesion_dict = {'data': W_lesion, 'row_labels': W_net_dict['row_labels'],
                     'col_labels': W_net_dict['col_labels']}
    graph_list.append(network_gen.import_weights_to_graph(W_lesion_dict,
                                                          directed=False))
    weight_list.append(deepcopy(W_lesion))

    # Compute feature dictionary for all areas
    lesion_results.append(area_compute.get_feature_dicts(
        graph_list[-1].nodes(), graph_list[-1], weight_list[-1], row_labels))

if show_stat_plots:
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
