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

# Network generation parameters
p_th = .01  # P-value threshold
w_th = 0  # Weight-value threshold

calc_nets = True
calc_features = True
show_example_plots = True
show_stat_plots = False

if calc_nets:
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
    # Convert to networkx graph object
    G = network_gen.import_weights_to_graph(W_net_dict)

    # Collect & sort areas & edges according to various attributes
    sorted_areas = collect_areas.collect_and_sort(G, W_net, labels=row_labels,
                                                  print_out=False)

if calc_features:
    # Compute feature dictionary for all areas
    area_dict = area_compute.get_feature_dicts(G.nodes(), G, W_net, row_labels)

if show_example_plots:
    # Visualize individual areas & their cxns
    num_top_deg = 1
    for top_deg_idx in [1]:
        # Get pair of neighbors for each area
        area0 = sorted_areas['ccoeff_labels'][2 * top_deg_idx]
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
        area_idxs = np.array([name in [area0, area1] for name in all_nodes])

        # Set sizes, colors, and alphas
        node_sizes = all_vols
        edge_sizes = 2 * np.ones((len(edges),))

        node_colors = np.array(['WhiteSmoke' for node_idx in
                                range(len(all_nodes))])
        node_colors[neighbor_idxs] = '#00B200'
        node_colors[area_idxs] = 'DodgerBlue'
        edge_colors = np.array(['#1565B2' for edge_idx in range(len(edges))])

        node_alphas = .2 * np.ones((len(all_nodes),), dtype=float)
        node_alphas[neighbor_idxs] = .5
        node_alphas[area_idxs] = .8
        edge_alphas = .6 * np.ones((len(edges),), dtype=float)

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
                                    save_movie='./movie/images/')

if show_stat_plots:
    feats_lists = [[['inj_volume', 'degree'], ['inj_volume', 'out_deg']],
                   [['degree', 'node_btwn'], ['degree', 'ccoeff']]]
    for feats in feats_lists:
        fig, axs = plt.subplots(1, len(feats))
        for ax_idx, ax in enumerate(axs):
            area_plot.scatter_2D(ax, area_dict, feats[ax_idx][0],
                                 feats[ax_idx][1], s=50, c='r')
