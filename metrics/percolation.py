"""
Created on Sat Nov 22 15:06:24 2014

@author wronk

Metrics for lesioning graphs
"""

import numpy as np
import networkx as nx

import auxiliary as aux
reload(aux)


def percolate_random(graph, prop_removed, func_list):
    """ Get size avg of largest cluster after randomly removing some proportion
    of nodes.

    Parameters
    ----------
    graph : networkx graph
        Graph to perform the percolation on
    prop_removed: list
        Occupation probabilities to measure.
    repeats: int
        Number of repetitions to average over for each proportion removed.

    Returns
    -------
    S : list
        Size of largest cluster in each lesion
    """

    # Instantiate matrix to hold calculated graph size
    #S = np.zeros((len(prop_removed)))
    #asp = np.zeros((len(prop_removed)))
    metrics = np.zeros((len(func_list), len(prop_removed)))
    n = graph.order()
    kwargs = {'orig_order': n}

    # Loop over each function
    for fi, func in enumerate(func_list):
        # Loop over each proportion
        for pi, prop in enumerate(prop_removed):
            temp_G, _ = aux.lesion_graph_randomly(graph, prop)

            #S[pi] = func_list[0](temp_G, **kwargs)

            if pi > 0 and metrics[fi, pi - 1] == np.nan:
                metrics[fi, pi] = np.nan
            else:
                metrics[fi, pi] = func_list[fi](temp_G, **kwargs)
            #metrics[fi, pi] = func(temp_G, **kwargs)

            '''
            # largest component
            components = sorted(nx.connected_components(temp_G), key=len,
                                reverse=True)
            if len(components) > 0:
                largest_component = len(components[0])
            else:
                largest_component = 0.
            S[pi] = largest_component / float(n)

            # Avg shortest path
            try:
                asp[pi] = nx.average_shortest_path_length(temp_G)
            except (nx.exception.NetworkXError,
                    nx.exception.NetworkXPointlessConcept):
                asp[pi:] = np.nan
            '''

    return metrics


def percolate_degree(graph, num_lesions, func_list):
    """ Get size avg of largest cluster after removing some number of nodes
        based on degree.

    Parameters
    ----------
    graph : networkx graph
        Graph to perform the percolation on
    num_lesions: list
        Number of lesions on network.

    Returns
    -------
    S : list
        Size of largest cluster in each lesion step.
    """

    # Instantiate matrix to hold calculated graph size
    #S = np.zeros(len(num_lesions))
    #asp = np.zeros(len(num_lesions))

    metrics = np.zeros((len(func_list), len(num_lesions)))
    n = graph.order()
    kwargs = {'orig_order': n}

    # Loop over each function
    for fi, func in enumerate(func_list):
    # Loop over each lesion
        for li, l in enumerate(num_lesions):
            temp_G, _ = aux.lesion_graph_degree(graph, l)

            if li > 0 and metrics[fi, li - 1] == np.nan:
                metrics[fi, li] = np.nan
            else:
                metrics[fi, li] = func_list[fi](temp_G, **kwargs)

    return metrics


def lesion_met_largest_component(G, **kwargs):
    # compute largest component
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(components) > 0:
        largest_component = len(components[0])
    else:
        largest_component = 0.

    return largest_component / float(kwargs['orig_order'])


def lesion_met_avg_shortest_path(G, **kwargs):
    # Avg shortest path
    try:
        asp = nx.average_shortest_path_length(G)
    except (nx.exception.NetworkXError, nx.exception.NetworkXPointlessConcept):
        asp = np.nan

    return asp


def lesion_met_diameter(G, **kwargs):
    # Diameter
    try:
        d = nx.diameter(G)
    except (nx.exception.NetworkXError, nx.exception.NetworkXPointlessConcept):
        d = np.nan

    return d


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from random_graph import binary_undirected as bu

    biophys, A, D = bu.biophysical()
    ER, A, D = bu.ER_distance()

    prop_removed = np.arange(0.05, 1, 0.1)
    S_bio = percolate_random(biophys, prop_removed, 3)
    S_ER = percolate_random(ER, prop_removed, 3)
    plt.plot(prop_removed, S_bio, prop_removed, S_ER)

    '''
    lesion_list = np.arange(0, 400, 10)
    S_bio = percolate_degree(biophys, lesion_list)
    S_ER = percolate_degree(ER, lesion_list)
    plt.plot(lesion_list, S_bio, lesion_list, S_ER)
    '''

    plt.show()
