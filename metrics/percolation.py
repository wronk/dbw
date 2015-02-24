"""
Created on Sat Nov 22 15:06:24 2014

@author wronk

Metrics for lesioning graphs
"""

import numpy as np
import networkx as nx

import auxiliary as aux
reload(aux)


def percolate_random(graph, prop_removed, func_list, repeats=1):
    """ Get metrics after randomly removing some proportion
    of nodes.

    Parameters
    ----------
    graph : networkx graph
        Graph to perform the percolation on
    prop_removed: list
        Occupation probabilities to measure.
    func_list : list of function
        Metric functions to calculate and return.
    repeats: int
        Number of repetitions to average over for each proportion removed.

    Returns
    -------
    metrics : array
        All metrics evaluated. Size (func x lesions x repeats)
    """

    # Instantiate matrix to hold calculated graph size
    metrics = np.zeros((len(func_list), len(prop_removed), repeats))
    n = graph.order()
    kwargs = {'orig_order': n}

    # Loop over each function
    for fi, func in enumerate(func_list):
        # Loop over each proportion
        for pi, prop in enumerate(prop_removed):
            for ri in np.arange(repeats):
                temp_G, _ = aux.lesion_graph_randomly(graph, prop)

                #S[pi] = func_list[0](temp_G, **kwargs)

                if pi > 0 and metrics[fi, pi - 1] == np.nan:
                    metrics[fi, pi, ri] = np.nan
                else:
                    metrics[fi, pi, ri] = func(temp_G, **kwargs)
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


def percolate_degree(graph, num_lesions, func_list, repeats=1):
    """ Get metric after removing some number of nodes based on degree.

    Parameters
    ----------
    graph : networkx graph
        Graph to perform the percolation on
    num_lesions: list
        Number of lesions on network.
    func_list : list of function
        Metric functions to calculate and return.
    repeats: int
        Number of repetitions to average over for each proportion removed.

    Returns
    -------
    metrics : array
        All metrics evaluated. Size (func x lesions x repeats)
    """

    # Instantiate matrix to hold calculated graph size
    #S = np.zeros(len(num_lesions))
    #asp = np.zeros(len(num_lesions))

    metrics = np.zeros((len(func_list), len(num_lesions), repeats))
    n = graph.order()
    kwargs = {'orig_order': n}

    # Loop over each function
    for fi, func in enumerate(func_list):
        # Loop over each lesion
        for li, l in enumerate(num_lesions):
            # Loop over each repeat
            for ri in np.arange(repeats):
                temp_G, _ = aux.lesion_graph_degree(graph, l)

                if li > 0 and metrics[fi, li - 1] == np.nan:
                    metrics[fi, li, ri] = np.nan
                else:
                    metrics[fi, li, ri] = func_list[fi](temp_G, **kwargs)

    return metrics


def lesion_met_largest_component(G, orig_order=None):
    """
    Get largest component size of a graph.

    Parameters
    ----------
    G : networkx graph
        Graph to compute largest component for
    orig_order : int
        Define orig_order if you'd like the largest component proportion

    Returns
    -------
    largest component size : int
        Proportion of largest remaning component size if orig_order
        is defined. Otherwise, return number of nodes in largest component.
    """
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(components) > 0:
        largest_component = len(components[0])
    else:
        largest_component = 0.

    # Check if original component size is defined
    if orig_order is not None:
        return largest_component / float(orig_order)
    else:
        return largest_component


def lesion_met_avg_shortest_path(G):
    """
    Get average geodesic (shortest path) distance between all nodes in
    a graph.

    Parameters
    ----------
    G : networkx graph
        Graph to compute average geodesic for

    Returns
    -------
    average geodesic : float
        Average geodesic distance of a graph.
    """

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
