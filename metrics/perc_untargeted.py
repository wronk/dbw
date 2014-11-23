"""
Created on Sat Nov 22 15:06:24 2014

@author wronk

Metrics for randomly lesioned graphs
"""

import numpy as np
import networkx as nx

import perc_auxiliary as perc_aux


def percolate_random(graph, phi_list, repeats=1):
    """ Get size avg of largest cluster after randomly removing some proportion
    of nodes.

    Parameters
    ----------
    graph : networkx graph
        Graph to perform the percolation on
    phi_list: list
        Occupation probabilities to measure.
    repeats: int
        Number of repetitions to average over for each phi.

    Returns
    -------
    S : list
        Size of largest cluster in each network
    """

    # Instantiate matrix to hold calculated graph size
    S = np.zeros((len(phi_list), repeats))
    n = graph.order()

    # Loop over each phi
    for pi, phi in enumerate(phi_list):
        # Loop over each repeat
        for ri, in range(repeats):
            temp_G, _ = perc_aux.lesion_graph_randomly(graph, phi)
            largest_component = len(sorted(nx.connected_components(temp_G)[0],
                                           key=len, reverse=True))
            S[pi, ri] = largest_component / float(n)

    # Average over repeats
    return np.mean(S, axis=1)


if __name__ == '__main__':
    import binary_undirected as bu

    biophys = bu.biophysical()
    phi_list = np.arange(0., 1., 0.1)

    S = percolate_random(biophys, phi_list, 5)
    print S
