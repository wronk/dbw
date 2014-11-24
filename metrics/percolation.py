"""
Created on Sat Nov 22 15:06:24 2014

@author wronk

Metrics for randomly lesioned graphs
"""

import numpy as np
import networkx as nx

import auxiliary as aux
reload(aux)


def percolate_random(graph, prop_removed, repeats=1):
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
    S = np.zeros((len(prop_removed), repeats))
    n = graph.order()

    # Loop over each proportion
    for pi, prop in enumerate(prop_removed):
        # Loop over each repeat
        for ri in range(repeats):
            temp_G, A = aux.lesion_graph_randomly(graph, prop)
            components = sorted(nx.connected_components(temp_G), key=len,
                                reverse=True)
            if len(components) > 0:
                largest_component = len(components[0])
            else:
                largest_component = 0.
            S[pi, ri] = largest_component / float(n)

    # Average over repeats
    return S


def percolate_degree(graph, num_lesions):
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
    S = np.zeros(len(num_lesions))
    n = graph.order()

    # Loop over each lesion
    for li, l in enumerate(num_lesions):
        temp_G, A = aux.lesion_graph_degree(graph, l)

        components = list(nx.connected_components(temp_G))
        sizes = [len(c) for c in components]
        #print sizes

        largest_component = max(sizes)
        #print sorted(nx.connected_components(temp_G), key=len, reverse=True)
        #print nx.connected_components(temp_G)

        S[li] = largest_component / float(n)

    # Average over repeats
    return S

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
