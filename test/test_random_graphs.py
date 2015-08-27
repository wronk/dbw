"""
Tests related to our random graphs.
"""
from __future__ import print_function, division
import unittest
from random_graph import binary_directed as random_graph


class RandomGraphsTestCase(unittest.TestCase):

    def test_all_random_graphs_yield_correct_number_of_nodes_and_edges(self):

        G, A, D = random_graph.biophysical_indegree(N=426, N_edges=2000)
        self.assertEqual(len(G.nodes()), 426)
        self.assertEqual(len(G.edges()), 2000)

        G, A, D = random_graph.biophysical_reverse_outdegree(N=426, N_edges=2000)
        self.assertEqual(len(G.nodes()), 426)
        self.assertEqual(len(G.edges()), 2000)


if __name__ == '__main__':
    unittest.main()