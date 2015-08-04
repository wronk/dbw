"""
Tests related to various metrics we use.
"""
from __future__ import print_function, division
import unittest
import numpy as np
import networkx as nx


class NetworkXDirectedGraphsTestCase(unittest.TestCase):

    def test_rows_are_sources_and_cols_are_targets_when_making_graph_from_adjacency_matrix(self):

        a = np.array([[0, 1, 1, 1, 1],
                      [1, 0, 0, 0, 1],
                      [0, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0]])

        G = nx.from_numpy_matrix(a, create_using=nx.DiGraph())

        out_degree_correct = {0: 4, 1: 2, 2: 1, 3: 1, 4: 0}
        in_degree_correct = {0: 1, 1: 1, 2: 2, 3: 2, 4: 2}

        self.assertEqual(out_degree_correct, G.out_degree())
        self.assertEqual(in_degree_correct, G.in_degree())

        # make sure we get back the same adjacency matrix
        np.testing.assert_array_equal(a, nx.adjacency_matrix(G).todense())


if __name__ == '__main__':
    unittest.main()