"""
Tests related to various metrics we use.
"""
from __future__ import print_function, division
import unittest
import numpy as np
import networkx as nx
from metrics import binary_directed as metrics_bd


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

    def test_power_law_fit_deg_cc(self):

        G = nx.erdos_renyi_graph(100, .2)
        fit = metrics_bd.power_law_fit_deg_cc(G)
        self.assertEqual(len(fit), 5)


class CustomMetricsTestCase(unittest.TestCase):

    def test_reciprocity(self):

        G = nx.DiGraph()
        G.add_nodes_from(range(4))
        G.add_edge(0, 2)
        G.add_edge(2, 0)
        G.add_edge(1, 2)
        G.add_edge(2, 1)
        G.add_edge(2, 3)
        G.add_edge(3, 1)

        reciprocity_correct = 1 / 3

        self.assertAlmostEqual(metrics_bd.reciprocity(G), reciprocity_correct)


if __name__ == '__main__':
    unittest.main()