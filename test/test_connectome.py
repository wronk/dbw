"""
Tests related to extracting the connectivity matrix.
"""
from __future__ import print_function, division
import unittest
import numpy as np
from extract import auxiliary
from extract import brain_graph


class BrainSymmetryTestCase(unittest.TestCase):

    def setUp(self):

        self.W, self.P, self.labels = auxiliary.load_W_and_P()
        self.G, self.A, self.labels = brain_graph.binary_directed()

    def test_loaded_connectome_is_correctly_symmetric(self):

        self.assertEqual(self.W.shape, self.P.shape)

        n_sources, n_targets = self.W.shape

        self.assertEqual(n_sources, n_targets)
        self.assertEqual(n_sources, len(self.labels))

        np.testing.assert_array_almost_equal(self.W[:n_sources/2, :n_sources/2],
                                             self.W[n_sources/2:, n_sources/2:])
        np.testing.assert_array_almost_equal(self.W[:n_sources/2, n_sources/2:],
                                             self.W[n_sources/2:, :n_sources/2])

        np.testing.assert_array_almost_equal(self.P[:n_sources/2, :n_sources/2],
                                             self.P[n_sources/2:, n_sources/2:])
        np.testing.assert_array_almost_equal(self.P[:n_sources/2, n_sources/2:],
                                             self.P[n_sources/2:, :n_sources/2])

    def test_every_label_has_its_partner(self):

        n_sources_one_side = int(len(self.labels) / 2)
        labels_left = self.labels[:n_sources_one_side]
        labels_right = self.labels[n_sources_one_side:]

        for label in labels_left:
            label_right = label.replace('_L', '_R')
            self.assertTrue(label_right in labels_right)

        for label in labels_right:
            label_left = label.replace('_R', '_L')
            self.assertTrue(label_left in labels_left)

    def test_symmetry_retained_in_in_and_out_degree_distributions(self):

        in_degs = self.G.in_degree().values()
        out_degs = self.G.out_degree().values()

        for node_ctr, in_deg in enumerate(in_degs[:213]):
            self.assertEqual(in_deg, in_degs[node_ctr + 213])
            self.assertEqual(out_degs[node_ctr], out_degs[node_ctr + 213])


class PValueTestCase(unittest.TestCase):

    def test_number_of_nonzero_connections_increases_as_p_value_increases(self):

        p_ths = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)

        n_edges = brain_graph.binary_directed(p_th=p_ths[0])[1].sum()

        for p_th in p_ths[1:]:
            new_n_edges = brain_graph.binary_directed(p_th=p_th)[1].sum()

            self.assertGreater(new_n_edges, n_edges)
            n_edges = new_n_edges


if __name__ == '__main__':
    unittest.main()