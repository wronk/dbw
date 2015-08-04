"""
Tests related to extracting the connectivity matrix.
"""
from __future__ import print_function, division
import unittest
import numpy as np
from extract import auxiliary


class BrainSymmetryTestCase(unittest.TestCase):

    def test_loaded_connectome_is_correctly_symmetric(self):
        W, P, labels = auxiliary.load_W_and_P()

        self.assertEqual(W.shape, P.shape)

        n_sources, n_targets = W.shape

        self.assertEqual(n_sources, n_targets)
        self.assertEqual(n_sources, len(labels))

        np.testing.assert_array_almost_equal(W[:n_sources/2, :n_sources/2],
                                             W[n_sources/2:, n_sources/2:])
        np.testing.assert_array_almost_equal(W[:n_sources/2, n_sources/2:],
                                             W[n_sources/2:, :n_sources/2])

        np.testing.assert_array_almost_equal(P[:n_sources/2, :n_sources/2],
                                             P[n_sources/2:, n_sources/2:])
        np.testing.assert_array_almost_equal(P[:n_sources/2, n_sources/2:],
                                             P[n_sources/2:, :n_sources/2])

    def test_every_label_has_its_partner(self):
        W, P, labels = auxiliary.load_W_and_P()

        n_sources_one_side = int(len(labels) / 2)
        labels_left = labels[:n_sources_one_side]
        labels_right = labels[n_sources_one_side:]

        for label in labels_left:
            label_right = label.replace('_L', '_R')
            self.assertTrue(label_right in labels_right)

        for label in labels_right:
            label_left = label.replace('_R', '_L')
            self.assertTrue(label_left in labels_left)


class PValueTestCase(unittest.TestCase):

    def test_number_of_nonzero_connections_decreases_as_p_value_decreases(self):
        pass

if __name__ == '__main__':
    unittest.main()