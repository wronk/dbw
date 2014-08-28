import pdb
"""
Created on Wed Aug 27 22:31:09 2014

@author: rkp

Set of functions for pulling out the nodes and edges according to specific
ranking criteria.
"""

import numpy as np
import operator


def out_in_ratio(W_net, labels):
    """Calculate the output/input ratio given the weight matrix."""
    W = (W_net > 0).astype(float)
    # Calculate total output & input connections
    out_total = W.sum(axis=1)
    in_total = W.sum(axis=0)
    out_in_vec = out_total.astype(float) / in_total
    # Put into dictionary format
    out_in_dict = {labels[idx]: out_in_vec[idx] for idx in range(len(labels))}

    return out_in_dict


def get_ranked(criteria_dict, high_to_low=True):
    """Get labels & criteria, sorted (ranked) by criteria."""

    dict_list_sorted = sorted(criteria_dict.iteritems(),
                              key=operator.itemgetter(1), reverse=high_to_low)

    labels_sorted = [item[0] for item in dict_list_sorted]
    criteria_sorted = [item[1] for item in dict_list_sorted]

    return labels_sorted, criteria_sorted
