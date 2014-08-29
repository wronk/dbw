import pdb
"""
Created on Thu Aug 28 15:58:46 2014

@author: rkp

Module containing functions for computing properties of specific areas.
"""

import numpy as np
import networkx as nx

import network_compute

from friday_harbor.structure import Ontology
from friday_harbor.mask import Mask
import friday_harbor.experiment as experiment

# Get ontological dictionary with acronyms as keys
DATA_DIR = '../data'
ONTO = Ontology(data_dir=DATA_DIR)

# Build experiment manager for looking up injection site masks
EXPT_MGR = experiment.ExperimentManager(data_dir=DATA_DIR)
# Calculate union of injection masks
INJ_MASKS = [e.injection_mask() for e in EXPT_MGR.experiment_list]
INJ_MASK_UNION = Mask.union(*INJ_MASKS)

def get_feature_dicts(area_list,G,W,W_labels):
    """Create a feature dictionary for each of a list of areas.
    
    Returns:
        dict of dicts
            keys are labels, values are feature dictionaries"""
    
    # Get value that are easier to compute all at once
    node_btwn_dict = nx.betweenness_centrality(G)
    ccoeff_dict = nx.clustering(G)
    out_dict, in_dict, out_in_dict = network_compute.out_in(W,W_labels,binarized=False)

    # Make area dictionary
    area_dict = {}
    for area in area_list:
        print 'Examining area %s'%area
        feat_dict = {}
        # Get structure for this area
        s = ONTO.structure_by_acronym(area[:-2])
        s_id = s.structure_id # id
        feat_dict['name'] = s.name # name
        feat_dict['acronym'] = s.acronym # non-lateralized acronym
        if area[-1] == 'L':
            mask = ONTO.get_mask_from_id_left_hemisphere_nonzero(s_id)
        elif area[-1] == 'R':
            mask = ONTO.get_mask_from_id_right_hemisphere_nonzero(s_id)
        feat_dict['centroid'] = mask.centroid
        feat_dict['volume'] = len(mask.mask[0])
        # Find volume covered by injection masks
        inj_mask_intsct = Mask.intersection(mask,INJ_MASK_UNION)
        feat_dict['inj_volume'] = float(len(inj_mask_intsct.mask[0]))
        # Find percent volume covered by injection masks
        feat_dict['inj_percent'] = feat_dict['inj_volume']/feat_dict['volume']
        
        # Get graph theory scalars
        feat_dict['node_btwn'] = node_btwn_dict[area]
        feat_dict['degree'] = G.degree()[area]
        feat_dict['ccoeff'] = ccoeff_dict[area]
        feat_dict['out_in'] = out_in_dict[area]
        feat_dict['out_deg'] = out_dict[area]
        feat_dict['in_deg'] = in_dict[area]
        # Graph theory lists
        feat_dict['neighbors'] = G.neighbors(area)
        
        # Associate this feature dictionary to this area
        area_dict[area] = feat_dict.copy()
        
    return area_dict