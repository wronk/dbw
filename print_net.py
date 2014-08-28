import pdb
"""
Created on Thu Aug 28 14:40:04 2014

@author: rkp

File containing functions to print out names, etc.
"""

import pprint as pp
from friday_harbor.structure import Ontology

# Get ontological dictionary with acronyms as keys
DIR_ONTO = '../data'
ONTO = Ontology(data_dir=DIR_ONTO)
ONTO_DICT = {s.acronym: s.name for s in ONTO.structure_list}

def print_node_list(labels,vals,num_nodes):
    """Print out a list of nodes in a nice way."""
    
    # Get names
    names = ['%s_%s' % (ONTO_DICT[label[:-2]], label[-2:])
                    for label in labels]
    # Print out top- or bottom- ranked nodes
    if num_nodes > 0:
        print 'Top nodes:'
        tuple_set = zip(labels[:num_nodes],
                        names[:num_nodes],
                        vals[:num_nodes])
    elif num_nodes < 0:
        print 'Bottom nodes:'
        tuple_set = zip(labels[num_nodes:],
                        names[num_nodes:],
                        vals[num_nodes:])
                                
    pp.pprint(tuple_set)
    
def print_edge_list(labels,vals,num_edges):
    """Print out a list of edges in a nice way."""
    
    # Get names
    names = [None for ii in range(len(labels))]
    for edge_idx,label in enumerate(labels):
        node0 = '%s%s'%(ONTO_DICT[label[0][:-2]],label[0][-2:])
        node1 = '%s%s'%(ONTO_DICT[label[1][:-2]],label[1][-2:])
        names[edge_idx] = '%s <--> %s'%(node0,node1)
    print 'Top edges by edge-betweenness:'
    top_edge_btwn_tuples = zip(labels[:num_edges],
                               names[:num_edges],
                               vals[:num_edges])
    pp.pprint(top_edge_btwn_tuples)
