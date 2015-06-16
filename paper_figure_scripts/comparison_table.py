import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from random_graph.binary_directed import biophysical_indegree, biophysical_reverse_outdegree
from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

import brain_constants as bc

import color_scheme
import in_out_plot_config as cf

from network_compute import reciprocity

from extract.brain_graph import binary_directed
Gattachment, _, _ = biophysical_reverse_outdegree(N=bc.num_brain_nodes,
                                              N_edges=bc.num_brain_edges_directed,
                                              L=0.75, gamma=0.)

Ggrowth, _, _ = biophysical_reverse_outdegree(N=bc.num_brain_nodes,
                                              N_edges=bc.num_brain_edges_directed,
                                              L=np.inf, gamma=1.)

Gpgpa, _, _ = biophysical_reverse_outdegree(N=bc.num_brain_nodes,
                                              N_edges=bc.num_brain_edges_directed,
                                              L=0.75, gamma=1.)
Gbrain, _, _ = binary_directed()


graphs = [Ggrowth,Gattachment,Gpgpa,Gbrain]
graph_names = ['Pref growth only','Proximal attachment only','PGPA','Connectome']

metrics = ['Reciprocity','Clustering','Degree','CC-deg rank','Prop in-total rank']


table = pd.DataFrame(index=metrics,col

for G in graphs:
    A = np.matrix(G.adjacency_list())
