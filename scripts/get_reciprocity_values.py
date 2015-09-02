import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os

from extract.brain_graph import binary_directed as brain_graph
from random_graph.binary_directed import biophysical_reverse_outdegree as biophysical

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

from network_compute import reciprocity

import brain_constants as bc
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network_plot.network_viz import plot_scatterAndMarginal

# IMPORT PLOT PARAMETERS
import in_out_plot_config as cf

plt.ion()

# PLOT PARAMETERS
FACECOLOR = 'w'
TEXTCOLOR = 'k'
MARKERCOLOR='m'
FONTSIZE = 24
NBINS = 15

Ls = np.linspace(0,2,21)
N = 10

reciprocity_matrix = np.zeros([len(Ls),N])

for i,L in enumerate(Ls):
    print "Running L " + str(i) + " of " + str(len(Ls))
    for j in range(N):
        
        G_PA,A,_ = biophysical(N=bc.num_brain_nodes,\
                               N_edges=bc.num_brain_edges_directed,
                               L=L,\
                               gamma=1.)
        
        reciprocity_matrix[i,j] = reciprocity(A)

current_reciprocity\_DF = pd.DataFrame(reciprocity_matrix)



data_dir = os.getenv('DBW_DATA_DIRECTORY')
current_reciprocity_DF.to_csv(data_dir+'/reciprocity.csv')

