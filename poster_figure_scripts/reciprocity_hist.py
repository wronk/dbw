# Author: sid
# Script to generate N graphs with length parameter L
# Plot a resulting histogram of these distributions
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from random_graph.binary_directed import biophysical_reverse_outdegree as biophysical_model
from network_plot.change_settings import set_all_text_fontsizes as set_fontsize
from extract.brain_graph import binary_directed as brain_graph

from brain_constants import *
from network_compute import reciprocity
import graph_tools.auxiliary as aux_tools

# Number of iterations per L
N = 250
load = 1
# Specify length coefficients
Ls = [np.inf,1,0.75]

modelReciprocity = np.zeros([len(Ls),N])

if not load:
    for n in range(N):
        for j,L in enumerate(Ls):
            G, A, labels = biophysical_model(N=num_brain_nodes,
                                             N_edges=num_brain_edges_directed,
                                             L=L,
                                             gamma=1.)
            # Compute reciprocity
            modelReciprocity[j,n] = reciprocity(A)
else:
    modelReciprocity = pd.read_csv('reciprocity.csv',header=None).as_matrix().transpose()
    

# Plot histograms
matplotlib.rc('axes',edgecolor='white')
fig,axs = plt.subplots(1,facecolor='white',edgecolor='white')

bins = np.linspace(0,0.2,41)
model0 = np.histogram(modelReciprocity[0,:],bins)
model1 = np.histogram(modelReciprocity[1,:],bins)
fsize = 40
a=axs.hist(modelReciprocity[0,:],bins,color='c')
b=axs.hist(modelReciprocity[1,:],bins)
c=axs.hist(modelReciprocity[2,:],bins,color='orange')
xticks = [0,0.05,0.1,0.15,0.2]
yticks = np.array([0,50,100,150,200])
axs.set_xlabel('Reciprocity',fontsize=fsize,color='white')
axs.set_ylabel('Count',fontsize=fsize,color='white')

axs.set_xticks(xticks)
axs.set_xticklabels(xticks,fontsize=fsize,color='white')

axs.set_yticks(yticks)
axs.set_yticklabels(yticks,fontsize=fsize,color='white')
axs.set_ylim([0,200])
axs.set_xlim([0,0.175])

G,A,labels = brain_graph()
brainReciprocity = reciprocity(A)

axs.plot([brainReciprocity,brainReciprocity],[0,100], color='white',linewidth=4)
fig.subplots_adjust(bottom=0.15,left=0.15)

fig.patch.set(alpha=0)
axs.patch.set(alpha=0)

myleg = axs.legend(['Reciprocity in Allen Atlas','No L', 'L=1','L=0.75'],\
           prop={'size':30},loc='upper right',fancybox=True)
text1,text2,text3,text4 = myleg.get_texts()
text1.set_color('white'); text2.set_color('white'); text3.set_color('white')
text4.set_color('white')
myleg.get_frame().set(alpha=0)

plt.show(block=False)
