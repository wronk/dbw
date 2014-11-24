"""
Created on Mon Nov 24 12:29:20 2014

@author: wronk

Create figures showing progressive percolation on standard and brain graphs.
"""
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

import network_gen as ng
from metrics import percolation as perc
from metrics import auxiliary as aux
from random_graph import binary_undirected as bu

repeats = 100  # Number of times to repeat random percolation
prob_rm = np.arange(0., 1.05, 0.05)
lesion_list = np.arange(0, 426, 10)
node_order = 426

linear_model_path = '/home/wronk/Builds/friday-harbor/linear_model'
##############################################################################
### Construct graphs
graph_list = []
graph_names = []
graph_col = ['k', 'r', 'g', 'b', 'c']

# Construct Allen brain graph
W_net, row_labels, col_labels = ng.quick_net(dir_name=linear_model_path)
W_net_dict = {'row_labels': row_labels, 'col_labels': col_labels,
              'data': W_net}
graph_list.append(ng.import_weights_to_graph(W_net_dict))
graph_names.append('Mouse')
print 'Added graph: ' + graph_names[-1]

# Construct Random (ER) graph
G_ER, A, D = bu.ER_distance()
graph_list.append(G_ER)
graph_names.append('Erdos-Renyi')
print 'Added graph: ' + graph_names[-1]

# Construct WS graph
graph_list.append(nx.watts_strogatz_graph(node_order, 36, 0.159))
graph_names.append('Small-World')
print 'Added graph: ' + graph_names[-1]

# Construct BA graph
graph_list.append(nx.barabasi_albert_graph(node_order, 20))
graph_names.append('Scale-Free')
print 'Added graph: ' + graph_names[-1]

# Construct biophysical graph
G_BIO, A, D = bu.biophysical()
graph_list.append(G_BIO)
graph_names.append('Biophysical')
print 'Added graph: ' + graph_names[-1]

assert len(graph_list) == len(graph_names), 'Graph list/names don\'t match'
print '\n'

##############################################################################
### Do percolation
S_rand = []
S_target = []

# Percolation
for gi, G in enumerate(graph_list):
    print 'Lesioning ' + graph_names[gi]
    S_rand.append(perc.percolate_random(G, prob_rm, repeats))
    S_target.append(perc.percolate_degree(G, lesion_list))
    print '\t ... Done'

S_rand_avg = [np.mean(s, axis=1) for s in S_rand]
S_rand_std = [np.std(s, axis=1) for s in S_rand]

##############################################################################
### Plot results
# Set font type for compatability with adobe if doing editting later
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42
plt.ion()

FONTSIZE = 18
FIGSIZE = (18, 8)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=FIGSIZE)

for si, trace in enumerate(S_rand_avg):
    ax1.plot(prob_rm, trace, label=graph_names[si], color=graph_col[si])
for si, trace in enumerate(S_target):
    ax2.plot(lesion_list, trace, label=graph_names[si], color=graph_col[si])

# Set title and labels
ax1.set_title('Random Attack', fontsize=FONTSIZE)
ax1.set_xlabel('Proportion Removed', fontsize=FONTSIZE)
ax1.set_ylabel('Size of largest cluster', fontsize=FONTSIZE)

ax2.set_title('Targeted Attack (by Degree)', fontsize=FONTSIZE)
ax2.set_xlabel('# Nodes Removed', fontsize=FONTSIZE)
ax2.set_ylabel('Size of largest cluster', fontsize=FONTSIZE)
ax2.legend()

for ax in [ax1, ax2]:
    for text in ax.get_ticklabels() + ax.get_yticklabels():
        text.set_fontsize(FONTSIZE)

plt.show()
