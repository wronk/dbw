from __future__ import print_function, division
from scipy import stats
import networkx as nx
import matplotlib.pyplot as plt
plt.ion()

from extract import brain_graph
from random_graph.binary_directed import biophysical_reverse_outdegree as pgpa
from metrics import binary_directed as metrics
from network_plot import change_settings

import brain_constants as bc
from config import COLORS, FACE_COLOR, AX_COLOR, FONT_SIZE

# parameters for this particular plot
FIG_SIZE = (12, 6)

# load brain graph
print('loading brain graph...')
G_brain, _, _ = brain_graph.binary_directed()

# create directed ER graph
print('generating directed ER graph...')
G_er = nx.erdos_renyi_graph(bc.num_brain_nodes,
                            bc.p_brain_edge_directed,
                            directed=True)

# create pgpa graph
print('generating model...')
G_pgpa, _, _ = pgpa(N=bc.num_brain_nodes, N_edges=bc.num_brain_edges_directed, L=.75)

print('calculating efficiencies')
# calculate local efficiency distribution
for ctr, G in enumerate((G_brain, G_er, G_pgpa)):
    print('for graph {}'.format(ctr))
    G.efficiency = metrics.efficiency_matrix(G)
    G.avg_local_efficiency = G.efficiency.sum(axis=1) / (bc.num_brain_nodes - 1)

# plot histograms of all three efficiency matrices
fig, axs = plt.subplots(1, 2, facecolor=FACE_COLOR, figsize=FIG_SIZE, tight_layout=True)

print('making histograms')
axs[0].hist([G_brain.efficiency[G_brain.efficiency >= 0],
             G_er.efficiency[G_er.efficiency >= 0],
             G_pgpa.efficiency[G_pgpa.efficiency >= 0]],
            bins=20, normed=True, lw=0,
            color=(COLORS['brain'], COLORS['er'], COLORS['pgpa']))

axs[1].hist([G_brain.avg_local_efficiency,
             G_er.avg_local_efficiency,
             G_pgpa.avg_local_efficiency],
            bins=20, normed=True, lw=0,
            color=(COLORS['brain'], COLORS['er'], COLORS['pgpa']),)

axs[0].set_xlabel('efficiency')
axs[0].set_ylabel('probability')
axs[1].set_xlabel('efficiency')

axs[0].set_title('all efficiencies')
axs[1].set_title('average local efficiencies')

for ax in axs:
    change_settings.set_all_colors(ax, AX_COLOR)
    change_settings.set_all_text_fontsizes(ax, FONT_SIZE)

# calculate KS-statistic and p-value between brain and ER, brain and PGPA
d_brain_er, p_brain_er = stats.ks_2samp(G_brain.avg_local_efficiency,
                                        G_er.avg_local_efficiency)
d_brain_pgpa, p_brain_pgpa = stats.ks_2samp(G_brain.avg_local_efficiency,
                                            G_pgpa.avg_local_efficiency)

print('KS p-value (brain vs. ER): {}'.format(p_brain_er))
print('KS p-value (brain vs. PGPA): {}'.format(p_brain_pgpa))

plt.draw()
plt.show(block=True)