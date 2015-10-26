"""
Plot proportion in-degree of mouse connectome
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from extract.brain_graph import binary_directed as brain_graph
import color_scheme

fontsize = 12
save_dir = os.environ['DBW_SAVE_CACHE']

# Load connectome graph
G = brain_graph()[0]


# Calculate proportion in-degree
outdeg = G.out_degree()
indeg = G.in_degree()
nodes = G.nodes()
sumdeg = [float(outdeg[node] + indeg[node]) for node in nodes]
prop_indeg = [indeg[node] / sumdeg[node] for node in nodes]

# Plot in-degree
fig, ax = plt.subplots(1, facecolor='w', figsize=(3.5, 2.75))
fig.subplots_adjust(bottom=0.2, left=0.2)
bins = np.linspace(0, 1, 11)
ax.hist(prop_indeg, bins, facecolor=color_scheme.ATLAS)

# Add labels and ticks
ax.set_xlabel('Proportion in-degree', fontsize=fontsize)
ax.set_ylabel('Count', fontsize=fontsize)
xticks = [0, 0.25, 0.5, 0.75, 1.0]
yticks = np.arange(0, 100, 20)
ax.set_xticks(xticks)
ax.set_xticklabels(xticks, size=fontsize)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, size=fontsize)

# Save plot
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.9)
plt.savefig(os.path.join(save_dir, 'figS4.png'), dpi=300)
plt.savefig(os.path.join(save_dir, 'figS4.pdf'), dpi=300)

plt.show()
