"""
Created on Mon Nov 17 11:13:06 2014

@author: rkp

Plot some distance metrics for the real brain network.
"""

# PLOT PARAMETERS
FACECOLOR = 'white'

import extract.brain_graph
import metrics.binary_undirected

import numpy as np
import matplotlib.pyplot as plt

# Load brain network
G_brain, A_brain, _, _ = extract.brain_graph.binary_undirected()
D_brain, _ = extract.brain_graph.distance_matrix()
# Calculate connection probability vs. distance
cxn_prob, dist_bins = metrics.binary_undirected.cxn_probability(A_brain, 
                                                                D_brain)

fig, axs = plt.subplots(2, 1, facecolor=FACECOLOR)

# Plot edge distance histogram
edge_distances = D_brain[np.triu(A_brain, k=1) == 1]
axs[0].hist(edge_distances, 50)

# Plot connection probability vs. distance
bin_centers = .5 * (dist_bins[:-1] + dist_bins[1:])
bin_width = dist_bins[1] - dist_bins[0]
axs[1].bar(bin_centers, cxn_prob, width=bin_width)

# Set axis limits
[ax.set_xlim(0,11) for ax in axs]

# Label axes
axs[0].set_xlabel('Edge distance (mm)')
axs[0].set_ylabel('Count')
axs[0].set_title('Edge distance histogram')
axs[1].set_xlabel('Distance between nodes (mm)')
axs[1].set_ylabel('Probability of cxn')
axs[1].set_title('Cxn probability vs. distance')

plt.draw()
plt.tight_layout()