from __future__ import division
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import os

from random_graph.binary_directed import growing_SGPA_1
from network_plot.change_settings import set_all_text_fontsizes, set_all_colors
from metrics import binary_directed as metrics_bd
from network_plot import change_settings
import in_out_plot_config as cf

N_NODES = 426
P_EDGE_SPLIT = 0.016
L = 0.75
BRAIN_SIZE = [7., 7., 7.]
MARKERSIZE = 25

FONT_SIZE = 12

# Use environmental variable
SAVE_DIR = os.environ['DBW_SAVE_CACHE']

# create model
G = growing_SGPA_1(N_NODES, P_EDGE_SPLIT, L, BRAIN_SIZE, remove_extra_ccs=True)
G = nx.convert_node_labels_to_integers(G)  # remove nodes not in main network

# Initialize the figure and axes objects

fig = plt.figure(figsize=(7.5, 4), facecolor='w')
plt.subplots_adjust(bottom=0.15, hspace=0.45, wspace=0.55)

left_main_ax = plt.subplot2grid(
    cf.subplot_divisions, cf.left_main_location,
    rowspan=cf.left_main_rowspan, colspan=cf.left_main_colspan)

right_main_ax = plt.subplot2grid(
    cf.subplot_divisions, cf.right_main_location,
    rowspan=cf.right_main_rowspan, colspan=cf.right_main_colspan)

top_margin_ax = plt.subplot2grid(
    cf.subplot_divisions, cf.top_margin_location,
    rowspan=cf.top_margin_rowspan, colspan=cf.top_margin_colspan,
    sharex=left_main_ax)

right_margin_ax = plt.subplot2grid(
    cf.subplot_divisions, cf.right_margin_location,
    rowspan=cf.right_margin_rowspan, colspan=cf.right_margin_colspan,
    sharey=left_main_ax)

# To get the log axes, create another axis on top/right of our existing ones
top_dummy_ax = top_margin_ax.twinx()
right_dummy_ax = right_margin_ax.twiny()

# Get in- & out-degree
nodes = np.sort(G.nodes())
indeg = np.array([G.in_degree()[node] for node in nodes])
outdeg = np.array([G.out_degree()[node] for node in nodes])
node_ages = nodes / nodes.max()
deg = indeg + outdeg

# Calculate proportion in degree
percent_indeg = indeg / deg.astype(float)

a1 = 1.0

# Left main plot (in vs out degree)
left_main_ax.scatter(indeg, outdeg, c=node_ages, cmap=cm.jet, s=MARKERSIZE, lw=0)

left_main_ax.set_xlabel('In-degree')
left_main_ax.set_ylabel('Out-degree')

left_main_ax.set_xlim([0, 100])
left_main_ax.set_ylim([-2, 100])
left_main_ax.set_aspect('auto')
left_main_ax.set_xticks(np.arange(0, 101, 25))
left_main_ax.set_yticks(np.arange(0, 101, 25))
left_main_ax.legend(loc='best')
left_main_ax.text(1.2, 1.2, 'a', fontsize=FONT_SIZE + 2, fontweight='bold',
                  transform=left_main_ax.transAxes)

# Top marginal (in-degree)
top_margin_ax.hist(indeg, bins=cf.OUTDEGREE_BINS, histtype='stepfilled',
                   color='k', normed=True, stacked=True)

# This is for the log-axis
indeg_hist = np.histogram(indeg, bins=cf.OUTDEGREE_BINS)
indeg_x = indeg_hist[1][0:len(indeg_hist[0])]
indeg_y = indeg_hist[0]
indeg_y = indeg_y / float(indeg_y.sum())

top_dummy_ax.plot(indeg_x, indeg_y, linestyle='-', lw=1.5, color='b')
top_dummy_ax.yaxis.tick_right()
top_dummy_ax.yaxis.set_label_position('right')
top_dummy_ax.set_yscale('log')

top_margin_ax.set_yticks([0, 0, 5, 1.0])
top_margin_ax.set_ylim([0, 1.0])

# Right marginal (out-degree)
right_margin_ax.hist(outdeg, bins=cf.OUTDEGREE_BINS, histtype='stepfilled',
                     color='k', orientation='horizontal', normed=True,
                     stacked=True)

# This is for the log-axis
outdeg_hist = np.histogram(outdeg, bins=cf.OUTDEGREE_BINS)
outdeg_x = outdeg_hist[1][0:len(outdeg_hist[0])]
outdeg_y = outdeg_hist[0]
outdeg_y = outdeg_y / float(outdeg_y.sum())

right_dummy_ax.plot(outdeg_y, outdeg_x, linestyle='-', lw=1.5, color='b')
right_dummy_ax.xaxis.tick_top()
right_dummy_ax.xaxis.set_label_position('top')
right_dummy_ax.set_xscale('log')

top_margin_ax.set_yticks([0, 0.04, 0.08])
top_margin_ax.set_ylim([0, 0.08])

plt.setp(right_margin_ax.get_yticklabels() + top_margin_ax.get_xticklabels(),
         visible=False)

right_margin_ax.set_xticks([0, 0.04, 0.08])
right_margin_ax.set_xlim([0, 0.08])

# Right main plot (proportion in vs total degree)
right_main_ax.scatter(deg, percent_indeg, s=MARKERSIZE, lw=0, c=node_ages,
                      cmap=cm.jet)

right_main_ax.set_xlabel('Total degree (in + out)')
right_main_ax.set_ylabel('Proportion in-degree')
right_main_ax.xaxis.set_major_locator(plt.MaxNLocator(4))
right_main_ax.set_yticks(np.arange(0, 1.1, .25))
right_main_ax.set_ylim([0., 1.05])
right_main_ax.set_xlim([0, 120])
right_main_ax.set_xticks(np.arange(0, 151, 50))
right_main_ax.text(1., 1.2, 'b', fontsize=FONT_SIZE + 2, fontweight='bold',
                   transform=right_main_ax.transAxes, ha='right')

top_margin_ax.set_ylabel('$P(K_\mathrm{in}=k)$')
right_margin_ax.set_xlabel('$P(K_\mathrm{out}=k)$')

for temp_ax in [left_main_ax, right_main_ax, top_margin_ax, right_margin_ax,
                top_dummy_ax, right_dummy_ax]:
    set_all_text_fontsizes(temp_ax, FONT_SIZE)
    set_all_colors(temp_ax, cf.LABELCOLOR)

top_lin_ticks = top_margin_ax.get_yticklabels()
right_lin_ticks = right_margin_ax.get_xticklabels()

top_log_ticks = top_dummy_ax.get_yticklabels()
right_log_ticks = right_dummy_ax.get_xticklabels()

for tick in top_lin_ticks+right_lin_ticks:
    tick.set_color('k')
    tick.set_fontsize(FONT_SIZE-2)

for tick in top_log_ticks+right_log_ticks:
    tick.set_color('blue')
    tick.set_fontsize(FONT_SIZE-5)

for tick in right_log_ticks+right_lin_ticks:
    tick.set_rotation(270)

fig.subplots_adjust(right=0.95, bottom=0.2)

# Add colorbar
rightAx_bbox = right_main_ax.get_position()
cbar_ax_rect = (rightAx_bbox.xmin, rightAx_bbox.y1 + 0.095, 0.2, .045)
cbar_ax = fig.add_axes(cbar_ax_rect, xticks=[0, 255],
                       xticklabels=['Old', 'New'], yticks=[])
cbar_ax.xaxis.set_tick_params(size=0)
cbar_ax.yaxis.set_tick_params(size=0)

jet_grad = np.linspace(0, 1, 256)  # Jet gradient for Old->New
cbar_ax.imshow(np.vstack((jet_grad, jet_grad)), aspect='auto', cmap=cm.jet)

fig.savefig(os.path.join(SAVE_DIR, 'node_by_node_in_and_out.png'), dpi=300)

# plot clustering vs degree and nodal efficiency
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.5), tight_layout=True,
                        facecolor='white')

cc_full = nx.clustering(G.to_undirected())
deg_full = nx.degree(G.to_undirected())
cc = [cc_full[node] for node in nodes]
deg = [deg_full[node] for node in nodes]

# calculate nodal efficiency
G.efficiency_matrix = metrics_bd.efficiency_matrix(G)
nodal_efficiency = np.sum(G.efficiency_matrix, axis=1) / (len(G.nodes()) - 1)

labels = ('a', 'b')

axs[0].scatter(deg, cc, c=node_ages, cmap=cm.jet, lw=0)
axs[0].set_xlim(0, 150)
axs[0].set_ylim(-0.025, 1.025)
axs[0].set_xlabel('Degree')
axs[0].set_ylabel('Clustering coefficient')
axs[0].locator_params(axis='x', nbins=6)

# Add colorbar
ax0_bbox = axs[0].get_position()
ax0_yCenter = np.mean([ax0_bbox.y0, ax0_bbox.y1])
cbar_ax_rect = (ax0_bbox.xmax - 0.05, ax0_yCenter - 0.065,
                0.03, .35)
cbar_ax = fig.add_axes(cbar_ax_rect, yticks=[0, 255],
                       yticklabels=['New', 'Old'], xticks=[])
cbar_ax.xaxis.set_tick_params(size=0)
cbar_ax.yaxis.set_tick_params(size=0)

jet_grad = np.linspace(1, 0, 256)  # Jet gradient for Old->New
cbar_ax.imshow(np.vstack((jet_grad, jet_grad)).T, aspect='auto', cmap=cm.jet)

axs[1].hist(nodal_efficiency, bins=20)
axs[1].set_xlim(0, 1)
axs[1].set_xlabel('Nodal efficiency')
axs[1].set_ylabel('Number of nodes')

for ax, label in zip(axs, labels):
    change_settings.set_all_text_fontsizes(ax, fontsize=FONT_SIZE)
    ax.text(0.915, 0.925, label, fontsize=FONT_SIZE + 2, fontweight='bold',
            transform=ax.transAxes,  ha='center', va='center')

fig.savefig(os.path.join(SAVE_DIR, 'node_by_node_cc_deg_nodal_eff.png'),
            dpi=300)
