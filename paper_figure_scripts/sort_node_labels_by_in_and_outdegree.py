import os
import numpy as np
from extract import brain_graph


SAVE_PATH_PREFIX = os.path.join(os.getenv('DBW_DATA_DIRECTORY'), 'nodes_sorted')
G, A, labels = brain_graph.binary_directed()

idxs_sorted = {}
labels_sorted = {}
values_sorted = {}

# sort by outdegree
out_deg = A.sum(axis=1)
idxs_sorted['out_deg'] = out_deg.argsort()
labels_sorted['out_deg'] = list(np.array(labels)[idxs_sorted['out_deg']])
values_sorted['out_deg'] = out_deg[idxs_sorted['out_deg']]

# sort by indegree
in_deg = A.sum(axis=0)
idxs_sorted['in_deg'] = in_deg.argsort()
labels_sorted['in_deg'] = list(np.array(labels)[idxs_sorted['in_deg']])
values_sorted['in_deg'] = in_deg[idxs_sorted['in_deg']]

# write a new file for each metric
for metric in ['out_deg', 'in_deg']:
    file_path = '{}_{}.txt'.format(SAVE_PATH_PREFIX, metric)
    with open(file_path, 'wb') as f:
        f.write('rank, node idx, node label, {}\n\n'.format(metric))
        for ctr, (idx, label, value) in \
                enumerate(zip(idxs_sorted[metric], labels_sorted[metric], values_sorted[metric])):
            f.write('{}, {}, {}, {}\n'.format(ctr, idx, label, value))