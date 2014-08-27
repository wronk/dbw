'''
author: sid
'''

import numpy as np
import networkx.algorithms.shortest_paths as SP

def ShortestPaths(G):
    nG = G.number_of_nodes()
    Names = G.nodes()
    PathLength = np.zeros([nG,nG])
    PathLength[:] = np.nan
    
    ShortestPaths = SP.generic.shortest_path_length(G)
    for i in range(nG):
        for j in range(nG):
            CurrentPath = ShortestPaths[Names[i]]
            if Names[j] in CurrentPath:
                PathLength[i,j] = ShortestPaths[Names[i]][Names[j]]

    return PathLength