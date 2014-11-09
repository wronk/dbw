import numpy as np
import matplotlib.pyplot as plt
import aux_random_graphs
import scipy.stats
import network_gen
import aux_random_graphs
import networkx as nx


plot_lambda = 0 			# Plot length parameter
run_degree_comparison = 0		# Compare the degree for the model
run_average_distance = 0		# Compare average distance between edges
run_node_dist = 0			# Plot a histogram of the distances between nodes in Allen
compare_edge_distances = 0		# Plot histogram of edge distances between graphs
compare_degree_distribution = 1
compare_connectivity_patterns = 0 # Don't run this.

plt.close('all')

# Model parameters; L is the length parameter, pwr is the pref attachment parameter
L = 10^12
pwr = 1.4#1.5

# Number of bins to plot in one of the histograms, and number of iterations to average over..
# I forget which these are used in, actually.
nbins = 30
N_iter = 10


# coords is a dictionary with coordinates
coords = aux_random_graphs.get_coords()

# First do all this stuff for the Allen atlas
G = network_gen.Allen_graph('../friday-harbor/linear_model',0.1,0)

# D is a distance matrix
D = aux_random_graphs.get_edge_distances(G)
D_flat = np.array(D.values())

fig2,axs2 = plt.subplots(1,facecolor='white')
hist_D = axs2.hist(D_flat,nbins)

x = hist_D[1]
x = x[0:-1]
y = hist_D[0]

b,a,r,p,stderr = scipy.stats.linregress(x,np.log(y))

x1 = np.linspace(0,max(x),500)
y1 = a + b*x1

# This is our estimate of lambda
if L == -1:
    L = -1/b

## So for the edge distribution:
## If I use small values of lambda (around 10-11 microns, I can replicate the 
## edge distance distribution of the Allen atlas... 


## Now for the model...
G_bio,W_bio,D_bio = aux_random_graphs.biophysical_graph\
(N=426,N_edges=7804,L=L,power=pwr,dims=[10.,10,10],mode=0,centroids=\
coords)

D_bio = aux_random_graphs.get_edge_distances(G_bio)
D_bio_flat = np.array(D_bio.values())

hist_bio_D = np.histogram(D_bio_flat,nbins)
bio_x = hist_bio_D[1][0:-1]
bio_y = hist_bio_D[0]
#axs2.plot(bio_x,bio_y,'-',color='black',linewidth=4)

if plot_lambda:
    #axs2.plot(x1,np.exp(a)*np.exp(x1*b), linewidth=3, color='black')
    #mymean = x[y == max(y)]
    #axs2.plot(x1,max(y)*np.exp(-(x1-mymean)**2 / (2*sigma**2)), linewidth=3, color='black')
    axs2.plot(x1,np.exp(a)*np.exp(-x1/L), linewidth=3, color='black')
    axs2.set_xlabel('Distance ($\mu m$)',fontsize=16)
    axs2.set_ylabel('Count', fontsize=16)
    axs2.set_title('Fit for $a\exp(-x/\lambda)$, where $\lambda = 1758 \mu m, \ a=1812$', fontsize=20)

    fig1, axs1 = plt.subplots(1,facecolor='white')
    axs1.plot(x,np.log(y),linewidth=3,color='black')
    axs1.plot(x1,y1,linewidth=3,color='red')

    y_ticks = [1,2,3,4,5,6,7,8]
    axs1.set_yticks(y_ticks)
    axs1.set_yticklabels(np.round(np.exp(y_ticks)))
    axs1.set_xlabel('Allen atlas edge distance ($\mu m$)', fontsize=16)
    axs1.set_ylabel('Count', fontsize=16)


    plt.show(block=False)


if run_degree_comparison:
    node_distances,keys = aux_random_graphs.get_distances()
    # First we load the Allen linear model...

    # Set parameters
    p_th = .01 # P-value threshold
    w_th = 0 # Weight-value threshold

    # Set relative directory path
    dir_name = '../friday-harbor/linear_model'

    G = network_gen.Allen_graph(dir_name,p_th,w_th)

    G_deg = G.degree()
    # Then we iterate through a number of model iterations
    
    # This is setting the nodes that we're going to subsample. FilterCategories
    #threshold = 78.5 # For 0.
    threshold = 0 # For 1
    #threshold = 0 # For 2
    target_keys = [node for node in G.nodes() if G.node[node][1] > threshold]
   
    
    x = [G_deg[k] for k in target_keys]
    x_new = [G_deg[k] for k in target_keys]
    
    for k in range(N_iter):
	biophys_G,biophys_A,biophys_D =\
	aux_random_graphs.biophysical_graph(N=426,N_edges=7804,L=L,power=pwr,dims=[10.,10,10],mode=0,centroids=\
	coords)
	
	biophys_deg = biophys_G.degree()
	
	if k == 0:
	    y = [biophys_deg[j] for j in target_keys]
	else:
	    y_new = [biophys_deg[j] for j in target_keys]
	    y.extend(y_new)
	    x.extend(x_new)
    
    x = np.array(x)
    y = np.array(y)
    x_unique = np.unique(x)
    y_mean = [np.mean(y[x == j]) for j in x_unique]
    fig3,axs3 = plt.subplots(1)
    axs3.plot(x_unique,y_mean,'o', color='black')
    b,a,r,p,stderr = scipy.stats.linregress(x,y)
    axs3.plot(x_unique,a+b*x_unique,'-',color='red',linewidth=3)
    plt.show(block=False)
    
# Run this if we want to compute the average distance of each node
if run_average_distance and N_iter > 0:
    # Compute the empirical distances
    Allen_distance = aux_random_graphs.compute_average_distance(G)
    
    # These are the model distances; we need one to initiate the dict
   
    all_distances = {node:[] for node in G.nodes()}
    for k in range(N_iter):
	# Generate new graph for each iteration
	biophys_G,biophys_A,biophys_D =\
	aux_random_graphs.biophysical_graph(N=426,N_edges=7804,L=L,power=pwr,dims=[10.,10,10],mode=0,centroids=\
	coords)
	
	# Compute distances and add to the "meta dictionary"
	bio_distance = aux_random_graphs.compute_average_distance(biophys_G)
	for node in biophys_G.nodes():
	    all_distances[node].extend([bio_distance[node]])
	    
    
    mean_distance = {node:np.mean(all_distances[node]) for node in G.nodes()}
    fig4,axs4 = plt.subplots(1,facecolor='white')
    nodes = G.nodes()
  
    
    
    
    dist_x = [Allen_distance[node] for node in nodes]
    dist_y = [mean_distance[node] for node in nodes]
    
    b,a,r,p,stderr = scipy.stats.linregress(dist_x,dist_y)
    
    axs4.plot(dist_x,dist_y,'o', color='black')
    axs4.plot(dist_x,a + b*np.array(dist_x), '-', linewidth=3,color='red')
    axs4.set_xlim([0,7000])
    axs4.set_ylim([0,7000])
    axs4.set_xlabel('Allen atlas mean edge distance', fontsize=16)
    axs4.set_ylabel('Model mean edge distance', fontsize=16)
    plt.show(block=False)

if run_node_dist:
    fig5,axs5 = plt.subplots(1,facecolor='white')
    node_distances_flat = np.ndarray.flatten(node_distances)
    axs5.hist(node_distances_flat,50)
    axs5.set_xlabel('Distance between nodes',fontsize=20)
    axs5.set_ylabel('Count', fontsize=20)
    plt.show(block=False)
    
if compare_edge_distances:
    allen_edge_distances = aux_random_graphs.get_edge_distances(G)
    model_edge_distances = aux_random_graphs.get_edge_distances(G_bio)
    
    fig6,axs6 = plt.subplots(1,facecolor='white')
    
    # First plot the Allen atlas edge distribution
    axs6_1 = plt.subplot(121)
    axs6_1.hist(allen_edge_distances.values(),50)
    axs6_1.set_xlabel('Edge distance', fontsize=18)
    axs6_1.set_ylabel('Count', fontsize=18)
    axs6_1.set_xlim([0,16000])
    axs6_1.set_ylim([0,1200])
    axs6_1.set_title('Allen Atlas edge distribution', fontsize=22)
    
    # Then plot the model edge distribution
    axs6_2 = plt.subplot(122)
    axs6_2.hist(model_edge_distances.values(),50)
    axs6_2.set_xlabel('Edge distance', fontsize=18)
    axs6_2.set_ylabel('Count', fontsize=18)
    axs6_2.set_xlim([0,16000])
    axs6_2.set_ylim([0,1200])
    axs6_2.set_title('Biophysical model edge distribution', fontsize=22)
    
    plt.show(block=False)
    
if compare_degree_distribution:
    nbins = 50
    fig7,axs7 = plt.subplots(1,facecolor='white')
    
    G_deg = G.degree()
    G_bio_deg = G_bio.degree()
    
    axs7_1 = plt.subplot(121)
    axs7_1.hist(G_deg.values(),nbins)
    
    axs7_2 = plt.subplot(122)
    axs7_2.hist(G_bio_deg.values(),nbins)
    
    plt.show(block=False)
    
# This will compute the adjacency matrix for a series of graphs and compare the connectivity pattern between them.
# It will compute a false negative, false positive, hit and miss rate.
# This isn't quite functional yet...
if compare_connectivity_patterns:
    A = nx.adjacency_matrix(G,weight=None)
    A_bio = nx.adjacency_matrix(G_bio)
    
    A_diff = A-A_bio
    
    A_error = sum(A_diff**2)
    
    # This gives us a matrix where the entries are 1 for any connection that we missed with our model
    miss_matrix = np.multiply(A,(np.multiply(A_diff,A_diff)))
    miss = sum(miss_matrix)/sum(A)
    hit = 1-miss
    
    FP_rate = np.sum(A_diff < 0,1)/sum(A_bio,1)
    FN_rate = np.sum(A_diff > 0,1)/sum(A_bio,1)