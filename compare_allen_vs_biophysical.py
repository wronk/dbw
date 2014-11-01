import numpy as np
import matplotlib.pyplot as plt
import aux_random_graphs
import scipy.stats
import network_gen
import aux_random_graphs

plot_lambda = 1
run_comparison = 1

plt.close('all')

pwr = 1.5
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
L = -1/b
L = L*0.01




## Now for the model...
G_bio,W_bio,D_bio = aux_random_graphs.biophysical_graph\
(N=426,N_edges=7804,L=L,power=pwr,dims=[10.,10,10],mode=0,centroids=\
coords)

D_bio = aux_random_graphs.get_edge_distances(G_bio)
D_bio_flat = np.array(D_bio.values())

hist_bio_D = np.histogram(D_bio_flat,nbins)
bio_x = hist_bio_D[1][0:-1]
bio_y = hist_bio_D[0]
axs2.plot(bio_x,bio_y,'-',color='black',linewidth=4)
#axs2.plot(x1,np.exp(a)*np.exp(-x1/L),'-',color='black',linewidth=4)



#axs2.plot(x1,np.exp(a)*np.exp(x1*b), linewidth=3, color='black')
mymean = x[y == max(y)]
#axs2.plot(x1,max(y)*np.exp(-(x1-mymean)**2 / (2*sigma**2)), linewidth=3, color='black')
#axs2.plot(x1,np.exp(a)*np.exp(-x1/L), linewidth=3, color='black')
axs2.set_xlabel('Distance ($\mu m$)',fontsize=16)
axs2.set_ylabel('Count', fontsize=16)
axs2.set_title('Fit for $a\exp(-x/\lambda)$, where $\lambda = 1758 \mu m, \ a=1812$', fontsize=20)

fig1, axs1 = plt.subplots(1)
axs1.plot(x,np.log(y),linewidth=3,color='black')
axs1.plot(x1,y1,linewidth=3,color='red')
if plot_lambda:
    plt.show(block=False)

    
if run_comparison:
    
    garbage,keys = aux_random_graphs.get_distances()
    # First we load the Allen linear model...
    
    # Set parameters
    p_th = .01 # P-value threshold
    w_th = 0 # Weight-value threshold
    
    # Set relative directory path
    dir_name = '../friday-harbor/linear_model'

    G = network_gen.Allen_graph(dir_name,p_th,w_th)
    
    G_deg = G.degree()
    
    
    # Then we iterate through a number of model iterations
    
    # This is setting the nodes that we're going to subsample.
    threshold = 78.5
    target_keys = [node for node in G.nodes() if G.node[node][0] > threshold]
    
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
    
