import pdb
"""
Created on Wed Sep  3 21:57:48 2014

@author: rkp

Functions for visualizing the node-swapping procedure.
"""

import os
import numpy as np
import matplotlib.pyplot as plt; plt.close('all')

NODE1_START = [1.2,.6]
NODE2_START = [-1.,.4]

def gen_swap_figs(ax,save_dir='/Users/rkp/Desktop/node_swap',only_first=True):
    """Generate figures demonstrating the swapping of two nodes."""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fixed_nodes = np.array([[1.,1],[-1,.9],[-1.3,-.2],[-.5,-.5],[1,-1.3],[.8,.2]])
    mobile_nodes = np.array([[.2,1.3],[-.7,-1.1]])
    
    dots_fixed, = ax.plot(fixed_nodes[:,0],fixed_nodes[:,1],'o',markersize=50,zorder=1)
    if only_first:
        dots_mobile, = ax.plot(mobile_nodes[:,0],mobile_nodes[:,1],'o',markersize=50,c='b')
        c0 = 1804.33
        c1 = 2209.79
        ax.set_xlim(-2,1.5)
        ax.set_ylim(-1.7,1.7)
        ax.set_axis_off()
        title = ax.set_title('Cost = %.2f'%c0,fontsize=20)
        fixed_edge_coords = [[0,1],[1,5],[3,5],[1,2],[2,4],[4,1],[4,5]]
        for edge_c in fixed_edge_coords:
            ax.plot(fixed_nodes[edge_c,0],fixed_nodes[edge_c,1],lw=4,c='k',zorder=0)
        mobile_edge_coords0 = [1,5]
        mobile_edge_coords1 = [0,3,4]
        edges0 = []
        edges1 = []
        for edge_c in mobile_edge_coords0:
            edges0 += ax.plot([mobile_nodes[0,0],fixed_nodes[edge_c,0]],
                              [mobile_nodes[0,1],fixed_nodes[edge_c,1]],
                              lw=4,c='k',zorder=0)
        for edge_c in mobile_edge_coords1:
            edges1 += ax.plot([mobile_nodes[1,0],fixed_nodes[edge_c,0]],
                              [mobile_nodes[1,1],fixed_nodes[edge_c,1]],
                              lw=4,c='k',zorder=0)  
    else:
        dots_mobile, = ax.plot(mobile_nodes[:,0],mobile_nodes[:,1],'o',markersize=50,c='m')
    
        c0 = 1804.33
        c1 = 2209.79
        ax.set_xlim(-2,1.5)
        ax.set_ylim(-1.7,1.7)
        ax.set_axis_off()
        title = ax.set_title('Cost = %.2f'%c0,fontsize=20)
        fixed_edge_coords = [[0,1],[1,5],[3,5],[1,2],[2,4],[4,1],[4,5]]
        for edge_c in fixed_edge_coords:
            ax.plot(fixed_nodes[edge_c,0],fixed_nodes[edge_c,1],lw=4,c='k',zorder=0)
        mobile_edge_coords0 = [1,5]
        mobile_edge_coords1 = [0,3,4]
        edges0 = []
        edges1 = []
        for edge_c in mobile_edge_coords0:
            edges0 += ax.plot([mobile_nodes[0,0],fixed_nodes[edge_c,0]],
                              [mobile_nodes[0,1],fixed_nodes[edge_c,1]],
                              lw=4,c='r',zorder=0)
        for edge_c in mobile_edge_coords1:
            edges1 += ax.plot([mobile_nodes[1,0],fixed_nodes[edge_c,0]],
                              [mobile_nodes[1,1],fixed_nodes[edge_c,1]],
                              lw=4,c='r',zorder=0)          
    if only_first:
        fig.savefig('%s/first_frame.png'%save_dir,
                    facecolor='white',edgecolor='white',bbox_inches='tight',pad_inches=0)
        dots_mobile, = ax.plot(mobile_nodes[:,0],mobile_nodes[:,1],'o',markersize=50,c='m')
        fig.savefig('%s/second_frame.png'%save_dir,
                    facecolor='white',edgecolor='white',bbox_inches='tight',pad_inches=0)
        return
    nsteps = 20
    x0_start = mobile_nodes[0.,0]
    x0_end = mobile_nodes[1.,0]
    x1_start = mobile_nodes[1.,0]
    x1_end = mobile_nodes[0.,0]
    y0_start = mobile_nodes[0.,1]
    y0_end = mobile_nodes[1.,1]
    y1_start = mobile_nodes[1.,1]
    y1_end = mobile_nodes[0.,1]
    
    x0 = np.linspace(x0_start,x0_end,nsteps)
    x1 = np.linspace(x1_start,x1_end,nsteps)
    y0 = np.linspace(y0_start,y0_end,nsteps)
    y1 = np.linspace(y1_start,y1_end,nsteps)
    c = np.linspace(c0,c1,nsteps)
    
    for step in range(nsteps):
        dots_mobile.set_xdata([x0[step],x1[step]])
        dots_mobile.set_ydata([y0[step],y1[step]])
        for edge in edges0:
            x_end = edge.get_xdata()[-1]
            y_end = edge.get_ydata()[-1]
            edge.set_xdata([x0[step],x_end])
            edge.set_ydata([y0[step],y_end])
        for edge in edges1:
            x_end = edge.get_xdata()[-1]
            y_end = edge.get_ydata()[-1]
            edge.set_xdata([x1[step],x_end])
            edge.set_ydata([y1[step],y_end])
        title.set_text('Cost = %.2f'%c[step])
        plt.draw()
        fig.savefig('%s/frame%d.png'%(save_dir,step),
                    facecolor='white',edgecolor='white',bbox_inches='tight',pad_inches=0)
    x0 = x0[::-1]
    x1 = x1[::-1]
    y0 = y0[::-1]
    y1 = y1[::-1]
        
if __name__ == '__main__':
    fig,ax = plt.subplots(1,1,facecolor='w')
    gen_swap_figs(ax)