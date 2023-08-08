import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib as mpl
import os
import glob
import networkx as nx
import argparse
import seaborn as sns
import matplotlib.style as style
style.use('seaborn-poster') 
mpl.rcParams['font.family'] = "sans-serif"
sns.set_context('poster')

def get_edge_points(pos_i,ax_n,sign_p):
    edge_n = np.zeros(2)
    edge_n = pos_i + sign_p[0]*ax_n[:,0]/2. + sign_p[1]*ax_n[:,1]/2.

    return edge_n

def rotation_matrix(theta):
    rot_mat = np.zeros((2,2))

    rot_mat[0,0] = np.cos(theta) 
    rot_mat[0,1] = -np.sin(theta)
    rot_mat[1,0] = np.sin(theta)
    rot_mat[1,1] = np.cos(theta)

    return rot_mat


def get_orient(v, rot_mat):
    return rot_mat.dot(v)

def read_bonds(filen):
	first_line_pair = [0,0,0,0]
	cut=False
	with open(filen, 'r') as f:
		network_list = []
		for line in f:
			if "#" in line:
				network_list.append([])
				first_line_pair = [0,0,0,0]
				cut=False

			else:
				line_counter=len(network_list[-1])
				pairs = list(map(int, line.split(" ")))
				if pairs == first_line_pair or cut==True:
					cut=True
				else:
					network_list[-1].append(np.array(pairs))

				if line_counter == 0:
					first_line_pair = pairs
	network_list = [ np.array(item) for item in network_list]

	return network_list

def get_hexcolor(i, cmap):
	rgb = cmap(i)[:3]
	return mpl.colors.rgb2hex(rgb)

def get_domain_colors(N_particles, bond_arr, length_color_dict):
    domain_colors = [length_color_dict[0]]*N

    if bond_arr.size:
        G = nx.Graph()
        G.add_edges_from(bond_arr[:,:2])
        domains = list(nx.connected_components(G))

        for cluster in domains:
            length = len(cluster)
            if length >=5:
                length=5
            for particle in cluster:
                domain_colors[particle] = length_color_dict[length]

    return domain_colors

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Particle drawing methods')
    parser.add_argument('-ptype', type=str, choices=['dma-as2', 'feq-as2', 'dma-as1', 'dmo-s1', 'dmo-s2', 'dmo-as1'])
    parser.add_argument('-delta', type=float)
    parser.add_argument('-radius', type=float)

    args = parser.parse_args()
    # get all check point values and sort them
    # import the bonds file:
    # file format:
    # ------------------------
    # #new time
    # particle_id1 particle_id2  patch_id1 patch_id2]
    #  ....
    # --------------------------

    #pn_file = "patch_network.dat"

	# colormap for cluster size
    cmap = plt.cm.get_cmap('cividis', 6)
    hex_color = [ get_hexcolor(i, cmap) for i in range(6)]
    hex_color[4] = '#8A2BE2'
    length_color_dict = dict(zip(np.arange(6),hex_color))

    # network_arr format: network_arr.shape = ( frame_i, bond_rows_frame_i )
    #network_arr = read_bonds("patch_network.dat")
    # patch position calculation
    radius=args.radius

    # make frame directory if it doesn't exist
    if not os.path.isdir("./frames"):
        os.mkdir("./frames")
    pos_i = np.fromfile("positions.bin")
    pos_i = np.reshape(pos_i, (-1,3))
    pos_i = pos_i[:,:2]
    orient_i = np.fromfile("orientations.bin")
    orient_i = np.reshape(orient_i, (-1,5))[:,4]
    N=len(pos_i)
    
    fig,ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    #domain_colors.shape = (N,) ( colors per particle )
    #domain_colors = get_domain_colors(N, network_arr[j], length_color_dict)

    sin60 = np.sin(np.pi/3.)
    cos60 = np.cos(np.pi/3.)

    ax0 = np.array([[1,cos60],[0,sin60]])
    edges = np.zeros((4,2))
    ax_n = np.zeros((2,2))

    particle_patches = np.zeros((4,2))

    cr = '#FF9999'
    cb = '#9999FF'
    patch_color_dict = {'dma-as1':[cr,cr,cb,cb],
                        'dma-as2':[cr,cr,cb,cb],
                        'dmo-s1' :[cr,cb,cr,cb],
                        'dmo-s2' :[cr,cb,cr,cb],
                        'dmo-as1':[cr,cb,cr,cb],
                        'feq-as2':[cr,cr,cr,cr]}

    dp=args.delta
    patch_delta_dict = {'dma-as1':[dp,1-dp,dp,dp],
                        'dma-as2':[dp,1-dp,1-dp,1-dp],
                        'dmo-s1' :[dp,dp,dp,1-dp],
                        'dmo-s2' :[dp,1-dp,dp,dp],
                        'dmo-as1':[dp,dp,1-dp,dp],
                        'feq-as2':[dp,1-dp,1-dp,1-dp]}


    for i in range(N):
        #rhombus_color=domain_colors[i]
        #rhombus_color = type_color_dict[patch_i[i,0]]
        #rhombus_color = '#C17DCB'
       
        # dma-as1 
       #rhombus_color = '#8E3952' 
        #rhombus_color = '#994C63'
        # dmo-as1 
        rhombus_color = '#595855'


        rotmat_i = rotation_matrix(orient_i[i])
        ax_n = get_orient(ax0, rotmat_i)

        edges[0] = get_edge_points(pos_i[i],ax_n,np.array([-1,-1]))
        edges[1] = get_edge_points(pos_i[i],ax_n,np.array([+1,-1]))
        edges[2] = get_edge_points(pos_i[i],ax_n,np.array([+1,+1]))
        edges[3] = get_edge_points(pos_i[i],ax_n,np.array([-1,+1]))

        pdelta = patch_delta_dict[args.ptype]
        particle_patches[0] = edges[0] + pdelta[0]*(edges[3]-edges[0])
        particle_patches[1] = edges[2] + pdelta[1]*(edges[3]-edges[2])
        particle_patches[2] = edges[0] + pdelta[2]*(edges[1]-edges[0])
        particle_patches[3] = edges[1] + pdelta[3]*(edges[2]-edges[1])

        rhombi = patches.Polygon(edges, linewidth=0.5, edgecolor='k',facecolor=rhombus_color)
        ax.add_patch(rhombi)
        pcolor = patch_color_dict[args.ptype]

        for pi in range(4):
            patch = patches.Circle((particle_patches[pi,0],particle_patches[pi,1]),
                                    radius=radius,
                                facecolor=pcolor[pi])
            ax.add_patch(patch)

    #ax.scatter(pos_i[:,0], pos_i[:,1],s=0.1)
    plt.xlim((-1,30))
    plt.ylim((-1,40))
    plt.axis("equal")
    plt.axis('off')
    plt.savefig("./frames/frame.pdf")
    
    plt.cla()
    plt.clf()
    plt.close('all')
