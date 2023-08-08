import numpy as np 
import pandas as pd 
import networkx as nx 
import matplotlib as pyplot 
from collections import defaultdict 
import pore_tool as pt 
from math import ceil
import yaml
from yaml.loader import SafeLoader
import argparse
#TODO Write out a combo of 27 cells with particles and pores as spheres in different colors 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str)
    args = parser.parse_args()

    print("read data...")
    fdir = args.f 
    df_pos = pd.read_csv(fdir, delim_whitespace=True, names=["type,",'x','y','z'])
    pos = df_pos[['x','y','z']].values 

    print("initialize variables...")
    with open('param_pore_biogel.yaml') as f:
        param = yaml.load(f, Loader=SafeLoader)

    ndim = int(param['dimension'])

    # generate particles object 
    sigma= float(param['sigma'])
    particles = pt.Spheres(pos,ndim,sigma)

    # generate box object 
    lx = 30 
    origin = (-15)*np.ones(ndim)
    box = pt.Box(origin,lx,ndim)

    # initalize cells object
    # absolute lmin = particles.sigma 
    lmin = 3*particles.sigma 
    cells = pt.Cells(lmin,box.lx,ndim)

    # initialize voxels object
    voxcels_per_cell = int(param['voxcels_per_cell'])
    vxl = cells.lx/voxcels_per_cell
    vxn = cells.nx*voxcels_per_cell 
    voxcels = pt.Voxcels(vxl,vxn, box.origin,ndim)
    
    print("generate cell lists...")
    # initalize cells object 
    cells.list_generate(particles.pos,voxcels.coords, voxcels.pos, box.origin)
  
    N_trial = int(param['N_trial'])
    threshold_overlap = float(param['threshold_overlap'])

    print("calculate voxcel state...")
    for ci in cells.coords:

        vcoords = cells.voxcel_list[ci]
        pcoords = []
        neighbours = cells.get_neighbour_coords(ci)
        for nci in neighbours:
            pcoords.extend(cells.particle_list[nci])

        if pcoords:
            pos_v = np.array([voxcels.pos[i] for i in vcoords])
            pos_c = particles.pos[pcoords]
            dist, ndist = pt.get_vdistance(pos_v, pos_c, box.lx)

            for i, j in np.argwhere(ndist < (particles.outer_radius+voxcels.outer_radius)):
                overlap_volume_i = particles.estimate_volume(
                    voxcels, vcoords[i], pcoords[j], box.lx, N_trial)
                if overlap_volume_i > threshold_overlap:
                    voxcels.set_to_filled(vcoords[i])

    print("generate links between empty voxels ...")
    voxcels.get_links()

    print("get all pore volumes and domain lengths ")
    pore_volumes, domain_lengths, domains, Graph = pt.get_pore_volume(voxcels)

    outfile = param['outfile']
    with open(outfile, 'w') as f:
        for v, d in zip(pore_volumes, domain_lengths):
            f.write("{},{}\n".format(v, d))

    N_empty = np.sum(domain_lengths)
    n=0 
    with open('voxels_biogel_{}.xyz'.format(param['outpos']), 'w') as f:
        f.write("{}\n".format(N_empty*8))
        f.write("Voxels for biogel (cubes)\n")
        for ci in cells.coords:
            vcoords = cells.voxcel_list[ci]
            for vi in vcoords: 
                if voxcels.fill_state[vi] == 0:
                    n+=1 
                    #pos = voxcels.pos[vi] 
                    vertices = voxcels.get_vertices(vi)
                    #lprint(vertices)
                    for vert in vertices:
                        f.write("V   {}   {}   {}\n".format(vert[0],vert[1],vert[2]))
