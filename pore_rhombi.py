import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
import cv2 as cv
import pore_tool as pt
from math import ceil
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import configparser
from os.path import exists
import networkx as nx
from collections import defaultdict
import multiprocessing
import glob
import re
import argparse
import random
import h5py


def generator_from_fsys(fsys_iterator, nvoxels):
    for dir_star in fsys_iterator:
        print(dir_star)
        dir_i = glob.glob(dir_star)[0]
        fid = dir_i
        run_id = dir_i.split("_")[-1].split("/")[0]

        config = configparser.ConfigParser()

        try:
            config.read("{}para.ini".format(dir_i))

            N = int(config["System"]["Number_of_Particles"])
            phi = float(config["System"]["Packing_Fraction"])
            temperature = float(config["System"]["Temperature"])
            ptype = config["Rhombus"]["rhombus_type"]
            delta = config["Rhombus"]["patch_delta"]

            pos_files = glob.glob("{}positions_*.bin".format(dir_i))

            # get the last value from the string
            def g(x):
                return int(re.findall(r"\d+", x)[-1])

            mc_times = list(map(g, pos_files))

            last_time = np.max(mc_times)

            pos_file = "{}positions_{}.bin".format(dir_i, last_time)
            pos = np.fromfile(pos_file)
            pos = np.reshape(pos, (-1, 3))
            pos = pos[:, :2]
            orient_file = "{}orientations_{}.bin".format(dir_i, last_time)
            orient = np.fromfile(orient_file)
            orient = np.reshape(orient, (-1, 5))[:, 4]

            box_file = "{}Box_{}.bin".format(dir_i, last_time)
            box = np.fromfile(box_file)
            blx = box[3]

        except:
            N = -1
            phi = -1
            temperature = -1
            delta = -1
            last_time = -1
            pos = None
            orient = None
            box = None
            ptype = None
            nvoxels = -1

        yield (
            fid,
            ptype,
            N,
            phi,
            temperature,
            delta,
            run_id,
            last_time,
            pos,
            orient,
            box,
            nvoxels,
        )


def get_edge_points(p, axes, sign_p):
    vertex_n = np.zeros(2)
    vertex_n = p + sign_p[0] * axes[:, 0] / 2.0 + sign_p[1] * axes[:, 1] / 2.0
    return vertex_n


def get_vertices(p, axes):
    vertices = np.zeros((4, 2))
    vertices[0] = get_edge_points(p, axes, np.array([-1, -1]))
    vertices[1] = get_edge_points(p, axes, np.array([+1, -1]))
    vertices[2] = get_edge_points(p, axes, np.array([+1, +1]))
    vertices[3] = get_edge_points(p, axes, np.array([-1, +1]))

    return vertices


def draw(particles, voxcels, box, cells, frame_name):
    scale = 100
    img = np.full((ceil(box.lx * scale), ceil(box.ly * scale), 3), 255, np.uint8)
    for vert_i in particles.vertices:
        cv.fillPoly(img, np.int32([vert_i * scale]), (0, 0, 0))

    for coord_vi in voxcels.coords:
        if voxcels.fill_state[coord_vi] == 0:
            vert_i = voxcels.get_vertices(coord_vi)
            cv.rectangle(
                img,
                np.int32(vert_i[2] * scale),
                np.int32(vert_i[0] * scale),
                (255, 0, 0),
                2,
            )

    outsize = (10000, 10000)
    out = cv.resize(img, outsize)
    cv.imwrite(frame_name, out)


def draw_pos(voxcel_pos, blx, frame_name, voxcels, L):
    scale = 100
    bf = 0.2
    img = np.full(
        (ceil((1 + bf) * blx * scale), ceil((1 + bf) * blx * scale), 3), 255, np.uint8
    )

    axes = np.array([[1, 0], [0, 1]]) * voxcels.lx

    for di in range(L):
        points = voxcel_pos[voxcel_pos[:, 0] == di][:, 1:3]

        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        for p in points:
            vert_i = get_vertices(p, axes) + bf * blx / 2
            cv.rectangle(
                img, np.int32(vert_i[2] * scale), np.int32(vert_i[0] * scale), color, -1
            )

            cv.rectangle(
                img,
                np.int32(vert_i[2] * scale),
                np.int32(vert_i[0] * scale),
                (0, 0, 0),
                1,
            )

    outsize = (5000, 5000)
    out = cv.resize(img, outsize)
    cv.imwrite(frame_name, out)


import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
import seaborn as sns


def draw_select_clusters_on_hull_axis(
    voxcel_pos, voxcels, hull_ratio, domain_lengths, figname
):
    fig, ax = plt.subplots(figsize=(50, 3))
    # sizes=[(0,5),(5,10),(10,20)]
    # selected_domain_ids=[]
    size = (0, 20)
    domain_ids_sizex = np.where(
        (domain_lengths > size[0]) & (domain_lengths < size[1])
    )[0]
    L_di = len(domain_ids_sizex)

    """
    T=10
    for size in sizes:
        domain_ids_sizex = np.where((domain_lengths>size[0]) & (domain_lengths<size[1]))[0]
        L_di = len(domain_ids_sizex)
        if L_di > 1:
            ids = np.random.choice(domain_ids_sizex, size=min(L_di,T))
            selected_domain_ids.extend(ids)
    """
    sns.set()
    sns.set_style("white")
    colors = sns.color_palette("Spectral", L_di).as_hex()
    for j, di in enumerate(domain_ids_sizex):
        points = voxcel_pos[voxcel_pos[:, 0] == di][:, 1:3]
        hr = hull_ratio[di]
        startp = points[0]
        scale = 0.0005
        i = np.random.randint(0, L_di)
        dx = random.uniform(-0.001, 0.001)
        dy = random.uniform(-0.01, 0.01)

        for p in points:
            pi = (p - startp) * scale
            x = pi[0] + hr + dx
            y = pi[1] + scale * domain_lengths[di] + dy
            lx = voxcels.lx * scale
            ax.add_patch(
                patches.Rectangle(
                    (x, y),
                    lx,
                    lx,
                    facecolor=colors[i],
                    alpha=0.5,
                    edgecolor="k",
                    lw=0.1,
                )
            )
        ax.set_xlabel("hull_ratio")
        ax.set_ylabel("domain size")
        ax.set_xlim((0.4, 1.1))
        lticks = np.array([0, 5, 10, 15, 20, 25, 30, 35])
        ax.set_ylim((0, lticks[-1] * scale))
        ax.set_yticks(lticks * scale)
        ax.set_yticklabels(lticks)
        plt.tight_layout()
        plt.savefig(figname)


def get_voxel_array(domains, voxcels):
    arr = []
    for di, domain in enumerate(domains):
        for coord_vi in domain:
            arr.append([di, coord_vi[0], coord_vi[1]])

    arr = np.array(arr)

    arr = np.array(arr)
    arr = np.reshape(arr, (-1, 3))
    return arr


def stitch_cluster(G, next_i, old_coords, new_coords):
    while len(old_coords) > 0:
        neigh = [n for n in G.neighbors(next_i)]
        leftover_neigh = [
            elem_1 for elem_1 in neigh for elem_2 in old_coords if elem_1 == elem_2
        ]
        for ni in leftover_neigh:
            nxi = ni[0]
            nyi = ni[1]

            dxi = next_i[0] - ni[0]
            dyi = next_i[1] - ni[1]

            if abs(dxi) > 1:
                nxi = next_i[0] + np.sign(dxi)
            if abs(dyi) > 1:
                nyi = next_i[1] + np.sign(dyi)

            new_ni = (nxi, nyi)
            new_coords.append(new_ni)
            old_coords.remove(ni)
            nx.relabel_nodes(G, {ni: new_ni}, copy=False)

        next_i = new_ni

        if len(leftover_neigh) == 0:
            k = 1
            while len(leftover_neigh) == 0:
                last_i = new_coords[new_coords.index(next_i) - k]
                neigh = [n for n in G.neighbors(last_i)]
                i = 0

                while len(leftover_neigh) == 0 and i < len(neigh):
                    next_test = neigh[i]
                    neighk = [n for n in G.neighbors(next_test)]
                    leftover_neigh = [
                        elem_1
                        for elem_1 in neighk
                        for elem_2 in old_coords
                        if elem_1 == elem_2
                    ]
                    i += 1

                k += 1
            next_i = next_test

    return old_coords, new_coords


def get_stitched_pos_greedy(voxcels, box, g, domains, arr):
    axes = np.array([[1, 0], [0, 1]]) * voxcels.lx
    voxcel_pos = []

    for di in range(len(domains)):
        coords = arr[arr[:, 0] == di][:, 1:3]
        cid = random.randint(0, len(coords) - 1)
        coord_i = coords[cid]
        distances = voxcels.get_distances_to_voxcel(coord_i, coords)

        new_coords = coord_i - distances
        for nni in new_coords:
            vx = box.origin[0] + voxcels.lx * nni[0] + voxcels.lx / 2
            vy = box.origin[1] + voxcels.lx * nni[1] + voxcels.lx / 2
            voxcel_pos.append([di, vx, vy])

    voxcel_pos = np.array(voxcel_pos)
    edge_pos = []
    for di in range(len(domains)):
        coords = voxcel_pos[voxcel_pos[:, 0] == di][:, 1:3]
        for coord_vi in coords:
            vert_i = get_vertices(coord_vi, axes)
            for vi in vert_i:
                edge_pos.append([di, vi[0], vi[1]])

    edge_pos = np.array(edge_pos)

    return voxcel_pos, edge_pos


def get_stitched_pos(voxcels, box, G, domains, arr):
    axes = np.array([[1, 0], [0, 1]]) * voxcels.lx
    voxcel_pos = []
    for di in range(len(domains)):
        coords = arr[arr[:, 0] == di][:, 1:3]
        old_coords = [(i, j) for i, j in coords]
        rand_c = coords[random.randint(0, len(coords) - 1)]
        ci = (rand_c[0], rand_c[1])
        new_coords = [ci]
        old_coords.remove(ci)

        old_coords, new_coords = stitch_cluster(G, ci, old_coords, new_coords)

        for entry in new_coords:
            nni = entry
            vx = box.origin[0] + voxcels.lx * nni[0] + voxcels.lx / 2
            vy = box.origin[1] + voxcels.lx * nni[1] + voxcels.lx / 2
            voxcel_pos.append([di, vx, vy])

    voxcel_pos = np.array(voxcel_pos)
    edge_pos = []

    for di in range(len(domains)):
        coords = voxcel_pos[voxcel_pos[:, 0] == di][:, 1:3]
        for coord_vi in coords:
            vert_i = get_vertices(coord_vi, axes)
            for vi in vert_i:
                edge_pos.append([di, vi[0], vi[1]])

    edge_pos = np.array(edge_pos)

    return voxcel_pos, edge_pos


def get_ALL_circumferences(G, domains, n_edges, vlx):
    def ete_degree(G):
        degree = defaultdict(int)
        for u, v in ((u, v) for u, v, d in G.edges(data=True) if d["ete"] == 1):
            degree[u] += 1
            degree[v] += 1
        return degree

    degree_dict = ete_degree(G)

    def get_ete_degree(u):
        return degree_dict[u]

    domain_dg = []
    for domain in domains:
        domain_dg.append(list(map(get_ete_degree, domain)))

    def get_circumference(domain_dgi):
        arr = (n_edges - np.array(domain_dgi)) * vlx
        cf = np.sum(arr)
        return cf

    circumferences = [get_circumference(dd) for dd in domain_dg]
    return circumferences


def calculate(vals):
    (
        fid,
        ptype,
        N,
        phi,
        temperature,
        delta,
        run_id,
        last_time,
        pos,
        orient,
        box,
        nvoxels,
    ) = vals

    meta = {}
    fid = fid.split("/")[1]

    if pos is None:
        meta["fid"] = "{}".format(fid)
        meta["ptype"] = ptype
        meta["phi"] = phi
        meta["temperature"] = temperature
        meta["delta"] = delta
        meta["last_time"] = last_time
        meta["run_id"] = run_id
        meta["N"] = N
        meta["voxel_side_length"] = -1

        df = pd.DataFrame()
        df["pore_area"] = -1
        df["circumference"] = -1

        if args.wgeom == "yes":
            df["percent_explained_variance"] = -1
            df["convex_hull_ratio"] = -1

    else:
        print("init")

        blx = box[3]
        ndim = 2
        side_length = 1
        voxcels_per_cell = nvoxels
        threshold_overlap = 0.25
        N_trial = 100

        # Generate particles object
        rlx = side_length
        particles = pt.Rhombi(pos, orient, rlx, ndim)

        # Generate box object
        origin = box[:2] - blx / 2.0
        box = pt.Box(origin, blx, ndim)

        # Initalize cells object
        # Absolute lmin = particles.sigma
        lmin = particles.sigma
        cells = pt.Cells(lmin, box.lx, ndim)

        # Initialize voxels object
        vxl = cells.lx / voxcels_per_cell
        vxn = cells.nx * voxcels_per_cell
        voxcels = pt.Voxcels(vxl, vxn, box.origin, ndim)

        # Initalize cells object
        cells.list_generate(particles.pos, voxcels.coords, voxcels.pos, box.origin)
        print("calc voxel state")
        # Calculate voxcel state
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

                for i, j in np.argwhere(
                    ndist < (particles.outer_radius + voxcels.outer_radius)
                ):
                    overlap_volume_i = particles.estimate_volume(
                        voxcels, vcoords[i], pcoords[j], box.lx, N_trial
                    )
                    if overlap_volume_i > threshold_overlap:
                        voxcels.set_to_filled(vcoords[i])

        print("get links")
        # Generate links between empty voxcels
        voxcels.get_links()
        frame_name = (
            "pore_results/{}_phi_{}_delta_{}_temp_{}_run_{}_nvoxels_{}.pdf".format(
                ptype, phi, delta, temperature, run_id, nvoxels
            )
        )

        print("draw voxels")
        draw(particles, voxcels, box, cells, frame_name)

        print("get pore area")
        # RESULT: pore area/ domain sizes
        pore_areas, domain_lengths, domains, G = pt.get_pore_volume(voxcels)

        if args.wgeom == "no":
            print("Get circumferences")
            circumferences = get_ALL_circumferences(
                G, domains, particles.N_edges, voxcels.lx
            )

            meta["fid"] = "{}_{}".format(fid, last_time)
            meta["ptype"] = ptype
            meta["phi"] = float(phi)
            meta["temperature"] = float(temperature)
            meta["delta"] = float(delta)
            meta["last_time"] = int(last_time)
            meta["run_id"] = int(run_id)
            meta["N"] = int(N)
            meta["voxel_side_length"] = float(vxl)

            df = pd.DataFrame()
            df["pore_area"] = pore_areas
            df["circumference"] = circumferences

        if args.wgeom == "yes":
            size_T = 1000
            for domain_i in domains:
                if len(domain_i) > size_T:
                    for node_j in domain_i:
                        G.remove_node(node_j)

            domains_mid = list(nx.connected_components(G))
            domain_lengths_mid = np.array(
                [len(domain_mid) for domain_mid in domains_mid]
            )
            pore_areas_mid = voxcels.lx * domain_lengths_mid

            print("get circumference")
            # RESULT: pore circumference
            circumferences_mid = get_ALL_circumferences(
                G, domains_mid, particles.N_edges, voxcels.lx
            )

            # Get PBC stitched clusters for convex hull to get asymmetry measures:
            arr = get_voxel_array(domains_mid, voxcels)
            voxel_pos, edge_pos = get_stitched_pos_greedy(
                voxcels, box, G, domains_mid, arr
            )

            print("draw shifted voxels")
            shifted_frame_name = "pore_results/{}_phi_{}_delta_{}_temp_{}_run_{}_voxcels_shifted.png".format(
                ptype, phi, delta, temperature, run_id
            )

            draw_pos(voxel_pos, box.lx, shifted_frame_name, voxcels, len(domains_mid))

            print("get hull ratio")
            # RESULT: ratio pore volume and convex hull: general asymmetry measure
            hull = []
            for di in range(len(domains_mid)):
                points = edge_pos[edge_pos[:, 0] == di][:, 1:3]
                chull = ConvexHull(points)
                hull.append([pore_areas_mid[di], chull.volume])

            hull = np.array(hull)
            hull_ratio = hull[:, 0] / hull[:, 1]
            hull_ratio = hull_ratio / 4

            figname = "pore_results/{}_phi_{}_delta_{}_temp_{}_run_{}_pore_hull_ratio.pdf".format(
                ptype, phi, delta, temperature, run_id
            )
            draw_select_clusters_on_hull_axis(
                voxel_pos, voxcels, hull_ratio, domain_lengths_mid, figname
            )

            print("get pca")
            # RESULT: explained variance by largest principal component:
            # how elongated are pores
            pca = PCA(n_components=2)
            xlambda = []
            for di in range(len(domains_mid)):
                points = edge_pos[edge_pos[:, 0] == di][:, 1:3]
                pca.fit(points)
                xlambda.append(pca.explained_variance_ratio_[0])

            meta["fid"] = "{}_{}".format(fid, last_time)
            meta["ptype"] = ptype
            meta["phi"] = float(phi)
            meta["temperature"] = float(temperature)
            meta["delta"] = float(delta)
            meta["last_time"] = int(last_time)
            meta["run_id"] = int(run_id)
            meta["N"] = int(N)
            meta["voxel_side_length"] = float(vxl)

            df = pd.DataFrame()
            df["pore_area"] = pore_areas_mid
            df["percent_explained_variance"] = xlambda
            df["convex_hull_ratio"] = hull_ratio
            df["circumference"] = circumferences_mid

    return meta, df


if __name__ == "__main__":
    ptypes = [
        "double_manta_asymm_1",
        "double_mouse_asymm_1",
        "double_mouse_symm_1",
        "double_mouse_symm_2",
    ]
    # read data either through files system via glob or via db
    parser = argparse.ArgumentParser()
    parser.add_argument("-run_id", type=str)
    parser.add_argument("-ncores", type=int)
    parser.add_argument("-ptype", type=str, choices=ptypes)
    parser.add_argument("-wgeom", type=str, choices=["yes", "no"], default="no")
    parser.add_argument("-nvoxels", type=int)

    args = parser.parse_args()

    temps = [0.01]
    phis = [0.05, 0.125, 0.15, 0.2, 0.3, 0.5]
    deltas = [0.2]
    ptype = args.ptype
    dir_list = []
    for phi in phis:
        for temp in temps:
            for delta in deltas:
                for run in range(1, 2):
                    dir_list.append(
                        "batch*/{}_phi_{}_delta_{}_temp_{}_run_{}/".format(
                            ptype, phi, delta, temp, run
                        )
                    )

    nvoxels = int(args.nvoxels)
    gen_fsys = generator_from_fsys(dir_list, nvoxels)

    N_CORES = int(args.ncores)
    N_CORES_MAX = 8

    if N_CORES > 1 and N_CORES <= N_CORES_MAX:
        print("Multiprocessing with {} cores".format(N_CORES))

        pool = multiprocessing.Pool(N_CORES)
        results = pool.map(calculate, gen_fsys)
        pool.close()
        pool.join()

        f = h5py.File("pore_results/pore_measures_{}.h5".format(args.run_id), "w")
        for i, res in enumerate(results):
            if res[0]["phi"] == -1:
                print(
                    "Warning: Folder {} has issues. Results not evaluated".format(
                        res[0]["fid"]
                    )
                )

            else:
                grp = f.create_group(res[0]["fid"])

                for key in res[0]:
                    grp.attrs[key] = res[0][key]

                dfi = res[1]
                for col in dfi.columns:
                    dset = grp.create_dataset(col, len(dfi), dtype="f")
                    dset[...] = dfi[col]

        f.close()

    if N_CORES == 1:
        print("Calculating for single core job")
        f = h5py.File("pore_results/pore_measures_{}.h5".format(args.run_id), "w")

        for vals in gen_fsys:
            res = calculate(vals)
            if res[0]["phi"] == -1:
                print(
                    "Warning: Folder {} has issues. Results not evaluated".format(
                        res[0]["fid"]
                    )
                )

            else:
                grp = f.create_group(res[0]["fid"])

                for key in res[0]:
                    grp.attrs[key] = res[0][key]

                dfi = res[1]
                for col in dfi.columns:
                    dset = grp.create_dataset(col, len(dfi), dtype="f")
                    dset[...] = dfi[col]

        f.close()

    if N_CORES > N_CORES_MAX:
        print(
            "Too many cores allocated, please do not use more than {} cores".format(
                N_CORES_MAX
            )
        )
        exit()
