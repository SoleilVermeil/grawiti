import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
import json
import logging
import glob
import argparse
import tarfile
import shutil
import multiprocessing
import time

def perm(M, max_complexity: int) -> float:
    n = M.shape[0]
    complexity = n * 2**n
    logging.debug(f"    Computing the permanent of a {n}x{n} matrix. Estimated complexity: {complexity:.0f}.")
    if complexity > max_complexity:
        logging.debug(f"    Complexity too high (threshold : {max_complexity:.0f}). Returning -1 as fallback.")
        return -1
    d = np.ones(n)
    j =  0
    s = 1
    f = np.arange(n)
    v = M.sum(axis=0)
    p = np.prod(v)
    while (j < n-1):
        v -= 2*d[j]*M[j]
        d[j] = -d[j]
        s = -s
        prod = np.prod(v)
        p += s*prod
        f[0] = 0
        f[j] = f[j+1]
        f[j+1] = j+1
        j = f[0]
    return p/2**(n-1)


def get_G(fsyst, energy: float, t: float) -> float:
    hamiltonian = fsyst.hamiltonian_submatrix(sparse=False)
    SMALL = 1e-10
    G = np.linalg.inv((energy + 1j * SMALL) * np.eye(hamiltonian.shape[0]) - hamiltonian)
    return G

def get_transmission(fsyst, energy: float, t: float) -> float:
    G = get_G(fsyst, energy, t)
    transmission = np.abs(np.trace(G))**2 * t**2
    return transmission

def get_paths(G, i: int, j: int) -> list:
    if nx.has_path(G, i, j):
        paths = list(nx.all_simple_paths(G, source=i, target=j))
        return paths
    else:
        return []
        
def get_pairs_if_pairable(G: nx.Graph, max_complexity: int) -> bool:
    matchings = list(nx.maximal_matching(G))
    # logging.debug(f"Matchings found: {matchings}")
    nodes = G.nodes
    if len(nodes) == 0:
        logging.debug(f"Pairable! (path goes through all atoms)")
        return [(-1, -1)], -1
    if len(matchings) == len(nodes) / 2:
        permanent = perm(nx.to_numpy_array(G), max_complexity=max_complexity)
        logging.debug(f"Pairable! (permanent: {permanent})")
        return matchings, permanent
    else:
        permanent = perm(nx.to_numpy_array(G), max_complexity=max_complexity)
        logging.debug(f"Not pairable. (permanent: {permanent})")
        return [], permanent



def study_paths(
    junction_name: str,
    max_path_count: int = 1e5,
    max_complexity: int = 1e7,
    truncation: bool = True,
    print_numbers: bool = False,
):
    
    # Loading junction
    # ----------------
    
    junction_path = f"data/{junction_name}"
    adjacancy_matrix = np.loadtxt(f"{junction_path}/adjacancy.txt", dtype=int)
    
    properties = json.load(open(f"{junction_path}/properties.json", "r"))
    i = properties["i"]
    j = properties["j"]
    pos = [[p["x"], p["y"]] for p in properties["sites"]]
    
    # Building the graph
    # ------------------

    G = nx.from_numpy_array(adjacancy_matrix)
    logging.info(f"  Number of nodes: {len(list(G.nodes))}")
    logging.info(f"  Number of edges: {len(list(G.edges))}")
    
    # Truncating G to remove unnecessary sites
    # ----------------------------------------
    # NOTE: "Uncecessary sites" means here sites which are equivalent to leads.
    
    if truncation:
        while True:
            i_neighbors = list(G.neighbors(i))
            if len(i_neighbors) == 1:
                G.remove_node(i)
                i = i_neighbors[0]
            else:
                break
        while True:
            j_neighbors = list(G.neighbors(j))
            if len(j_neighbors) == 1:
                G.remove_node(j)
                j = j_neighbors[0]
            else:
                break
            
    # G = nx.relabel_nodes(G, {i: 0, 0: i})
    # last_node_index = max(G.nodes)
    # G = nx.relabel_nodes(G, {j: last_node_index, last_node_index: j})
    # pos[0], pos[i] = pos[i], pos[0]
    # pos[-1], pos[j] = pos[j], pos[-1]
    # i = 0
    # j = last_node_index
            
    # Getting the name of the graph
    # -----------------------------
    
    if os.path.exists(f"data/{junction_name}/plot.png"):
        logging.warning(f"  Junction already studied. Skipping.")
        return True
    
    # Getting all paths connecting sites i and j
    # ------------------------------------------
    
    estimated_path_count = 10**(0.1 * len(list(G.nodes)) - 1) # Empirical estimation
    logging.info(f"  Estimated number of paths: {estimated_path_count:.0f}")
    if estimated_path_count > max_path_count:
        logging.warning(f"  Too many paths. Removing this junction from list.")
        return False
    logging.info(f"  Looking for paths...")
    paths = get_paths(G=G, i=i, j=j)
    paths = sorted(paths, key=lambda x: len(x))
    logging.info(f"  Number of paths found: {len(paths)}")
    # logging.debug(f"Paths found:\n{paths}")
    
    # Drawing the graph: nodes, edges, labels
    # ---------------------------------------
    
    _, ax = plt.subplots(figsize=(5, 5), ncols=1, nrows=1)
    colors = ["red" if K in [i, j] else "blue" for K in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray")
    if print_numbers:
        nx.draw_networkx_labels(G, pos, ax=ax, font_color="white")
    ax.set_aspect("equal")

    # For each path, check if it is pairable
    # --------------------------------------
    
    pairable_paths = []
    permanents = []
    path_drawn = False
    shortest_path_drawn = False
    
    logging.info(f"  Looking at individual paths...")
    for path in paths:
        
        logging.debug(f"    Looking at path: {path}")
        G_without_path = G.copy()
        G_without_path.remove_nodes_from(path)
        pairs, permanent = get_pairs_if_pairable(G_without_path, max_complexity=max_complexity)
        pairable_path = len(pairs) > 0
        pairable_paths.append(pairable_path)
        permanents.append(permanent)
        
        if pairable_path and not path_drawn:
            
            for i in range(len(path)-1):
                
                pos_1 = pos[path[i]]
                pos_2 = pos[path[i+1]]
                ax.plot([pos_1[0], pos_2[0]], [pos_1[1], pos_2[1]], color="red", linewidth=5)
            
            for pair in pairs:
                
                index_1 = pair[0]
                index_2 = pair[1]
                pos_1 = pos[index_1]
                pos_2 = pos[index_2]
                ax.plot([pos_1[0], pos_2[0]], [pos_1[1], pos_2[1]], color="blue", linewidth=5)
                
            path_drawn = True
            
            # NOTE: Drawing the shortest path here is a way to, if multiple
            # paths are "the shortest", to favor paths which are pairable.
            if len(path) == len(paths[0]) and not shortest_path_drawn:
                
                for i in range(len(path)-1):
                    
                    pos_1 = pos[path[i]]
                    pos_2 = pos[path[i+1]]
                    ax.plot(
                        [pos_1[0], pos_2[0]],
                        [pos_1[1], pos_2[1]],
                        color="red",
                        linewidth=10,
                        linestyle=(0, (0.2, 0.2)), # (offset, (line width, space width))
                    )
                    shortest_path_drawn = True
    
    # NOTE: If none of the shortest path(s) is pairable, then we draw it now.
    if not shortest_path_drawn:
        for i in range(len(paths[0])-1):
            pos_1 = pos[paths[0][i]]
            pos_2 = pos[paths[0][i+1]]
            ax.plot(
                [pos_1[0], pos_2[0]],
                [pos_1[1], pos_2[1]],
                color="red",
                linewidth=10,
                linestyle=(0, (0.2, 0.2)), # (offset, (line width, space width))
            )
            shortest_path_drawn = True
    
    # Finishing the drawing
    # ---------------------
    
    ax.axis('off')
    
    # Saving the data
    # ---------------
    # TODO: Store the data in a more efficient way. The field 'paths' should
    # not store each individual path, but rather, for each possible path
    # length, the number of pairable paths and non-pairable paths.
    
    lengths = [l for l in range(1, max([len(path) for path in paths])+1)]
    pairable = []
    unpairable = []
    for i, length in enumerate(lengths):
        pairable.append(sum([len(path) == length and pairable_path for path, pairable_path in zip(paths, pairable_paths)]))
        unpairable.append(sum([len(path) == length and not pairable_path for path, pairable_path in zip(paths, pairable_paths)]))
    
    logging.info(f"  Computing final properties...")
    
    d = {
        "name": junction_name,
        "nodes": len(list(G.nodes)),
        "edges": len(list(G.edges)),
        "pairable": any(pairable_paths),
        "determinant": int(np.linalg.det(nx.to_numpy_array(G))),
        "permanent": int(perm(nx.to_numpy_array(G), max_complexity=max_complexity)),
        "pathsdetailed": [
            {
                "path": path,
                "length": len(path),
                "pairable": pairable_result,
                "permanent": permanent
            }
            for path, pairable_result, permanent in zip(paths, pairable_paths, permanents)
        ],
        "paths": [
            {
                "length": lengths[i],
                "pairable": pairable[i],
                "unpairable": unpairable[i],
            }
            for i in range(len(lengths)) if pairable[i] + unpairable[i] > 0
        ]
    }
    
    if not os.path.exists(f"data_finalized/{junction_name}"):
        os.makedirs(f"data_finalized/{junction_name}")
    
    json.dump(d, open(f"data_finalized/{junction_name}/paths.json", "w"), indent=4)
    
    file_name = f"data_finalized/{junction_name}/plot.png"
    plt.savefig(file_name)
    plt.close()
    
    truncated_adjacancy_matrix = nx.to_numpy_array(G)
    np.savetxt(f"data_finalized/{junction_name}/adjacancy_truncated.txt", truncated_adjacancy_matrix, fmt="%d")
    
    # Copy the other files to the new folder
    # --------------------------------------
    
    shutil.copytree(f"data/{junction_name}", f"data_finalized/{junction_name}", dirs_exist_ok=True)
    
    logging.warning(f"Done studying junction {junction_name}!")
    
    return True
    
if __name__ == '__main__':
    
    # Setting up logging
    # ------------------
    
    logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] [%(levelname)8s] --- %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    # Making the parser
    # -----------------
    
    parser = argparse.ArgumentParser(description='Study paths in a junction.')
    parser.add_argument('--maxpaths', type=int, default=1000, help='Maximum number of paths to study.')
    parser.add_argument('--maxcomplexity', type=int, default=1000000, help='Maximum complexity of the permanent.')
    parser.add_argument('--restrict', type=int, default=0, help='Restrict the indices of junctions to study.')
    
    max_path_count = parser.parse_args().maxpaths
    max_complexity = parser.parse_args().maxcomplexity
    
    # Decompressing the data
    # ----------------------
    # NOTE: The archive is called 'data.tar.gz'
    
    # with tarfile.open("data.tar.gz", "r:gz") as tar:
    #     tar.extractall()
    
    
    if parser.parse_args().restrict == 0:
        junctions = glob.glob("data/junction_*")
    else:
        time.sleep(parser.parse_args().restrict) # Delay to avoid conflicts
        junctions = glob.glob(f"data/junction_{parser.parse_args().restrict}*")
    logging.warning(f"Found {len(junctions)} junctions.")
        
    if not os.path.exists("data_finalized"):
        os.makedirs("data_finalized")
    
    number_of_processes = multiprocessing.cpu_count()
    logging.warning(f"Using {number_of_processes} processes.")
    with multiprocessing.Pool(number_of_processes) as pool:
        pool.starmap(
            study_paths,
            [
                (
                    os.path.basename(junction),
                    max_path_count,
                    max_complexity,
                    True,
                    False,
                )
                for junction in junctions
            ]
        )

    # Compressing the data again
    # --------------------------
    
    # with tarfile.open("data_finalized.tar.gz", "w:gz") as tar:
    #     tar.add("data")
        
    # shutil.rmtree("data")