"""
the implementation of prior/likelihood/posterior probilities
infer the MAP path
"""
import os
import pickle
import itertools
import osmnx as ox
import numpy as np
import pandas as pd
from math import exp
from networkx import shortest_simple_paths
from collections import defaultdict
from tqdm import tqdm

base_folder = "../dataset"

G = pickle.load(open(f"{base_folder}/road_graph.pkl", "rb"))
G_di = ox.utils_graph.get_digraph(G, "length")

K = 10                  # 默认考虑的最短路径的数量
PRIOR_W = 5             # 先验权重
SIGMA_RATIO = 0.8       # 用于高斯函数的σ值的比率
SPEED_MULTIPLE = 1.25   # 速度的乘数，用于调整速度值

# speed_dicts: 列表，其中每个元素都是一个字典，表示不同时间段的速度数据, 数据源于 train_speed.py
speed_dicts = pickle.load(open(f"{base_folder}/road_speed_completed_slice.pkl", "rb"))
for speed_dict in speed_dicts:
    for key, value in speed_dict.items():
        speed_dict[key] = value * SPEED_MULTIPLE

# 相机节点到节点的映射
camera_node_to_node_to_A = pickle.load(
    open(f"{base_folder}/camera_node_to_node_to_A.pkl", "rb")
)
camera_node_to_node_to_A_p = pickle.load(
    open(f"{base_folder}/camera_node_to_node_to_A_p.pkl", "rb")
)
node_to_A = pickle.load(open(f"{base_folder}/node_to_A.pkl", "rb"))
node_to_A_p = pickle.load(open(f"{base_folder}/node_to_A_p.pkl", "rb"))

edge_info_dict = {}
for u, v, k in G.edges:
    edge_info = G.edges[u, v, k]
    edge_info_dict[edge_info["id"]] = [u, v, k, edge_info]

def get_edge_to_pred_succ_index(G):
    # TODO 搞清楚这个的含义，以`edge`为单位
    """
    Generates a dictionary mapping each edge in the given NetworkX graph to its corresponding indices
    for predecessors and successors.

    Parameters:
    G (nx.Graph): A NetworkX graph.

    Returns:
    Dict[str, Dict[str, int]]: A dictionary where keys are edge identifiers (as strings), and values are
                               dictionaries containing "pred" and "succ" keys, each mapping to an integer
                               representing the index of the edge in the predecessor and successor lists
                               of the nodes connected by the edge.
    """
    edge_to_pred_succ_index = defaultdict(dict)
    for _, info in G.nodes(data=True):
        # 均为唯一值，不会被覆盖
        for i, edge in enumerate(info["pred"]):
            edge_to_pred_succ_index[edge]["pred"] = i
        for i, edge in enumerate(info["succ"]):
            edge_to_pred_succ_index[edge]["succ"] = i

    return edge_to_pred_succ_index

edge_to_pred_succ_index = get_edge_to_pred_succ_index(G)


def my_k_shortest_paths(u, v, k):
    """_summary_

    Args:
        u (_type_): _description_
        v (_type_): _description_
        k (_type_): _description_

    Yields:
        list: 节点列表
    """
    paths_gen = shortest_simple_paths(G_di, u, v, "length")
    for path in itertools.islice(paths_gen, 0, k):
        yield path


# 计算摄像头两两之间的 top 10 最短路径
path = "./data/shortest_path_results.pkl"
if os.path.exists(path):
    shortest_path_results = pickle.load(open(path, "rb"))
else:
    print("Pre-calculating paths between cameras...")
    cameras = pickle.load(open(f"{base_folder}/camera_info.pkl", "rb"))
    camera_nodes = set(x['node_id'] for x in cameras)
    shortest_path_results = {}
    for u in tqdm(camera_nodes):
        for v in camera_nodes:
            if u != v:
                try:
                    shortest_path_results[(u, v)] = [x for x in my_k_shortest_paths(u, v, 10)]
                except:
                    pass
    pickle.dump(shortest_path_results, open(path, "wb"))


def gauss(v, mu):
    sigma = mu * SIGMA_RATIO
    return exp(-((v - mu) ** 2) / sigma**2 / 2)


def route_likelihood(route, ttm, slot, route_type="node"):
    """cal route likelihood in a special `time slot`, considering the travel time. 

    Args:
        route (list): Route in nodes seq or edges seq.
        ttm (float): Travel time
        slot (int): Time slot, 0~23
        route_type (str, optional): _description_. Defaults to "node".

    Returns:
        flaot: likelihood
        list: edges list
    """
    # Formula 5
    speed_dict = speed_dicts[slot]
    total_etm = 0
    if route_type == "node":
        edges = []
        for n1, n2 in zip(route, route[1:]):
            edge = G.edges[n1, n2, 0]
            length = edge["length"]
            edge_id = edge["id"]
            speed = speed_dict[edge_id]
            etm = length / speed
            total_etm += etm
            edges.append(edge_id)
        v = length / (ttm * etm / total_etm)
        return gauss(v, speed), edges
    elif route_type == "edge":
        for edge in route:
            length = edge_info_dict[edge][-1]["length"]
            speed = speed_dict[edge]
            etm = length / speed
            total_etm += etm
        v = length / (ttm * etm / total_etm)
        return gauss(v, speed)


def route_prior(route, return_p_nostart=False):
    """
    Calculates the prior probability of a given route based on camera data and the structure of the road network graph.

    Parameters:
        route (list): A list of edges representing the route.
        return_p_nostart (bool, optional): If True, returns the probability of the route without considering the start node. Default is False.

    Returns:
        float: The prior probability of the route.
        (float, float): If return_p_nostart is True, returns a tuple containing the prior probability of the route and the probability without considering the start node.

    Example:
        >>> route_prior([edge1, edge2, edge3])
        0.05
        >>> route_prior([edge1, edge2, edge3], return_p_nostart=True)
        (0.05, 0.04)
    """
    # Formula 4
    u = edge_info_dict[route[0]][0]
    v = edge_info_dict[route[-1]][1]

    # 从摄像头数据中获取的转移概率矩阵
    A_dict = camera_node_to_node_to_A.get(v, node_to_A)
    A_dict_p = camera_node_to_node_to_A_p.get(v, node_to_A_p)
    
    # p_start
    if u in A_dict:
        tmp = np.sum(A_dict[u], axis=0)
        tmp += np.ones(tmp.shape) * PRIOR_W * A_dict[u].shape[0]
        tmp /= np.sum(tmp)
        p_start = tmp[edge_to_pred_succ_index[route[0]]["succ"]]
    else:
        p_start = 1 / len(G.nodes[u]["succ"])
    
    # p_nostart, 一个累积概率，表示不从起始节点开始的概率
    p_nostart = 1.0
    nodes = [edge_info_dict[x][0] for x in route[1:]]
    for rin, rout, node in zip(route, route[1:], nodes):
        A = A_dict_p.get(node, node_to_A_p.get(node, None))
        if A is None:
            p_nostart *= 1 / len(G.nodes[node]["succ"])
        else:
            prv = edge_to_pred_succ_index[rin]["pred"]
            nxt = edge_to_pred_succ_index[rout]["succ"]
            p_nostart *= A[prv][nxt]
    
    if return_p_nostart:
        return p_start * p_nostart, p_nostart
    else:
        return p_start * p_nostart


def read_k_shortest_path(u, v, k):
    t = shortest_path_results.get((u, v), [])
    return t[:k]


def get_proposals(u, v, k):
    if u == v:
        proposals = []
        for inter in G[u].keys():
            tmp = my_k_shortest_paths(inter, v, 5)
            for t in tmp:
                proposals.append([u] + t)
        return proposals
    return read_k_shortest_path(u, v, k)


def MAP_routing(start_node, end_node, start_time, end_time, num_paths=K, result_type="probability"):
    """
    Infers the most probable route between two nodes based on Maximum A Posteriori (MAP) estimation.

    Parameters:
        start_node (int): The starting node.
        end_node (int): The ending node.
        start_time (float): The starting time.
        end_time (float): The ending time.
        end_node (int, optional): The number of shortest paths to consider. Default is K.
        result_type (str, optional): The type of result to return. Can be one of:
            - "probability": Returns only the maximum posterior probability.
            - "route": Returns the most probable node route and its posterior probability.
            - "edge_route": Returns the most probable edge route and its posterior probability.
            Default is "probability".

    Returns:
        float: If result_type is "probability", returns the maximum posterior probability.
        (list, float): If result_type is "route", returns a tuple containing the most probable node route and its posterior probability.
        (list, float): If result_type is "edge_route", returns a tuple containing the most probable edge route and its posterior probability.

    Example:
        >>> MAP_routing(node1, node2, time1, time2)
        0.05
        >>> MAP_routing(node1, node2, time1, time2, result_type="route")
        ([node3, node4], 0.05)
        >>> MAP_routing(node1, node2, time1, time2, result_type="edge_route")
        ([edge1, edge2], 0.05)
    """
    travel_time = end_time - start_time
    if travel_time == 0:
        travel_time += 0.01

    time_slot = int(start_time / 3600)
    assert travel_time > 0

    proposals = get_proposals(start_node, end_node, num_paths)
    if not proposals:
        if result_type == "probability":
            return 1e-12
        else:
            return [], 1e-12

    posteriors = []
    node_routes = []
    edge_routes = []

    for vpath in proposals:
        likelihood, epath = route_likelihood(vpath, travel_time, time_slot)
        prior = route_prior(epath)
        # Formula 3
        posteriors.append(likelihood * prior)
        
        node_routes.append(vpath[1:-1])
        edge_routes.append(epath)

    max_posterior_index = posteriors.index(max(posteriors))
    max_posterior_value = max(posteriors[max_posterior_index], 1e-12)

    if result_type == "probability":
        return max_posterior_value
    elif result_type == "route":
        return node_routes[max_posterior_index], max_posterior_value
    elif result_type == "edge_route":
        return edge_routes[max_posterior_index], max_posterior_value


def pre_compute_shortest_path(cache_path):
    if os.path.exists(cache_path):
        return 
    
    print("pre-computing...")
    camera_nodes = [x["node_id"] for x in cameras]
    camera_nodes = set(camera_nodes)
    shortest_path_results = {}
    for u in tqdm(camera_nodes):
        for v in camera_nodes:
            if u != v:
                try:
                    paths = [x for x in my_k_shortest_paths(u, v, 10)]
                    shortest_path_results[(u, v)] = paths
                except:
                    pass
    
    print(len(shortest_path_results))
    pickle.dump(shortest_path_results, open(cache_path, "wb"))
