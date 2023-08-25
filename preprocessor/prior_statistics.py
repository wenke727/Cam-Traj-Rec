"""
estimate the prior road transition probability based on the map-matched historical trajectories
"""
import pickle
from collections import defaultdict
import numpy as np
from tqdm import tqdm

N_PRED = 15
PRIOR_W = 3

def save_pickle(fn, obj):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)


G = pickle.load(open("../dataset/road_graph.pkl", "rb"))

cameras = pickle.load(
    open("../dataset/camera_info.pkl", "rb")
)  # [{'id', 'node_id', 'gps', 'gps_orig'}]
nodes = [x["node_id"] for x in cameras]
edge_info_dict = {}
for u, v, k in G.edges:
    edge_info = G.edges[u, v, k]
    edge_info_dict[edge_info["id"]] = [u, v, k, edge_info]


def get_camera_node_to_pred_routes(matched_traj):
    """
    根据匹配的轨迹，获取每个摄像头节点对应的之前途径的路段序列。

    该函数接收一个匹配的轨迹列表作为输入，返回一个字典，将每个摄像头节点映射到一个之前途径的路段序列的列表。
    该函数遍历每个轨迹，从path属性中提取路段id和节点id。
    然后找到位于路径中间的摄像头节点，并记录它们在路径中的位置。
    对于每个摄像头节点，它从路径的开始到它的位置切片，并保留最后N_PRED个路段作为之前途径的路段序列。
    该函数返回一个字典，将每个摄像头节点映射到一个之前途径的路段序列的列表。

    Parameters
    ----------
    matched_traj : list
        一个匹配的轨迹列表，每个轨迹是一个字典，包含path属性，表示轨迹经过的路段和时间戳。

    Returns
    -------
    dict
        一个字典，将每个摄像头节点映射到一个之前途径的路段序列的列表。

    Examples
    --------
    >>> matched_traj = [{"path": [(1, 10), (2, 20), (3, 30), (4, 40)]}, {"path": [(5, 50), (6, 60), (7, 70), (8, 80)]}]
    >>> get_camera_node_to_previous_road_sequences(matched_traj)
    {3: [[1, 2]], 7: [[5, 6]]}
    """
    camera_node_to_pred_routes = defaultdict(list)
    for item in tqdm(matched_traj):
        route = [x[0] for x in item["path"]]
        nodes = [edge_info_dict[x][0] for x in route]
        nodes.append(edge_info_dict[route[-1]][1])
        
        t_dict = {}
        for i, node in enumerate(nodes):
            if i > 1:
                if "camera" in G.nodes[node]:
                    t_dict[node] = i
        for node, pos in t_dict.items():
            t = route[: pos]
            t = t[-N_PRED: ]
            camera_node_to_pred_routes[node].append(t)

    return camera_node_to_pred_routes

def prior_statistic():
    matched_traj = pickle.load(open("data/matched_traj.pkl", "rb"))
    camera_node_to_pred_routes = get_camera_node_to_pred_routes(matched_traj)
    
    # camera_node_to_node_to_A: camera_node_to_node_to_transition_matrix, 以`边`为单位的
    camera_node_to_node_to_A = defaultdict(dict)
    for camera_node, pred_routes in tqdm(camera_node_to_pred_routes.items()):
        for route in pred_routes:
            nodes = [edge_info_dict[x][0] for x in route][1:]
            for (rin, rout), node in zip(zip(route, route[1:]), nodes):
                preds = G.nodes[node]["pred"] # 接入边
                succs = G.nodes[node]["succ"] # 输出边
                len_succs = len(succs)
                if node not in camera_node_to_node_to_A[camera_node]:
                    camera_node_to_node_to_A[camera_node][node] = np.zeros(
                        (len(preds), len_succs), dtype=float
                    )
                elif len_succs > 1:
                    rin = preds.index(rin)
                    rout = succs.index(rout)
                    camera_node_to_node_to_A[camera_node][node][rin][rout] += 1
    save_pickle("../dataset/camera_node_to_node_to_A.pkl", camera_node_to_node_to_A)
    
    # node_to_A
    node_to_A = {}
    for item in tqdm(matched_traj):
        route = [x[0] for x in item["path"]]
        if len(route) < 2:
            continue
        nodes = [edge_info_dict[x][0] for x in route[1:]]
        for (rin, rout), node in zip(zip(route, route[1:]), nodes):
            preds = G.nodes[node]["pred"]
            succs = G.nodes[node]["succ"]
            len_succs = len(succs)
            if node not in node_to_A:
                node_to_A[node] = np.zeros((len(preds), len_succs), dtype=float)
            elif len_succs > 1:
                rin = preds.index(rin)
                rout = succs.index(rout)
                node_to_A[node][rin][rout] += 1
    save_pickle("../dataset/node_to_A.pkl", node_to_A)

    # node_to_A_p
    for node, A in node_to_A.items():
        rin, rout = A.shape
        if rout == 1:
            node_to_A[node] = np.ones((rin, rout))
            continue
        A += np.ones((rin, rout)) * (PRIOR_W + 1)
        A /= np.sum(A, axis=1).reshape(-1, 1)
        node_to_A[node] = A
        assert np.allclose(np.sum(A, axis=1), 1)
    save_pickle("../dataset/node_to_A_p.pkl", node_to_A)

    # camera_node_to_node_to_A_p
    for v1 in camera_node_to_node_to_A.values():
        # ? why
        for node, A in v1.items():
            rin, rout = A.shape
            if rout == 1:
                v1[node] = np.ones((rin, rout))
                continue
            B = A.copy()
            A += np.ones((rin, rout)) * PRIOR_W
            A /= np.sum(A, axis=1).reshape(-1, 1)
            assert np.allclose(np.sum(A, axis=1), 1)
            
            if np.sum(B) >= 4 * rout:
                t = np.sum(B, axis=0)
                t += np.ones(t.shape) * PRIOR_W * rin
                t /= np.sum(t)
                for i, (a, b) in enumerate(zip(A, B)):
                    sum_b = np.sum(b)
                    tmp = max(3, min(rout, 10))
                    if sum_b <= tmp:
                        tmp = sum_b / tmp
                        tmp = 0.3 + 0.3 * tmp
                        A[i] = a * tmp + t * (1 - tmp)
            if np.sum(B) < 1.5 * rout:
                A2 = node_to_A[node]
                A = A * 0.6 + A2 * 0.4
            assert np.allclose(np.sum(A, axis=1), 1)
            v1[node] = A
    save_pickle("../dataset/camera_node_to_node_to_A_p.pkl", camera_node_to_node_to_A)

if __name__ == "__main__":
    prior_statistic()
