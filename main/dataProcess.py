import pickle
import numpy as np
from copy import deepcopy
from collections import defaultdict

from cfg import *


def load_data(folder="../dataset"):
    cameras = pickle.load(open(f"{folder}/camera_info.pkl", "rb"))
    G       = pickle.load(open(f"{folder}/road_graph.pkl", "rb"))
    records = pickle.load(open(f"{folder}/records_100w_pca_64.pkl", "rb"))

    cameras_dict = {x["id"]: x for x in cameras}
    f_car = [x["car_feature"] for x in records]
    f_plate = [x["plate_feature"] for x in records]
    f_emb = deepcopy(f_car)

    return cameras, cameras_dict, G, records, f_car, f_plate, f_emb

def get_vid_2_rids(records):
    vid_to_rids = defaultdict(list)
    for i, r in enumerate(records):
        t = r["vehicle_id"]
        if t is not None:
            vid_to_rids[t].append(i)

    return vid_to_rids

def get_cid_2_rids(labels):
    """
    Creates a dictionary mapping cluster IDs (cids) to a list of corresponding record IDs (rids).

    Args:
        labels (list): A list of cluster labels where each element represents the cluster ID for a record.

    Returns:
        dict: A defaultdict object that maps cluster IDs to a list of record IDs.

    Example:
        labels = [0, 1, 1, 0, -1, 2, 2]
        cid_to_rids = get_cid_2_rids(labels)
        cid_to_rids would be {0: [0, 3], 1: [1, 2], 2: [5, 6]}

    Note:
        - The function assumes that the cluster IDs are non-negative integers, and -1 is used to indicate an unassigned record.
        - The function uses a defaultdict(list) to handle the mapping, ensuring that each cluster ID has an associated list, 
        even if no records are assigned to it.
    """
    cid_to_rids = defaultdict(list)
    for i, c in enumerate(labels):
        if c != -1:
            cid_to_rids[c].append(i)

    return cid_to_rids

def get_feat_by_rid(rid, f_car, f_plate):
    car2 = f_car[rid]
    plate2 = f_plate[rid]
    car2 /= np.linalg.norm(car2) + 1e-12
    if plate2 is not None:
        plate2 /= np.linalg.norm(plate2) + 1e-12

    return car2, plate2

def merge_and_split_points(points):
    points = merge_tm_adj_points(points, adj_range=ADJ_RANGE)
    points = merge_tm_adj_points(points, adj_range=ADJ_RANGE)
    points.sort(key=lambda x: x[1]) # (node id, time, rid list)
    cuts = cut_distant_points(points, tm_gap_gate=TM_GAP_GATE)
    
    return points, cuts

def trans_rids_to_points(rids, records, cameras_dict) -> tuple:
    """
    This function transforms a list of record IDs into a list of points, where each point is a tuple containing node ID, time, and record ID. rid -> (node_id, time, rid)

    Parameters:
    rids (list): A list of record IDs.
    records (dict): A dictionary where the key is a record ID and the value is a dictionary containing information about the record, such as camera ID and time.
    cameras_dict (dict): A dictionary where the key is a camera ID and the value is a dictionary containing information about the camera, such as node ID.

    The function iterates over the list of record IDs, and for each record ID, it retrieves the corresponding record from 'records', gets the camera ID from the record, retrieves the corresponding camera from 'cameras_dict', gets the node ID from the camera, and gets the time from the record. It then forms a point as a tuple containing the node ID, time, and record ID, and adds the point to the list of points.

    Returns:
    poin
    """
    points = [ 
        (
            cameras_dict[records[i]["camera_id"]]["node_id"], 
            records[i]["time"], 
            i
        ) for i in rids
    ]

    return points

def tms_adj_range(tms, adj_range=MERGE_CLUSTER_ADJ_RANGE):
    """
    This function transforms a list of `time points` into a list of `time intervals` by extending each time point by a certain range and merging overlapping intervals.

    Parameters:
    tms (list): A list of time points.
    adj_range (float, optional): The range to extend each time point. Defaults to MERGE_CLUSTER_ADJ_RANGE.

    The function iterates over the list of time points, and for each time point, it forms an interval by subtracting and adding 'adj_range' from/to the time point. It then merges overlapping intervals into a single interval.

    Returns:
    adj_ranges (list): A list of time intervals, where each interval is a list containing two elements: the start time and the end time.

    Example:
    input: [33497, 34283, 34525, 34891, 35013, 36512, 42987, 45909, 46650, 47902, 55962, 58448, 67032]
    output: [
                [33197, 33797]
                [33983, 35313]
                [36212, 36812]
                [42687, 43287]
                [45609, 46209]
                [46350, 46950]
                [47602, 48202]
                [55662, 56262]
                [58148, 58748]
                [66732, 67332]
            ]
    """
    tm = tms[0]
    adj_ranges = [[max(tm - adj_range, 0), tm + adj_range]]
    for tm in tms[1:]:
        tm_m = tm - adj_range
        tm_p = tm + adj_range
        if tm_m <= adj_ranges[-1][1]:
            adj_ranges[-1][1] = tm_p
        else:
            adj_ranges.append([tm_m, tm_p])

    return adj_ranges

def merge_tm_adj_points(points, adj_range=ADJ_RANGE):
    """将同一个节点时间上(< 180s)相邻的点合并, rids 列表合并，时间取平均

    Args:
        points (_type_): _description_
        adj_range (_type_, optional): _description_. Defaults to ADJ_RANGE.

    Returns:
        _type_: _description_
    
    Example:
        input(node_id, time, rid):
            [(290, 54547, 7740), (976, 54799, 7741), (976, 54735, 7742), (1098, 55661, 73553), ('camera85', 41992, 73554), (577, 48956, 285205), (577, 58667, 285206), (113, 42250, 385203), (295, 46585, 487655), (295, 35702, 579003), (122, 60842, 579005), (295, 41992, 815166), (38, 58599, 202233), ('camera17', 60583, 361541)]
        output:
            [
                (290, 54547, [7740])
                (976, 54767.0, [7742, 7741])
                (1098, 55661, [73553])
                ('camera85', 41992, [73554])
                (577, 48956.0, [285205])
                (577, 58667.0, [285206])
                (113, 42250, [385203])
                (295, 35702.0, [579003])
                (295, 41992.0, [815166])
                (295, 46585.0, [487655])
                (122, 60842, [579005])
                (38, 58599, [202233])
                ('camera17', 60583, [361541])
            ]
    """

    node_to_tms = defaultdict(list)
    if isinstance(points[0][-1], list):
        for node, tm, rid in points:
            node_to_tms[node].append((tm, rid))
    else:
        for node, tm, rid in points:
            node_to_tms[node].append((tm, [rid]))
    
    merge_points = []
    for node, tms in node_to_tms.items():
        if len(tms) == 1:
            merge_points.append((node, tms[0][0], tms[0][1]))
        else:
            # tms: [(time, rids)]
            tms.sort(key=lambda x: x[0])
            min_tm = tms[0][0]
            one_cluster = [tms[0]]
            for tm, i in tms[1:]:
                if tm - min_tm <= adj_range:
                    one_cluster.append((tm, i))
                else:
                    a, b = list(zip(*one_cluster))
                    merge_points.append((node, np.mean(a), sum(b, [])))
                    one_cluster = [(tm, i)]
                    min_tm = tm
            a, b = list(zip(*one_cluster))
            merge_points.append((node, np.mean(a), sum(b, [])))
    
    return merge_points

def cut_distant_points(points, tm_gap_gate=TM_GAP_GATE):
    """将 点集 按照时间聚类, 相当于 DBSCAN(eps=TM_GAP_GATE, min_samples=1)

    Args:
        points (_type_): _description_
        tm_gap_gate (_type_, optional): _description_. Defaults to TM_GAP_GATE.

    Returns:
        _type_: _description_
    
    Example:
        input:
            [
                (717, 33497, [490378])
                (957, 34283, [340463])
                (738, 34525, [340462])
                (577, 34844, [525621])
                (945, 34891, [340464])
                (284, 35013, [340465])
                (1206, 36512.0, [334125])
                (1093, 42987.0, [553450])
                (1206, 45909.0, [570714])
                (1206, 46650.0, [570715])
                (1206, 47902.0, [570717])
                (404, 55962, [24456])
                (1206, 58448.0, [476519])
                (1093, 67032.0, [0])
            ]
        output:
            [
                [( 717, 33497, [490378])], 
                [( 957, 34283, [340463]), (738, 34525, [340462]), (577, 34844, [525621]), (945, 34891, [340464]), (284, 35013, [340465])],
                [(1206, 36512, [334125])], 
                [(1093, 42987, [553450])], 
                [(1206, 45909, [570714])], 
                [(1206, 46650, [570715])], 
                [(1206, 47902, [570717])], 
                [( 404, 55962, [24456])], 
                [(1206, 58448, [476519])], 
                [(1093, 67032, [0])]
            ]

    """
    cut_points = []
    one_cut = [points[0]]
    tm_last = points[0][1]
    for point in points[1:]:
        tm = point[1]
        if tm - tm_last > tm_gap_gate:
            cut_points.append(one_cut)
            one_cut = [point]
        else:
            one_cut.append(point)
        tm_last = tm
    cut_points.append(one_cut)

    return cut_points

def get_merge_point_idxs(ori_points, all_points):
    """ Identify the `merged` point index, processed in merge_tm_adj_points.
    This function identifies the indices of `points that are merged` during the process of merging temporally adjacent points.
    

    Args:
        ori_points (list): Origin points in the cluster.
        all_points (list): ori_points + noise_points(filter by similarity and time)

    Returns:
        set: merge point indexes
    """
    # orig_idxs: 一开始簇内 idxs
    # points_all: 经过 `similarity` 和 `time` 筛选后 noise set + points
    # merge_point_idxs: 在 points_all 中但不在 points_nc 中的记录, 只能是 merge 后的记录

    ori_ridxs = {x[-1] for x in ori_points}
    merge_point_idxs = set()
    for p in all_points:
        idxs = {x for x in p[-1]}
        diff = idxs - ori_ridxs
        if len(diff) >= len(idxs): 
            continue

        # 有共同元素
        for idx in diff:
            merge_point_idxs.add(idx)    

    return merge_point_idxs

def load_ori_metric(fn):
    """ Load the metric from the initial model result"""
    with open(fn, 'r') as f:
        data = " ".join(f.readlines())
        ori_metric = np.array(eval(data))

    return ori_metric    


def get_node_to_noises(cid_to_noises, records, cameras_dict):
    """get noises to each `node`

    Args:
        cid_to_noises (dict): Cluster index to noise records indexes
        records (_type_): _description_
        cameras_dict (_type_): _description_

    Returns:
        dict((avg_tms, idx)): node to noises. 
    """
    # TODO Params
    node_to_noises = defaultdict(list)
    noises = (noise[0] for noises in cid_to_noises.values() 
                            for noise in noises)
    for idxs in noises:
        tms = [records[i]["time"] for i in idxs]
        camera_id = records[idxs[0]]["camera_id"]
        node = cameras_dict[camera_id]["node_id"]
        node_to_noises[node].append((np.mean(tms), idxs))
    
    return node_to_noises





