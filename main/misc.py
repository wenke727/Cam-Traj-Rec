import numpy as np
from math import ceil, sqrt
from itertools import combinations
from collections import defaultdict
from routing import MAP_routing
from multiprocessing import Pool
from cfg import ONE_NOISE, TWO_NOISE, LONG_NOISE, BLACK_LIST_NOISE, OUT_OF_SUBSET_NOISE

MERGE_CLUSTER_ADJ_RANGE = 300
ADJ_RANGE = 180
TM_GAP_GATE = 720



def trans_rids_to_points(rids, records, cameras_dict) -> tuple:
    """rid -> (node_id, time, rid)

    Args:
        rids (_type_): _description_
        records (_type_): _description_
        cameras_dict (_type_): _description_

    Returns:
        tuple: _description_
    """
    # (node_id, time, rid)
    points = [ 
        (
            cameras_dict[records[i]["camera_id"]]["node_id"], 
            records[i]["time"], 
            i
        ) for i in rids
    ]

    return points

def subsets(arr, k=0, max_return=1000):
    """enumerate the k largest subsets of `arr`

    Args:
        arr (_type_): _description_
        k (int, optional): _description_. Defaults to 0.
        max_return (int, optional): _description_. Defaults to 1000.

    Yields:
        _type_: _description_
    """
    cnt = 0
    if cnt >= max_return:
        return
    for i in range(len(arr), max(0, k - 1), -1):
        # 打印出列表list中所有长度为r的子集
        for j in combinations(arr, i):
            yield j
            cnt += 1
            if cnt >= max_return:
                return

def tms_adj_range(tms, adj_range=MERGE_CLUSTER_ADJ_RANGE):
    """将`时间序列`转换成`时间区间`, 即将每一个节点变更为 [t - 300s, t + 300s],
    然后将各个区间合并

    Args:
        tms (list): _description_
        adj_range (flaot, optional): _description_. Defaults to MERGE_CLUSTER_ADJ_RANGE.

    Returns:
        list: time intervals
    
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

def parralle_process(func, arges, chunksize, workers=32):
    with Pool(processes=workers) as pool:
        results = pool.map(func, arges, chunksize=chunksize)
    
    return results

def greedy_search_trajs(seg, ratio=.5):
    _len = len(seg)
    p_dict = {}
    sub_ps_raw = []
    sub_ps = []
    sub_idxs = []
    
    # enumerate the k largest subsets
    for sub_idx in subsets(list(range(_len)), k=ceil(_len * ratio)):
        ps = []
        for i, j in zip(sub_idx, sub_idx[1:]):
            p = p_dict.get((i, j), None)
            if p is None:
                u, ut = seg[i][:2]
                v, vt = seg[j][:2]
                p = MAP_routing(u, v, ut, vt)
                p_dict[(i, j)] = p
            ps.append(p)
        
        p = np.exp(np.mean(np.log(ps)))
        _p = np.exp(np.sum(np.log(ps)) / (len(ps) + 2))
        sub_ps_raw.append(p)
        sub_ps.append(_p)
        sub_idxs.append(sub_idx)
    
    return _len, sub_ps_raw, sub_ps, sub_idxs

def process_segs(cuts):
    noises = [] # rids
    long_cuts = []
    
    # process short cuts
    for i, one_cut in enumerate(cuts):
        if len(one_cut) == 1:
            noises.append(one_cut[0][-1])
        elif len(one_cut) == 2:
            (u, ut, _), (v, vt, _) = one_cut
            p = MAP_routing(u, v, ut, vt)
            if p < 0.3:
                noises += [x[-1] for x in one_cut]
        else:
            long_cuts.append(one_cut)
    
    # process long cuts
    for one_cut in long_cuts:
        len_cut, sub_ps_raw, sub_ps, sub_idxs = greedy_search_trajs(one_cut, .6)
        # formula 6: sub_idxs 子集
        max_sub_p = max(sub_ps)
        if max_sub_p < 0.3:
            noises += [x[-1] for x in one_cut]
            continue
        
        black_list = set()
        white_list = set()
        for i in range(len_cut):
            p = max(p for p, idx in zip(sub_ps_raw, sub_idxs) if i in idx)
            if p < 0.01:
                black_list.add(i)
                noises.append(one_cut[i][-1])
            else:
                ps = [p for p, idx in zip(sub_ps_raw, sub_idxs)
                        if i in idx and p >= 0.01]
                if (len(ps) >= min(len_cut / 3, 3) and np.mean(ps) > 0.2 and max(ps) > 0.5):
                    white_list.add(i)
        
        # optimal subsets
        opt_cands = (x for x in zip(sub_ps, sub_idxs) 
                        if x[0] > 0.8 * max_sub_p)
        opt_sub_p, opt_sub_idx = max(opt_cands, key = lambda x: (len(x[1]), x[0]))
        # the records outsides the optimal subsets are recognized as `noises`
        for i in set(range(len_cut)) - set(opt_sub_idx) - black_list - white_list:
            noises.append(one_cut[i][-1])

    return noises

def filter_noises(noises, recalled_noises, cls='strong'):
    assert cls in ['strong', 'ordinary']
    types_dict = {'strong': {ONE_NOISE, TWO_NOISE, LONG_NOISE, BLACK_LIST_NOISE},
                  'ordinary': {OUT_OF_SUBSET_NOISE}}

    res = [x[0] for x in noises
           if x[1] in types_dict[cls] \
            and x[0] not in recalled_noises]
    
    return set(sum(res, []))

def merge_and_split_points(points):
    points = merge_tm_adj_points(points, adj_range=ADJ_RANGE)
    points = merge_tm_adj_points(points, adj_range=ADJ_RANGE)
    points.sort(key=lambda x: x[1]) # (node id, time, rid list)
    cuts = cut_distant_points(points, tm_gap_gate=TM_GAP_GATE)
    
    return points, cuts
