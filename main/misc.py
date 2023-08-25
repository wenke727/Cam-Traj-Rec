import numpy as np
from math import ceil, sqrt
from itertools import combinations
from collections import defaultdict
from module.routing import MAP_routing
from multiprocessing import Pool
from cfg import ONE_NOISE, TWO_NOISE, LONG_NOISE, BLACK_LIST_NOISE, OUT_OF_SUBSET_NOISE

MERGE_CLUSTER_ADJ_RANGE = 300
ADJ_RANGE = 180
TM_GAP_GATE = 720


def subsets(arr, k=0, max_return=1000):
    """
    Enumerate the k largest subsets of the given array `arr`.
    
    Parameters:
        arr (list): The array for which subsets are to be generated.
        k (int, optional): The size of the subset. Defaults to 0.
        
    Yields:
        tuple: Subsets of the array `arr`.
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

def parralle_process(func, arges, chunksize, workers=32):
    with Pool(processes=workers) as pool:
        results = pool.map(func, arges, chunksize=chunksize)
    
    return results

def cal_traj_score(points, sub_idx, memo={}):
    # TODO
    _probs = []
    for i, j in zip(sub_idx, sub_idx[1:]):
        prob = memo.get((i, j), None)
        if prob is None:
            u, ut = points[i][:2]
            v, vt = points[j][:2]
            prob = MAP_routing(u, v, ut, vt)
            memo[(i, j)] = prob
        _probs.append(prob)
    
    prob = np.exp(np.mean(np.log(_probs)))
    # fomula 7
    score = np.exp(np.sum(np.log(_probs)) / (len(_probs) + 2))

    return prob, score

def greedy_search_trajs(seg, ratio=.5):
    """
    Perform a greedy search on trajectories by enumerating over the largest subsets of the input segment.
    
    Parameters:
    - seg (list): The input segment containing trajectory data.
    - ratio (float, optional): The ratio to determine the size of the subsets. Defaults to 0.5.
    
    Returns:
    - tuple: Contains the following elements:
        - _len (int): Length of the input segment.
        - sub_ps_raw (list): Raw scores for each subset.
        - sub_ps (list): Adjusted scores for each subset.
        - sub_idxs (list): Indices of each subset.
    """
    _len = len(seg)
    memo = {}
    sub_ps_raw = []
    sub_ps = []
    sub_idxs = []
    
    # enumerate the k largest subsets
    for sub_idx in subsets(list(range(_len)), k=ceil(_len * ratio)):
        _probs = []
        for i, j in zip(sub_idx, sub_idx[1:]):
            p = memo.get((i, j), None)
            if p is None:
                u, ut = seg[i][:2]
                v, vt = seg[j][:2]
                p = MAP_routing(u, v, ut, vt)
                memo[(i, j)] = p
            _probs.append(p)
        
        p = np.exp(np.mean(np.log(_probs)))
        # fomula 7
        _p = np.exp(np.sum(np.log(_probs)) / (len(_probs) + 2))
        sub_ps_raw.append(p)
        sub_ps.append(_p)
        sub_idxs.append(sub_idx)
    
    return _len, sub_ps_raw, sub_ps, sub_idxs

def process_segs(cuts):
    """ for merge cluster """
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
        _noises, _ = identify_noisy_and_recall_points(one_cut)
        noises.extend([i for i, _ in _noises])

    return noises

def filter_noises(noises, recalled_noises, cls='strong'):
    assert cls in ['strong', 'ordinary']
    types_dict = {'strong': {ONE_NOISE, TWO_NOISE, LONG_NOISE, BLACK_LIST_NOISE},
                  'ordinary': {OUT_OF_SUBSET_NOISE}}

    res = [x[0] for x in noises
           if x[1] in types_dict[cls] \
            and x[0] not in recalled_noises]
    
    return set(sum(res, []))

def identify_noisy_and_recall_points(one_cut, subset_ratio=.6,
                                     noise_prob=0.3, min_prob=0.3,
                                     min_len=3, min_avg_p=0.2, min_max_p=0.5,
                                     do_recall_attempt=False, net=None
                                     ):
    """
    Identify noisy trajectory points and potential recall attempts from a given trajectory segment.
    
    Parameters:
        one_cut (list): The trajectory segment to be processed.
        subset_ratio (float, optional): Ratio for the greedy_search_trajs function. Defaults to 0.6.
        noise_prob (float, optional): Probability threshold for noisy trajectories. Defaults to 0.3.
        min_prob (float, optional): Minimum probability for valid trajectories. Defaults to 0.3.
        min_len (int, optional): Minimum length for certain criteria. Defaults to 3.
        min_avg_p (float, optional): Minimum average probability for certain criteria. Defaults to 0.2.
        min_max_p (float, optional): Minimum maximum probability for certain criteria. Defaults to 0.5.
        do_recall_attempt (bool, optional): Flag to determine recall attempts. Defaults to False.
        net (object, optional): Network structure containing node information. Defaults to None.
        
    Returns:
        tuple: A tuple containing:
            - noises (list): Identified noisy trajectory points with their labels.
            - recall_attempts (list): Potential recall attempts, if any.
    """
    # 1. Initialization
    noises = []
    _len, sub_ps_raw, sub_ps, sub_idxs = greedy_search_trajs(one_cut, subset_ratio)

    opt_sub_p = 0
    max_sub_p = max(sub_ps)

    # Identifying Noisy Trajectories
    # formula 6: sub_idxs 子集
    if max_sub_p < noise_prob:
        noises += [(x[-1], LONG_NOISE) for x in one_cut]
    elif max_sub_p >= min_prob:
        black_list = set()
        white_list = set()
        for i in range(_len):
            filterd_traj_ps = [p for p, idx in zip(sub_ps_raw, sub_idxs) if i in idx]
            max_prob = max(filterd_traj_ps)
            if max_prob < 0.01:
                black_list.add(i)
                noises.append((one_cut[i][-1], BLACK_LIST_NOISE)) # diff
            else:
                ps = list(filter(lambda p: p >= 0.01, filterd_traj_ps))
                if (len(ps) >= min(_len / 3, min_len) \
                        and np.mean(ps) > min_avg_p \
                        and max(ps) > min_max_p):
                    white_list.add(i)
        
        # optimal subsets
        opt_cands = (x for x in zip(sub_ps, sub_idxs) if x[0] > 0.8 * max_sub_p)
        opt_sub_p, opt_sub_idx = max(opt_cands, key = lambda x: (len(x[1]), x[0]))
        
        # the records outsides the optimal subsets are recognized as `noises`
        for i in set(range(_len)) - set(opt_sub_idx) - black_list - white_list:
            noises.append((one_cut[i][-1], OUT_OF_SUBSET_NOISE)) # diff

    if not do_recall_attempt or opt_sub_p <= 0.3:
        return noises, []

    recall_attempts = []
    for i, j in zip(opt_sub_idx, opt_sub_idx[1:]):
        u, ut, _ = one_cut[i]
        v, vt, _ = one_cut[j]
        inter_nodes, _ = MAP_routing(u, v, ut, vt, result_type='route')
        inter_camera_nodes = [
            node for node in inter_nodes if "camera" in net.nodes[node]
        ]
        if inter_camera_nodes:
            recall_attempts.append(((u, ut), (v, vt), inter_camera_nodes))    
        
    
    return noises, recall_attempts

