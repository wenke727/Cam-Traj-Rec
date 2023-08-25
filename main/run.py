"""
implementation of the iterative framework:
do the clustering and the feedback
"""
import os
os.environ["NUMEXPR_MAX_THREADS"] = '32'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import time
import json
import pickle
import random
import numpy as np
from math import ceil, sqrt
from eval import evaluate
from collections import defaultdict
from loguru import logger

from cfg import *
from utils.parallel_helper import parallel_process
from module.cluster_algorithm import SigCluster, FlatSearcher
from module.routing import MAP_routing, pre_compute_shortest_path
from dataProcess import load_data, get_cid_2_rids, get_feat_by_rid, merge_and_split_points, trans_rids_to_points, tms_adj_range, load_ori_metric, get_merge_point_idxs, get_node_to_noises

from misc import process_segs, filter_noises, identify_noisy_and_recall_points

random.seed(233)
import time


""" aux data process funs """
def calculate_ave_f(rids): # , f_car=f_car, f_plate=f_plate
    """calculate the average feather of `rids`, considering the normalization.

    Args:
        rids (list): the rid list.

    Returns:
        tuple: car_feat, plate_feat
    """
    car1 = [f_car[i] for i in rids]
    plate1 = [f_plate[i] for i in rids]
    car1 = np.mean(np.asarray(car1), axis=0)
    car1 /= np.linalg.norm(car1) + 1e-12
    plate1 = [x for x in plate1 if x is not None]
    if plate1:
        plate1 = np.mean(np.asarray(plate1), axis=0)
        plate1 /= np.linalg.norm(plate1) + 1e-12
    else:
        plate1 = None
    
    return car1, plate1    

def ave_f_unit(rids):
    """calculate the average feather of `rids`

    Args:
        rids (list): the rid list.

    Returns:
        tuple: car_feat, plate_feat
    """
    if len(rids) == 1:
        # BUG
        # i = rids[0]
        return f_car[i], f_plate[i]
    else:
        fs_car = [f_car[i] for i in rids]
        fs_plate = [f_plate[i] for i in rids]
        ave_car = np.mean(np.asarray(fs_car), axis=0)
        # FIXME 归一化？
        fs_plate = [x for x in fs_plate if x is not None]
        if fs_plate:
            ave_plate = np.mean(np.asarray(fs_plate), axis=0)
        else:
            ave_plate = None
        
        return ave_car, ave_plate

cameras, cameras_dict, G, records, f_car, f_plate, f_emb = load_data()

""" aux cluster funcs """
def cal_multi_similarity(car1, plate1, car2, plate2):
    sim_car = car1 @ car2
    if plate1 is not None and plate2 is not None:
        sim_plate = plate1 @ plate2
        sim = 0.2 * sim_car + 0.8 * sim_plate
    else:
        sim = sim_car
    
    return sim

def sim_filter(car, plate, candidates, sim_gate=0.7):
    # TODO 函数示例
    candidates_filter = []
    for noise in candidates:
        idxs2 = noise[1]
        car2, plate2 = calculate_ave_f(idxs2)
        sim = cal_multi_similarity(car, plate, car2, plate2)

        if sim > sim_gate:
            candidates_filter.append(noise)
            
    return candidates_filter

def filter_noise_by_similarity(cid, noise_cids, cid_to_rids):
    """filter by `multi-model similarity`, for each record in each cluster

    Args:
        cid (int): The cluster index.
        noise_cids (list): The list of noise cluster index.
        cid_to_rids (dict): The dict that map `cluser index` to `records index` list.

    Returns:
        list: noise list
    """
    idxs1 = cid_to_rids[cid]
    car1, plate1 = calculate_ave_f(idxs1)

    noise_rids = []
    for nc in noise_cids:
        idxs2 = cid_to_rids[nc]
        for i in idxs2:
            car2, plate2 = get_feat_by_rid(i, f_car, f_plate)
            sim = cal_multi_similarity(car1, plate1, car2, plate2)
            if sim > MERCLUSTER_SIM_GATE: # 0.8
                noise_rids.append(i)

    return noise_rids    

def filter_noise_by_time(rids, noise_rids, adj_range = MERGE_CLUSTER_ADJ_RANGE):
    """
    This function filters out noise data points from a list of data points based on their time intervals with a list of trajectory points.

    Parameters:
    rids (list): A list of record IDs representing trajectory points.
    noise_rids (list): A list of record IDs representing noise data points.
    adj_range (float, optional): The range to extend each time point when forming time intervals. Defaults to MERGE_CLUSTER_ADJ_RANGE (300).

    The function first transforms the record IDs into points, where each point is a tuple containing node ID, time, and record ID. Then, it forms time intervals by extending each time point in 'rids' by 'adj_range' and merging overlapping intervals. Finally, it filters out noise data points in 'noise_rids' that do not fall within any of the time intervals.

    Returns:
    points, points_nc_filter (tuple): A tuple containing two lists of points. The first list 'points' contains points transformed from 'rids', and the second list 'points_nc_filter' contains noise data points in 'noise_rids' that fall within the time intervals. Point format: `(node_id, time, rid)`
    """

    points_nc_filter = []
    points_nc   = trans_rids_to_points(noise_rids, records, cameras_dict)
    points      = trans_rids_to_points(rids, records, cameras_dict) 
    tm_ranges   = tms_adj_range(sorted([x[1] for x in points]), adj_range = adj_range)
    
    for p in points_nc:
        t = p[1]
        flag = False
        for min_t, max_t in tm_ranges:
            if min_t < t < max_t:
                flag = True
                break
        if flag:
            points_nc_filter.append(p)
    
    return points, points_nc_filter


""" ori funs """
def avg_cluster_feat(cs, cid_to_rids, chunksize=500):
    t = time.time()
    args = [cid_to_rids[c] for c in cs]
    results = parallel_process(ave_f_unit, args, min(ceil(len(args) / workers), chunksize), workers)
    car_feat = np.asarray([x[0] for x in results])
    
    tmp = [(x[1], c) for x, c in zip(results, cs) if x[1] is not None]
    plate_feat = np.asarray([x[0] for x in tmp])
    plate_feat_c = [x[1] for x in tmp]

    return car_feat, plate_feat, plate_feat_c


def detect_many_noise(cuts):
    """detect noise by spatial-temporal constrain.

    Args:
        cuts (tuple): (node, time, rid lst)

    Returns:
        (tuple): noises, recall_attempts
    """
    noises = [] # [(rids, code), ..., ]
    recall_attempts = [] # ([u, ut], [v, vt], inter_camera_nodes)
    long_cuts = []
    
    # case: len <= 2 
    for i, one_cut in enumerate(cuts):
        if len(one_cut) == 1:
            total_point_num = sum([len(cut) for cut in cuts])
            if total_point_num == 1 and DO_NOISE_SINGLE_CLUSTER:
                noises.append((one_cut[0][-1], SINGLE_CLUSTER))
            elif total_point_num > 6:
                c, ct = one_cut[0][:2]
                flag = True
                # check merge with previous point
                if i > 0:
                    u, ut = cuts[i - 1][-1][:2]
                    if MAP_routing(u, c, ut, ct) > 0.1:
                        flag = False
                # check merge with next point
                if flag and i < len(cuts) - 1:
                    v, vt = cuts[i + 1][0][:2]
                    if MAP_routing(c, v, ct, vt) > 0.1:
                        flag = False
                
                if flag:
                    noises.append((one_cut[0][-1], ONE_NOISE))
                    
        elif len(one_cut) == 2:
            (u, ut, _), (v, vt, _) = one_cut
            inter_nodes, p = MAP_routing(u, v, ut, vt, result_type='route')
            if p < 0.05:
                noises += [(x[-1], TWO_NOISE) for x in one_cut]
            elif DO_RECALL_ATTEMPT and p > 0.4:
                inter_camera_nodes = [
                    node for node in inter_nodes if "camera" in G.nodes[node]
                ]
                if inter_camera_nodes:
                    recall_attempts.append(([u, ut], [v, vt], inter_camera_nodes))
        else:
            long_cuts.append(one_cut)

    # case: len > 3 
    for one_cut in long_cuts:
        _noises, _recall_candidates = identify_noisy_and_recall_points(
            one_cut, subset_ratio=.5, noise_prob=0.05, min_prob=.45, min_len=2, min_avg_p=.1, min_max_p=.3, do_recall_attempt=True, net=G)
        noises.extend(_noises)
        recall_attempts.extend(_recall_candidates)

    if DO_MERGE_ATTEMPT:
        noise_idxss = {tuple(x[0]) for x in noises}
        non_noise_points = [
            (node, tm)
                for one_cut in cuts
                    for node, tm, idxs in one_cut
                        if tuple(idxs) not in noise_idxss
        ]
        recall_attempts += non_noise_points

    return noises, recall_attempts


def noise_detect_unit(rids):
    """根据时间聚类 rids, 然后类内识别 nose 和 recal attempt

    Args:
        rids (_type_): _description_

    Returns:
        (tuple): noises, recall_attempts
    """
    points = trans_rids_to_points(rids, records, cameras_dict)
    points, cuts = merge_and_split_points(points)

    return detect_many_noise(cuts)


def recall_unit(args):
    recall_attempts, cr_idxs, node_to_noises = args
    car1, plate1 = None, None
    accept_recalls = []
    
    for tmp in recall_attempts:
        if len(tmp) == 3:
            (u, ut), (v, vt), inter_camera_nodes = tmp
            p_base = None
            for node in inter_camera_nodes:
                candidates = [
                    noise for noise in node_to_noises[node] if ut < noise[0] < vt
                ]
                if not candidates:
                    continue
                if car1 is None:
                    car1, plate1 = calculate_ave_f(cr_idxs)
                candidates_filter = sim_filter(car1, plate1, candidates, sim_gate=DENOISE_SIM_GATE) # 0.7
                
                # fomula 9
                if candidates_filter:
                    if p_base is None:
                        p_base = MAP_routing(u, v, ut, vt)
                    for tm, idxs in candidates_filter:
                        p_new = sqrt(MAP_routing(u, node, ut, tm) * MAP_routing(node, v, tm, vt))
                        t = p_new * (1 - MISS_SHOT_P) - p_base * MISS_SHOT_P # MISS_SHOT_P: 0.6
                        if t > 0:
                            accept_recalls.append((idxs, t))
        else:
            node, tm = tmp
            candidates = [
                noise for noise in node_to_noises[node]
                        if tm - MERGE_ATTEMPT_ADJ_RANGE < noise[0] < tm + MERGE_ATTEMPT_ADJ_RANGE # 90
            ]
            if not candidates:
                continue
            if car1 is None:
                car1, plate1 = calculate_ave_f(cr_idxs)
            candidates_filter = sim_filter(car1, plate1, candidates, sim_gate=0.78)
            for tm, idxs in candidates_filter:
                accept_recalls.append((idxs, 0))
                
    return accept_recalls


def update_f_emb(labels, do_update=True):
    global f_emb

    # 0. preprocess: cid_to_rids
    cid_to_rids = get_cid_2_rids(labels)

    # 1. detecting noise ahd recall attempt
    cid_to_noises = {}
    cid_to_recall_attempts = {}

    logger.info("detecting noise...")
    start_time = time.time()
    results = parallel_process(noise_detect_unit, cid_to_rids.values(), 
                               chunksize=min(ceil(len(cid_to_rids) / workers), 200),
                               workers=workers)
    for cid, (noises, recall_attempts) in zip(cid_to_rids.keys(), results):
        if noises:
            cid_to_noises[cid] = noises
        if recall_attempts:
            cid_to_recall_attempts[cid] = recall_attempts
    logger.info("detect noise use time:", time.time() - start_time)

    # 2. recall
    cid_to_accept_recalls = defaultdict(list)
    if DO_RECALL_ATTEMPT:
        logger.debug("recalling...")
        start_time = time.time()
        
        # node_to_noises
        node_to_noises = get_node_to_noises(cid_to_noises, records, cameras_dict)
        args = [(recall_attempts, cid_to_rids[cid], node_to_noises)
                    for cid, recall_attempts in cid_to_recall_attempts.items()]
        results = parallel_process(recall_unit, args, 
                                   chunksize=min(ceil(len(args) / workers), 200),
                                   workers=workers)
        # ? why rids
        rids_to_cid_reward = defaultdict(list)
        for cid, accept_recalls in zip(cid_to_recall_attempts.keys(), results):
            if accept_recalls:
                for idxs, reward in accept_recalls:
                    rids_to_cid_reward[tuple(idxs)].append((cid, reward))
        
        for idxs, cid_reward in rids_to_cid_reward.items():
            max_cid = max(cid_reward, key=lambda x: x[1])[0]
            cid_to_accept_recalls[max_cid] += idxs
        
        logger.info(f"recall use time: {time.time() - start_time:.2f}")

    if not do_update:
        return cid_to_noises, cid_to_accept_recalls

    # 3. update:
    recalled_noises = []
    to_update = []
    for cid, idxs in cid_to_accept_recalls.items():
        recalled_noises += idxs
        rids = cid_to_rids[cid]
        noise_rids = [y for x in cid_to_noises.get(cid, []) for y in x[0]]
        t = set(rids) - set(noise_rids)
        if len(t) >= len(rids) / 2:
            tmp = [f_emb[i] for i in t]
        else:
            tmp = [f_emb[i] for i in rids]
        # formula 10
        tmp = np.mean(np.asarray(tmp), axis=0)
        to_update.append((tmp, idxs))

    # 3.1 update cluster records 
    for tmp, idxs in to_update:
        for i in idxs:
            f_emb[i] = tmp
    
    # 3.2 update noises' feat
    for cid, noises in cid_to_noises.items():
        strong_noises = filter_noises(noises, recalled_noises, 'strong')
        ordinary_noises = filter_noises(noises, recalled_noises, 'ordinary')
        noises = strong_noises | ordinary_noises
        
        rids = cid_to_rids[cid]
        _f_emb = [f_emb[i] for i in rids if i not in noises]
        if _f_emb:
            _f_emb = np.mean(np.asarray(_f_emb), axis=0)
        else:
            _f_emb = np.mean(np.asarray([f_emb[i] for i in rids]), axis=0)
        
        # formula 8
        for i in strong_noises:
            f_emb[i] += 0.3 * (f_emb[i] - _f_emb)
        for i in ordinary_noises:
            f_emb[i] += 0.2 * (f_emb[i] - _f_emb)
    
    return cid_to_noises, cid_to_accept_recalls, cid_to_recall_attempts


def merge_cluster_unit(args):
    """
    This function merges a cluster with its nearest clusters based on their similarity and temporal adjacency.

    Parameters:
    args (tuple): A tuple containing two elements: 
                  1) a tuple (c, ncs), where 'c' is the label of the cluster to be merged and 'ncs' is a set of labels of its nearest clusters;
                  2) a dictionary 'cid_to_rids', where the key is a cluster label and the value is a list of indices of data points belonging to the cluster.

    The function performs the following steps:

    1. Prepares variables for the cluster 'c' and its nearest clusters 'ncs'.
    2. Filters out noise data points from 'ncs' based on their multi-model similarity with 'c'.
    3. Filters out noise data points from 'ncs' based on their time intervals with 'c'.
    4. Merges 'c' and 'ncs' into a new cluster and splits it into segments based on their temporal adjacency.
    5. Processes each segment and identifies noise data points.
    6. Returns the indices of data points in 'ncs' that are not identified as noise and are accepted to be merged with 'c'.

    Returns:
    accept_idxs (list): A list of indices of data points in 'ncs' that are accepted to be merged with 'c'.
    """
    # 1. prepare variables
    (c, ncs), cid_to_rids = args
    rids = cid_to_rids[c]

    # 2. filter by `multi-model similarity`, similarity > .8
    cands = filter_noise_by_similarity(c, ncs, cid_to_rids)
    if not cands:
        return []

    # 3. filter by time intervals, [-300, +300]
    points, cands = filter_noise_by_time(rids, cands)
    if not cands:
        return []

    # 4. merge and split points into `segs`
    points_all = points + cands
    points_all, segs = merge_and_split_points(points_all)

    # 5. process cuts
    noises = process_segs(segs)

    # 6. get accepted idxs
    cand_noise_idxs = {x[-1] for x in cands}
    noise_idxs  = {idx for idxs in noises for idx in idxs}
    merge_point_idxs = get_merge_point_idxs(points, points_all)
    accept_idxs = cand_noise_idxs - noise_idxs - merge_point_idxs
    
    return list(accept_idxs)


def merge_clusters(labels, step, ngpu=1, car_topk=15, plate_topk=30):
    """
    This function merges clusters based on their similarity and temporal adjacency.

    Parameters:
    - labels (list): A list of cluster labels for each data point.
    - step (int): The current iteration step.
    - ngpu (int): The number of GPUs to use for computation. Default is 1.
    - car_topk (int): The number of top similar car clusters to consider for merging. Default is 15.
    - plate_topk (int): The number of top similar plate clusters to consider for merging. Default is 30.

    The function first groups data points into clusters based on the input labels. 
    Then, it iteratively performs the following steps for three rounds:

    1. Divides clusters into two groups: 'big' clusters and 'small' clusters, based on their sizes.
    2. For each 'big' cluster, computes its average feature vector and finds its nearest neighbors among 'small' clusters.
    3. Merges each 'big' cluster with its nearest 'small' clusters if they are temporally adjacent and their similarity exceeds a threshold.

    The function updates the labels and the global feature vectors during the process.

    Returns:
    - labels (list): A list of updated cluster labels for each data point.
    """
    global f_emb
    print("cluster merging...")
    start_time = time.time()
    cid_to_rids = get_cid_2_rids(labels)

    for nn in range(3):
        if nn == 0:
            cs_big = []
            cs_small = []
            for c, rids in cid_to_rids.items():
                t = len(rids)
                if 10 < t <= 20:
                    cs_big.append(c)
                elif t <= 10:
                    cs_small.append(c)
        elif nn == 1:
            cs_big = []
            cs_small = []
            for c, rids in cid_to_rids.items():
                t = len(rids)
                if 20 < t <= 30:
                    cs_big.append(c)
                elif t <= 20:
                    cs_small.append(c)
        elif nn == 2:
            cs_big = []
            cs_small = []
            for c, rids in cid_to_rids.items():
                t = len(rids)
                if 30 < t:
                    cs_big.append(c)
                elif t <= 30:
                    cs_small.append(c)

        car_query, plate_query, plate_query_c  = \
            avg_cluster_feat(cs_big, cid_to_rids, 100)
        car_gallery, plate_gallery, plate_gallery_c = \
            avg_cluster_feat(cs_small, cid_to_rids, 500)

        # rough search
        car_searcher = FlatSearcher(feat_len=64, ngpu=ngpu)
        plate_searcher = FlatSearcher(feat_len=64, ngpu=ngpu)
        car_topk_idxs = car_searcher.search_by_topk(
            query=car_query, gallery=car_gallery, topk=car_topk)[1].tolist()
        plate_topk_idxs = plate_searcher.search_by_topk(
            query=plate_query, gallery=plate_gallery, topk=plate_topk)[1].tolist()
        
        # `cluster id` to `noise cluster ids`
        cid_to_ncids = defaultdict(set)
        for c, rids in zip(cs_big, car_topk_idxs):
            for i in rids:
                cid_to_ncids[c].add(cs_small[i])
        for c, rids in zip(plate_query_c, plate_topk_idxs):
            for i in rids:
                cid_to_ncids[c].add(plate_gallery_c[i])
        
        # merge_cluster
        args = [((c, ncs), cid_to_rids) for c, ncs in cid_to_ncids.items()]
        results = parallel_process(merge_cluster_unit, args,
                                   chunksize=min(ceil(len(args) / workers), 200),
                                   workers=workers)

        # cid_2_accept_rids: `cluster id` -> `accept idxs`  
        accept_rid_2_cids = defaultdict(list)
        cid_2_accept_rids = defaultdict(list)
        for c, accept_idxs in zip(cid_to_ncids.keys(), results):
            for rid in accept_idxs:
                accept_rid_2_cids[rid].append(c)
        for rid, cs in accept_rid_2_cids.items():
            if len(cs) == 1:
                cid_2_accept_rids[cs[0]].append(rid)
            else:
                cid_2_accept_rids[random.sample(cs, 1)[0]].append(rid)

        logger.info(f"\t{step}/{nn}, cluster mapping {len(cid_to_ncids)} -> {sum([len(i) for i in cid_to_ncids.values()])}, "
                    f"accept_rid_2_cids: {len(accept_rid_2_cids)}, cid_2_accept_rids: {len(cid_2_accept_rids)}")
        
        # update `label` and `featrues`
        for c, rids in cid_2_accept_rids.items():
            for rid in rids:
                labels[rid] = c
        for c, rids in cid_2_accept_rids.items():
            tmp = [f_emb[i] for i in cid_to_rids[c]]
            tmp = np.mean(np.asarray(tmp), axis=0)
            
            tmp2 = [f_emb[i] for i in rids]
            tmp2 = np.mean(np.asarray(tmp2), axis=0)
            
            delta = tmp - tmp2
            for rid in rids:
                f_emb[rid] += delta

    logger.info(f"merging consume time: {time.time() - start_time:.2f}")
    return labels


if __name__ == "__main__":
    N_iter = 10
    s = 0.8
    topK = 128
    ngpu = 2
    metrics = []
    load_data = False

    pre_compute_shortest_path("data/shortest_path_results_test.pkl")
    ori_metric = load_ori_metric('./metric/metrics_ori.json')

    for i, operation in zip(range(N_iter), ["merge", "denoise"] * (N_iter // 2)):
        print(f"---------- iter {i}: {operation} -----------")

        if i > -1 and not load_data:
            print("clustering...")
            start_time = time.time()
            cluster = SigCluster(feature_dims=[64, 64, 64], ngpu=ngpu)
            data = [[a, b, c] for a, b, c in zip(f_car, f_plate, f_emb)]
            labels = cluster.fit(data, weights=[0.1, 0.8, 0.1], similarity_threshold=s, topK=topK)
            
            print("clustering consume time:", time.time() - start_time)
            pickle.dump(labels, open(f"label/labels_iter_{i}.pkl", "wb"))
        else:
            labels = pickle.load(open(f"label/labels_iter_{i}.pkl", "rb"))

        precision, recall, fscore, expansion, vid_to_cid = evaluate(records, labels)
        metrics.append((precision, recall, fscore, expansion))
        
        all_close = np.allclose(ori_metric[i], np.array((precision, recall, fscore, expansion)))
        print(f"all_close: {all_close}\n")
        assert all_close, "Check accuracy"

        if operation == "merge":
            merge_clusters(labels, step=i, ngpu=ngpu)
        elif operation == "denoise":
            res = update_f_emb(labels, do_update=True)
            cid_to_noises, cid_to_accept_recalls, cid_to_recall_attempts = res

    json.dump(metrics, open(f"metric/metrics.json", "w"))
