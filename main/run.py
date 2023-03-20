"""
implementation of the iterative framework:
do the clustering and the feedback
"""
import os
os.environ["NUMEXPR_MAX_THREADS"] = '32'

import time
import json
import pickle
import random
import numpy as np
from tqdm import tqdm
from math import ceil, sqrt
from eval import evaluate
from copy import deepcopy
from multiprocessing import Pool
from collections import defaultdict

from cluster_algorithm import SigCluster, FlatSearcher
from routing import MAP_routing, MAP_routing_return_route, my_k_shortest_paths

from misc import subsets, tms_adj_range, trans_rids_to_points, merge_tm_adj_points, cut_distant_points
from misc import get_merge_point_idxs, process_segs, filter_noises, merge_and_split_points, greedy_search_trajs
from cfg import *
from utils import parallel

random.seed(233)
import time
DO_RECALL_ATTEMPT = True
DO_MERGE_ATTEMPT = False
DO_NOISE_SINGLE_CLUSTER = False

MERCLUSTER_SIM_GATE = 0.8
DENOISE_SIM_GATE = 0.7
MISS_SHOT_P = 0.6
ADJ_RANGE = 180
MERGE_ATTEMPT_ADJ_RANGE = ADJ_RANGE / 2
TM_GAP_GATE = 720
MERGE_CLUSTER_ADJ_RANGE = 300

workers = 64


cameras = pickle.load(open("../dataset/camera_info.pkl", "rb"))
cameras_dict = {x["id"]: x for x in cameras}

G = pickle.load(open("../dataset/road_graph.pkl", "rb"))

records = pickle.load(open("../dataset/records_100w_pca_64.pkl", "rb"))

f_car = [x["car_feature"] for x in records]
f_plate = [x["plate_feature"] for x in records]
f_emb = deepcopy(f_car)

vid_to_rids = defaultdict(list)
for i, r in enumerate(records):
    t = r["vehicle_id"]
    if t is not None:
        vid_to_rids[t].append(i)

""" aux funs """
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

def get_feat_by_rid(rid):
    car2 = f_car[rid]
    plate2 = f_plate[rid]
    car2 /= np.linalg.norm(car2) + 1e-12
    if plate2 is not None:
        plate2 /= np.linalg.norm(plate2) + 1e-12

    return car2, plate2

def cal_multi_similarity(car1, plate1, car2, plate2):
    sim_car = car1 @ car2
    if plate1 is not None and plate2 is not None:
        sim_plate = plate1 @ plate2
        sim = 0.2 * sim_car + 0.8 * sim_plate
    else:
        sim = sim_car
    
    return sim

def sim_filter(car, plate, candidates, sim_gate=0.7):
    candidates_filter = []
    for noise in candidates:
        idxs2 = noise[1]
        car2, plate2 = calculate_ave_f(idxs2)
        sim = cal_multi_similarity(car, plate, car2, plate2)

        if sim > sim_gate:
            candidates_filter.append(noise)
            
    return candidates_filter

def get_node_to_noises(cid_to_noises):
    node_to_noises = defaultdict(list) # [(avg_tms, idx), ..., ]
    noises = (noise[0] for noises in cid_to_noises.values() 
                            for noise in noises)
    for idxs in noises:
        tms = [records[i]["time"] for i in idxs]
        camera_id = records[idxs[0]]["camera_id"]
        node = cameras_dict[camera_id]["node_id"]
        node_to_noises[node].append((np.mean(tms), idxs))
    
    return node_to_noises


""" ori funs """
def avg_cluster_feat(cs, cid_to_rids, chunksize=500):
    args = [cid_to_rids[c] for c in cs]
    with Pool(processes=workers) as pool:
        results = pool.map(
            ave_f_unit, args, chunksize=min(ceil(len(args) / workers), chunksize))
    
    car_feat = np.asarray([x[0] for x in results])

    # TODO 模块化
    # t = time.time()
    # res = parallel(ave_f_unit, args, False, n_jobs=workers, chunksize=min(ceil(len(args) / workers), chunksize))
    # print(f'avg_cluster_feat_1: {time.time() - t: .4f}')
    # _car_feat = np.asarray([x[0] for x in res])
    # assert np.allclose(car_feat, _car_feat), "check"
    
    # `plate_feat` maybe None
    tmp = [(x[1], c) for x, c in zip(results, cs) if x[1] is not None]
    plate_feat = np.asarray([x[0] for x in tmp])
    plate_feat_c = [x[1] for x in tmp]

    return car_feat, plate_feat, plate_feat_c


def detect_many_noise(cuts):
    """_summary_

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
            p = MAP_routing(u, v, ut, vt)
            if p < 0.05:
                noises += [(x[-1], TWO_NOISE) for x in one_cut]
            elif DO_RECALL_ATTEMPT and p > 0.4:
                inter_nodes, _ = MAP_routing_return_route(u, v, ut, vt)
                inter_camera_nodes = [
                    node for node in inter_nodes if "camera" in G.nodes[node]
                ]
                if inter_camera_nodes:
                    recall_attempts.append(([u, ut], [v, vt], inter_camera_nodes))
        else:
            long_cuts.append(one_cut)

    # case: len > 3 
    for one_cut in long_cuts:
        _len, sub_ps_raw, sub_ps, sub_idxs = greedy_search_trajs(one_cut, .5)
        
        max_sub_p = max(sub_ps)
        if max_sub_p < 0.05:
            noises += [(x[-1], LONG_NOISE) for x in one_cut]
        elif max_sub_p > 0.45:
            black_list = set()
            white_list = set()
            
            # iter nodes
            for i in range(_len):
                filterd_traj_ps = [p for p, idx in zip(sub_ps_raw, sub_idxs) if i in idx]
                max_prob = max(filterd_traj_ps)
                if max_prob < 0.01:
                    black_list.add(i)
                    noises.append((one_cut[i][-1], BLACK_LIST_NOISE))
                else:
                    ps = list(filter(lambda p: p >= 0.01, filterd_traj_ps))
                    if (len(ps) >= min(_len / 3, 2) and np.mean(ps) > 0.1 and max(ps) > 0.3):
                        white_list.add(i)

            # formula 6: sub_idxs 子集
            opt_cands = (x for x in zip(sub_ps, sub_idxs) if x[0] > 0.8 * max_sub_p)
            opt_sub_p, opt_sub_idx = max(opt_cands, key = lambda x: (len(x[1]), x[0]))

            for i in set(range(_len)) - set(opt_sub_idx) - black_list - white_list:
                noises.append((one_cut[i][-1], OUT_OF_SUBSET_NOISE))

            if DO_RECALL_ATTEMPT and opt_sub_p > 0.3:
                for i, j in zip(opt_sub_idx, opt_sub_idx[1:]):
                    u, ut, _ = one_cut[i]
                    v, vt, _ = one_cut[j]
                    inter_nodes, _ = MAP_routing_return_route(u, v, ut, vt)
                    inter_camera_nodes = [
                        node for node in inter_nodes if "camera" in G.nodes[node]
                    ]
                    if inter_camera_nodes:
                        recall_attempts.append(((u, ut), (v, vt), inter_camera_nodes))

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
        # 判断加入概率是否更高
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
                candidates_filter = sim_filter(car1, plate1, candidates, sim_gate=DENOISE_SIM_GATE)
                
                # fomula 9
                if candidates_filter:
                    if p_base is None:
                        p_base = MAP_routing(u, v, ut, vt)
                    for tm, idxs in candidates_filter:
                        p_new = sqrt(MAP_routing(u, node, ut, tm) * MAP_routing(node, v, tm, vt))
                        t = p_new * (1 - MISS_SHOT_P) - p_base * MISS_SHOT_P
                        if t > 0:
                            accept_recalls.append((idxs, t))
        else:
            node, tm = tmp
            candidates = [
                noise for noise in node_to_noises[node]
                        if tm - MERGE_ATTEMPT_ADJ_RANGE < noise[0] < tm + MERGE_ATTEMPT_ADJ_RANGE
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
    cid_to_rids = defaultdict(list)
    for i, c in enumerate(labels):
        if c != -1:
            cid_to_rids[c].append(i)

    # 1. detecting noise ahd recall attempt
    cid_to_noises = {}
    cid_to_recall_attempts = {}

    print("detecting noise...")
    start_time = time.time()
    chunksize = min(ceil(len(cid_to_rids) / workers), 200)
    with Pool(processes=workers) as pool:
        results = pool.map(
            noise_detect_unit, cid_to_rids.values(), chunksize=chunksize) # list((noises, recall_attempts))
    
    for cid, (noises, recall_attempts) in zip(cid_to_rids.keys(), results):
        if noises:
            cid_to_noises[cid] = noises
        if recall_attempts:
            cid_to_recall_attempts[cid] = recall_attempts
    print("detect noise use time:", time.time() - start_time)

    # 2. recall
    cid_to_accept_recalls = defaultdict(list)
    if DO_RECALL_ATTEMPT:
        print("recalling...")
        start_time = time.time()
        
        # node_to_noises
        node_to_noises = get_node_to_noises(cid_to_noises)

        args = [
            (recall_attempts, cid_to_rids[cid], node_to_noises)
                for cid, recall_attempts in cid_to_recall_attempts.items()
        ]
        chunksize = min(ceil(len(args) / workers), 200)
        with Pool(processes=workers) as pool:
            results = pool.map(recall_unit, args, chunksize=chunksize) # list(accept_recalls)
        
        # ? why rids
        rids_to_cid_reward = defaultdict(list)
        for cid, accept_recalls in zip(cid_to_recall_attempts.keys(), results):
            if accept_recalls:
                for idxs, reward in accept_recalls:
                    rids_to_cid_reward[tuple(idxs)].append((cid, reward))
        
        for idxs, cid_reward in rids_to_cid_reward.items():
            max_cid = max(cid_reward, key=lambda x: x[1])[0]
            cid_to_accept_recalls[max_cid] += idxs
        
        print("recall use time:", time.time() - start_time)

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


def filter_noise_by_similarity(cid, noise_cids, cid_to_rids):
    """filter by `multi-model similarity`, for each record in each cluster

    Args:
        cid (int): _description_
        noise_cids (list): _description_
        cid_to_rids (dict): _description_

    Returns:
        list: noise list
    """
    idxs1 = cid_to_rids[cid]
    car1, plate1 = calculate_ave_f(idxs1)

    nidxs_filter = []
    for nc in noise_cids:
        idxs2 = cid_to_rids[nc]
        for i in idxs2:
            car2, plate2 = get_feat_by_rid(i)
            sim = cal_multi_similarity(car1, plate1, car2, plate2)
            if sim > MERCLUSTER_SIM_GATE:
                nidxs_filter.append(i)

    return nidxs_filter    

def filter_noise_by_time(cid, nidxs_filter, cid_to_rids):
    points_nc_filter = []
    points      = trans_rids_to_points(cid_to_rids[cid], records, cameras_dict) # (node_id, time, rid)
    points_nc   = trans_rids_to_points(nidxs_filter, records, cameras_dict)
    tm_ranges   = tms_adj_range(sorted([x[1] for x in points]), 
                                adj_range=MERGE_CLUSTER_ADJ_RANGE)
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

def merge_cluster_unit(args):
    # 1. prepare variables
    (c, ncs), cid_to_rids = args
    rids = cid_to_rids[c]

    # 2. filter by `multi-model similarity`, for each record in each cluster
    nidxs_filter = filter_noise_by_similarity(c, ncs, cid_to_rids)
    if not nidxs_filter:
        return []

    # 3. filter by time intervals
    points, points_nc_filter = filter_noise_by_time(c, nidxs_filter, cid_to_rids)
    if not points_nc_filter:
        return []

    # 4. merge and split points into `segs`
    points_nc = points_nc_filter
    points_all = points + points_nc
    points_all, cuts = merge_and_split_points(points_all)

    # 5. process cuts
    noises = process_segs(cuts)

    # 6. get accepted idxs
    cand_noise_idxs = {x[-1] for x in points_nc}
    noise_idxs  = {idx for idxs in noises for idx in idxs}
    merge_point_idxs = get_merge_point_idxs(points, points_all)
    accept_idxs = cand_noise_idxs - noise_idxs - merge_point_idxs
    
    return list(accept_idxs)


def merge_clusters(labels, ngpu=1, car_topk=15, plate_topk=30):
    global f_emb
    print("cluster merging...")
    start_time = time.time()

    cid_to_rids = defaultdict(list)
    # i: rid; c: cid
    for i, c in enumerate(labels):
        if c != -1:
            cid_to_rids[c].append(i)

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

        # search
        car_searcher = FlatSearcher(feat_len=64, ngpu=ngpu)
        plate_searcher = FlatSearcher(feat_len=64, ngpu=ngpu)
        car_topk_idxs = car_searcher.search_by_topk(
            query=car_query, gallery=car_gallery, topk=car_topk)[1].tolist()
        plate_topk_idxs = plate_searcher.search_by_topk(
            query=plate_query, gallery=plate_gallery, topk=plate_topk)[1].tolist()
        
        # `cluster id` to noise `cluster ids`
        c_to_nc = defaultdict(set)
        for c, rids in zip(cs_big, car_topk_idxs):
            for i in rids:
                c_to_nc[c].add(cs_small[i])
        for c, rids in zip(plate_query_c, plate_topk_idxs):
            for i in rids:
                c_to_nc[c].add(plate_gallery_c[i])

        # merge_cluster
        args = [((c, ncs), cid_to_rids) for c, ncs in c_to_nc.items()]
        with Pool(processes=workers) as pool:
            results = pool.map(
                merge_cluster_unit, args, chunksize=min(ceil(len(args) / workers), 200))

        # cid_2_accept_rids: `cluster id` -> `accept idxs`  
        accept_rid_2_cids = defaultdict(list)
        cid_2_accept_rids = defaultdict(list)
        for c, accept_idxs in zip(c_to_nc.keys(), results):
            for rid in accept_idxs:
                accept_rid_2_cids[rid].append(c)
        for rid, cs in accept_rid_2_cids.items():
            if len(cs) == 1:
                cid_2_accept_rids[cs[0]].append(rid)
            else:
                cid_2_accept_rids[random.sample(cs, 1)[0]].append(rid)

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

    print("merging consume time:", time.time() - start_time)
    return labels


if __name__ == "__main__":
    N_iter = 10
    s = 0.8
    topK = 128
    ngpu = 2
    metrics = []
    load_data = False

    cache_path = "data/shortest_path_results_test.pkl"
    if not os.path.exists(cache_path):
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

    for i, operation in zip(range(N_iter), ["merge", "denoise"] * (N_iter // 2)):
        print(f"---------- iter {i}: {operation} -----------")

        if i > -1 and not load_data:
            print("clustering...")
            start_time = time.time()
            cluster = SigCluster(feature_dims=[64, 64, 64], ngpu=ngpu)
            labels = cluster.fit(
                [[a, b, c] for a, b, c in zip(f_car, f_plate, f_emb)],
                weights=[0.1, 0.8, 0.1],
                similarity_threshold=s,
                topK=topK,
            )
            print("clustering consume time:", time.time() - start_time)
            pickle.dump(labels, open(f"label/labels_iter_{i}.pkl", "wb"))
        else:
            labels = pickle.load(open(f"label/labels_iter_{i}.pkl", "rb"))

        precision, recall, fscore, expansion, vid_to_cid = evaluate(records, labels)
        metrics.append((precision, recall, fscore, expansion))

        if operation == "merge":
            merge_clusters(labels, ngpu=ngpu)
        elif operation == "denoise":
            res = update_f_emb(labels, do_update=True)
            cid_to_noises, cid_to_accept_recalls, cid_to_recall_attempts = res

    json.dump(metrics, open(f"metric/metrics.json", "w"))
