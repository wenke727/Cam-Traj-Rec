
import time
from math import ceil
from multiprocessing import Pool
from cfg import ADJ_RANGE, TM_GAP_GATE
from misc import merge_and_split_points, trans_rids_to_points

import numpy as np
from collections import defaultdict


def get_cid_to_rids(labels):
    cid_to_rids = defaultdict(list)
    for i, c in enumerate(labels):
        if c != -1:
            cid_to_rids[c].append(i)

    return cid_to_rids

def get_node_2_noises(cid_to_noises, records, cameras_dict):
    # [(avg_tms, idx), ..., ]
    node_to_noises = defaultdict(list) 
    for idxs in (noise[0] for noises in cid_to_noises.values() 
                            for noise in noises):
        tms = [records[i]["time"] for i in idxs]
        camera_id = records[idxs[0]]["camera_id"]
        node = cameras_dict[camera_id]["node_id"]
        node_to_noises[node].append((np.mean(tms), idxs))

    return node_to_noises

