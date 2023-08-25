

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
    """filter by time within the time intervals of each trajectory points. 
    The rids is transfer to points at first.

    Args:
        rids (list): The records index list
        noise_rids (list): The indexs of noisy records.
        adj_range (float, optional): The adjence time to form the time intervals. 
            Defaults to MERGE_CLUSTER_ADJ_RANGE (300).

    Returns:
        (points, points_nc_filter): Point format: `(node_id, time, rid)`
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