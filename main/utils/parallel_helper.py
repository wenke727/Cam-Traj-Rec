import time
from math import ceil
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count


def parallel_process(func, args, chunksize, workers=64):
    with Pool(processes=workers) as pool:
        results = pool.map(func, args, chunksize=chunksize)
    
    return results

#! FIXME 为何调用反而耗时更长
def parallel(func, args, asynch=True, n_jobs=-1, multiple=False, chunksize=None):
    if n_jobs > cpu_count() or n_jobs == -1:
        n_jobs = cpu_count()

    pool = Pool(processes=n_jobs) 

    # if asynch:
    #     if multiple:
    #         res = pool.starmap_async(func, args, chunksize).get()
    #     else:
    #         res = pool.map_async(func, args, chunksize).get()
    # else:
    #     if multiple:
    #         res = pool.starmap(func, args, chunksize)
    #     else:
    #         res = pool.map(func, args, chunksize)

    # pool.close()
    # pool.join()

    with Pool(processes=n_jobs) as pool:
        if asynch:
            res = pool.map_async(func, args, chunksize).get()
        else:
            res = pool.map(func, args, chunksize)

    return res

def _add(args):
    x, y = args
    res = x + y
    a = np.random.random(100000)

    return {x: a.sum()}

if __name__ == "__main__":
    res = parallel(_add, ((i, i) for i in range(int(1e6))), True, n_jobs=64)