from multiprocessing import Pool
from time import sleep
from typing import Protocol
from tqdm import tqdm

import numpy as np
import pandas as pd


def run_async(function, args_list, njobs, sleep_time_s=0.05):
    with Pool(njobs) as pool:
        with tqdm(total=len(args_list)) as pbar:
            results = {i: pool.apply_async(function, args=args)
                       for i, args in enumerate(args_list)}
            previous_state = 0
            while not all(future.ready() for future in results.values()):
                this_state = np.sum([future.ready() for future in results.values()])
                pbar.update(this_state - previous_state)
                previous_state = this_state
                sleep(sleep_time_s)
            results = [results[i].get() for i in range(len(results))]
    return results
