from multiprocessing import Pool
from time import sleep
from typing import Protocol

import pandas as pd


def run_async(function, args_list, njobs, sleep_time_s=0.05):
    pool = Pool(njobs)
    results = {i: pool.apply_async(function, args=args)
               for i, args in enumerate(args_list)}
    while not all(future.ready() for future in results.values()):
        sleep(sleep_time_s)
    results = [results[i].get() for i in range(len(results))]
    pool.close()
    return results


def separate_name_and_dimensions(ser: pd.Series):
    name = ser.apply(lambda x: x.split("(")[0])
    dims = ser.apply(lambda x: int(x
                                   .split("(")[-1]
                                   .split(")")[0])
                     )
    return name, dims


class HasName(Protocol):
    def name(self) -> str:
        raise NotImplementedError