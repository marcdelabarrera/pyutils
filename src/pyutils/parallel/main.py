# Standard libraries
import concurrent.futures
from functools import partial

def map(fn:callable, arg:list, max_workers:int = 1, **kwargs)-> list:
    '''
    Executes map in parallel.
    '''
    if max_workers == 1:
        return [fn(i, **kwargs) for i in arg]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers = max_workers) as executor:
            return list(executor.map(partial(fn, **kwargs), arg))