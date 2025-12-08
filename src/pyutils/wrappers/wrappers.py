import time

def timeit(func):
    def wrapper(*args, **kwargs):
        function_call = f"{func.__name__}({', '.join([str(i) for i in args])})"
        print(f"Calling {function_call}", end='')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        if elapsed < 1:
            print(f"\r{function_call} took {elapsed*1000:.3f} milliseconds")
        elif elapsed < 60:
            print(f"\r{function_call} took {elapsed:.3f} seconds")
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            print(f"\r{function_call} took {minutes} minutes and {seconds:.3f} seconds")
        return result
    return wrapper



import time

import pandas as pd

from .tools import format_time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f'{func.__name__} completed in {format_time(time.time()-start_time)}')
        return result
    return wrapper



def filter(func, msg:str=None, logger=None, silent = False):
    '''
    Wrapper that prints the size of an input dataframe and the size of the output dataframe
    '''
    def wrapper(*args, **kwargs):
        
        if not isinstance(args[0], pd.DataFrame):
              raise ValueError(f'The first argument of {func.__name__} must be a pd.DataFrame')

    
        df = func(*args, **kwargs)
        
        if not isinstance(df, pd.DataFrame):
              raise ValueError(f'The outcome of {func.__name__} must be a pd.DataFrame')
        
        if silent is False:
            if func.__name__ != "<lambda>":
                print(f'Filter using {func.__name__}')
            if msg:
                print(msg)
            print(f'Input dataframe: {len(args[0]):,}\nOutput dataframe: {len(df):,} \nObs lost: {(1-len(df)/len(args[0]))*100:.1f}%')
        
        if logger:
            logger.info(f'Filter using {func.__name__}')
            logger.info(f'\n\tInput dataframe: {len(args[0]):,}\n\tOutput dataframe: {len(df):,} \n\tObs lost: {(1-len(df)/len(args[0]))*100:.1f}%')
        
        return df
    
    return wrapper