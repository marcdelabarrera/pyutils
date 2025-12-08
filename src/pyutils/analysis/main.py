import pandas as pd


def windsorize(
    series: pd.Series,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    verbose: bool = False,
) -> pd.Series:
    """
    Clip a Series at the given quantiles.

    Parameters
    ----------
    series : pd.Series
        Input data.
    lower_quantile : float, default 0.01
        Lower quantile for clipping.
    upper_quantile : float, default 0.99
        Upper quantile for clipping.
    verbose : bool, default False
        If True, print the cutoffs used.

    Returns
    -------
    pd.Series
        Quantile-clipped series.
    """
    lower_bound = series.quantile(lower_quantile)
    upper_bound = series.quantile(upper_quantile)

    if verbose:
        print(
            f"Windsorizing series with q_{lower_quantile}={lower_bound:0.2f} "
            f"and q_{upper_quantile}={upper_bound:0.2f}"
        )

    return series.clip(lower=lower_bound, upper=upper_bound)



# Standard imports
from pathlib import Path
import time
import warnings
import re
import os
# Third party imports
import numpy as np
import pandas as pd
import pandas.api.types as ptypes

ArrayLike = pd.Series

def interval_to_category(x:pd.Series, 
                         labels = None) -> pd.Series:
    '''
    Given an interval from pd.cut, convert to category
    '''
    x = x.astype(str)
    x[x=='nan'] = pd.NA
    categories = pd.Series(x.unique()).dropna()
    x = pd.Categorical(x,
                       sorted(categories, key=lambda x: x.split(',')[0][1:]),
                       ordered=True)
    if labels:
        x = x.rename_categories(labels)
    return x

def percentile_bin(df:pd.DataFrame, 
                   var:str,
                   bins:list[float],
                   by:list[str]=None,
                   as_categorical:bool=False,
                   labels:list[str]=None,
                   duplicates:str='raise') -> pd.Series:
    '''
    Bins a variable by group
    Parameters:
    -------------
    bins: list[float]
        List of bins, starting at 0 and ending at 1.
    '''
    if bins[0]!=0:
        raise ValueError('bin must start at 0')
    if bins[-1]!=1:
        raise ValueError('bin must end in 1')
    if labels and len(labels)!=len(bins)-1:
        raise ValueError('len(labels) must be consistent with len(bins)-1')
    
    if by is None:
        out = _percentile_bin(df[var], bins, duplicates = duplicates)
    else:
        out = (df.groupby(by)[var].transform(lambda x: _percentile_bin(x, bins, duplicates = duplicates)))
    
    if as_categorical or labels:
        out = interval_to_category(out, labels = labels)
    return out

def _percentile_bin(x:pd.Series, 
                    bins:list[float],
                    duplicates:str='raise') -> pd.Series:
    '''
    Splits a series in quantiles
    Parameters:
    -----------
    x: pd.Series
        Series to bin
    bins: list[float]
        List that defines the bins. Must start at 0 and end at 1
    duplicates: str
        Either raise or ignore

    Notes:
    -------
    In most of the cases, works as expected. This is when the quantiles of x are all different.
    If some of the quantiles are duplicated and duplicates = 'ignore', then the bin is the leftmost bin.
    If all elements of x are the same, then it returns a series of None.
    '''

    if duplicates not in ['raise','ignore']:
        raise ValueError('duplicates must be either "raise" or "ignore"')
    
    if len(x.unique())==1:
        return pd.Series([None]*len(x), dtype='float')
    
    quantiles = x.quantile(bins)

    labels = [f'({q1},{q2}]' for q1,q2 in zip(bins, bins[1:])]
    labels[0] = labels[0].replace('(','[')
    all_labels = labels

    if quantiles.is_unique is False:
        if duplicates=='raise':
            raise ValueError(f'Duplicate quantiles: {quantiles}')
        else:
            new_labels = []
            for i in range(1, len(quantiles)):
                if quantiles.iloc[i]==quantiles.iloc[i-1]:
                    continue
                else:
                    new_labels.append(labels[i-1])
            labels = new_labels
            quantiles = quantiles.drop_duplicates()
    
    bins = pd.cut(x, quantiles, labels=labels, include_lowest=True)

    if len(bins.cat.categories) != len(all_labels):
        bins = bins.cat.set_categories(all_labels, ordered = True)
    
    return bins



def drop_duplicates(df:pd.DataFrame, verbose:bool=False, subset:list[str]=None)->pd.DataFrame:
    if verbose:
        initial_len = len(df)
        print("Before dropping duplicates:", len(df))
    df = df.drop_duplicates(subset=subset)
    if verbose:
        print("After dropping duplicates:", len(df), f"({len(df)/initial_len:.2%} of initial)")
    return df





# Standard imports
from pathlib import Path
import time
import warnings
import re
import json
import os
# Third party imports
import numpy as np
import pandas as pd
import pandas.api.types as ptypes

ArrayLike = pd.Series



def memory_usage(df:pd.DataFrame, units:str='B', index=True, deep=False, verbose=False)->float:
    conversion = {"GB":1024**3, "MB": 1024**2, "KB":1024, "B": 1}
    if units not in conversion:
        raise ValueError("Units must be one of 'GB', 'MB', 'KB' or 'B'")
    df_size = df.memory_usage(index, deep).sum()/conversion[units]

    if verbose:
        print(f'{df_size:.2f} {units}')
    return df_size
    



def to_parquet(df:pd.DataFrame, path:Path, overwrite:bool=True)->None:
    '''
    Functions that saves a dataframe to parquet but is more flexible.
        - Transforms series to dataframe.
        - Allows overwritting control.
        - Creates parent paths if they do not exists.
    '''
    path = Path(path) if isinstance(path, str) else path
    df = df.to_frame() if isinstance(df, pd.Series) else df

    if path.exists():
        if overwrite:
             pass
        else:
             return None
    os.makedirs(path.parent, exist_ok=True)
    df.to_parquet(path)



def add_readme(readme:str, display:bool=True, filename:Path=None)->None:
    '''
    Adds a readme file
    '''
    filename = 'readme.md' if filename is None else filename

    readme = readme + f'\nReadme created: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
    with open(filename,'w') as file:
        file.write(readme)

    if display:
        print(readme+'\n')

def fill_na(x:pd.Series, y:pd.Series, verbose:bool=False)->pd.Series:
    """
    Fills elements of x that are NA with elements of y
    """
    if len(x)!=len(y):
        raise ValueError("x and y must have the same length")
    if x.dtype!=y.dtype:
        raise ValueError("x and y must have the same type")
    if verbose:
        print(f"NA share in {x.name}: {x.isna().mean()}")
        print(f"Share replaced in {x.name}: {y[x.isna()].notna().sum()/len(x)}")

    return pd.Series(np.where(x.isna(), y, x), dtype = x.dtype, index = x.index)

def format_time(seconds:int)->str:
    '''
    Writes seconds in min and secods
    '''
    if seconds<60:
        return f'{seconds:.0f} sec'
    elif seconds>=60:
        return f'{seconds//60:.0f} min {seconds%60:.0f} sec'

def aggregate_by(df: pd.DataFrame,
                 col: str,
                 by:list[str],
                 functions:list[str],
                 custom_functions:dict=None)->pd.DataFrame:
    '''
    Given a tax return dataset, aggregates and computes the
    count, mean, std, q10, q50 and q90
    ['year','age_bin','iagi_bin','area_code','ifil']
    Parameters
    ----------
    data
    col
    groups
    Returns
    -------
    out
    '''
    
    return_series = False

    if not ptypes.is_numeric_dtype(df[col]):
        raise ValueError(f'{col} must be numeric')
    
    if isinstance(functions, str):
        functions = [functions]
        return_series = True

    FUNCTIONS = {'n_obs': pd.NamedAgg(column = col, aggfunc = "count"),
              'mean': pd.NamedAgg(column = col, aggfunc = "mean"),
              'sd':  pd.NamedAgg(column = col, aggfunc = 'std'),
              'var':  pd.NamedAgg(column = col, aggfunc = 'var'),
	      'min': pd.NamedAgg(column = col, aggfunc = "min"),
              'q10': pd.NamedAgg(column = col, aggfunc = lambda x: x.quantile(0.1)),
              'q50': pd.NamedAgg(column = col, aggfunc = lambda x: x.quantile(0.5)),
              'q90': pd.NamedAgg(column = col, aggfunc = lambda x: x.quantile(0.9)),
              'max': pd.NamedAgg(column = col, aggfunc = "max"),
              'sum': pd.NamedAgg(column = col, aggfunc = "sum")}
    
    functions = {i:FUNCTIONS[i] for i in functions}
    if custom_functions:
        functions = functions | custom_functions
    out = df.groupby(by, observed=True).agg(**functions)
    if return_series:
        out = out.iloc[:,0]
    return out


def interval_to_caregory(x:pd.Series, labels=None)->pd.Series:
    '''
    Given a interval from pd.cut, convert to category
    ''' 
    x = x.astype(str)
    x[x=='nan'] = pd.NA
    categories = pd.Series(x.unique()).dropna()
    x = pd.Categorical(x, 
                       sorted(categories, key=lambda x: x.split(',')[0][1:]),
                       ordered=True)
    if labels:
        x = x.rename_categories(labels)
    return x

def bin(x:pd.Series,
           bins:list[float],
           as_categorical:bool=False,
           labels:list[str]=None,
           **kwargs)->pd.Series:
    '''
    Bins a variable by group
    Parameters:
    ----------
    bins: list[float]
        List of bins, starting at 0 and ending at 1.
    '''
    if labels and len(labels)!=len(bins)-1:
        raise ValueError('len(labels) must be consistent with len(bins)-1')

    out = pd.cut(x ,bins = bins, **kwargs)
    if as_categorical or labels:
        out = interval_to_caregory(out, labels=labels)
    return out


def percentile_bin(df:pd.DataFrame,
           var:str, 
           bins:list[float],
           by:list[str]=None,
           as_categorical:bool=False,
           labels:list[str]=None,
           duplicates:str='raise')->pd.Series:
    '''
    Bins a variable by group
    Parameters:
    ----------
    bins: list[float]
        List of bins, starting at 0 and ending at 1.
    '''
    if bins[0]!=0:
        raise ValueError('bin must start at 0')
    if bins[-1]!=1:
        raise ValueError('bin must end in 1')
    if labels and len(labels)!=len(bins)-1:
        raise ValueError('len(labels) must be consistent with len(bins)-1')

    if by is None:
        out = _percentile_bin(df[var], bins, duplicates = duplicates)
    else:
        out = (df.groupby(by)[var]
                             .transform(lambda x: _percentile_bin(x, bins,duplicates = duplicates)))
    
    if as_categorical or labels:
        out = interval_to_caregory(out, labels=labels)
    return out



def _percentile_bin(x:pd.Series, bins:list[float], duplicates:str='raise')->pd.Series:
    '''
    Splits a series in quantiles.
    Parameters:
    ----------
    x: pd.Series
        Series to bin
    bins: list[float]
        List that defines the bins. Must start at 0 and end at 1
    duplicates: str
        Either raise or ignore.

    Notes:
    ------
    In most of the cases, works as expected. This is when the quantiles of x
    are all different.
    If some of the quantiles are duplicated and duplicates='ignore', 
    then the bin is the leftmost bin. 
    If all elements of x are the same, then it returns a series of None.
    
    '''
    if duplicates not in ['raise','ignore']:
        raise ValueError('duplicates must be either "raise" or "ignore"')
    
    
    if len(x.unique())==1:
        return pd.Series([None]*len(x),dtype='float')

    quantiles = x.quantile(bins)

   
    labels = [f'({q1},{q2}]' for q1,q2 in zip(bins, bins[1:])]
    labels[0] = labels[0].replace('(','[')
    all_labels = labels

    if quantiles.is_unique is False:
        if duplicates=='raise':
            raise ValueError(f'Duplicate quantiles: {quantiles}')
        else:
            new_labels = []
            for i in range(1,len(quantiles)):
                if quantiles.iloc[i]==quantiles.iloc[i-1]:
                    continue
                else:
                    new_labels.append(labels[i-1])
            labels = new_labels
            quantiles = quantiles.drop_duplicates()
    bins =  pd.cut(x,
                    quantiles,
                    labels = labels,
                    include_lowest=True)

    if  len(bins.cat.categories) != len(all_labels):
        bins = bins.cat.set_categories(all_labels, ordered=True)
    return bins

    



def find_corrupted(files:list[Path])->list[Path]:
    """
    Checks that it can open all files
    """
    out = []
    for file in files:
        try:
            pd.read_parquet(file)
        except:
            out.append(file)
    return out


def log(x:ArrayLike)->ArrayLike:
    '''
    Computes the log of x, returns np.nan if x=0 or negative without raising a warning.
    If x is a pd.Series, it returns a pd.Series
    #TODO: most likely does not work if x is a numpy array 
    '''
    if isinstance(x, np.ndarray):
        x[np.isnan(x)] = 0
        x_log = np.log(x, where=x>0)
        x_log = np.where(x>0, x_log, np.nan)
    elif isinstance(x, pd.Series):
        x = x.fillna(0)
        x_log = np.log(x, where=x>0)
        x_log = x_log.where(x>0)
    else:
        raise ValueError(f'Unrecognized type {type(x)}')
    return x_log



def compute_log_growth(x:pd.Series, y:pd.Series)->pd.Series:
    """
    Computes log growth between series x and y
    """
    return log(x)-log(y)
    


def compute_allen_growth(x:pd.Series, y:pd.Series)->pd.Series:
    """
    Computes log growth between series x and y
    """
    return 2*(x-y)/(x+y)
    



def winsorize(x:pd.Series, lower: float=0, upper:float=1)->pd.Series:
    '''
    Winsorizes a pd.Series 
    Example
    winsorize(x, lower=0.01, upper = 0.99)
    '''

    if pd.api.types.is_integer_dtype(x):
    	return x.clip(int(x.quantile(lower)), int(x.quantile(upper)))
    else:
        return x.clip(x.quantile(lower), x.quantile(upper))

def load_json(path:Path)->dict:
    with open(path,'r') as f:
        out = json.load(f)
    return out

def parse_number(x:str)->int:
    return int(re.findall('\d+',x)[0])


def _to_Int(x:pd.Series, dtype: str='Int64')->pd.Series:
    if dtype not in ['Int32','Int64']:
        raise ValueError('dtype must be either "Int32" or "Int64"')
    
    
    if not pd.api.types.is_numeric_dtype(x):
        try:
            x=x.astype(float)
        except:
            print(x[pd.to_numeric(x, errors='coerce').isna() & x.notna()])
            raise ValueError(f'{x.name} ({x.dtype}) must be of numeric dtype')

    if dtype=='Int32' and x.max()> 2**31-1:
        raise ValueError(f'Max of {x.name} is too large to be casted to Int32 {x.max()} is larger than 2**31-1')

    if dtype=='Int64' and x.max()> 2**63-1:
        raise ValueError(f'Max of {x.name} is too large to be casted to Int32 {x.max()} is larger than 2**63-1')
    
    x = np.round(x, 5) #avoids 10.9999 or 10.00001
 
    if (x.fillna(0)%1 == 0).all():
        x = x.astype(dtype)
    else:
        print(x[x.fillna(0)%1 != 0])
        raise ValueError(f'Trying to cast {x.name} ({x.dtype}) into {dtype}. Info would be lost.')
    return x
 
#x[~x.isin(['MANPNXY 1','0III','60997S520','11    410','11    910'])].astype(float)

def astype(df: pd.DataFrame, dtypes:dict)->pd.DataFrame:
    '''
    Robust version of pd.astype which is more informative when there is an error.
    '''
    for var,var_type in dtypes.items():
        try:
            df[var] = df[var].astype(var_type)
        except Exception as e:
            if var_type in ['Int32','Int64']:
                df[var]=_to_Int(df[var], var_type)
            else:

                print(df[var].head())
                raise ValueError(f"Unable to convert {var} ({df[var].dtype}) to {var_type}")
    return df



def test_na(df:pd.DataFrame, columns:list[str]=None)->None:
     """
     Prints the share of NA in a given column
     """
     if columns:
         print(f'{df[columns].isna().mean().rename("NA share")}')
     else:
         print(f'{df.isna().mean().rename("NA share")}')

def dropna(df: pd.DataFrame, subset:list[str]=None)->pd.DataFrame:
    initial_rows = len(df)
    df = df.dropna(subset=subset)
    print(f'dropna({subset}). {initial_rows-len(df):,} ({100*(initial_rows-len(df))/initial_rows:.3f}%) columns droped')
    return df


def sort_values(df:pd.DataFrame, by, verbose:bool=True)->pd.DataFrame:
    start_time = time.time()
    df = df.sort_values(by)
    if verbose:
       print(f'Sort completed in {format_time(time.time()-start_time)}')
    return df

    
def find_duplicates(df: pd.DataFrame, subset:list[str]=None)->pd.DataFrame:
    '''
    Returns a dataframe containing the duplicated rows based on a subset of columns.
    '''
    duplicates = df[df.duplicated(subset=subset, keep=False)]
    if len(duplicates)==0:
        print(f'No duplicates {subset} in the dataframe')
        return None
    else:
        return duplicates

def drop_duplicates(df: pd.DataFrame, subset: list[str]=None, verbose=False)->pd.DataFrame:
    initial_rows = len(df)
    df = df[~df.duplicated(subset=subset, keep='first')]
    if verbose:
        print(f'{initial_rows-len(df):,}  ({100*(initial_rows-len(df))/initial_rows:.3f}%) duplicated rows deleted')
    return df


def overwrite(path:Path, overwrite:bool):
    """
    Checks if a file exists. If does, it checks if you want to overwrite it.
    """
    if path.exists():
        if overwrite:
             return True
        else:
             print("path already exists and overwrite is set to False")
             return False
    else:
        return True


def left_merge(left:pd.DataFrame, right:pd.DataFrame, on: list[str], verbose:bool=False, keep_index=True, **kwargs):
    '''
    Enchanced left merge. It checks that the left dataframe has unique rows, it outputs the success rate, and the time it takes to merge.
    '''
    initial_length = len(left)
    
    if keep_index:
        index_name = left.index.name
        left = left.reset_index()
    if verbose:
        start_time = time.time()
        out = pd.merge(left, right, how='left', on=on, indicator=True, **kwargs)
        print(f'Mergin on {on}.')
        print(f'Matching rate {100*(out._merge == "both").mean():.2f}%')
        out = out.drop(columns='_merge')
    else:
        out = pd.merge(left, right, how='left', on=on, indicator=False, **kwargs)

    if keep_index:
        out = out.set_index(index_name)

    if len(out)>initial_length:
        warnings.warn('Extra rows created by the merge')
    if verbose:
        print(f'Merge completed in {format_time(time.time()-start_time)}')
    return out


def roll_categories(x:pd.Series, shift:int=1)->pd.Series:
    '''
    Rolls categoriey orders.
    '''
    return pd.Categorical(x,np.roll(x.unique(),shift),ordered=True)

def min_cell_size(df:pd.DataFrame, cell:list[str], min_size:int)->pd.DataFrame:
    '''
    Filters cells with a minimum cell size
    '''
    return df.loc[df.groupby(cell, observed=True).transform('size')>=min_size]






