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