
# Standard imports
import warnings
import platform
import sys
from pathlib import Path
# Third party imports
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from statsmodels.tsa.x13 import x13_arima_analysis
from statsmodels.tsa import filters
from pandas.api.types import is_datetime64_any_dtype as is_datetime
# Local imports
sys.path.append('/bbkinghome/mbarrera/git_supply')
#from utils.econometrics.utils import lagmat, add_const

PATH = Path(__file__).parent


def growth_to_levels(x:pd.Series):
    growth = 1+x.values
    levels = np.zeros_like(growth)
    levels[0] = growth[0]**(1/4)
    levels[1] = growth[1]**(1/2)
    levels[2] = growth[2]**(3/4)
    levels[3] = growth[0]**(3/4) * growth[3]
    for i in range(4, len(x)):
        levels[i] = levels[i-4] * growth[i]
    return pd.Series(levels, index=x.index)



def hamilton_filter(x:pd.Series, p:int=4, h:int=8, dropna = False)->pd.Series:
    r'''
    Runs the regression y_{t+h} = \alpha + \beta_1 y_{t} + \beta_2 y_{t-1} + ... + \beta_p y_{t-p+1} + \epsilon_{t+h}
    and returns \epsilon_{t+1}
    '''
    if dropna and np.any(x.isna()):
        warnings.warn(f'Series contains {x.isna().sum} NaNs')
        x = x.dropna()
    X = add_const(lagmat(x, p-1, 0))
    X.index = X.index.get_level_values(0).shift(h, freq = 'QS')
    X = X.iloc[p-1:-h]
    y = x.iloc[h+p-1:]
    beta = inv(X.T@X)@(X.T@y)
    return x-X@beta, X@beta


def hp_filter(x, lamb=1600):
    return filters.hp_filter.hpfilter(x, lamb=lamb)


def seasonally_adjust_series(x:Series, **kwargs)->Series:
    '''
    Seasonally adjusts a series using X13-ARIMA-SEATS
    '''
    if platform.system() == 'Linux':
        x12path = PATH / 'x13arima/linux/x13as_ascii-v1-1-b60/x13as'
    elif platform.system() == 'Windows':
        x12path = PATH / 'x13arima/windows/x13as_ascii-v1-1-b60/x13as'
    if x.isna().any():
        raise ValueError('Series contains NaNs')
    return x13_arima_analysis(x, x12path=x12path, **kwargs).seasadj 

def seasonally_adjust_panel(data:Series, id:str|list[str], **kwargs)->Series:
    '''
    Seasonally adjusts a panel
    Parameters:
    -----------
    data: DataFrame
        Panel data with a multiindex, [id, t]
    id: str or list
        Name of the column(s) or indices that identifies the id
    '''
    if not isinstance(data.index, pd.MultiIndex):
        raise ValueError('Series must have a MultiIndex')
    
    return data.groupby(id, group_keys=True).apply(lambda x: seasonally_adjust_series(x.reset_index(id, drop=True), **kwargs))


def year_quarter_to_datetime(year:pd.Series,quarter:pd.Series)->pd.Series:
    '''
    Given a series of year and quarters, returns a datetime.
    
    '''
    return pd.to_datetime(year.astype(str) + 'Q'+quarter.astype(str))


def parse_date(x:Series)->Series:
    return pd.to_datetime(x)

def floor_date(x:Series, freq:str)->Series:
    '''
    Rounds a date to the beginning of the period
    Parameters:
    -----------
    x: Series
        Series of dates to floor
    freq: str
        Frequency of the period to floor to. Must be one of 'W','W-MON','M','Q','Y'
    '''
    
    FREQUENCIES = ['W','W-MON','M','Q','Y','QS']
    
    if freq not in FREQUENCIES:
        raise ValueError(f'Invalid frequency {freq}.')
    return pd.PeriodIndex(x, freq = freq).to_timestamp()

#def plot_coverage(ax, data:DataFrame, date_col:str, group_col:str):
#    ax.scatter(data[date_col], data[group_col], s=1)
#    return ax

def check_balanced_panel(df:DataFrame, id:str|list[setattr], t:str)->bool:
    '''
    Checks if a panel is balanced by id and time
    Parameters:
    -----------
    data: DataFrame
        Panel data
    id: str or list
        Name of the column(s) or indices that identifies the id
    t: str
        Name of the column or index that identifies the time periods
    TODO: Currently not working.
    '''
    if df.groupby(id).size().min() != df.groupby(id).size().max():
        return False
        #warnings.warn(f'Panel is not balanced by {id}. Some id have more observations than others.')
    df = df.reset_index()
    if t in df.columns and id in df.columns:
        periods = df[t].drop_duplicates()
        balanced_periods = df.groupby(id)[t].aggregate(lambda x: np.all(periods.isin(x)))
    else:
        raise ValueError(f'df must contain columns or index {id} and {t}')
    if np.all(balanced_periods):
        return True
    else:
        return False
    #if not np.all(balanced_periods):
    #    warnings.warn(f'Panel is not balanced by {t}. Not all periods have the same sequence of {t}.')
    #    return False
    #else:
    #    return True

# def multiindex_resample(series, freq:str, interpolate:bool=False)->pd.Series:
#     '''
#     Resamples a series to a given frequency. The series must have a DatetimeIndex.
#     Currently, index is id,date.
#     Takes the first observation of each period.
#     Only works if resampling at a lower frequency.
#     '''
#     if not isinstance(series.index, pd.MultiIndex):
#         raise ValueError('Series must have a MultiIndex')
#     index_name = series.index.names[0]
#     if index_name is None:
#         index_name = 'level_0'
#     out = series.reset_index(level=0).groupby(index_name, sort=False).resample(freq)
    
#     out = out.interpolate() if interpolate else out.first()
    
#     out = out.drop(columns = index_name)
#     #out.index.freq = freq
#     return out

# def balance_panel(df, freq:str = None):
#     '''
#     Balances a panel id, t. 
#     '''
#     if not isinstance(df.index, pd.MultiIndex):
#         raise ValueError('Series must have a MultiIndex')
#     if freq is None:
#         freq = pd.infer_freq(df.index.get_level_values(1).sort_values().unique())
#         if freq is None:
#             raise ValueError('Could not infer frequency')

#     min_date = df.index.get_level_values(1).min()
#     max_date = df.index.get_level_values(1).max()
    
#     index = pd.MultiIndex.from_product([df.index.get_level_values(0).unique(), pd.date_range(min_date, max_date, freq=freq)], names=df.index.names)
    
#     df = df.reindex(index)
    
#     return df

#def balance_panel(df, t:str):
#    if not isinstance(df.index, pd.MultiIndex):
#        raise ValueError('Series must have a MultiIndex')
#    
#    min_date = df.index.get_level_values(t).min()
#    max_date = df.index.get_level_values(t).max()




def prepend_index(df:DataFrame|Series, value:str, name:str)->DataFrame|Series:
    '''
    Prepends a constant outer index to a dataframe or series
    Parameters:
    -----------
    df: DataFrame or Series
        Dataframe or series to prepend the index to
    value: str
        Value of the outer index
    name: str
        Name of the outer index
    Example:
    --------
    >>> df = pd.DataFrame({'a':[1,2,3]})
    >>> prepend_index(df, 'value', 'name')
    name  a
    value 1
    value 2
    value 3
    '''
    
    return pd.concat([df],keys = [value], names = [name])


# def _resample_group(group, id, t:str, freq:str):
#     '''
#     Given a group, resamples it to a given frequency.
#     Dataframe has index id, t. Id must be constant.
#     #TODO: impfove
#     '''
#     id = [id] if isinstance(id, str) else id
    
#     #Checks
#     group.reset_index(t,drop=True).index.unique()
#     for i in id:
#         if len(group.index.get_level_values(i).unique())>1:
#             raise Exception(f'Index {i} has more than one value')
    
#     if not group.index.get_level_values(t).is_unique:
#         raise Exception(f'Index {t} is not unique')

#     out = group.reset_index(id,drop=True).resample(freq).asfreq()
    
#     for i in id[::-1]:
#         out = prepend_index(out, group.index.get_level_values(i).unique()[0], i)
    
#     return out


# def multiindex_resample(data:pd.DataFrame|pd.Series, id:list[str]|str, t:str, freq: str)->pd.DataFrame|pd.Series:
#     '''
#     Given a dataset with a multiindex, resamples it to a given frequency.
#     '''
#     if not isinstance(data.index, pd.MultiIndex):
#         raise ValueError('Series must have a MultiIndex')
#     return pd.concat([_resample_group(g, id, t, freq) for _,g in data.groupby(id)])

def resample_panel(data:pd.DataFrame|pd.Series, id:str|list[str], freq:str)->pd.DataFrame|pd.Series:
    '''
    Resamples a panel to a given frequency.
    Panel is multiindex
    Parameters:
    -----------
    data: DataFrame
        Panel data with a multiindex, [id, t]
    id: str or list
        Name of the column(s) or indices that identifies the id
    freq: str
        Frequency to resample to. Must be one of 'W','W-MON','M','Q','Y'
    Returns:
    --------
    out: DataFrame
        Resampled panel
    Example:
    --------
    >>> df = pd.DataFrame({'a':[1,2,3]}, index = pd.MultiIndex.from_product([['id1','id2'],pd.date_range('2020-01-01','2020-01-03')], names=['id','date']))
    >>> resample_panel(df, 'id', 'W')
        date        a
    id  date
    id1 2020-01-05  1.0
        2020-01-12  NaN
    id2 2020-01-05  2.0
        2020-01-12  NaN
    '''
    # TODO: avoid changing the or
    if data.index.has_duplicates:
        print('duplicate indices')
        print(data[data.index.duplicated()].sort_index().index)
        raise ValueError('Index has duplicates')
    
    return data.groupby(id, group_keys = True, sort=False).apply(lambda x: x.reset_index(id, drop=True).resample(freq).asfreq())





def interpolate_panel(df:pd.DataFrame, id:list[str], **kwargs):
    '''
    Interpolates a panel.()
    '''
    return df.groupby(id, group_keys = False).apply(lambda x: x.interpolate(**kwargs))



def _is_continuous(x:pd.Series, freq:str)->bool:
    '''
    Checks if a series is continuous
    '''
    if freq == 'M':
        return (x == (x.shift() + pd.DateOffset(months=1)))[1:].all()
    if freq == 'Y':
        if pd.api.types.is_integer_dtype(x):
            return (x==x.shift()+1).iloc[1:].all()
        elif pd.api.types.is_datetime64_any_dtype(x):
            return (x == (x.shift() + pd.DateOffset(years=1)))[1:].all()
        else:
            raise ValueError('Invalid type for yearly frequency')
    else:
        raise ValueError('Only monthly and yearly frequency are supported')




def is_continuous(data:pd.DataFrame, id:list[str], t:str, freq:str)->bool:
    """
    Check if a panel is continuous
    """
    if data.index.has_duplicates:
        raise ValueError('Index has duplicates')
    
    data = data.reset_index()
    
    return data.groupby(id)[t].apply(lambda x: _is_continuous(x, freq)).all()


def plot_coverage(df:pd.DataFrame, id:str, t:str, **kwargs)->None:
    df.reset_index().plot.scatter(x=t,y=id, **kwargs)


def growth_rate_by(data:pd.DataFrame|pd.Series,
                   by:list[str] = None, 
                   h:int=1, 
                   direction='forward'):
    """
    
    """
    if isinstance(data, pd.Series):
        if direction == 'forward':
            return (data.groupby(by, group_keys=False).apply(lambda x: np.log(x).shift(-h)-np.log(x)))
        if direction == 'backward':
            return (data.groupby(by, group_keys=False).apply(lambda x: np.log(x)-np.log(x).shift(h)))
    else:
        raise ValueError("not implemented for pandas dataframe")
    




from functools import reduce
import warnings

import numpy as np
import pandas as pd



def compute_growth_by(df:pd.DataFrame, var:str, id:list, t:str, freq:str, k:int=1)->pd.Series:
     '''
     Computse growth of variable var_growth_t+k = log(x_t+k) - log(x_t) or var_growth_t-k = log(x_t)-log(x_t-k) depending on the sign of k 
     '''
     
     if not is_continuous(df, id = id,t=t,freq=freq):
        raise ValueError(f'data provided is not continuous by {id} at frequency {freq}')

     id = [id] if isinstance(id, str) else id

     out = (df.groupby(id, group_keys=False)[var]
                .apply(lambda x: np.log(x.shift(-k))-np.log(x))).rename(f'{var}_growth_t{+k:+}')
     out = -out if k<0 else out
     out.index =  df.reset_index().set_index(id+[t]).index
     return out


def index_to_year(df:pd.DataFrame, t:str='date')->pd.DataFrame:
    '''
    Given a index t that is a datetime, substitute it for year.
    '''
    df['year'] = df.index.get_level_values(t).year
    new_index = [i if i!=t else 'year' for i in df.index.names]
    df = df.reset_index(t,drop=True)
    return df.reset_index().set_index(new_index)


def resample_panel(data:pd.DataFrame, id:list[str], freq:str)->pd.DataFrame:
    '''
    Resamples an indexed dataframe
    '''
    if data.index.has_duplicates:
        raise ValueError('Index has duplicates')
    return data.groupby(id, group_keys=True).apply(lambda x: x.reset_index(id, drop=True).resample(freq).asfreq())




def _is_continuous(x:pd.Series, freq:str)->bool:
    '''
    Checks if a series is contnuous at the montly or yearly level
    '''
    if freq not in ['M','Y']:
        raise ValueError('Implemented frequencies are M and Y')

    if freq=='M':
        return (x==x.shift()+pd.DateOffset(months=1)).iloc[1:].all()

    if freq=='Y':
        if pd.api.types.is_integer_dtype(x):
            return (x==x.shift()+1).iloc[1:].all()
        elif pd.api.types.is_datetime64_any_dtype(x):
            return (x==x.shift()+pd.DateOffset(years=1)).iloc[1:].all()

def is_continuous(data:pd.DataFrame, id:list[str], t:str, freq:str)->bool:
    if data.index.has_duplicates:
        raise ValueError('Index has duplicates')

    return data.reset_index().groupby(id)[t].apply(lambda x: _is_continuous(x,freq)).all()

class NonContinuousError(Exception):
    def __init__(self, message='data is not continuous'):
          super().__init__(message)


def check_continuous(data:pd.DataFrame, id:list[str], t:str, freq:str)->None:
    """
    Checks if a dataset is continuous. Returns none if it is, and raises an error if it is not.
    """
    if is_continuous(data, id, t, freq):
         return None
    else:
         raise NonContinuousError()







def is_balanced(df:pd.DataFrame, id:list[str])->bool:
    obs_per_id = df.groupby(id).size()
    return all(obs_per_id==obs_per_id.max())

def balance_panel(df:pd.DataFrame, id:list[str], verbose=False)->pd.DataFrame:
    '''
    Ensures that a panel has the same number of observations by id, droping unbalanced terms.
    #TODO: make it more general. 
    Parameters:
    -----------
    df: pd.DataFrame
        Indexed dataframe, index must be id, time.
    '''
    if isinstance(id, list):
        raise NotImplementedError('Only works for one level id')
    if id != df.index.names[0]:
        raise ValueError('id must be indexed first')
    obs_per_id = df.groupby(id).size()
    if verbose:
        id_dropped = (obs_per_id != obs_per_id.max()).sum()
        print(f'{id_dropped} of {id} ({100*id_dropped/obs_per_id.size:.1f}%) dropped')
    balanced_id = obs_per_id.index[obs_per_id == obs_per_id.max()]
    return df.loc[balanced_id]



def balance_panel(df, id:list[str], t:str, freq:str=None):
    """
    Completes a panel.
    """
    #warnings.warn("balance panel has changed")
    id = [id] if isinstance(id, str) else id
    index = df.index.names
    df = df.reset_index()
    if True:
        t_range = pd.Series(range(df[t].min(), df[t].max()+1), name=t)
    else:
        t_range = pd.date_range(df[t].min(), df[t].max(), freq=freq, name=t)

    out =  pd.merge(df[id].drop_duplicates(),
                    t_range, how='cross')
    out = pd.merge(out, df, on = id+[t], how='left')
    return out.set_index(index)

def interpolate_by(df, id:list[str], **kwargs):
    return df.groupby(id,group_keys=False).apply(lambda x: x.interpolate(**kwargs))



def constant_column_by(df:pd.DataFrame, column:list[str], id:list[str],verbose=False)->pd.Series:
    '''
    Ensures a column is constant within id.
    Returns:
    --------
    pd.Series

    '''
    if isinstance(column, list):
        id_list = [constant_column_by(df, i, id, verbose) for i in column]
        return reduce(lambda x,y: x.intersection(y), id_list)

    constant_column = df.groupby(id)[column].nunique(dropna=False).eq(1)

    if verbose:
        print(f'{constant_column.mean()*100:.1f} % of {id} have a constant value for {column}')
    return constant_column.index[constant_column]

  