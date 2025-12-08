from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from dataclasses import dataclass
import pandas as pd
from .converters import python_to_r, r_to_python, pandas_to_r

@dataclass
class FixestModel:
    coeftable: pd.DataFrame

    def __repr__(self)->str:
        return self.coeftable.__repr__()
    

def install_fixest()->None:
    utils = importr('utils')
    utils.chooseCRANmirror(ind=1)
    utils.install_packages('fixest')


def feols(fml:str, data:pd.DataFrame)->FixestModel:
    ro.r("library(fixest)")
    ro.globalenv["data"] = pandas_to_r(data)
    ro.r(f"fitted_model <- feols({fml}, data = data)")
    fitted_model = ro.globalenv['fitted_model']
    coeftable = r_to_python(fitted_model.rx2('coeftable'))
    coeftable.index.name = 'variable'
    return FixestModel(coeftable=coeftable)



# Standard imports
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property 
import warnings
# Third party imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List 
import rpy2.robjects as ro
# Local imports
from .converters import pandas_to_r, r_to_pandas, formula_to_string


@dataclass
class FixestModel:
    coeftable: pd.DataFrame
    formula: str
    nobs: int
    dep_var: str
    fixef_vars:str = None
    sample: dict[str] = None
    subset:str = None
    cluster: str = None
    r_model: ro.ListVector = None 

    def __repr__(self)->str:
        repr =  'feols regression: ' + self.formula
        if self.sample:
            repr = repr + '\n' + str(self.sample)
        if self.subset:
            repr = repr + '\n' + self.subset
        return repr

    @classmethod
    def from_r(cls,r_model:ro.ListVector)->FixestModel:
        formula = formula_to_string(r_model.rx2('fml'))
        
        if r_model.rx2('model_info').rx2('sample'):
            var = r_model.rx2('model_info').rx2('sample').rx2('var')[0]
            value = r_model.rx2('model_info').rx2('sample').rx2('value')[0]
            value = True if value=='TRUE' else value
            value = False if value=='FALSE' else value
            sample = {'var':var,'value':value}
        else:
            sample = None

        if r_model.rx2('model_info').rx2('subset'):
            subset = r_model.rx2('model_info').rx2('subset')[0]
        else:
            subset = None

        if r_model.rx2('fixef_vars'):
            fixef_vars = [i for i in r_model.rx2('fixef_vars')]
        else:
            fixef_vars = None

        if r_model.rx2('summary_flags'):
            if r_model.rx2('summary_flags').rx2('vcov'):
                cluster = formula_to_string(r_model.rx2('summary_flags').rx2('vcov')).split('~')[1].lstrip()
        else:
            cluster = None
    
        return cls(coeftable = coeftable_to_pandas(r_model.rx2('coeftable')),
                            formula= formula,
                            nobs = r_model.rx2('nobs')[0],
                            dep_var = formula.split('~')[0].replace(" ",""),
                            fixef_vars = fixef_vars,
                            sample = sample,
                            subset = subset,
                            cluster = cluster, 
                            r_model = r_model)


    @cached_property 
    def vcov(self)->pd.DataFrame:
        '''
        Computes the variance covariance matrix of the coefficients 
        ''' 
        ro.globalenv['model'] = self.r_model 
        return pd.DataFrame(r_to_pandas(ro.r('''vcov(model)''')), 
                            index=self.coeftable.index, 
                            columns = self.coeftable.index)

    
    def to_frame(self, expand=True)->pd.DataFrame:
        out = expand_coeftable(self.coeftable) if expand else self.coeftable
        out['dep_var'] = self.dep_var
        out['formula'] = self.formula
        out['fixed_effects'] = '+'.join(self.fixef_vars)
        if self.cluster:
            out['cluster'] = self.cluster
        if self.sample:
            out['sample_var'] = self.sample['var']
            out['sample_value'] = self.sample['value']
        out['n'] = self.nobs
        return out

    
def coeftable_to_pandas(coeftable_r:ro.vectors.DataFrame)->pd.DataFrame:
    coeftable = np.array(coeftable_r)
    if coeftable.shape[0]==4 and coeftable.shape[1]!=4: #not sure why some times the shape is transposed
        coeftable = coeftable.T

    if coeftable.shape == (4,4):
        if np.any((coeftable[:,3]>1) | (coeftable[:,3]<0)):
            coeftable = coeftable.T
    
    return pd.DataFrame(coeftable, 
                            index = list(ro.r.rownames(coeftable_r)), 
                            columns = list(ro.r.colnames(coeftable_r)))


def feols(fml: str, 
            data: pd.DataFrame, 
            cluster: str = None,
            fsplit: str = None,
            subset: str = None,
            weights: str= None, 
	    panel_id: list[str]=None,
            pre_process: list[str] = None)->FixestModel:
#CW notes: added vcov:bool=False (for vcov matrix) 
    '''
    Runs a the R feols function in python.
    Parameters
    ----------
    fml: str
	String representing the R formula
    pre_process: list[str]
        List of r code lines to be run before running feols.
    '''
    ro.globalenv['data'] = pandas_to_r(data.reset_index())

    cluster = 'NULL' if cluster is None else f'"{cluster}"'
    fsplit = 'NULL' if fsplit is None else f'~{fsplit}'
    subset = 'NULL' if subset is None else f'~{subset}'
    weights = 'NULL' if weights is None else f'~{weights}'
    panel_id = 'NULL' if panel_id is None else ','.join(panel_id)

 

    # Check if list of elements to pass through 
    if pre_process: 
        r_code = '\n'.join(pre_process)
        ro.r(r_code)

    ro.r(f'''
        library(fixest)
        fitted_model<-feols({fml}, 
                            data=data, 
                            cluster={cluster},
                            fsplit = {fsplit}, split=NULL,
                            subset = {subset},
                            weights = {weights},
			    panel.id = {panel_id}
                            )

        fixest_multi <- class(fitted_model) == 'fixest_multi'
        ''')
    fitted_model = ro.globalenv['fitted_model']
    fixest_multi = bool(ro.globalenv['fixest_multi'][0])

    if fixest_multi:
        return [FixestModel.from_r(i) for i in fitted_model]
    else: 
        return FixestModel.from_r(fitted_model) 



def expand_coeftable(coeftable:pd.DataFrame)->pd.DataFrame:
    '''
    Expands a coeftable that has interactions, adding the columns factor, value, interaction.
    If there are more than one sorting, then columns are factor1, value1, factor2, value2, interaction.
    '''
    index = coeftable.index.to_series().str.split(':',expand=True)

    if index.shape[1]==4:
        interactions = index.iloc[:,[0,2,3]]
        interactions.columns = ['factor', 'value','interaction']
        return coeftable.join(interactions)
    
    elif index.shape[1]==6:
        interactions = index.iloc[:,[0,3,1,4,5]]
        interactions.columns = ['factor1', 'value1','factor2','value2','interaction']
        return coeftable.join(interactions)
    
    elif index.shape[1]==1:
        warnings.warn('Trying to expand a table without interactions')
        coeftable[['factor', 'value','interaction']] = None
        return coeftable






def iplot(model:FixestModel, interaction:str, alpha:float=0.05, ax=None,
          hline:bool=True, legend:bool=True, savefig:str=None, **kwargs)->None:
    """
    Plots the interaction
    """
    coeftable = expand_coeftable(model.coeftable)
    coeftable = coeftable.query(f'interaction == "{interaction}"')
    if coeftable.empty:
        raise ValueError(f'{interaction} not found in coeftable')
    if ax is None:
        fig,ax = plt.subplots()
    
    if 'value' in coeftable:
        _iplot_single_sort(ax, coeftable, **kwargs)
    elif 'value1' in coeftable:
        _iplot_double_sort(ax, coeftable, **kwargs)
    ax.tick_params(axis='x',rotation=90)
    ax.set(ylabel=interaction, title = model.dep_var)
    if hline:
        ax.axhline(0,lw=1)
    if savefig:
        plt.savefig(savefig)
        plt.close()
        return None
    else:
        return fig, ax
   


def ttest(model:FixestModel, a:str, b:str):
    """
    Tests a=b
    """
    if a not in model.coeftable.index:
        raise ValueError(f"{a} not in model")
    if b not in model.coeftable.index:
        raise ValueError(f"{b} not in model")

    coeftable = model.coeftable["Estimate"]   
    vcov = model.vcov
    return (coeftable.loc[a]-coeftable.loc[b])/np.sqrt(vcov.loc[a,a]+vcov.loc[b,b]-2*vcov.loc[a,b])

    


def _iplot_single_sort(ax, coeftable, **kwargs):
    return ax.errorbar(x = coeftable['value'], y = coeftable['Estimate'], yerr = 1.96*coeftable['Std. Error'], **kwargs)
    
def _iplot_double_sort(ax, coeftable, **kwargs)->None:
    for (value2,icoef) in coeftable.groupby('value2'):
        ax.errorbar(x = icoef['value1'], y = icoef['Estimate'], yerr = 1.96*icoef['Std. Error'], label=value2, **kwargs)
    ax.legend()
    return ax