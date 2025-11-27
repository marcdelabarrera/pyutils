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