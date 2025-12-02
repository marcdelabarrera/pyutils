import pandas as pd
from pystata import stata

def ivreg(reg:str, data:pd.DataFrame):
    stata.pdataframe_to_data(data, force=True)
    stata.run(reg)
    return  stata.get_ereturn()


def ivreg(estimator:str, depvar:str, exog:list[str], endog:str, instruments:list[str], data:pd.DataFrame):
    stata.pdataframe_to_data(data, force=True)
    stata.run(f'ivregress {estimator} {depvar} {" ".join(exog)} ({endog} = {" ".join(instruments)})')
    return  stata.get_ereturn()