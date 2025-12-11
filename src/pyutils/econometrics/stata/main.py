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

def xtset(pvar):
    stata.run('xtset fe')
    stata.run(f'xtreg y x1 x2, fe')
    return stata.get_ereturn()

def reghdfe(depvar:str, exog:list[str], absorb:str, vce:str, data:pd.DataFrame):
    stata.pdataframe_to_data(data, force=True)
    stata.run(f'reghdfe {depvar} {" ".join(exog)}, absorb({absorb}) vce({vce})')
    return stata.get_ereturn()
