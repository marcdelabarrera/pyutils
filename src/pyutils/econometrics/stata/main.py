import pandas as pd
from pystata import stata # type: ignore

import numpy as np
from scipy.stats import norm

# def ivreg(reg:str, data:pd.DataFrame):
#     stata.pdataframe_to_data(data, force=True)
#     stata.run(reg)
#     return  stata.get_ereturn()


def ivreg(estimator:str, depvar:str, exog:list[str], endog:str, instruments:list[str], data:pd.DataFrame):
    stata.pdataframe_to_data(data, force=True)
    stata.run(f'ivregress {estimator} {depvar} {" ".join(exog)} ({endog} = {" ".join(instruments)})')
    return  stata.get_ereturn()

def xtset(pvar):
    stata.run('xtset fe')
    stata.run(f'xtreg y x1 x2, fe')
    return stata.get_ereturn()


from dataclasses import dataclass
@dataclass
class RegressionResult:
    coeftable:pd.DataFrame
    call: str


    @classmethod
    def from_ereturn(cls, ereturn:dict):
        coeftable = pd.DataFrame({"coef": ereturn["e(b)"].flatten(),
                                  "se": np.diag(np.sqrt(ereturn["e(V)"])),
                                  }, index = ereturn["e(indepvars)"].split())
        coeftable["t-stat"] = coeftable["coef"] / coeftable["se"]

        coeftable["p-value"] = 2 * (1 - norm.cdf(np.abs(coeftable["t-stat"])))

        return cls(coeftable=coeftable, call=ereturn["e(cmdline)"])


def reghdfe(depvar:str, exog:list[str], absorb:str, vce:str, data:pd.DataFrame) -> RegressionResult:
    stata.pdataframe_to_data(data, force=True)
    stata.run(f'reghdfe {depvar} {" ".join(exog)}, absorb({absorb}) vce({vce})')
    return RegressionResult.from_ereturn(stata.get_ereturn())

def install_reghdf():
    stata.run('ssc install require, replace')
    stata.run('ssc install ftools, replace')
    stata.run('ssc install reghdfe, replace')
