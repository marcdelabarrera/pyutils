import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

def python_to_r(data: pd.DataFrame):
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_obj = ro.conversion.py2rpy(data)
    return r_obj

def r_to_python(r_obj):
    with localconverter(ro.default_converter + pandas2ri.converter):
        py_data = ro.conversion.rpy2py(r_obj)
    return py_data


def pandas_to_r(data:pd.DataFrame):
    with (ro.default_converter + pandas2ri.converter).context():
        r_dataframe = ro.conversion.get_conversion().py2rpy(data)
    return r_dataframe