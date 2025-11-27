import pandas as pd


def percentile_by():
    pass


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