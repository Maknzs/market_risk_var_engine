from __future__ import annotations

import pandas as pd
from scipy.stats import norm


def historical_var(returns: pd.Series, window: int = 250, alpha: float = 0.05) -> pd.Series:
    """
    Historical VaR on a return series.
    alpha=0.05 => 95% VaR threshold (5th percentile)
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1).")
    return returns.rolling(window).quantile(alpha)


def parametric_var_normal(
    returns: pd.Series,
    window: int = 250,
    alpha: float = 0.05,
    use_mean: bool = True,
) -> pd.Series:
    """
    Parametric (Normal) VaR using rolling mean/std of returns:
      VaR_alpha = mu + z_alpha * sigma
    where z_alpha = norm.ppf(alpha) (negative for alpha<0.5).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1).")

    z = norm.ppf(alpha)  # e.g., alpha=0.05 -> ~ -1.645

    mu = returns.rolling(window).mean() if use_mean else 0.0
    sigma = returns.rolling(window).std(ddof=1)

    return mu + z * sigma
