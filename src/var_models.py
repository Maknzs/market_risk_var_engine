from __future__ import annotations

import numpy as np
import pandas as pd


def historical_var(returns: pd.Series, window: int = 250, alpha: float = 0.05) -> pd.Series:
    """
    Historical VaR on a return series.
    alpha=0.05 => 95% VaR threshold (5th percentile)
    Returns a series aligned to returns index with NaNs for initial window.
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
    Parametric (Normal) VaR using rolling mean/std of portfolio returns:
      VaR_alpha = mu + z_alpha * sigma
    where z_alpha is the alpha-quantile of standard normal (negative for alpha<0.5).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1).")

    # z for Normal(0,1) quantile at alpha
    # Use numpy approximation via scipy? Keep clean: hardcode common alphas.
    z_map = {0.05: -1.6448536269514722, 0.01: -2.3263478740408408}
    z = z_map.get(alpha)
    if z is None:
        # Fallback: approximate using inverse error function if scipy not desired
        # If you installed scipy, we can do it properly; for now require common alphas.
        raise ValueError("For clean version, use alpha=0.05 or alpha=0.01.")

    mu = returns.rolling(window).mean() if use_mean else 0.0
    sigma = returns.rolling(window).std(ddof=1)

    return mu + z * sigma
