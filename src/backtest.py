from __future__ import annotations

import pandas as pd


def var_exceptions(returns: pd.Series, var_series: pd.Series) -> pd.Series:
    """
    Exception (breach) occurs when realized return is less than VaR threshold.
    Both should be aligned time series (VaR usually negative threshold).
    """
    aligned = pd.concat([returns, var_series], axis=1).dropna()
    aligned.columns = ["ret", "var"]
    breaches = aligned["ret"] < aligned["var"]
    breaches.name = "breach"
    return breaches


def exception_summary(breaches: pd.Series, alpha: float) -> dict:
    """
    Basic exception stats for VaR backtest.
    alpha = 0.05 => expected breach rate 5%
    """
    n = int(breaches.shape[0])
    x = int(breaches.sum())
    expected = alpha * n if n > 0 else 0.0
    rate = (x / n) if n > 0 else 0.0
    return {
        "n_obs": n,
        "alpha": alpha,
        "breaches": x,
        "expected_breaches": expected,
        "breach_rate": rate,
    }
