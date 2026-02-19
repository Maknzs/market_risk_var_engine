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

def exception_table(
    returns: pd.Series,
    var_series: pd.Series,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Build an exceptions report with dates where realized return/P&L breaches VaR.
    Returns a DataFrame sorted by worst realized return/P&L (most negative first).

    Columns:
      - realized: realized return or $ P&L
      - var: VaR threshold (same units)
      - breach: boolean
      - exceedance: realized - var (more negative => worse breach)
    """
    df = pd.concat([returns, var_series], axis=1).dropna()
    df.columns = ["realized", "var"]

    df["breach"] = df["realized"] < df["var"]
    df["exceedance"] = df["realized"] - df["var"]

    breaches = df[df["breach"]].copy()
    breaches = breaches.sort_values("realized")  # worst days first

    if top_n is not None and top_n > 0:
        breaches = breaches.head(top_n)

    return breaches
