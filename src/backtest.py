from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1).")


def _align_returns_var(returns: pd.Series, var_series: pd.Series) -> pd.DataFrame:
    aligned = pd.concat([returns, var_series], axis=1).dropna()
    aligned.columns = ["realized", "var"]
    return aligned


def var_exceptions(returns: pd.Series, var_series: pd.Series) -> pd.Series:
    """
    Exception (breach) occurs when realized return is less than VaR threshold.
    Both should be aligned time series (VaR usually negative threshold).
    """
    aligned = _align_returns_var(returns, var_series)
    breaches = aligned["realized"] < aligned["var"]
    breaches.name = "breach"
    return breaches


def exception_summary(breaches: pd.Series, alpha: float) -> dict:
    """
    Basic exception stats for VaR backtest.
    alpha = 0.05 => expected breach rate 5%
    """
    _validate_alpha(alpha)
    b = breaches.dropna().astype(bool)
    n = int(b.shape[0])
    x = int(b.sum())
    expected = alpha * n if n > 0 else 0.0
    rate = (x / n) if n > 0 else 0.0
    return {
        "n_obs": n,
        "alpha": alpha,
        "breaches": x,
        "expected_breaches": expected,
        "breach_rate": rate,
        "coverage_gap": rate - alpha,
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
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be a positive integer or None.")

    df = _align_returns_var(returns, var_series)

    df["breach"] = df["realized"] < df["var"]
    df["exceedance"] = df["realized"] - df["var"]

    breaches = df[df["breach"]].copy()
    breaches = breaches.sort_values("realized")  # worst days first

    if top_n is not None:
        breaches = breaches.head(top_n)

    breaches.index.name = "date"
    return breaches


def kupiec_test_uc(breaches: pd.Series, alpha: float) -> dict:
    """
    Kupiec (1995) Unconditional Coverage (UC) test for VaR exceptions.

    H0: exception probability == alpha
    Inputs:
      - breaches: boolean Series (True if realized < VaR)
      - alpha: expected exception rate (e.g., 0.05 for 95% VaR)

    Returns dict with LR statistic and p-value.
    """
    _validate_alpha(alpha)
    b = breaches.dropna().astype(bool)
    n = int(b.shape[0])
    x = int(b.sum())

    if n == 0:
        raise ValueError("No observations in breaches series.")

    # empirical exception rate
    phat = x / n

    # Guard against log(0) issues for x=0 or x=n
    eps = 1e-12
    phat = min(max(phat, eps), 1 - eps)
    p = min(max(alpha, eps), 1 - eps)

    # Likelihood under H0 and under MLE
    ll_h0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll_mle = (n - x) * np.log(1 - phat) + x * np.log(phat)

    lr_uc = -2.0 * (ll_h0 - ll_mle)
    p_value = 1.0 - chi2.cdf(lr_uc, df=1)

    return {
        "n_obs": n,
        "breaches": x,
        "alpha": alpha,
        "breach_rate": x / n,
        "lr_uc": lr_uc,
        "p_value": p_value,
    }
