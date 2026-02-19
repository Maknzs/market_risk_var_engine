from __future__ import annotations

import numpy as np
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

def parametric_var_ewma_normal(
    returns: pd.Series,
    alpha: float = 0.05,
    lam: float = 0.94,
    use_mean: bool = False,
    burn_in: int = 30,
) -> pd.Series:
    """
    Parametric VaR using EWMA volatility (RiskMetrics-style) and Normal quantile.

    EWMA variance recursion:
      sigma2_t = lam * sigma2_{t-1} + (1-lam) * r_{t-1}^2

    VaR threshold:
      VaR_t = mu_t + z_alpha * sigma_t

    Notes:
    - use_mean=False is common in RiskMetrics (assume mean ~ 0 daily)
    - burn_in: number of initial periods to set as NaN for stability
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1).")
    if not 0 < lam < 1:
        raise ValueError("lam must be in (0,1).")

    r = returns.dropna().astype(float)
    z = norm.ppf(alpha)

    # Initialize variance with sample variance of first ~60 obs (or all if shorter)
    init_n = min(60, len(r))
    if init_n < 2:
        raise ValueError("Not enough return observations for EWMA initialization.")

    sigma2 = np.empty(len(r))
    sigma2[0] = float(r.iloc[:init_n].var(ddof=1))

    # recursion uses lagged return
    for t in range(1, len(r)):
        sigma2[t] = lam * sigma2[t - 1] + (1.0 - lam) * (r.iloc[t - 1] ** 2)

    sigma = pd.Series(np.sqrt(sigma2), index=r.index, name="ewma_sigma")

    if use_mean:
        mu = r.ewm(alpha=(1.0 - lam), adjust=False).mean()
    else:
        mu = 0.0

    var = mu + z * sigma
    var = pd.Series(var, index=r.index, name=f"VaR_EWMA_{int((1-alpha)*100)}")

    if burn_in and burn_in > 0:
        var.iloc[:burn_in] = np.nan

    # Reindex to original returns index (preserve any missing dates)
    return var.reindex(returns.index)

def parametric_var_cov_matrix(
    asset_returns: pd.DataFrame,
    weights: dict,
    window: int = 250,
    alpha: float = 0.05,
    use_mean: bool = True,
) -> pd.Series:
    """
    Parametric VaR using rolling covariance matrix:
      sigma_p(t) = sqrt(w^T Sigma(t) w)
      mu_p(t) = rolling mean of portfolio returns (optional)
      VaR(t) = mu_p(t) + z_alpha * sigma_p(t)

    asset_returns: DataFrame with columns as tickers
    weights: dict {ticker: weight}
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1).")

    tickers = list(weights.keys())
    missing = [t for t in tickers if t not in asset_returns.columns]
    if missing:
        raise ValueError(f"Missing tickers in asset_returns: {missing}")

    r = asset_returns[tickers].dropna().copy()

    w = pd.Series(weights, index=tickers, dtype=float)
    w = w / w.sum()
    wv = w.values.reshape(-1, 1)

    z = norm.ppf(alpha)

    # Optional rolling mean of portfolio returns
    if use_mean:
        mu_p = (r @ w).rolling(window).mean()
    else:
        mu_p = 0.0

    # Rolling covariance -> portfolio sigma -> VaR series
    var_list = []
    idx_list = []

    # Use a loop for clarity (fast enough for 4 assets)
    for i in range(window - 1, len(r)):
        window_slice = r.iloc[i - window + 1 : i + 1]
        sigma = window_slice.cov().values  # Sigma(t)
        sigma_p = float(np.sqrt((wv.T @ sigma @ wv)[0, 0]))
        idx_list.append(r.index[i])
        var_list.append(mu_p.iloc[i] + z * sigma_p if use_mean else z * sigma_p)

    out = pd.Series(var_list, index=pd.Index(idx_list), name=f"VaR_Cov_{int((1-alpha)*100)}")
    return out.reindex(asset_returns.index)

def historical_es(returns: pd.Series, window: int = 250, alpha: float = 0.05) -> pd.Series:
    """
    Historical Expected Shortfall (ES): average return in the tail beyond VaR.
    ES_t = mean( r | r <= VaR_alpha ) over rolling window.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1).")

    def es_func(x: pd.Series) -> float:
        v = x.quantile(alpha)
        tail = x[x <= v]
        return float(tail.mean()) if len(tail) else float("nan")

    return returns.rolling(window).apply(es_func, raw=False)

def component_var_cov_matrix(
    asset_returns: pd.DataFrame,
    weights: dict,
    window: int = 250,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Component VaR under multivariate normal with covariance matrix Sigma (rolling).

    For each date t:
      sigma_p = sqrt(w' Sigma w)
      marginal_sigma = (Sigma w) / sigma_p
      component_VaR_i = w_i * z_alpha * marginal_sigma_i

    Returns a DataFrame of component VaR contributions (same units as returns),
    indexed by date, columns=tickers.

    Note: This is a parametric decomposition; does not require portfolio return series.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1).")

    tickers = list(weights.keys())
    r = asset_returns[tickers].dropna().copy()

    w = pd.Series(weights, index=tickers, dtype=float)
    w = w / w.sum()
    wv = w.values.reshape(-1, 1)

    z = norm.ppf(alpha)

    rows = []
    idx_vals = []

    for i in range(window - 1, len(r)):
        window_slice = r.iloc[i - window + 1 : i + 1]
        sigma = window_slice.cov().values

        sigma_w = sigma @ wv  # (n,1)
        sigma_p = float(np.sqrt((wv.T @ sigma_w)[0, 0]))

        if sigma_p == 0:
            comp = np.zeros(len(tickers))
        else:
            marginal_sigma = (sigma_w.flatten() / sigma_p)  # (n,)
            comp = (w.values * z * marginal_sigma)          # (n,)

        rows.append(comp)
        idx_vals.append(r.index[i])

    out = pd.DataFrame(rows, index=pd.Index(idx_vals), columns=tickers)
    out.name = f"ComponentVaR_{int((1-alpha)*100)}"
    return out.reindex(asset_returns.index)
