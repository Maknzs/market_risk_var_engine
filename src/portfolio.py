from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Portfolio:
    """
    Simple long-only portfolio.
    weights should sum to 1.0.
    """
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("weights must not be empty.")
        w = pd.Series(self.weights, dtype=float)
        if w.isna().any() or not np.isfinite(w.to_numpy()).all():
            raise ValueError("weights must be finite numeric values.")
        if (w < 0).any():
            raise ValueError("weights must be non-negative for a long-only portfolio.")

    def normalized(self) -> "Portfolio":
        w = pd.Series(self.weights, dtype=float)
        s = float(w.sum())
        if np.isclose(s, 0.0):
            raise ValueError("Weights sum to 0.")
        w = w / s
        return Portfolio(weights=w.to_dict())

    @property
    def tickers(self) -> Iterable[str]:
        return list(self.weights.keys())


def portfolio_returns(asset_returns: pd.DataFrame, portfolio: Portfolio) -> pd.Series:
    """
    Compute portfolio daily returns = R * w.
    asset_returns: columns are tickers.
    """
    p = portfolio.normalized()
    missing = [t for t in p.tickers if t not in asset_returns.columns]
    if missing:
        raise ValueError(f"Missing tickers in returns data: {missing}")

    w = pd.Series(p.weights).reindex(asset_returns.columns).fillna(0.0).values
    pr = asset_returns.values @ np.asarray(w, dtype=float)
    return pd.Series(pr, index=asset_returns.index, name="portfolio_return")
