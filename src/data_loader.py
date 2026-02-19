from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class MarketDataConfig:
    tickers: List[str]
    start: str = "2015-01-01"
    end: Optional[str] = None  # None => up to latest
    price_field: str = "Adj Close"  # yfinance columns
    use_log_returns: bool = True


def fetch_prices(cfg: MarketDataConfig) -> pd.DataFrame:
    """
    Fetch adjusted close prices for tickers from Yahoo Finance via yfinance.
    Returns a DataFrame indexed by date with one column per ticker.
    """
    raw = yf.download(
        tickers=cfg.tickers,
        start=cfg.start,
        end=cfg.end,
        auto_adjust=False,
        progress=False,
    )

    if cfg.price_field not in raw.columns:
        # When multiple tickers, columns are MultiIndex: (Field, Ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                prices = raw[cfg.price_field].copy()
            except KeyError as e:
                raise KeyError(f"Could not find '{cfg.price_field}' in downloaded data.") from e
        else:
            raise KeyError(f"Could not find '{cfg.price_field}' in downloaded data.")
    else:
        prices = raw[cfg.price_field].copy()

    # Ensure DataFrame shape
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=cfg.tickers[0])

    prices = prices.sort_index()
    prices = prices.dropna(how="all")

    # Forward-fill occasional missing values, then drop any remaining rows
    prices = prices.ffill().dropna()

    return prices


def compute_returns(prices: pd.DataFrame, use_log: bool = True) -> pd.DataFrame:
    """
    Compute daily returns from prices.
    - log returns: log(Pt/Pt-1)
    - simple returns: Pt/Pt-1 - 1
    """
    if use_log:
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()

    rets = rets.dropna()
    return rets
