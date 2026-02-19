from __future__ import annotations

import pandas as pd


def scale_to_dollars(series: pd.Series, notional: float) -> pd.Series:
    """
    Convert a return-series (or VaR return threshold) into dollars given a notional.
    Example:
      pnl_$ = notional * return
      VaR_$ = notional * VaR_return_threshold
    """
    if notional <= 0:
        raise ValueError("notional must be positive.")
    out = series * float(notional)
    out.name = f"{series.name}_dollars" if series.name else "dollars"
    return out
