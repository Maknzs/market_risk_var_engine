from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_var_overlay(
    returns: pd.Series,
    var_hist: pd.Series,
    var_param: pd.Series,
    title: str,
    outpath: Optional[str] = None,
) -> None:
    """
    Plot realized returns and VaR thresholds (historical + parametric).
    Marks breaches vs historical VaR.
    """
    df = pd.concat([returns, var_hist, var_param], axis=1).dropna()
    df.columns = ["return", "VaR_hist", "VaR_param"]

    breaches = df["return"] < df["VaR_hist"]

    plt.figure()
    plt.plot(df.index, df["return"], label="Portfolio Return")
    plt.plot(df.index, df["VaR_hist"], label="Historical VaR")
    plt.plot(df.index, df["VaR_param"], label="Parametric VaR (Normal)")

    # breach markers
    plt.scatter(df.index[breaches], df.loc[breaches, "return"], label="Breaches", marker="x")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()

    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=150)

    plt.show()
