from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import pandas as pd


def _default_y_label(series: pd.Series) -> str:
    name = (series.name or "").lower()
    if any(token in name for token in ("$", "pnl", "dollar", "usd")):
        return "P&L ($)"
    return "Return"


def plot_var_overlay(
    returns: pd.Series,
    var_hist: pd.Series,
    var_param: pd.Series,
    title: str,
    outpath: Optional[str] = None,
    y_label: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot realized returns and VaR thresholds (historical + parametric).
    Marks breaches vs historical VaR.
    """
    df = pd.concat([returns, var_hist, var_param], axis=1).dropna()
    df.columns = ["return", "VaR_hist", "VaR_param"]

    breaches = df["return"] < df["VaR_hist"]

    return_label = returns.name.replace("_", " ").title() if returns.name else "Portfolio Return"
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["return"], label=return_label)
    plt.plot(df.index, df["VaR_hist"], label="Historical VaR")
    plt.plot(df.index, df["VaR_param"], label="Parametric VaR (Normal)")

    # breach markers
    plt.scatter(df.index[breaches], df.loc[breaches, "return"], label="Breaches", marker="x")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_label or _default_y_label(returns))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def plot_var_multi(
    returns: pd.Series,
    var_lines: Mapping[str, pd.Series],
    title: str,
    outpath: Optional[str] = None,
    mark_breaches_against: Optional[str] = None,
    y_label: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot realized returns (or $ P&L) and multiple VaR threshold series.

    var_lines: dict of {label: series}
    mark_breaches_against: label key in var_lines to mark breaches (optional)
    """
    if not var_lines:
        raise ValueError("var_lines must not be empty.")

    df = pd.DataFrame({"return": returns})
    for label, s in var_lines.items():
        df[label] = s

    df = df.dropna()

    return_label = returns.name.replace("_", " ").title() if returns.name else "Portfolio Series"
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["return"], label=return_label)

    for label in var_lines.keys():
        plt.plot(df.index, df[label], label=label)

    if mark_breaches_against and mark_breaches_against in df.columns:
        breaches = df["return"] < df[mark_breaches_against]
        plt.scatter(df.index[breaches], df.loc[breaches, "return"], label="Breaches", marker="x")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_label or _default_y_label(returns))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()
