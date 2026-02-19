# Market Risk VaR Engine

Multi-asset portfolio market risk toolkit with historical and parametric Value-at-Risk (VaR), Expected Shortfall (ES), backtesting, stress testing, and component VaR decomposition.

This project mirrors a simplified daily risk workflow used in market risk monitoring.

## Overview

Core capabilities:

- Portfolio return aggregation from asset-level returns
- Historical VaR (95%, 99%)
- Parametric Normal VaR (rolling volatility)
- EWMA VaR (RiskMetrics style)
- Rolling covariance-matrix VaR
- Historical ES (CVaR)
- VaR backtesting (exception counts + Kupiec UC test)
- Exception report export (CSV-ready table)
- Stress scenario analysis
- Component VaR decomposition and pre-trade risk impact checks

All risk measures can be computed in return space and scaled to dollar P&L.

## Models

### 1. Historical VaR

Empirical rolling quantile over a lookback window:

```text
VaR_alpha(t) = Quantile_alpha(r_{t-window+1:t})
```

### 2. Parametric VaR (Normal)

```text
VaR_alpha(t) = mu_t + z_alpha * sigma_t
```

- `mu_t`: rolling mean (optional)
- `sigma_t`: rolling standard deviation
- `z_alpha`: Normal quantile from `scipy.stats.norm.ppf(alpha)`

### 3. EWMA VaR (RiskMetrics)

```text
sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2
VaR_alpha(t) = mu_t + z_alpha * sigma_t
```

### 4. Covariance-Matrix VaR

```text
sigma_p(t) = sqrt(w^T * Sigma_t * w)
VaR_alpha(t) = mu_p(t) + z_alpha * sigma_p(t)
```

### 5. Expected Shortfall (ES)

```text
ES_alpha(t) = E[r | r <= VaR_alpha(t)]
```

### 6. Backtesting

- Exception (breach) tracking
- Kupiec (1995) unconditional coverage test
- PASS/FAIL interpretation via p-value threshold

### 7. Component VaR

Parametric contribution of each asset to total VaR:

```text
ComponentVaR_i = w_i * z_alpha * ((Sigma * w)_i / sigma_p)
```

## Project Structure

```text
src/
  data_loader.py   # prices and return construction
  portfolio.py     # portfolio definition and aggregation
  var_models.py    # VaR / ES / component VaR models
  backtest.py      # breaches, summaries, Kupiec test
  scaling.py       # return <-> dollar scaling
  plots.py         # plotting utilities
notebooks/
  07_var_analysis.ipynb
  08_var_analysis.ipynb
  09_var_analysis.ipynb
```

## Typical Outputs

- VaR overlays (historical, parametric, EWMA, covariance)
- Exception tables (worst breaches first)
- Kupiec UC summary metrics
- Stress scenario comparison tables/charts
- Component VaR contribution charts

Output locations used in notebooks:

- Figures: `outputs/figures/`
- Exception and summary tables: `outputs/`

## Tech Stack

- Python 3.x
- pandas
- numpy
- scipy
- matplotlib
- yfinance
- Jupyter Notebook

## Extension Ideas

- Student-t VaR
- Conditional coverage (Christoffersen)
- EWMA covariance matrix
- Factor-model VaR
- GARCH volatility
- Monte Carlo VaR
