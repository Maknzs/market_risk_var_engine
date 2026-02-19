# 📊 Market Risk VaR Engine

A multi-asset portfolio risk monitoring toolkit implementing historical and parametric Value-at-Risk (VaR), Expected Shortfall (ES), backtesting, stress testing, and portfolio risk decomposition.

This project replicates a simplified daily market risk monitoring workflow for a trading portfolio.

---

## 🔎 Overview

This engine performs:

- Portfolio return aggregation
- Historical VaR (95%, 99%)
- Parametric Normal VaR (rolling volatility)
- EWMA (RiskMetrics-style) VaR
- Rolling covariance-matrix VaR
- Historical Expected Shortfall (ES / CVaR)
- VaR backtesting (exception counts)
- Kupiec Unconditional Coverage test
- Exception reporting (CSV export)
- Stress scenario analysis
- Component (marginal) VaR decomposition
- Pre-trade risk impact simulation

All metrics are computed in both return space and dollar P&L space.

---

## 🧠 Models Implemented

### 1️⃣ Historical VaR

Rolling empirical quantile over a 252-day window.

VaR*α(t) = Quantile*α(r\_{t-252:t})

---

### 2️⃣ Parametric VaR (Rolling Normal)

VaR*α(t) = μ_t + z*α σ_t

Where:

- μ_t = rolling mean
- σ_t = rolling standard deviation
- z_α = Normal quantile via `scipy.stats.norm.ppf`

---

### 3️⃣ EWMA VaR (RiskMetrics)

Volatility estimated using:

σ²*t = λ σ²*{t-1} + (1 − λ) r²\_{t-1}

Captures volatility clustering and faster regime shifts.

---

### 4️⃣ Covariance-Matrix VaR

Portfolio volatility computed via:

σ_p = sqrt(wᵀ Σ w)

Incorporates cross-asset correlations and structural risk concentration.

---

### 5️⃣ Expected Shortfall (ES / CVaR)

ES*α = E[r | r ≤ VaR*α]

Captures tail severity beyond the VaR threshold.

---

### 6️⃣ VaR Backtesting & Validation

- Exception rate tracking
- Kupiec (1995) Unconditional Coverage test
- PASS / FAIL statistical interpretation

---

### 7️⃣ Stress Testing

Includes:

- Worst historical portfolio days
- Empirical quantile shocks
- Cross-asset quantile scenarios
- VaR vs ES comparison dashboard

---

### 8️⃣ Component (Marginal) VaR

Parametric decomposition of total portfolio VaR into asset contributions:

Component VaR*i = w_i · z*α · ((Σ w)\_i / σ_p)

Enables identification of concentration risk drivers.

---

## 📂 Notebook Structure

| Notebook                | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| `07_var_analysis.ipynb` | Full VaR suite (Historical, Parametric, EWMA, Cov-Matrix, ES) |
| `08_var_analysis.ipynb` | Stress testing and scenario analysis                          |
| `09_var_analysis.ipynb` | Component VaR and risk contribution reporting                 |

---

## 📈 Example Outputs

- Consolidated 95% and 99% VaR overlay charts
- VaR breach tables (CSV export)
- Kupiec test summary table
- Stress scenario results
- Component VaR bar chart

Figures saved to:
outputs/figures/

Exception reports saved to:
outputs/

---

## ⚙️ Tech Stack

- Python 3.x
- pandas
- numpy
- scipy
- matplotlib
- yfinance
- Jupyter Notebook (VS Code + WSL)

---

## 🎯 Purpose

This project demonstrates:

- Market risk monitoring workflow
- Statistical model validation
- Portfolio risk aggregation
- Tail risk analysis
- Pre-trade risk impact assessment
- Risk decomposition and concentration analysis

Designed to simulate core responsibilities of a bank Market Risk team overseeing multi-asset trading activity.

---

## 🚀 Possible Extensions

- EWMA covariance matrix
- Student-t parametric VaR
- Conditional Coverage (Christoffersen) test
- Factor-based VaR
- GARCH volatility modeling
- Monte Carlo simulation VaR
