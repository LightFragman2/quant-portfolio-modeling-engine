# Quant Portfolio Modeling Engine

[![Python Tests](https://github.com/LightFragman2/quant-portfolio-modeling-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/LightFragman2/quant-portfolio-modeling-engine/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-red)](https://quant-portfolio-modeling-engine.streamlit.app/)

A quantitative portfolio analysis, optimization, and backtesting engine built in Python from first principles.

The project retrieves real market data, calculates portfolio statistics, simulates thousands of portfolios, performs constrained optimization, constructs the efficient frontier, and evaluates strategies using walk-forward out-of-sample backtesting.

A major goal of the project is to understand and implement the mathematics behind portfolio modeling instead of treating financial libraries as a black box.

---

## Live Demo

### [Launch the Quant Portfolio Modeling Engine](https://quant-portfolio-modeling-engine.streamlit.app/)

The Streamlit application allows users to:

- Enter their own U.S. stock or ETF tickers
- Choose a benchmark
- Select a historical period
- Configure portfolio constraints
- Run Monte Carlo simulations
- Optimize portfolios
- View the efficient frontier
- Run walk-forward backtests
- Compare portfolio strategies

No local Python installation is required to use the deployed application.

---

# Features

## Market Data

- Historical daily market prices
- Latest market prices
- User-defined U.S. stock and ETF tickers
- Custom benchmark selection
- Custom historical date ranges
- Alpaca Market Data API integration
- Historical data export to CSV

## Quantitative Statistics

- Simple returns
- Arithmetic mean return
- Variance
- Standard deviation / volatility
- Covariance
- Correlation
- Annualized arithmetic return
- Annualized volatility
- Cumulative return
- Compounded annualized return
- Sharpe ratio
- Beta
- Simple linear regression

## Portfolio Analysis

- Multi-asset expected return
- Multi-asset portfolio variance
- Portfolio volatility
- Diversification analysis
- Correlation analysis
- Equal-weight portfolios
- Long-only portfolios
- Maximum position-size constraints

## Monte Carlo Simulation

The engine can simulate thousands of possible portfolio allocations and calculate each portfolio's:

- Expected annual return
- Annualized volatility
- Sharpe ratio
- Asset weights

The resulting portfolio cloud provides a visual representation of historical risk-return combinations.

## Portfolio Optimization

The engine calculates:

- Maximum-Sharpe portfolio
- Minimum-volatility portfolio
- Target-return portfolios
- Efficient frontier
- Portfolios subject to maximum position constraints

Monte Carlo results can be compared directly against mathematically optimized portfolios.

## Walk-Forward Backtesting

The engine supports rolling out-of-sample backtesting using:

- Trailing historical training windows
- Portfolio re-optimization
- Periodic rebalancing
- Natural portfolio-weight drift
- Transaction costs
- Portfolio turnover
- Position constraints
- Benchmark comparison
- Growth of $1
- Maximum drawdown
- Annualized return
- Annualized volatility
- Sharpe ratio

The backtesting process is designed to avoid look-ahead bias by ensuring that each allocation uses only information available before its testing period.

---

# Web Application

The project includes an interactive Streamlit dashboard.

Users can configure:

- Portfolio tickers
- Benchmark
- Historical start date
- Historical end date
- Risk-free rate
- Number of Monte Carlo simulations
- Maximum asset weight
- Backtest training window
- Rebalance frequency
- Transaction costs

The application includes:

- Overview
- Correlation
- Monte Carlo
- Optimization
- Efficient Frontier
- Backtesting
- Historical Market Data

### [Open the Live Application](https://quant-portfolio-modeling-engine.streamlit.app/)

---

# Project Philosophy

This project was intentionally built from first principles.

Instead of immediately using high-level financial functions to perform every calculation, important quantitative operations are implemented directly from their mathematical definitions whenever practical.

The goal is to understand the complete chain:

```text
Prices
  ↓
Returns
  ↓
Expected Return
  ↓
Variance / Volatility
  ↓
Covariance
  ↓
Correlation
  ↓
Portfolio Risk
  ↓
Diversification
  ↓
Sharpe Ratio
  ↓
Beta / Regression
  ↓
Annualization
  ↓
Monte Carlo Simulation
  ↓
Portfolio Optimization
  ↓
Efficient Frontier
  ↓
Walk-Forward Backtesting
```

Scientific libraries are still used where appropriate for numerical optimization, market-data handling, visualization, and supporting infrastructure.

---

# Mathematics

## Simple Return

For an asset moving from price $P_{t-1}$ to price $P_t$:

```math
R_t = \frac{P_t - P_{t-1}}{P_{t-1}}
```

This measures the percentage change in an asset's price from one period to the next.

---

## Arithmetic Mean Return

For returns $R_1, R_2, \ldots, R_n$:

```math
\bar{R}
=
\frac{1}{n}
\sum_{i=1}^{n} R_i
```

The arithmetic mean represents the average return across the observed periods.

---

## Variance

Population variance:

```math
\sigma^2
=
\frac{1}{n}
\sum_{i=1}^{n}
(R_i-\bar{R})^2
```

Sample variance:

```math
s^2
=
\frac{1}{n-1}
\sum_{i=1}^{n}
(R_i-\bar{R})^2
```

The sample form is useful when historical market observations are treated as a sample used to estimate underlying risk.

---

## Standard Deviation / Volatility

```math
\sigma
=
\sqrt{\sigma^2}
```

Standard deviation measures the dispersion of returns and is used as a measure of historical volatility.

---

# Covariance

For assets A and B:

```math
\mathrm{Cov}(A,B)
=
\frac{1}{n}
\sum_{i=1}^{n}
(R_{A,i}-\bar{R}_A)
(R_{B,i}-\bar{R}_B)
```

Covariance measures whether the returns of two assets tend to move together.

Positive covariance indicates that they tend to move in the same direction.

Negative covariance indicates that they tend to move in opposite directions.

---

# Correlation

Correlation standardizes covariance:

```math
\rho_{A,B}
=
\frac{
\mathrm{Cov}(A,B)
}{
\sigma_A \sigma_B
}
```

Correlation is bounded by:

```math
-1
\le
\rho_{A,B}
\le
1
```

Lower correlation between portfolio assets can create diversification benefits.

---

# Portfolio Mathematics

## Portfolio Expected Return

For a portfolio containing $n$ assets:

```math
E[R_p]
=
\sum_{i=1}^{n}
w_i E[R_i]
```

For a fully invested portfolio:

```math
\sum_{i=1}^{n} w_i = 1
```

where:

- $w_i$ is the weight of asset $i$
- $E[R_i]$ is the expected return of asset $i$

---

## Two-Asset Portfolio Variance

For two assets A and B:

```math
\sigma_p^2
=
w_A^2\sigma_A^2
+
w_B^2\sigma_B^2
+
2w_Aw_B\mathrm{Cov}(A,B)
```

Using correlation:

```math
\sigma_p^2
=
w_A^2\sigma_A^2
+
w_B^2\sigma_B^2
+
2w_Aw_B\rho_{A,B}\sigma_A\sigma_B
```

The covariance term is the mathematical reason diversification can reduce portfolio risk.

---

## Multi-Asset Portfolio Variance

For multiple assets:

```math
\sigma_p^2
=
\sum_i w_i^2\sigma_i^2
+
2
\sum_{i<j}
w_iw_j\mathrm{Cov}(i,j)
```

The compact matrix form is:

```math
\sigma_p^2
=
w^T \Sigma w
```

where:

- $w$ is the portfolio weight vector
- $w^T$ is the transpose of the weight vector
- $\Sigma$ is the covariance matrix

The project implements the expanded variance and covariance calculation directly.

---

## Portfolio Volatility

Portfolio volatility is the square root of portfolio variance:

```math
\sigma_p
=
\sqrt{\sigma_p^2}
```

---

# Sharpe Ratio

The Sharpe ratio measures excess return per unit of volatility.

```math
S
=
\frac{
R_p-R_f
}{
\sigma_p
}
```

where:

- $R_p$ = portfolio return
- $R_f$ = risk-free rate
- $\sigma_p$ = portfolio volatility

Higher Sharpe ratios indicate more historical excess return relative to the amount of volatility taken.

---

# Beta

Beta measures an asset's historical sensitivity to movements in a market benchmark.

```math
\beta
=
\frac{
\mathrm{Cov}(R_s,R_m)
}{
\mathrm{Var}(R_m)
}
```

where:

- $R_s$ = stock returns
- $R_m$ = market returns

A beta greater than 1 indicates greater historical sensitivity to market movements.

A beta below 1 indicates lower historical sensitivity.

---

# Simple Linear Regression

The engine implements the model:

```math
Y
=
\alpha
+
\beta X
+
\epsilon
```

For stock returns relative to market returns:

```math
R_s
=
\alpha
+
\beta R_m
+
\epsilon
```

The regression slope is:

```math
\beta
=
\frac{
\mathrm{Cov}(X,Y)
}{
\mathrm{Var}(X)
}
```

The intercept is:

```math
\alpha
=
\bar{Y}
-
\beta\bar{X}
```

---

# Annualization

The engine assumes approximately 252 trading days per year for daily market data.

## Arithmetic Annualized Return

```math
R_{\mathrm{annual}}
\approx
252\bar{R}_{\mathrm{daily}}
```

---

## Annualized Volatility

```math
\sigma_{\mathrm{annual}}
=
\sigma_{\mathrm{daily}}
\sqrt{252}
```

Variance scales approximately with time, causing standard deviation to scale with the square root of time.

---

## Cumulative Return

Returns must be compounded through time.

```math
R_{\mathrm{cumulative}}
=
\left[
\prod_{t=1}^{n}
(1+R_t)
\right]
-1
```

---

## Compounded Annualized Return

```math
R_{\mathrm{annualized}}
=
\left(
\prod_{t=1}^{n}
(1+R_t)
\right)^{252/n}
-1
```

---

# Monte Carlo Portfolio Simulation

Monte Carlo simulation explores many possible portfolio allocations.

For a long-only portfolio:

```math
w_i
\ge
0
```

For a fully invested portfolio:

```math
\sum_i w_i
=
1
```

For every simulated portfolio, the engine calculates expected return:

```math
E[R_p]
```

volatility:

```math
\sigma_p
```

and Sharpe ratio:

```math
S_p
```

The simulation creates a cloud of possible historical risk-return combinations.

Monte Carlo simulation does not mathematically prove that a portfolio is optimal. It samples the available portfolio space.

The numerical optimization engine separately attempts to solve directly for optimal portfolios.

---

# Portfolio Optimization

The project uses constrained numerical optimization through SciPy while retaining its own implementations of portfolio return, variance, volatility, and Sharpe ratio.

## Minimum-Volatility Portfolio

The minimum-volatility objective is:

```math
\min_w \sigma_p^2
```

subject to:

```math
\sum_i w_i
=
1
```

and for a long-only portfolio:

```math
0
\le
w_i
\le
w_{\max}
```

---

## Maximum-Sharpe Portfolio

The objective is:

```math
\max_w
\left(
\frac{
E[R_p]-R_f
}{
\sigma_p
}
\right)
```

Because numerical optimization software generally minimizes objective functions, the implementation minimizes the negative Sharpe ratio:

```math
\min_w (-S)
```

This is mathematically equivalent to maximizing $S$.

---

# Efficient Frontier

For a series of target returns $R^*$, the engine solves:

```math
\min_w
\sigma_p^2
```

subject to:

```math
E[R_p]
=
R^*
```

```math
\sum_i w_i
=
1
```

and:

```math
0
\le
w_i
\le
w_{\max}
```

The resulting minimum-risk portfolios form the efficient frontier.

A portfolio below the efficient frontier is inefficient because another available portfolio can provide either:

- More expected return for the same risk
- Less risk for the same expected return

---

# Walk-Forward Backtesting

A major focus of the project is preventing look-ahead bias.

The backtest follows this process:

```text
Historical Training Window
          ↓
Estimate Historical Statistics
          ↓
Optimize Portfolio
          ↓
Lock Target Weights
          ↓
Future Out-of-Sample Test Period
          ↓
Observe Actual Returns
          ↓
Allow Portfolio Weights to Drift
          ↓
Rebalance Using Newly Available Data
          ↓
Repeat
```

A common configuration is:

```text
Training window:        504 trading days
Rebalance frequency:     63 trading days
```

This approximates:

- Two years of trailing historical data
- Quarterly portfolio rebalancing

At every rebalance date, only information available before that date is used to calculate the portfolio for the next testing period.

---

# Natural Portfolio Weight Drift

Portfolio weights do not remain artificially constant between rebalances.

Suppose a portfolio begins with:

```text
Asset A: 50%
Asset B: 50%
```

If Asset A rises significantly more than Asset B, Asset A naturally becomes a larger percentage of the portfolio.

The engine allows this weight drift to occur until the next rebalance.

---

# Portfolio Turnover

One-way portfolio turnover is:

```math
T
=
\frac{1}{2}
\sum_i
\left|
w_{i,\mathrm{new}}
-
w_{i,\mathrm{old}}
\right|
```

where $T$ represents turnover.

For example, moving from:

```text
Asset A: 50%
Asset B: 50%
```

to:

```text
Asset A: 100%
Asset B: 0%
```

results in:

```math
T
=
0.50
```

or 50% turnover.

---

# Transaction Costs

Let:

- $T$ = portfolio turnover
- $b$ = transaction cost in basis points
- $C$ = transaction-cost rate

Then:

```math
C
=
T
\left(
\frac{b}{10000}
\right)
```

This allows the backtest to recognize that portfolio rebalancing is not free.

The current transaction-cost model is intentionally simplified and does not attempt to reproduce every component of real-world trade execution.

---

# Maximum Position Constraints

The optimizer supports limits on the amount invested in any individual asset.

For example, a 40% concentration cap imposes:

```math
0
\le
w_i
\le
0.40
```

This makes it possible to compare an unrestricted optimizer against a more diversified portfolio.

For example:

```text
Maximum Sharpe

vs.

Maximum Sharpe with 40% Position Cap
```

---

# Maximum Drawdown

Let $V_t$ be portfolio value at time $t$.

The running peak is:

```math
P_t
=
\max_{s \le t}
V_s
```

Drawdown is:

```math
D_t
=
\frac{V_t}{P_t}
-1
```

Maximum drawdown is:

```math
D_{\max}
=
\min_t D_t
```

It represents the largest peak-to-trough decline observed during the backtest.

---

# Strategy Comparison

The backtesting engine can compare:

- Maximum Sharpe
- Maximum Sharpe with a concentration cap
- Minimum Volatility
- Equal Weight
- Market benchmark

Strategies are evaluated using:

| Metric | Purpose |
|---|---|
| CAGR | Long-run compounded growth |
| Volatility | Historical variability of returns |
| Sharpe Ratio | Excess return per unit of volatility |
| Maximum Drawdown | Largest peak-to-trough decline |
| Turnover | Amount of portfolio reallocation |
| Transaction Costs | Estimated cost of rebalancing |

---

# Architecture

```text
                    Streamlit Web Application
                              |
                              v
                       User Parameters
                              |
              +---------------+---------------+
              |                               |
              v                               v
        Asset Universe                  Model Settings
              |                               |
              +---------------+---------------+
                              |
                              v
                      Alpaca Market Data
                              |
                              v
                    Historical Price Data
                              |
                              v
                  First-Principles Engine
                              |
       +----------------------+----------------------+
       |                      |                      |
       v                      v                      v
   Statistics           Portfolio Math        Regression / Beta
       |                      |
       +-----------+----------+
                   |
                   v
             Monte Carlo
                   |
                   v
             Optimization
                   |
                   v
          Efficient Frontier
                   |
                   v
        Walk-Forward Backtesting
                   |
                   v
       Interactive Streamlit Results
```

---

# Project Structure

```text
quant-portfolio-modeling-engine/
│
├── src/
│   ├── __init__.py
│   ├── annualization.py
│   ├── backtesting.py
│   ├── backtest_visualization.py
│   ├── input_validation.py
│   ├── market_data.py
│   ├── monte_carlo.py
│   ├── optimization.py
│   ├── portfolio.py
│   ├── regression.py
│   ├── returns.py
│   ├── statistics.py
│   └── visualization.py
│
├── scripts/
│   ├── __init__.py
│   └── run_robustness_backtest.py
│
├── tests/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── streamlit_app.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# Technology Stack

## Language

- Python

## Quantitative and Scientific Computing

- SciPy
- NumPy
- pandas

Important financial calculations are implemented manually where practical.

Scientific libraries are primarily used for:

- Numerical optimization
- Data handling
- Infrastructure
- Visualization
- Verification

## Visualization

- Altair
- Matplotlib
- Streamlit

## Market Data

- Alpaca Market Data API

## Testing

- pytest
- GitHub Actions

## Deployment

- Streamlit Community Cloud

---

# Running Locally

## 1. Clone the Repository

```bash
git clone https://github.com/LightFragman2/quant-portfolio-modeling-engine.git
```

Move into the project:

```bash
cd quant-portfolio-modeling-engine
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## 4. Configure Alpaca Credentials

Create a `.env` file in the project root:

```text
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
```

Do not commit this file.

The repository contains `.env.example` as a safe template.

## 5. Launch the Application

```bash
python -m streamlit run streamlit_app.py
```

The local application will normally be available at:

```text
http://localhost:8501
```

---

# Running the Test Suite

Run:

```bash
python -m pytest
```

Tests cover major parts of the engine including:

- Returns
- Arithmetic mean
- Variance
- Standard deviation
- Covariance
- Correlation
- Portfolio expected return
- Portfolio variance
- Portfolio volatility
- Sharpe ratio
- Beta
- Regression
- Annualization
- Cumulative returns
- Monte Carlo simulation
- Random portfolio weight generation
- Maximum position constraints
- Portfolio optimization
- Efficient frontier
- Walk-forward backtesting
- Maximum drawdown
- Natural portfolio-weight drift
- Portfolio turnover
- Transaction costs
- Ticker input validation

---

# Continuous Integration

GitHub Actions automatically runs the project's test suite whenever code is pushed to the `main` branch or included in a pull request.

```text
Code Push
   ↓
GitHub Actions
   ↓
Create Python Environment
   ↓
Install Dependencies
   ↓
Run pytest
   ↓
Pass / Fail
```

This helps prevent new changes from silently breaking previously working functionality.

---

# Security

Alpaca API credentials are never committed to the repository.

Local development uses:

```text
.env
```

The deployed Streamlit application stores credentials using Streamlit's secrets system.

Secret-containing files are excluded through `.gitignore`.

---

# Important Limitations

This project is an educational and research-oriented portfolio modeling engine.

Historical results should not be interpreted as predictions of future performance.

Important limitations include:

- Historical performance does not guarantee future performance.
- Expected returns are estimated from historical observations.
- Historical covariance and correlation may not persist.
- Portfolio optimization is highly sensitive to expected-return estimates.
- Optimizers can create unstable or concentrated portfolios.
- Asset selection may introduce selection or survivorship bias.
- Results depend heavily on the selected historical period.
- Transaction-cost modeling is simplified.
- Taxes are not modeled.
- Bid-ask spreads are not explicitly modeled.
- Market impact is not modeled.
- Liquidity constraints are not modeled.
- Historical risk-free rates are not yet dynamically incorporated.
- Free market-data feeds may differ from consolidated institutional-grade feeds.
- The current model primarily targets long-only U.S. stocks and ETFs.
- A successful historical backtest does not prove that a strategy will perform similarly in the future.

These limitations are part of the reason the project includes:

- Out-of-sample testing
- Benchmark comparisons
- Concentration constraints
- Transaction costs
- Turnover measurement
- Multiple competing portfolio strategies

---

# Roadmap

Potential future additions include:

- Sensitivity analysis
- Rolling returns
- Rolling volatility
- Rolling Sharpe ratio
- Drawdown visualizations
- Historical Treasury risk-free rates
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Alternative expected-return estimators
- Additional portfolio constraints
- Larger asset universes
- More benchmark comparisons
- Parameter sensitivity testing
- Transaction-cost sensitivity analysis
- CAPM analysis
- Alpha analysis
- R-squared analysis
- Factor models
- Fama-French factors
- Additional risk metrics

---

# Why I Built This

Many quantitative-finance projects can produce sophisticated-looking results while hiding most of the underlying mathematics behind libraries.

I wanted to build the opposite.

The goal was to understand how each stage connects to the next:

```text
Market Prices
     ↓
Returns
     ↓
Expected Return
     ↓
Variance / Volatility
     ↓
Covariance
     ↓
Correlation
     ↓
Portfolio Risk
     ↓
Diversification
     ↓
Sharpe Ratio
     ↓
Beta / Regression
     ↓
Annualization
     ↓
Monte Carlo Simulation
     ↓
Portfolio Optimization
     ↓
Efficient Frontier
     ↓
Walk-Forward Backtesting
```

The result is a project where both the Python implementation and the quantitative reasoning behind it can be explained.

---

# Live Application

## [Open the Quant Portfolio Modeling Engine](https://quant-portfolio-modeling-engine.streamlit.app/)

GitHub Repository:

## [LightFragman2/quant-portfolio-modeling-engine](https://github.com/LightFragman2/quant-portfolio-modeling-engine)

---

# Disclaimer

This project is for educational and research purposes only.

Nothing generated by this application should be interpreted as financial advice, an investment recommendation, or a guarantee of future performance.