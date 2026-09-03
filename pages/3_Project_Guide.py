import streamlit as st


st.set_page_config(
    page_title="Project Guide",
    page_icon="📘",
    layout="wide",
)


def main():
    st.title(
        "Project Guide"
    )

    st.caption(
        "How the Quant Portfolio Modeling Engine "
        "works and how to interpret its results."
    )

    st.info(
        "This application is an educational and "
        "research project. Results describe historical "
        "behavior and should not be interpreted as "
        "investment recommendations."
    )

    st.header(
        "What This Project Does"
    )

    st.markdown(
        """
The **Quant Portfolio Modeling Engine** is an
interactive quantitative-finance application for
studying portfolio risk, return, diversification,
optimization, and historical strategy behavior.

Users can select their own U.S. stocks or ETFs and
run the same quantitative pipeline on different
portfolios.
"""
    )

    st.markdown(
        """
### Analysis Pipeline

1. Retrieve historical market prices
2. Convert prices into returns
3. Calculate return and risk statistics
4. Measure covariance and correlation
5. Simulate thousands of portfolios
6. Optimize portfolio allocations
7. Construct the efficient frontier
8. Run walk-forward out-of-sample backtests
9. Analyze parameter sensitivity
10. Measure downside and rolling risk
"""
    )

    st.divider()

    st.header(
        "Core Mathematics"
    )

    left, right = (
        st.columns(2)
    )

    with left:
        st.subheader(
            "Simple Return"
        )

        st.latex(
            r"""
            R_t =
            \frac{
                P_t-P_{t-1}
            }{
                P_{t-1}
            }
            """
        )

        st.subheader(
            "Expected Portfolio Return"
        )

        st.latex(
            r"""
            E[R_p]
            =
            \sum_{i=1}^{n}
            w_iE[R_i]
            """
        )

        st.subheader(
            "Portfolio Variance"
        )

        st.latex(
            r"""
            \sigma_p^2
            =
            w^T\Sigma w
            """
        )

    with right:
        st.subheader(
            "Portfolio Volatility"
        )

        st.latex(
            r"""
            \sigma_p
            =
            \sqrt{
                \sigma_p^2
            }
            """
        )

        st.subheader(
            "Sharpe Ratio"
        )

        st.latex(
            r"""
            S
            =
            \frac{
                R_p-R_f
            }{
                \sigma_p
            }
            """
        )

        st.subheader(
            "Beta"
        )

        st.latex(
            r"""
            \beta
            =
            \frac{
                \mathrm{Cov}(R_s,R_m)
            }{
                \mathrm{Var}(R_m)
            }
            """
        )

    st.caption(
        "Important portfolio calculations are "
        "implemented directly in Python rather than "
        "being completely delegated to financial "
        "libraries."
    )

    st.divider()

    st.header(
        "Monte Carlo Simulation"
    )

    st.markdown(
        """
Monte Carlo simulation generates many possible
portfolio allocations.

Each simulated portfolio has weights satisfying:
"""
    )

    st.latex(
        r"""
        \sum_{i=1}^{n}w_i=1
        """
    )

    st.markdown(
        """
The engine calculates the expected return,
volatility, and Sharpe ratio of each portfolio.

The resulting cloud shows the range of historical
risk-return combinations available from the
selected assets.

Monte Carlo simulation **samples** the portfolio
space. It does not prove that the best sampled
portfolio is mathematically optimal.
"""
    )

    st.divider()

    st.header(
        "Portfolio Optimization"
    )

    optimization_one, optimization_two = (
        st.columns(2)
    )

    with optimization_one:
        st.subheader(
            "Minimum Volatility"
        )

        st.latex(
            r"""
            \min_w
            \sigma_p^2
            """
        )

    with optimization_two:
        st.subheader(
            "Maximum Sharpe"
        )

        st.latex(
            r"""
            \max_w
            \frac{
                E[R_p]-R_f
            }{
                \sigma_p
            }
            """
        )

    st.markdown(
        """
The application supports long-only portfolios and
optional maximum-position constraints.

For example, a 40% position cap requires:
"""
    )

    st.latex(
        r"""
        0
        \le
        w_i
        \le
        0.40
        """
    )

    st.divider()

    st.header(
        "Efficient Frontier"
    )

    st.markdown(
        """
The efficient frontier contains portfolios that
offer the minimum achievable volatility for a given
target expected return.

A portfolio below the frontier is inefficient
because another available portfolio can provide:

- more expected return for approximately the same risk, or
- less risk for approximately the same expected return.
"""
    )

    st.divider()

    st.header(
        "Walk-Forward Backtesting"
    )

    st.markdown(
        """
The backtesting engine is designed to avoid one of
the most dangerous mistakes in strategy research:

**look-ahead bias.**

The model does not optimize using the entire
historical dataset and then pretend that portfolio
was known in the past.
"""
    )

    st.code(
        """
Historical Training Window
          ↓
Optimize Portfolio
          ↓
Lock Weights
          ↓
Future Test Period
          ↓
Observe Real Returns
          ↓
Rebalance
          ↓
Repeat
""",
        language="text",
    )

    st.markdown(
        """
A typical configuration uses:

- **504 trading days** of training data
- **63 trading days** between rebalances

This roughly corresponds to a two-year trailing
window with quarterly rebalancing.
"""
    )

    st.divider()

    st.header(
        "Portfolio Turnover"
    )

    st.latex(
        r"""
        T
        =
        \frac{1}{2}
        \sum_i
        \left|
        w_{i,\mathrm{new}}
        -
        w_{i,\mathrm{old}}
        \right|
        """
    )

    st.markdown(
        """
Turnover measures how much of the portfolio must be
traded when moving from the current allocation to a
new target allocation.

Transaction costs are then applied using the
selected basis-point assumption.
"""
    )

    st.divider()

    st.header(
        "Sensitivity Analysis"
    )

    st.markdown(
        """
A historical strategy can appear impressive simply
because one exact set of assumptions happened to
work well.

Sensitivity analysis tests this by changing one
assumption at a time.

The application can test changes in:

- Maximum asset weight
- Training-window length
- Rebalance frequency
- Transaction costs

A more robust strategy should generally change
gradually as reasonable assumptions change.

A strategy that collapses after a tiny parameter
change may be overfit or fragile.
"""
    )

    st.divider()

    st.header(
        "Risk Analytics"
    )

    st.subheader(
        "Value at Risk"
    )

    st.markdown(
        """
Historical Value at Risk estimates a loss threshold
from the historical return distribution.

For example, a 95% daily VaR of 3% means that
approximately 5% of historical daily observations
were worse than a 3% loss threshold.
"""
    )

    st.subheader(
        "Conditional Value at Risk"
    )

    st.markdown(
        """
CVaR, also called Expected Shortfall, goes beyond
VaR and asks:

**When returns enter the bad tail, how bad have the
losses been on average?**
"""
    )

    st.subheader(
        "Maximum Drawdown"
    )

    st.latex(
        r"""
        D_t
        =
        \frac{
            V_t
        }{
            P_t
        }
        -1
        """
    )

    st.markdown(
        """
where the running peak is:
"""
    )

    st.latex(
        r"""
        P_t
        =
        \max_{s \le t}
        V_s
        """
    )

    st.markdown(
        """
Maximum drawdown measures the largest historical
peak-to-trough decline.
"""
    )

    st.divider()

    st.header(
        "How to Interpret Results"
    )

    st.markdown(
        """
### High Return Is Not Automatically Better

A portfolio producing a larger return may also have:

- greater volatility
- deeper drawdowns
- greater concentration
- greater turnover
- larger tail losses

### A High Sharpe Ratio Is Not a Guarantee

Sharpe is based on historical estimates.

Future:

- expected returns
- volatility
- covariance
- correlations

can all differ from their historical values.

### Optimizers Can Become Concentrated

Maximum-Sharpe optimization can place a large
weight in an asset whose historical return was
especially strong.

That is why the engine includes maximum-position
constraints and sensitivity analysis.

### Backtests Can Still Be Misleading

Even when look-ahead bias is avoided, historical
results can be affected by:

- asset-selection bias
- survivorship bias
- market regime
- parameter choices
- transaction-cost assumptions
- estimation error
"""
    )

    st.divider()

    st.header(
        "Technology"
    )

    technology_one, technology_two, technology_three = (
        st.columns(3)
    )

    with technology_one:
        st.subheader(
            "Quant Engine"
        )

        st.markdown(
            """
- Python
- NumPy
- SciPy
- pandas
- Custom portfolio mathematics
"""
        )

    with technology_two:
        st.subheader(
            "Application"
        )

        st.markdown(
            """
- Streamlit
- Altair
- Matplotlib
- Alpaca Market Data
"""
        )

    with technology_three:
        st.subheader(
            "Engineering"
        )

        st.markdown(
            """
- pytest
- Git
- GitHub
- GitHub Actions
- Streamlit Community Cloud
"""
        )

    st.divider()

    st.header(
        "Project Links"
    )

    st.link_button(
        "Live Portfolio Engine",
        (
            "https://quant-portfolio-modeling-engine."
            "streamlit.app/"
        ),
        use_container_width=True,
    )

    st.link_button(
        "GitHub Repository",
        (
            "https://github.com/"
            "LightFragman2/"
            "quant-portfolio-modeling-engine"
        ),
        use_container_width=True,
    )

    st.divider()

    st.caption(
        "Educational and research use only. "
        "Nothing produced by this application "
        "constitutes financial advice."
    )


if __name__ == "__main__":
    main()