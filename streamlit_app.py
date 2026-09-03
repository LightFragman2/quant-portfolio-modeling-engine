from datetime import (
    date,
    datetime,
    time,
    timezone,
)

import altair as alt
import pandas as pd
import streamlit as st

from src.market_data import (
    get_historical_close_prices,
    get_latest_trade_prices,
)

from src.returns import (
    calculate_simple_returns,
)

from src.statistics import (
    arithmetic_mean,
    standard_deviation,
    correlation,
)

from src.annualization import (
    annualized_arithmetic_return,
    annualized_volatility,
    annualized_compounded_return,
    cumulative_return,
)

from src.input_validation import (
    normalize_symbol,
    parse_symbol_input,
    validate_date_range,
)

from src.monte_carlo import (
    simulate_portfolios,
    maximum_sharpe_portfolio as sampled_maximum_sharpe,
    minimum_volatility_portfolio as sampled_minimum_volatility,
)

from src.optimization import (
    maximum_sharpe_portfolio as optimized_maximum_sharpe,
    minimum_volatility_portfolio as optimized_minimum_volatility,
    efficient_frontier,
)

from src.backtesting import (
    rolling_backtest,
)


st.set_page_config(
    page_title=(
        "Quant Portfolio Modeling Engine"
    ),
    page_icon="📈",
    layout="wide",
)


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def load_historical_data(
    symbols,
    start_date,
    end_date,
):
    start_datetime = (
        datetime.combine(
            start_date,
            time.min,
            tzinfo=timezone.utc,
        )
    )

    end_datetime = (
        datetime.combine(
            end_date,
            time.max,
            tzinfo=timezone.utc,
        )
    )

    return (
        get_historical_close_prices(
            symbols=list(
                symbols
            ),
            start_date=(
                start_datetime
            ),
            end_date=(
                end_datetime
            ),
        )
    )


@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def load_latest_prices(
    symbols,
):
    return (
        get_latest_trade_prices(
            list(
                symbols
            )
        )
    )


@st.cache_data(
    show_spinner=False,
)
def run_monte_carlo_cached(
    asset_returns_tuple,
    number_of_portfolios,
    annual_risk_free_rate,
    max_weight,
):
    asset_returns = [
        list(
            returns
        )
        for returns
        in asset_returns_tuple
    ]

    return simulate_portfolios(
        asset_returns=(
            asset_returns
        ),
        number_of_portfolios=(
            number_of_portfolios
        ),
        annual_risk_free_rate=(
            annual_risk_free_rate
        ),
        periods_per_year=252,
        sample=True,
        seed=42,
        max_weight=max_weight,
    )


@st.cache_data(
    show_spinner=False,
)
def run_optimization_cached(
    asset_returns_tuple,
    annual_risk_free_rate,
    max_weight,
):
    asset_returns = [
        list(
            returns
        )
        for returns
        in asset_returns_tuple
    ]

    max_sharpe = (
        optimized_maximum_sharpe(
            asset_returns,
            annual_risk_free_rate=(
                annual_risk_free_rate
            ),
            periods_per_year=252,
            sample=True,
            max_weight=max_weight,
        )
    )

    min_volatility = (
        optimized_minimum_volatility(
            asset_returns,
            annual_risk_free_rate=(
                annual_risk_free_rate
            ),
            periods_per_year=252,
            sample=True,
            max_weight=max_weight,
        )
    )

    frontier = (
        efficient_frontier(
            asset_returns,
            annual_risk_free_rate=(
                annual_risk_free_rate
            ),
            periods_per_year=252,
            sample=True,
            number_of_points=60,
            max_weight=max_weight,
        )
    )

    return {
        "max_sharpe": (
            max_sharpe
        ),
        "min_volatility": (
            min_volatility
        ),
        "frontier": (
            frontier
        ),
    }


@st.cache_data(
    show_spinner=False,
)
def run_backtest_cached(
    asset_returns_tuple,
    benchmark_returns_tuple,
    dates_tuple,
    strategy,
    train_window,
    rebalance_frequency,
    annual_risk_free_rate,
    max_weight,
    transaction_cost_bps,
):
    asset_returns = [
        list(
            returns
        )
        for returns
        in asset_returns_tuple
    ]

    benchmark_returns = list(
        benchmark_returns_tuple
    )

    dates = list(
        dates_tuple
    )

    return rolling_backtest(
        asset_returns=(
            asset_returns
        ),
        benchmark_returns=(
            benchmark_returns
        ),
        dates=dates,
        strategy=strategy,
        train_window=(
            train_window
        ),
        rebalance_frequency=(
            rebalance_frequency
        ),
        annual_risk_free_rate=(
            annual_risk_free_rate
        ),
        periods_per_year=252,
        sample=True,
        max_weight=max_weight,
        transaction_cost_bps=(
            transaction_cost_bps
        ),
    )


def build_price_dataframe(
    historical_prices,
    dates,
):
    dataframe = (
        pd.DataFrame(
            historical_prices,
            index=dates,
        )
    )

    dataframe.index.name = (
        "Date"
    )

    return dataframe


def build_returns_by_symbol(
    symbols,
    historical_prices,
):
    results = {}

    for symbol in symbols:
        results[
            symbol
        ] = (
            calculate_simple_returns(
                historical_prices[
                    symbol
                ]
            )
        )

    return results


def build_asset_statistics(
    asset_symbols,
    historical_prices,
    latest_prices,
):
    rows = []

    for symbol in asset_symbols:
        prices = (
            historical_prices[
                symbol
            ]
        )

        returns = (
            calculate_simple_returns(
                prices
            )
        )

        average_daily_return = (
            arithmetic_mean(
                returns
            )
        )

        daily_volatility = (
            standard_deviation(
                returns,
                sample=True,
            )
        )

        rows.append(
            {
                "Ticker": symbol,
                "Latest Price": (
                    latest_prices[
                        symbol
                    ][
                        "price"
                    ]
                ),
                "Annual Return": (
                    annualized_arithmetic_return(
                        average_daily_return
                    )
                ),
                "Compounded Annual Return": (
                    annualized_compounded_return(
                        returns
                    )
                ),
                "Annual Volatility": (
                    annualized_volatility(
                        daily_volatility
                    )
                ),
                "Cumulative Return": (
                    cumulative_return(
                        returns
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def format_asset_statistics(
    dataframe,
):
    formatted = (
        dataframe.copy()
    )

    formatted[
        "Latest Price"
    ] = formatted[
        "Latest Price"
    ].map(
        lambda value: (
            f"${value:,.2f}"
        )
    )

    percentage_columns = [
        "Annual Return",
        "Compounded Annual Return",
        "Annual Volatility",
        "Cumulative Return",
    ]

    for column in (
        percentage_columns
    ):
        formatted[
            column
        ] = formatted[
            column
        ].map(
            lambda value: (
                f"{value:.2%}"
            )
        )

    return formatted


def build_correlation_matrix(
    asset_symbols,
    returns_by_symbol,
):
    matrix = []

    for symbol_a in (
        asset_symbols
    ):
        row = []

        for symbol_b in (
            asset_symbols
        ):
            value = correlation(
                returns_by_symbol[
                    symbol_a
                ],
                returns_by_symbol[
                    symbol_b
                ],
                sample=True,
            )

            row.append(
                value
            )

        matrix.append(
            row
        )

    return pd.DataFrame(
        matrix,
        index=asset_symbols,
        columns=asset_symbols,
    )


def build_portfolio_weight_table(
    portfolio,
    asset_symbols,
):
    return pd.DataFrame(
        {
            "Ticker": (
                asset_symbols
            ),
            "Weight": [
                f"{weight:.2%}"
                for weight
                in portfolio[
                    "weights"
                ]
            ],
        }
    )


def display_portfolio_summary(
    portfolio,
    asset_symbols,
):
    metric_one, metric_two, metric_three = (
        st.columns(3)
    )

    metric_one.metric(
        "Expected Return",
        f"{portfolio['annual_return']:.2%}",
    )

    metric_two.metric(
        "Volatility",
        f"{portfolio['annual_volatility']:.2%}",
    )

    metric_three.metric(
        "Sharpe Ratio",
        f"{portfolio['sharpe_ratio']:.3f}",
    )

    st.dataframe(
        build_portfolio_weight_table(
            portfolio,
            asset_symbols,
        ),
        use_container_width=True,
        hide_index=True,
    )


def build_monte_carlo_dataframe(
    simulation_results,
    asset_symbols,
):
    rows = []

    for portfolio in (
        simulation_results
    ):
        weight_text = ", ".join(
            f"{symbol}: {weight:.1%}"
            for symbol, weight
            in zip(
                asset_symbols,
                portfolio[
                    "weights"
                ],
            )
        )

        rows.append(
            {
                "Volatility": (
                    portfolio[
                        "annual_volatility"
                    ]
                ),
                "Return": (
                    portfolio[
                        "annual_return"
                    ]
                ),
                "Sharpe": (
                    portfolio[
                        "sharpe_ratio"
                    ]
                ),
                "Weights": (
                    weight_text
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_portfolio_analysis_chart(
    simulation_results,
    frontier,
    max_sharpe,
    min_volatility,
    asset_symbols,
):
    monte_carlo_dataframe = (
        build_monte_carlo_dataframe(
            simulation_results,
            asset_symbols,
        )
    )

    frontier_dataframe = (
        pd.DataFrame(
            {
                "Volatility": [
                    portfolio[
                        "annual_volatility"
                    ]
                    for portfolio
                    in frontier
                ],
                "Return": [
                    portfolio[
                        "annual_return"
                    ]
                    for portfolio
                    in frontier
                ],
            }
        )
    )

    points_dataframe = (
        pd.DataFrame(
            [
                {
                    "Portfolio": (
                        "Maximum Sharpe"
                    ),
                    "Volatility": (
                        max_sharpe[
                            "annual_volatility"
                        ]
                    ),
                    "Return": (
                        max_sharpe[
                            "annual_return"
                        ]
                    ),
                },
                {
                    "Portfolio": (
                        "Minimum Volatility"
                    ),
                    "Volatility": (
                        min_volatility[
                            "annual_volatility"
                        ]
                    ),
                    "Return": (
                        min_volatility[
                            "annual_return"
                        ]
                    ),
                },
            ]
        )
    )

    cloud = (
        alt.Chart(
            monte_carlo_dataframe
        )
        .mark_circle(
            size=35,
            opacity=0.40,
        )
        .encode(
            x=alt.X(
                "Volatility:Q",
                title=(
                    "Annualized Volatility"
                ),
                axis=alt.Axis(
                    format=".0%"
                ),
            ),
            y=alt.Y(
                "Return:Q",
                title=(
                    "Annualized Expected Return"
                ),
                axis=alt.Axis(
                    format=".0%"
                ),
            ),
            color=alt.Color(
                "Sharpe:Q",
                title="Sharpe Ratio",
                scale=alt.Scale(
                    scheme="viridis"
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "Return:Q",
                    format=".2%",
                ),
                alt.Tooltip(
                    "Volatility:Q",
                    format=".2%",
                ),
                alt.Tooltip(
                    "Sharpe:Q",
                    format=".3f",
                ),
                "Weights:N",
            ],
        )
    )

    frontier_line = (
        alt.Chart(
            frontier_dataframe
        )
        .mark_line(
            size=4
        )
        .encode(
            x="Volatility:Q",
            y="Return:Q",
        )
    )

    optimized_points = (
        alt.Chart(
            points_dataframe
        )
        .mark_point(
            size=250,
            filled=True,
        )
        .encode(
            x="Volatility:Q",
            y="Return:Q",
            shape=alt.Shape(
                "Portfolio:N",
                title=(
                    "Optimized Portfolio"
                ),
            ),
            tooltip=[
                "Portfolio:N",
                alt.Tooltip(
                    "Return:Q",
                    format=".2%",
                ),
                alt.Tooltip(
                    "Volatility:Q",
                    format=".2%",
                ),
            ],
        )
    )

    return (
        alt.layer(
            cloud,
            frontier_line,
            optimized_points,
        )
        .properties(
            height=550
        )
        .interactive()
    )


def build_backtest_comparison_table(
    results_by_name,
    benchmark_name,
):
    rows = []

    for name, result in (
        results_by_name.items()
    ):
        metrics = (
            result[
                "portfolio_metrics"
            ]
        )

        rows.append(
            {
                "Strategy": name,
                "CAGR": (
                    metrics[
                        "annual_compounded_return"
                    ]
                ),
                "Volatility": (
                    metrics[
                        "annual_volatility"
                    ]
                ),
                "Sharpe": (
                    metrics[
                        "sharpe_ratio"
                    ]
                ),
                "Max Drawdown": (
                    metrics[
                        "max_drawdown"
                    ]
                ),
                "Turnover": (
                    result[
                        "total_turnover"
                    ]
                ),
            }
        )

    first_result = next(
        iter(
            results_by_name.values()
        )
    )

    benchmark = (
        first_result[
            "benchmark_metrics"
        ]
    )

    rows.append(
        {
            "Strategy": (
                benchmark_name
            ),
            "CAGR": (
                benchmark[
                    "annual_compounded_return"
                ]
            ),
            "Volatility": (
                benchmark[
                    "annual_volatility"
                ]
            ),
            "Sharpe": (
                benchmark[
                    "sharpe_ratio"
                ]
            ),
            "Max Drawdown": (
                benchmark[
                    "max_drawdown"
                ]
            ),
            "Turnover": None,
        }
    )

    dataframe = (
        pd.DataFrame(
            rows
        )
    )

    formatted = (
        dataframe.copy()
    )

    for column in [
        "CAGR",
        "Volatility",
        "Max Drawdown",
    ]:
        formatted[
            column
        ] = formatted[
            column
        ].map(
            lambda value: (
                f"{value:.2%}"
            )
        )

    formatted[
        "Sharpe"
    ] = formatted[
        "Sharpe"
    ].map(
        lambda value: (
            f"{value:.3f}"
        )
    )

    formatted[
        "Turnover"
    ] = formatted[
        "Turnover"
    ].map(
        lambda value: (
            "—"
            if pd.isna(
                value
            )
            else f"{value:.2f}x"
        )
    )

    return formatted


def build_backtest_growth_dataframe(
    results_by_name,
    benchmark_name,
):
    first_result = next(
        iter(
            results_by_name.values()
        )
    )

    dataframe = (
        pd.DataFrame(
            index=first_result[
                "dates"
            ]
        )
    )

    for name, result in (
        results_by_name.items()
    ):
        dataframe[
            name
        ] = result[
            "portfolio_growth"
        ]

    dataframe[
        benchmark_name
    ] = first_result[
        "benchmark_growth"
    ]

    dataframe.index.name = (
        "Date"
    )

    return dataframe


def build_rebalance_dataframe(
    result,
    asset_symbols,
):
    rows = []

    for rebalance in (
        result[
            "rebalances"
        ]
    ):
        row = {
            "Date": (
                rebalance[
                    "rebalance_date"
                ]
            ),
            "Turnover": (
                rebalance[
                    "turnover"
                ]
            ),
        }

        for symbol, weight in zip(
            asset_symbols,
            rebalance[
                "weights"
            ],
        ):
            row[
                symbol
            ] = weight

        rows.append(
            row
        )

    dataframe = (
        pd.DataFrame(
            rows
        )
    )

    formatted = (
        dataframe.copy()
    )

    formatted[
        "Date"
    ] = formatted[
        "Date"
    ].map(
        lambda value: (
            value.date()
            if hasattr(
                value,
                "date",
            )
            else value
        )
    )

    formatted[
        "Turnover"
    ] = formatted[
        "Turnover"
    ].map(
        lambda value: (
            f"{value:.2%}"
        )
    )

    for symbol in (
        asset_symbols
    ):
        formatted[
            symbol
        ] = formatted[
            symbol
        ].map(
            lambda value: (
                f"{value:.2%}"
            )
        )

    return formatted


def main():
    st.title(
        "Quant Portfolio Modeling Engine"
    )

    st.caption(
        "Real market data, portfolio optimization, "
        "Monte Carlo simulation, and walk-forward "
        "backtesting from first principles."
    )

    with st.sidebar:
        st.header(
            "Portfolio"
        )

        with st.form(
            "portfolio_settings"
        ):
            raw_assets = (
                st.text_area(
                    "Assets",
                    value=(
                        "AAPL, MSFT, NVDA, JPM"
                    ),
                    help=(
                        "Enter up to 10 U.S. "
                        "stock or ETF tickers."
                    ),
                )
            )

            raw_benchmark = (
                st.text_input(
                    "Benchmark",
                    value="SPY",
                )
            )

            start_date = (
                st.date_input(
                    "Historical start",
                    value=date(
                        2021,
                        1,
                        1,
                    ),
                    min_value=date(
                        2016,
                        1,
                        1,
                    ),
                    max_value=(
                        date.today()
                    ),
                )
            )

            end_date = (
                st.date_input(
                    "Historical end",
                    value=(
                        date.today()
                    ),
                    min_value=date(
                        2016,
                        1,
                        2,
                    ),
                    max_value=(
                        date.today()
                    ),
                )
            )

            st.divider()

            st.subheader(
                "Model"
            )

            risk_free_percent = (
                st.number_input(
                    "Risk-free rate (%)",
                    min_value=0.0,
                    max_value=25.0,
                    value=4.0,
                    step=0.25,
                )
            )

            max_weight_percent = (
                st.number_input(
                    "Maximum asset weight (%)",
                    min_value=1.0,
                    max_value=100.0,
                    value=100.0,
                    step=5.0,
                    help=(
                        "Set to 40 for a "
                        "40% concentration cap."
                    ),
                )
            )

            monte_carlo_count = (
                st.number_input(
                    "Monte Carlo portfolios",
                    min_value=1000,
                    max_value=25000,
                    value=10000,
                    step=1000,
                )
            )

            st.divider()

            st.subheader(
                "Backtest"
            )

            train_window = (
                st.number_input(
                    "Training window (trading days)",
                    min_value=126,
                    max_value=1260,
                    value=504,
                    step=63,
                )
            )

            rebalance_frequency = (
                st.number_input(
                    "Rebalance every (trading days)",
                    min_value=21,
                    max_value=252,
                    value=63,
                    step=21,
                )
            )

            transaction_cost_bps = (
                st.number_input(
                    "Transaction cost (bps)",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=1.0,
                )
            )

            submitted = (
                st.form_submit_button(
                    "Run Analysis",
                    type="primary",
                    use_container_width=True,
                )
            )

        st.caption(
            "Market data: Alpaca"
        )

        st.caption(
            "No trading orders are placed."
        )

    if submitted:
        try:
            asset_symbols = (
                parse_symbol_input(
                    raw_assets,
                    max_symbols=10,
                )
            )

            benchmark_symbol = (
                normalize_symbol(
                    raw_benchmark
                )
            )

            validate_date_range(
                start_date,
                end_date,
            )

            max_weight = (
                max_weight_percent
                / 100
            )

            minimum_feasible_weight = (
                1
                / len(
                    asset_symbols
                )
            )

            if (
                max_weight
                < minimum_feasible_weight
                - 1e-12
            ):
                raise ValueError(
                    "That maximum-weight cap is "
                    "not feasible. With "
                    f"{len(asset_symbols)} assets, "
                    "the cap must be at least "
                    f"{minimum_feasible_weight:.1%}."
                )

            st.session_state[
                "analysis_request"
            ] = {
                "asset_symbols": (
                    asset_symbols
                ),
                "benchmark_symbol": (
                    benchmark_symbol
                ),
                "start_date": (
                    start_date
                ),
                "end_date": (
                    end_date
                ),
                "risk_free_rate": (
                    risk_free_percent
                    / 100
                ),
                "max_weight": (
                    max_weight
                ),
                "monte_carlo_count": int(
                    monte_carlo_count
                ),
                "train_window": int(
                    train_window
                ),
                "rebalance_frequency": int(
                    rebalance_frequency
                ),
                "transaction_cost_bps": (
                    transaction_cost_bps
                ),
            }

        except ValueError as error:
            st.error(
                str(
                    error
                )
            )

            return

    if (
        "analysis_request"
        not in st.session_state
    ):
        st.info(
            "Configure your portfolio in the "
            "sidebar and click **Run Analysis**."
        )

        return

    request = (
        st.session_state[
            "analysis_request"
        ]
    )

    asset_symbols = (
        request[
            "asset_symbols"
        ]
    )

    benchmark_symbol = (
        request[
            "benchmark_symbol"
        ]
    )

    all_symbols = []

    for symbol in (
        asset_symbols
        + [
            benchmark_symbol
        ]
    ):
        if symbol not in (
            all_symbols
        ):
            all_symbols.append(
                symbol
            )

    with st.spinner(
        "Loading market data..."
    ):
        try:
            (
                historical_prices,
                dates,
            ) = load_historical_data(
                tuple(
                    all_symbols
                ),
                request[
                    "start_date"
                ],
                request[
                    "end_date"
                ],
            )

            latest_prices = (
                load_latest_prices(
                    tuple(
                        all_symbols
                    )
                )
            )

        except Exception as error:
            st.error(
                "Market-data request failed."
            )

            st.code(
                str(
                    error
                )
            )

            return

    if len(dates) < 3:
        st.error(
            "Not enough market data "
            "was returned."
        )

        return

    returns_by_symbol = (
        build_returns_by_symbol(
            all_symbols,
            historical_prices,
        )
    )

    return_dates = (
        dates[1:]
    )

    asset_returns = [
        returns_by_symbol[
            symbol
        ]
        for symbol in (
            asset_symbols
        )
    ]

    benchmark_returns = (
        returns_by_symbol[
            benchmark_symbol
        ]
    )

    immutable_asset_returns = tuple(
        tuple(
            values
        )
        for values in (
            asset_returns
        )
    )

    st.write(
        "**Assets:** "
        + ", ".join(
            asset_symbols
        )
    )

    st.write(
        f"**Benchmark:** "
        f"{benchmark_symbol}"
    )

    st.write(
        f"**Period:** "
        f"{dates[0].date()} → "
        f"{dates[-1].date()}"
    )

    (
        overview_tab,
        correlation_tab,
        monte_carlo_tab,
        optimization_tab,
        backtest_tab,
        data_tab,
    ) = st.tabs(
        [
            "Overview",
            "Correlation",
            "Monte Carlo",
            "Optimization",
            "Backtesting",
            "Data",
        ]
    )

    with overview_tab:
        statistics = (
            build_asset_statistics(
                asset_symbols,
                historical_prices,
                latest_prices,
            )
        )

        metric_one, metric_two, metric_three = (
            st.columns(3)
        )

        metric_one.metric(
            "Assets",
            len(
                asset_symbols
            ),
        )

        metric_two.metric(
            "Trading Days",
            f"{len(dates):,}",
        )

        metric_three.metric(
            "Benchmark",
            benchmark_symbol,
        )

        st.subheader(
            "Asset Statistics"
        )

        st.dataframe(
            format_asset_statistics(
                statistics
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "Normalized Growth"
        )

        price_dataframe = (
            build_price_dataframe(
                {
                    symbol: (
                        historical_prices[
                            symbol
                        ]
                    )
                    for symbol
                    in asset_symbols
                },
                dates,
            )
        )

        normalized_prices = (
            price_dataframe
            / price_dataframe.iloc[
                0
            ]
        )

        st.line_chart(
            normalized_prices,
            use_container_width=True,
        )

    with correlation_tab:
        st.subheader(
            "Return Correlation Matrix"
        )

        correlation_matrix = (
            build_correlation_matrix(
                asset_symbols,
                returns_by_symbol,
            )
        )

        st.dataframe(
            correlation_matrix.style.format(
                "{:.3f}"
            ),
            use_container_width=True,
        )

        st.caption(
            "These correlations are calculated "
            "by the project's own covariance and "
            "standard-deviation functions."
        )

    optimization_results = None
    simulation_results = None

    with monte_carlo_tab:
        with st.spinner(
            "Simulating portfolios..."
        ):
            try:
                simulation_results = (
                    run_monte_carlo_cached(
                        immutable_asset_returns,
                        request[
                            "monte_carlo_count"
                        ],
                        request[
                            "risk_free_rate"
                        ],
                        request[
                            "max_weight"
                        ],
                    )
                )

            except Exception as error:
                st.error(
                    str(
                        error
                    )
                )

        if simulation_results:
            sampled_max = (
                sampled_maximum_sharpe(
                    simulation_results
                )
            )

            sampled_min = (
                sampled_minimum_volatility(
                    simulation_results
                )
            )

            st.write(
                f"Simulated "
                f"{len(simulation_results):,} "
                f"portfolios."
            )

            left, right = (
                st.columns(2)
            )

            with left:
                st.subheader(
                    "Best Sampled Sharpe"
                )

                display_portfolio_summary(
                    sampled_max,
                    asset_symbols,
                )

            with right:
                st.subheader(
                    "Lowest Sampled Volatility"
                )

                display_portfolio_summary(
                    sampled_min,
                    asset_symbols,
                )

    with optimization_tab:
        with st.spinner(
            "Optimizing portfolios..."
        ):
            try:
                optimization_results = (
                    run_optimization_cached(
                        immutable_asset_returns,
                        request[
                            "risk_free_rate"
                        ],
                        request[
                            "max_weight"
                        ],
                    )
                )

            except Exception as error:
                st.error(
                    str(
                        error
                    )
                )

        if (
            optimization_results
            and simulation_results
        ):
            max_sharpe = (
                optimization_results[
                    "max_sharpe"
                ]
            )

            min_volatility = (
                optimization_results[
                    "min_volatility"
                ]
            )

            frontier = (
                optimization_results[
                    "frontier"
                ]
            )

            st.subheader(
                "Efficient Frontier"
            )

            chart = (
                build_portfolio_analysis_chart(
                    simulation_results,
                    frontier,
                    max_sharpe,
                    min_volatility,
                    asset_symbols,
                )
            )

            st.altair_chart(
                chart,
                use_container_width=True,
            )

            left, right = (
                st.columns(2)
            )

            with left:
                st.subheader(
                    "Maximum Sharpe"
                )

                display_portfolio_summary(
                    max_sharpe,
                    asset_symbols,
                )

            with right:
                st.subheader(
                    "Minimum Volatility"
                )

                display_portfolio_summary(
                    min_volatility,
                    asset_symbols,
                )

    with backtest_tab:
        if (
            len(
                benchmark_returns
            )
            <= request[
                "train_window"
            ]
        ):
            st.warning(
                "There is not enough historical "
                "data for the selected training "
                "window. Choose an earlier start "
                "date or a shorter training window."
            )

        else:
            st.subheader(
                "Out-of-Sample Strategy Comparison"
            )

            st.caption(
                "Every rebalance uses only information "
                "available before that test period."
            )

            strategy_configs = {
                "Max Sharpe": {
                    "strategy": (
                        "max_sharpe"
                    ),
                    "max_weight": (
                        1.0
                    ),
                },
                "Minimum Volatility": {
                    "strategy": (
                        "min_volatility"
                    ),
                    "max_weight": (
                        1.0
                    ),
                },
                "Equal Weight": {
                    "strategy": (
                        "equal_weight"
                    ),
                    "max_weight": (
                        1.0
                    ),
                },
            }

            if (
                request[
                    "max_weight"
                ]
                < 1.0
            ):
                strategy_configs[
                    (
                        "Max Sharpe "
                        f"({request['max_weight']:.0%} Cap)"
                    )
                ] = {
                    "strategy": (
                        "max_sharpe"
                    ),
                    "max_weight": (
                        request[
                            "max_weight"
                        ]
                    ),
                }

            results_by_name = {}

            with st.spinner(
                "Running walk-forward backtests..."
            ):
                try:
                    for (
                        name,
                        configuration,
                    ) in (
                        strategy_configs.items()
                    ):
                        results_by_name[
                            name
                        ] = (
                            run_backtest_cached(
                                immutable_asset_returns,
                                tuple(
                                    benchmark_returns
                                ),
                                tuple(
                                    return_dates
                                ),
                                configuration[
                                    "strategy"
                                ],
                                request[
                                    "train_window"
                                ],
                                request[
                                    "rebalance_frequency"
                                ],
                                request[
                                    "risk_free_rate"
                                ],
                                configuration[
                                    "max_weight"
                                ],
                                request[
                                    "transaction_cost_bps"
                                ],
                            )
                        )

                except Exception as error:
                    st.error(
                        str(
                            error
                        )
                    )

            if results_by_name:
                comparison_table = (
                    build_backtest_comparison_table(
                        results_by_name,
                        benchmark_symbol,
                    )
                )

                st.dataframe(
                    comparison_table,
                    use_container_width=True,
                    hide_index=True,
                )

                growth_dataframe = (
                    build_backtest_growth_dataframe(
                        results_by_name,
                        benchmark_symbol,
                    )
                )

                st.subheader(
                    "Growth of $1"
                )

                st.line_chart(
                    growth_dataframe,
                    use_container_width=True,
                )

                selected_strategy = (
                    st.selectbox(
                        "Inspect rebalance history",
                        options=list(
                            results_by_name.keys()
                        ),
                    )
                )

                st.dataframe(
                    build_rebalance_dataframe(
                        results_by_name[
                            selected_strategy
                        ],
                        asset_symbols,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    f"Transaction-cost assumption: "
                    f"{request['transaction_cost_bps']:.1f} "
                    f"basis points per unit of "
                    f"one-way turnover."
                )

    with data_tab:
        st.subheader(
            "Latest Prices"
        )

        latest_rows = []

        for symbol in (
            all_symbols
        ):
            latest = (
                latest_prices[
                    symbol
                ]
            )

            latest_rows.append(
                {
                    "Ticker": symbol,
                    "Price": (
                        f"${latest['price']:,.2f}"
                    ),
                    "Timestamp": (
                        latest[
                            "timestamp"
                        ]
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(
                latest_rows
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "Historical Prices"
        )

        full_price_dataframe = (
            build_price_dataframe(
                historical_prices,
                dates,
            )
        )

        st.dataframe(
            full_price_dataframe.tail(
                100
            ),
            use_container_width=True,
        )

        csv_data = (
            full_price_dataframe
            .to_csv()
            .encode(
                "utf-8"
            )
        )

        st.download_button(
            "Download Historical Data",
            data=csv_data,
            file_name=(
                "quant_market_data.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()