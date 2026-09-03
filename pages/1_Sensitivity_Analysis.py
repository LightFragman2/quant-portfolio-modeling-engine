from datetime import (
    date,
    datetime,
    time,
    timezone,
)

import altair as alt
import pandas as pd
import streamlit as st

from src.input_validation import (
    normalize_symbol,
    parse_symbol_input,
    validate_date_range,
)

from src.market_data import (
    get_historical_close_prices,
)

from src.returns import (
    calculate_simple_returns,
)

from src.sensitivity import (
    parse_numeric_values,
    run_sensitivity_analysis,
)


PARAMETER_NAMES = {
    "Maximum Asset Weight": (
        "max_weight"
    ),
    "Training Window": (
        "train_window"
    ),
    "Rebalance Frequency": (
        "rebalance_frequency"
    ),
    "Transaction Costs": (
        "transaction_cost_bps"
    ),
}


DEFAULT_VALUES = {
    "max_weight": (
        "25, 40, 60, 80, 100"
    ),
    "train_window": (
        "252, 378, 504, 756"
    ),
    "rebalance_frequency": (
        "21, 63, 126, 252"
    ),
    "transaction_cost_bps": (
        "0, 5, 10, 25, 50"
    ),
}


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
    show_spinner=False,
)
def run_sensitivity_cached(
    asset_returns_tuple,
    benchmark_returns_tuple,
    dates_tuple,
    parameter_name,
    parameter_values_tuple,
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

    return (
        run_sensitivity_analysis(
            asset_returns=(
                asset_returns
            ),
            benchmark_returns=list(
                benchmark_returns_tuple
            ),
            dates=list(
                dates_tuple
            ),
            parameter_name=(
                parameter_name
            ),
            parameter_values=list(
                parameter_values_tuple
            ),
            strategy="max_sharpe",
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
    )


def build_returns(
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


def parameter_display_value(
    parameter_name,
    value,
):
    if parameter_name == (
        "max_weight"
    ):
        return (
            value
            * 100
        )

    return value


def parameter_axis_title(
    parameter_name,
):
    if parameter_name == (
        "max_weight"
    ):
        return (
            "Maximum Asset Weight (%)"
        )

    if parameter_name == (
        "train_window"
    ):
        return (
            "Training Window "
            "(Trading Days)"
        )

    if parameter_name == (
        "rebalance_frequency"
    ):
        return (
            "Rebalance Frequency "
            "(Trading Days)"
        )

    return (
        "Transaction Cost "
        "(Basis Points)"
    )


def build_results_dataframe(
    results,
    parameter_name,
):
    rows = []

    for result in results:
        rows.append(
            {
                "Parameter Value": (
                    parameter_display_value(
                        parameter_name,
                        result[
                            "parameter_value"
                        ],
                    )
                ),
                "CAGR": (
                    result[
                        "cagr"
                    ]
                ),
                "Annual Return": (
                    result[
                        "annual_return"
                    ]
                ),
                "Volatility": (
                    result[
                        "annual_volatility"
                    ]
                ),
                "Sharpe": (
                    result[
                        "sharpe_ratio"
                    ]
                ),
                "Max Drawdown": (
                    result[
                        "max_drawdown"
                    ]
                ),
                "Total Return": (
                    result[
                        "total_return"
                    ]
                ),
                "Turnover": (
                    result[
                        "total_turnover"
                    ]
                ),
                "Rebalances": (
                    result[
                        "number_of_rebalances"
                    ]
                ),
            }
        )

    dataframe = (
        pd.DataFrame(
            rows
        )
    )

    dataframe = dataframe.sort_values(
        "Parameter Value"
    )

    return dataframe


def format_results_table(
    dataframe,
    parameter_name,
):
    formatted = (
        dataframe.copy()
    )

    if parameter_name == (
        "max_weight"
    ):
        formatted[
            "Parameter Value"
        ] = formatted[
            "Parameter Value"
        ].map(
            lambda value: (
                f"{value:.0f}%"
            )
        )

    elif parameter_name in (
        "train_window",
        "rebalance_frequency",
    ):
        formatted[
            "Parameter Value"
        ] = formatted[
            "Parameter Value"
        ].map(
            lambda value: (
                f"{int(value)} days"
            )
        )

    else:
        formatted[
            "Parameter Value"
        ] = formatted[
            "Parameter Value"
        ].map(
            lambda value: (
                f"{value:g} bps"
            )
        )

    percentage_columns = [
        "CAGR",
        "Annual Return",
        "Volatility",
        "Max Drawdown",
        "Total Return",
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
            f"{value:.2f}x"
        )
    )

    return formatted


def build_metric_chart(
    dataframe,
    parameter_name,
    metric,
    title,
    percent=False,
):
    axis_format = (
        ".0%"
        if percent
        else ".2f"
    )

    chart = (
        alt.Chart(
            dataframe
        )
        .mark_line(
            point=True,
            strokeWidth=3,
        )
        .encode(
            x=alt.X(
                "Parameter Value:Q",
                title=(
                    parameter_axis_title(
                        parameter_name
                    )
                ),
            ),
            y=alt.Y(
                f"{metric}:Q",
                title=title,
                axis=alt.Axis(
                    format=(
                        axis_format
                    )
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "Parameter Value:Q",
                    title="Setting",
                ),
                alt.Tooltip(
                    f"{metric}:Q",
                    title=title,
                    format=(
                        ".2%"
                        if percent
                        else ".3f"
                    ),
                ),
            ],
        )
        .properties(
            height=300
        )
    )

    return chart


def main():
    st.title(
        "Sensitivity Analysis"
    )

    st.write(
        "Test whether portfolio performance "
        "remains stable when important model "
        "assumptions change."
    )

    st.info(
        "A robust strategy should not depend "
        "on one extremely specific parameter "
        "setting. Large performance changes "
        "from small assumption changes can "
        "indicate model fragility."
    )

    with st.sidebar:
        st.header(
            "Portfolio"
        )

        raw_assets = (
            st.text_area(
                "Assets",
                value=(
                    "AAPL, MSFT, NVDA, JPM"
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

        st.header(
            "Base Model"
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
                "Base maximum asset weight (%)",
                min_value=1.0,
                max_value=100.0,
                value=100.0,
                step=5.0,
            )
        )

        train_window = (
            st.number_input(
                "Base training window",
                min_value=126,
                max_value=1260,
                value=504,
                step=63,
            )
        )

        rebalance_frequency = (
            st.number_input(
                "Base rebalance frequency",
                min_value=21,
                max_value=252,
                value=63,
                step=21,
            )
        )

        transaction_cost_bps = (
            st.number_input(
                "Base transaction cost (bps)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
            )
        )

        st.divider()

        st.header(
            "Sensitivity Test"
        )

        parameter_label = (
            st.selectbox(
                "Parameter to vary",
                options=list(
                    PARAMETER_NAMES.keys()
                ),
            )
        )

        parameter_name = (
            PARAMETER_NAMES[
                parameter_label
            ]
        )

        raw_values = (
            st.text_input(
                "Values to test",
                value=(
                    DEFAULT_VALUES[
                        parameter_name
                    ]
                ),
            )
        )

        if parameter_name == (
            "max_weight"
        ):
            st.caption(
                "Enter percentages, for example: "
                "25, 40, 60, 80, 100"
            )

        elif parameter_name in (
            "train_window",
            "rebalance_frequency",
        ):
            st.caption(
                "Values are trading days."
            )

        else:
            st.caption(
                "Values are basis points."
            )

        run_analysis = (
            st.button(
                "Run Sensitivity Analysis",
                type="primary",
                use_container_width=True,
            )
        )

    if not run_analysis:
        st.subheader(
            "What this tests"
        )

        st.markdown(
            """
Sensitivity analysis changes one assumption
while holding the others constant.

Examples:

- **Maximum asset weight:** Does performance disappear when concentration is limited?
- **Training window:** Does the strategy only work with one lookback period?
- **Rebalance frequency:** Does performance depend heavily on trading timing?
- **Transaction costs:** Does realistic trading friction destroy the result?
"""
        )

        return

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

        parameter_values = (
            parse_numeric_values(
                raw_values
            )
        )

        if parameter_name == (
            "max_weight"
        ):
            parameter_values = [
                value / 100
                for value
                in parameter_values
            ]

        elif parameter_name in (
            "train_window",
            "rebalance_frequency",
        ):
            for value in (
                parameter_values
            ):
                if (
                    int(
                        value
                    )
                    != value
                ):
                    raise ValueError(
                        "Trading-day sensitivity "
                        "values must be whole numbers."
                    )

            parameter_values = [
                int(
                    value
                )
                for value
                in parameter_values
            ]

        minimum_feasible_weight = (
            1
            / len(
                asset_symbols
            )
        )

        base_max_weight = (
            max_weight_percent
            / 100
        )

        if (
            parameter_name
            != "max_weight"
            and base_max_weight
            < minimum_feasible_weight
            - 1e-12
        ):
            raise ValueError(
                "The base maximum-weight setting "
                "is not feasible for this number "
                "of assets."
            )

    except ValueError as error:
        st.error(
            str(
                error
            )
        )

        return

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

    try:
        with st.spinner(
            "Loading market data..."
        ):
            (
                historical_prices,
                dates,
            ) = (
                load_historical_data(
                    tuple(
                        all_symbols
                    ),
                    start_date,
                    end_date,
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

    returns_by_symbol = (
        build_returns(
            all_symbols,
            historical_prices,
        )
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

    return_dates = (
        dates[1:]
    )

    immutable_asset_returns = tuple(
        tuple(
            values
        )
        for values
        in asset_returns
    )

    try:
        with st.spinner(
            "Running walk-forward "
            "sensitivity backtests..."
        ):
            results = (
                run_sensitivity_cached(
                    immutable_asset_returns,
                    tuple(
                        benchmark_returns
                    ),
                    tuple(
                        return_dates
                    ),
                    parameter_name,
                    tuple(
                        parameter_values
                    ),
                    int(
                        train_window
                    ),
                    int(
                        rebalance_frequency
                    ),
                    (
                        risk_free_percent
                        / 100
                    ),
                    base_max_weight,
                    transaction_cost_bps,
                )
            )

    except Exception as error:
        st.error(
            "Sensitivity analysis failed."
        )

        st.code(
            str(
                error
            )
        )

        return

    dataframe = (
        build_results_dataframe(
            results,
            parameter_name,
        )
    )

    st.subheader(
        f"{parameter_label} Results"
    )

    st.caption(
        "All other model assumptions remain "
        "constant while this parameter changes."
    )

    best_sharpe_index = (
        dataframe[
            "Sharpe"
        ].idxmax()
    )

    lowest_volatility_index = (
        dataframe[
            "Volatility"
        ].idxmin()
    )

    best_sharpe = (
        dataframe.loc[
            best_sharpe_index
        ]
    )

    lowest_volatility = (
        dataframe.loc[
            lowest_volatility_index
        ]
    )

    metric_one, metric_two, metric_three = (
        st.columns(3)
    )

    metric_one.metric(
        "Best Sharpe",
        f"{best_sharpe['Sharpe']:.3f}",
    )

    metric_two.metric(
        "Best-Sharpe Setting",
        f"{best_sharpe['Parameter Value']:g}",
    )

    metric_three.metric(
        "Lowest Volatility",
        f"{lowest_volatility['Volatility']:.2%}",
    )

    st.dataframe(
        format_results_table(
            dataframe,
            parameter_name,
        ),
        use_container_width=True,
        hide_index=True,
    )

    left, right = (
        st.columns(2)
    )

    with left:
        st.altair_chart(
            build_metric_chart(
                dataframe,
                parameter_name,
                metric="CAGR",
                title="CAGR",
                percent=True,
            ),
            use_container_width=True,
        )

    with right:
        st.altair_chart(
            build_metric_chart(
                dataframe,
                parameter_name,
                metric="Sharpe",
                title="Sharpe Ratio",
                percent=False,
            ),
            use_container_width=True,
        )

    left, right = (
        st.columns(2)
    )

    with left:
        st.altair_chart(
            build_metric_chart(
                dataframe,
                parameter_name,
                metric="Volatility",
                title="Annual Volatility",
                percent=True,
            ),
            use_container_width=True,
        )

    with right:
        st.altair_chart(
            build_metric_chart(
                dataframe,
                parameter_name,
                metric="Max Drawdown",
                title="Maximum Drawdown",
                percent=True,
            ),
            use_container_width=True,
        )

    st.subheader(
        "How to Interpret This"
    )

    st.markdown(
        """
A strategy looks more robust when performance
changes gradually across reasonable parameter
values.

Be cautious when:

- Sharpe collapses after a small parameter change
- One exact setting massively outperforms every nearby setting
- Drawdown changes dramatically from small assumption changes
- Returns disappear after modest transaction costs
- The strategy only works with one training window or rebalance frequency

Sensitivity analysis does **not** prove that a
strategy will work in the future. It helps reveal
whether historical results are overly dependent on
specific modeling choices.
"""
    )


if __name__ == "__main__":
    main()