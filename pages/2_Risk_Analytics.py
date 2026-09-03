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
    parse_symbol_input,
    validate_date_range,
)

from src.market_data import (
    get_historical_close_prices,
)

from src.returns import (
    calculate_simple_returns,
)

from src.statistics import (
    standard_deviation,
)

from src.annualization import (
    annualized_volatility,
)

from src.risk import (
    historical_var,
    historical_cvar,
    drawdown_series,
    maximum_drawdown,
    rolling_volatility,
    rolling_sharpe_ratio,
    equal_weight_portfolio_returns,
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

    return get_historical_close_prices(
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


def build_returns(
    symbols,
    historical_prices,
):
    return {
        symbol: (
            calculate_simple_returns(
                historical_prices[
                    symbol
                ]
            )
        )
        for symbol in symbols
    }


def build_daily_return_dataframe(
    dates,
    returns,
):
    return pd.DataFrame(
        {
            "Date": dates,
            "Return": returns,
        }
    )


def main():
    st.title(
        "Risk Analytics"
    )

    st.write(
        "Analyze downside risk, drawdowns, "
        "rolling volatility, rolling Sharpe "
        "ratios, and historical return tails."
    )

    with st.sidebar:
        st.header(
            "Risk Settings"
        )

        raw_assets = (
            st.text_area(
                "Assets",
                value=(
                    "AAPL, MSFT, NVDA, JPM"
                ),
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

        confidence_percent = (
            st.slider(
                "VaR / CVaR confidence level",
                min_value=90,
                max_value=99,
                value=95,
                step=1,
            )
        )

        rolling_window = (
            st.number_input(
                "Rolling window (trading days)",
                min_value=21,
                max_value=252,
                value=63,
                step=21,
            )
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

        run_analysis = (
            st.button(
                "Run Risk Analysis",
                type="primary",
                use_container_width=True,
            )
        )

    if not run_analysis:
        st.info(
            "Configure the analysis in the "
            "sidebar and click **Run Risk Analysis**."
        )

        return

    try:
        asset_symbols = (
            parse_symbol_input(
                raw_assets,
                max_symbols=10,
            )
        )

        validate_date_range(
            start_date,
            end_date,
        )

    except ValueError as error:
        st.error(
            str(
                error
            )
        )

        return

    try:
        with st.spinner(
            "Loading market data..."
        ):
            (
                historical_prices,
                dates,
            ) = load_historical_data(
                tuple(
                    asset_symbols
                ),
                start_date,
                end_date,
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
            asset_symbols,
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
        for symbol in asset_symbols
    ]

    equal_weight_returns = (
        equal_weight_portfolio_returns(
            asset_returns
        )
    )

    series_options = {
        "Equal Weight Portfolio": (
            equal_weight_returns
        ),
    }

    for symbol in asset_symbols:
        series_options[
            symbol
        ] = returns_by_symbol[
            symbol
        ]

    selected_name = (
        st.selectbox(
            "Analyze",
            options=list(
                series_options.keys()
            ),
        )
    )

    selected_returns = (
        series_options[
            selected_name
        ]
    )

    if (
        int(
            rolling_window
        )
        > len(
            selected_returns
        )
    ):
        st.error(
            "The rolling window is longer "
            "than the available return history."
        )

        return

    confidence_level = (
        confidence_percent
        / 100
    )

    var_value = (
        historical_var(
            selected_returns,
            confidence_level,
        )
    )

    cvar_value = (
        historical_cvar(
            selected_returns,
            confidence_level,
        )
    )

    max_drawdown_value = (
        maximum_drawdown(
            selected_returns
        )
    )

    daily_volatility = (
        standard_deviation(
            selected_returns,
            sample=True,
        )
    )

    annual_volatility_value = (
        annualized_volatility(
            daily_volatility
        )
    )

    metric_one, metric_two, metric_three, metric_four = (
        st.columns(4)
    )

    metric_one.metric(
        f"{confidence_percent}% Daily VaR",
        f"{var_value:.2%}",
    )

    metric_two.metric(
        f"{confidence_percent}% Daily CVaR",
        f"{cvar_value:.2%}",
    )

    metric_three.metric(
        "Maximum Drawdown",
        f"{max_drawdown_value:.2%}",
    )

    metric_four.metric(
        "Annualized Volatility",
        f"{annual_volatility_value:.2%}",
    )

    st.caption(
        f"Historical VaR means approximately "
        f"{100 - confidence_percent}% of observed "
        f"daily returns were worse than the "
        f"{confidence_percent}% VaR threshold. "
        f"CVaR measures the average loss inside "
        f"that historical tail."
    )

    tab_one, tab_two, tab_three, tab_four = (
        st.tabs(
            [
                "Drawdown",
                "Rolling Risk",
                "Return Distribution",
                "Extreme Days",
            ]
        )
    )

    with tab_one:
        drawdowns = (
            drawdown_series(
                selected_returns
            )
        )

        drawdown_dataframe = (
            pd.DataFrame(
                {
                    "Date": (
                        return_dates
                    ),
                    "Drawdown": (
                        drawdowns
                    ),
                }
            )
        )

        drawdown_chart = (
            alt.Chart(
                drawdown_dataframe
            )
            .mark_area()
            .encode(
                x=alt.X(
                    "Date:T",
                    title="Date",
                ),
                y=alt.Y(
                    "Drawdown:Q",
                    title="Drawdown",
                    axis=alt.Axis(
                        format=".0%"
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Date:T"
                    ),
                    alt.Tooltip(
                        "Drawdown:Q",
                        format=".2%",
                    ),
                ],
            )
            .properties(
                height=450
            )
        )

        st.altair_chart(
            drawdown_chart,
            use_container_width=True,
        )

    with tab_two:
        volatility_values = (
            rolling_volatility(
                selected_returns,
                window=int(
                    rolling_window
                ),
            )
        )

        sharpe_values = (
            rolling_sharpe_ratio(
                selected_returns,
                window=int(
                    rolling_window
                ),
                annual_risk_free_rate=(
                    risk_free_percent
                    / 100
                ),
            )
        )

        rolling_dates = (
            return_dates[
                int(
                    rolling_window
                )
                - 1:
            ]
        )

        rolling_dataframe = (
            pd.DataFrame(
                {
                    "Date": (
                        rolling_dates
                    ),
                    "Rolling Volatility": (
                        volatility_values
                    ),
                    "Rolling Sharpe": (
                        sharpe_values
                    ),
                }
            )
        )

        st.subheader(
            "Rolling Annualized Volatility"
        )

        volatility_chart = (
            alt.Chart(
                rolling_dataframe
            )
            .mark_line()
            .encode(
                x="Date:T",
                y=alt.Y(
                    "Rolling Volatility:Q",
                    axis=alt.Axis(
                        format=".0%"
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Date:T"
                    ),
                    alt.Tooltip(
                        "Rolling Volatility:Q",
                        format=".2%",
                    ),
                ],
            )
            .properties(
                height=300
            )
        )

        st.altair_chart(
            volatility_chart,
            use_container_width=True,
        )

        st.subheader(
            "Rolling Sharpe Ratio"
        )

        sharpe_chart = (
            alt.Chart(
                rolling_dataframe
            )
            .mark_line()
            .encode(
                x="Date:T",
                y="Rolling Sharpe:Q",
                tooltip=[
                    alt.Tooltip(
                        "Date:T"
                    ),
                    alt.Tooltip(
                        "Rolling Sharpe:Q",
                        format=".3f",
                    ),
                ],
            )
            .properties(
                height=300
            )
        )

        st.altair_chart(
            sharpe_chart,
            use_container_width=True,
        )

    with tab_three:
        return_dataframe = (
            build_daily_return_dataframe(
                return_dates,
                selected_returns,
            )
        )

        histogram = (
            alt.Chart(
                return_dataframe
            )
            .mark_bar()
            .encode(
                x=alt.X(
                    "Return:Q",
                    bin=alt.Bin(
                        maxbins=60
                    ),
                    title="Daily Return",
                    axis=alt.Axis(
                        format=".1%"
                    ),
                ),
                y=alt.Y(
                    "count():Q",
                    title="Observations",
                ),
                tooltip=[
                    alt.Tooltip(
                        "count():Q",
                        title="Observations",
                    )
                ],
            )
            .properties(
                height=450
            )
        )

        st.altair_chart(
            histogram,
            use_container_width=True,
        )

    with tab_four:
        extreme_dataframe = (
            build_daily_return_dataframe(
                return_dates,
                selected_returns,
            )
        )

        worst_days = (
            extreme_dataframe
            .sort_values(
                "Return"
            )
            .head(
                10
            )
            .copy()
        )

        best_days = (
            extreme_dataframe
            .sort_values(
                "Return",
                ascending=False,
            )
            .head(
                10
            )
            .copy()
        )

        for dataframe in [
            worst_days,
            best_days,
        ]:
            dataframe[
                "Return"
            ] = dataframe[
                "Return"
            ].map(
                lambda value: (
                    f"{value:.2%}"
                )
            )

        left, right = (
            st.columns(2)
        )

        with left:
            st.subheader(
                "10 Worst Historical Days"
            )

            st.dataframe(
                worst_days,
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.subheader(
                "10 Best Historical Days"
            )

            st.dataframe(
                best_days,
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    st.subheader(
        "Risk Interpretation"
    )

    st.markdown(
        """
- **VaR** estimates a historical loss threshold at the selected confidence level.
- **CVaR / Expected Shortfall** looks beyond VaR and measures the average loss among the worst historical observations.
- **Maximum drawdown** measures the largest historical peak-to-trough decline.
- **Rolling volatility** shows how risk changes through different market regimes.
- **Rolling Sharpe** shows whether historical risk-adjusted performance remains stable through time.
- **Return distributions** help reveal asymmetry, fat tails, and unusually large daily moves.

These statistics describe historical behavior. They do not guarantee the size or frequency of future losses.
"""
    )


if __name__ == "__main__":
    main()