from datetime import (
    datetime,
    timezone,
)

from src.market_data import (
    get_historical_close_prices,
)

from src.returns import (
    calculate_simple_returns,
)

from src.backtesting import (
    rolling_backtest,
)

from src.backtest_visualization import (
    plot_backtest_comparison,
)


def build_returns(
    symbols,
    historical_prices,
):
    results = {}

    for symbol in symbols:
        results[
            symbol
        ] = calculate_simple_returns(
            historical_prices[
                symbol
            ]
        )

    return results


def print_comparison_table(
    results,
    benchmark_name,
):
    print(
        "\n"
        "============================================================"
    )

    print(
        "                 BACKTEST ROBUSTNESS RESULTS"
    )

    print(
        "============================================================"
    )

    header = (
        f"{'Strategy':<30}"
        f"{'CAGR':>10}"
        f"{'Vol':>10}"
        f"{'Sharpe':>10}"
        f"{'Drawdown':>12}"
        f"{'Turnover':>12}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    for name, result in results.items():
        metrics = (
            result[
                "portfolio_metrics"
            ]
        )

        print(
            f"{name:<30}"
            f"{metrics['annual_compounded_return']:>9.2%} "
            f"{metrics['annual_volatility']:>9.2%} "
            f"{metrics['sharpe_ratio']:>9.3f} "
            f"{metrics['max_drawdown']:>11.2%} "
            f"{result['total_turnover']:>11.2f}x"
        )

    first_result = next(
        iter(
            results.values()
        )
    )

    benchmark = (
        first_result[
            "benchmark_metrics"
        ]
    )

    print(
        f"{benchmark_name:<30}"
        f"{benchmark['annual_compounded_return']:>9.2%} "
        f"{benchmark['annual_volatility']:>9.2%} "
        f"{benchmark['sharpe_ratio']:>9.3f} "
        f"{benchmark['max_drawdown']:>11.2%} "
        f"{'—':>12}"
    )


def main():
    asset_symbols = [
        "AAPL",
        "MSFT",
        "NVDA",
        "JPM",
    ]

    benchmark_symbol = (
        "SPY"
    )

    all_symbols = (
        asset_symbols
        + [benchmark_symbol]
    )

    start_date = datetime(
        2021,
        1,
        1,
        tzinfo=timezone.utc,
    )

    annual_risk_free_rate = (
        0.04
    )

    train_window = (
        504
    )

    rebalance_frequency = (
        63
    )

    transaction_cost_bps = (
        10.0
    )

    print(
        "Downloading historical data..."
    )

    historical_prices, dates = (
        get_historical_close_prices(
            symbols=all_symbols,
            start_date=start_date,
        )
    )

    returns_by_symbol = (
        build_returns(
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
        for symbol in asset_symbols
    ]

    benchmark_returns = (
        returns_by_symbol[
            benchmark_symbol
        ]
    )

    strategies = {
        "Max Sharpe": {
            "strategy": (
                "max_sharpe"
            ),
            "max_weight": (
                1.0
            ),
        },
        "Max Sharpe (40% Cap)": {
            "strategy": (
                "max_sharpe"
            ),
            "max_weight": (
                0.40
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

    results = {}

    for name, configuration in (
        strategies.items()
    ):
        print(
            f"Running {name}..."
        )

        results[
            name
        ] = rolling_backtest(
            asset_returns=(
                asset_returns
            ),
            benchmark_returns=(
                benchmark_returns
            ),
            dates=(
                return_dates
            ),
            strategy=(
                configuration[
                    "strategy"
                ]
            ),
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
            max_weight=(
                configuration[
                    "max_weight"
                ]
            ),
            transaction_cost_bps=(
                transaction_cost_bps
            ),
        )

    print_comparison_table(
        results,
        benchmark_symbol,
    )

    capped_result = (
        results[
            "Max Sharpe (40% Cap)"
        ]
    )

    highest_capped_weight = max(
        max(
            rebalance[
                "weights"
            ]
        )
        for rebalance
        in capped_result[
            "rebalances"
        ]
    )

    print(
        "\n40% cap verification:"
    )

    print(
        f"Highest portfolio weight observed: "
        f"{highest_capped_weight:.2%}"
    )

    print(
        "\nTransaction-cost assumption:"
    )

    print(
        f"{transaction_cost_bps:.1f} basis points "
        f"per unit of one-way turnover."
    )

    plot_path = (
        plot_backtest_comparison(
            results,
            benchmark_name=(
                benchmark_symbol
            ),
        )
    )

    print(
        "\nComparison plot saved to:"
    )

    print(
        plot_path
    )


if __name__ == "__main__":
    main()