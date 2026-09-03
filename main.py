from datetime import (
    datetime,
    timezone,
)

from src.market_data import (
    get_historical_close_prices,
    get_latest_trade_prices,
)

from src.returns import (
    calculate_simple_returns,
)

from src.statistics import (
    arithmetic_mean,
    variance,
    standard_deviation,
    covariance,
    correlation,
)

from src.portfolio import (
    portfolio_expected_return,
    multi_asset_portfolio_variance,
    portfolio_volatility,
    sharpe_ratio,
)

from src.regression import (
    beta,
    linear_regression,
)

from src.annualization import (
    annualized_arithmetic_return,
    annualized_volatility,
    cumulative_return,
    annualized_compounded_return,
)

from src.monte_carlo import (
    simulate_portfolios,
    maximum_sharpe_portfolio,
    minimum_volatility_portfolio,
)

from src.visualization import (
    plot_monte_carlo_portfolios,
)


def print_portfolio(
    title,
    portfolio,
    asset_names,
):
    print(
        f"\n{title}"
    )

    print(
        f"Annual return: "
        f"{portfolio['annual_return']:.2%}"
    )

    print(
        f"Annual volatility: "
        f"{portfolio['annual_volatility']:.2%}"
    )

    print(
        f"Sharpe ratio: "
        f"{portfolio['sharpe_ratio']:.4f}"
    )

    print(
        "Weights:"
    )

    for name, weight in zip(
        asset_names,
        portfolio["weights"],
    ):
        print(
            f"  {name}: "
            f"{weight:.2%}"
        )


def main():
    print(
        "======================================="
    )

    print(
        "   Quant Portfolio Modeling Engine"
    )

    print(
        "======================================="
    )

    asset_symbols = [
        "AAPL",
        "MSFT",
        "NVDA",
        "JPM",
    ]

    benchmark_symbol = "SPY"

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

    print(
        "\nDownloading historical market data..."
    )

    historical_prices, dates = (
        get_historical_close_prices(
            symbols=all_symbols,
            start_date=start_date,
        )
    )

    print(
        f"Loaded {len(dates):,} "
        f"aligned trading days."
    )

    print(
        f"Historical start: "
        f"{dates[0].date()}"
    )

    print(
        f"Historical end: "
        f"{dates[-1].date()}"
    )

    print(
        "\n--- Latest Market Prices ---"
    )

    latest_prices = (
        get_latest_trade_prices(
            all_symbols
        )
    )

    for symbol in all_symbols:
        latest = (
            latest_prices[
                symbol
            ]
        )

        print(
            f"{symbol}: "
            f"${latest['price']:.2f}"
        )

    print(
        "\nCalculating returns..."
    )

    returns_by_symbol = {}

    for symbol in all_symbols:
        returns_by_symbol[
            symbol
        ] = calculate_simple_returns(
            historical_prices[
                symbol
            ]
        )

    asset_returns = [
        returns_by_symbol[
            symbol
        ]
        for symbol in asset_symbols
    ]

    market_returns = (
        returns_by_symbol[
            benchmark_symbol
        ]
    )

    print(
        "\n--- Historical Asset Statistics ---"
    )

    for symbol in asset_symbols:
        returns = (
            returns_by_symbol[
                symbol
            ]
        )

        mean_return = (
            arithmetic_mean(
                returns
            )
        )

        asset_variance = (
            variance(
                returns,
                sample=True,
            )
        )

        daily_volatility = (
            standard_deviation(
                returns,
                sample=True,
            )
        )

        annual_return = (
            annualized_arithmetic_return(
                mean_return
            )
        )

        annual_volatility_value = (
            annualized_volatility(
                daily_volatility
            )
        )

        compounded_return = (
            annualized_compounded_return(
                returns
            )
        )

        total_return = (
            cumulative_return(
                returns
            )
        )

        print(
            f"\n{symbol}"
        )

        print(
            f"  Annual arithmetic return: "
            f"{annual_return:.2%}"
        )

        print(
            f"  Annual compounded return: "
            f"{compounded_return:.2%}"
        )

        print(
            f"  Annual volatility: "
            f"{annual_volatility_value:.2%}"
        )

        print(
            f"  Historical cumulative return: "
            f"{total_return:.2%}"
        )

        print(
            f"  Daily variance: "
            f"{asset_variance:.8f}"
        )

    print(
        "\n--- Correlation Example ---"
    )

    correlation_value = (
        correlation(
            returns_by_symbol["AAPL"],
            returns_by_symbol["MSFT"],
            sample=True,
        )
    )

    covariance_value = (
        covariance(
            returns_by_symbol["AAPL"],
            returns_by_symbol["MSFT"],
            sample=True,
        )
    )

    print(
        f"AAPL / MSFT covariance: "
        f"{covariance_value:.8f}"
    )

    print(
        f"AAPL / MSFT correlation: "
        f"{correlation_value:.4f}"
    )

    print(
        "\n--- Beta / Regression vs SPY ---"
    )

    for symbol in asset_symbols:
        stock_returns = (
            returns_by_symbol[
                symbol
            ]
        )

        stock_beta = beta(
            stock_returns,
            market_returns,
            sample=True,
        )

        alpha, regression_beta = (
            linear_regression(
                market_returns,
                stock_returns,
                sample=True,
            )
        )

        print(
            f"\n{symbol}"
        )

        print(
            f"  Beta: "
            f"{stock_beta:.4f}"
        )

        print(
            f"  Regression alpha: "
            f"{alpha:.6f}"
        )

        print(
            f"  Regression beta: "
            f"{regression_beta:.4f}"
        )

    print(
        "\n--- Equal-Weight Portfolio ---"
    )

    number_of_assets = len(
        asset_symbols
    )

    equal_weights = [
        1 / number_of_assets
    ] * number_of_assets

    expected_daily_returns = [
        arithmetic_mean(
            returns_by_symbol[
                symbol
            ]
        )
        for symbol in asset_symbols
    ]

    daily_portfolio_return = (
        portfolio_expected_return(
            equal_weights,
            expected_daily_returns,
        )
    )

    daily_portfolio_variance = (
        multi_asset_portfolio_variance(
            equal_weights,
            asset_returns,
            sample=True,
        )
    )

    daily_portfolio_volatility = (
        portfolio_volatility(
            daily_portfolio_variance
        )
    )

    annual_portfolio_return = (
        annualized_arithmetic_return(
            daily_portfolio_return
        )
    )

    annual_portfolio_volatility = (
        annualized_volatility(
            daily_portfolio_volatility
        )
    )

    annual_risk_free_rate = 0.04

    equal_weight_sharpe = (
        sharpe_ratio(
            annual_portfolio_return,
            annual_risk_free_rate,
            annual_portfolio_volatility,
        )
    )

    print(
        f"Annual expected return: "
        f"{annual_portfolio_return:.2%}"
    )

    print(
        f"Annual volatility: "
        f"{annual_portfolio_volatility:.2%}"
    )

    print(
        f"Sharpe ratio: "
        f"{equal_weight_sharpe:.4f}"
    )

    print(
        "\nWeights:"
    )

    for symbol, weight in zip(
        asset_symbols,
        equal_weights,
    ):
        print(
            f"  {symbol}: "
            f"{weight:.2%}"
        )

    print(
        "\n--- Monte Carlo Simulation ---"
    )

    number_of_portfolios = 10000

    simulation_results = (
        simulate_portfolios(
            asset_returns=asset_returns,
            number_of_portfolios=number_of_portfolios,
            annual_risk_free_rate=annual_risk_free_rate,
            periods_per_year=252,
            sample=True,
            seed=42,
        )
    )

    print(
        f"Simulated "
        f"{len(simulation_results):,} "
        f"portfolios."
    )

    max_sharpe = (
        maximum_sharpe_portfolio(
            simulation_results
        )
    )

    min_volatility = (
        minimum_volatility_portfolio(
            simulation_results
        )
    )

    print_portfolio(
        "Highest-Sharpe Sampled Portfolio",
        max_sharpe,
        asset_symbols,
    )

    print_portfolio(
        "Lowest-Volatility Sampled Portfolio",
        min_volatility,
        asset_symbols,
    )

    plot_path = (
        plot_monte_carlo_portfolios(
            simulation_results
        )
    )

    print(
        "\nMonte Carlo plot saved to:"
    )

    print(
        plot_path
    )


if __name__ == "__main__":
    main()