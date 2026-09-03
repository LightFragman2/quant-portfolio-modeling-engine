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
    maximum_sharpe_portfolio as sampled_maximum_sharpe,
    minimum_volatility_portfolio as sampled_minimum_volatility,
)

from src.optimization import (
    maximum_sharpe_portfolio as optimized_maximum_sharpe,
    minimum_volatility_portfolio as optimized_minimum_volatility,
    efficient_frontier,
)

from src.visualization import (
    plot_monte_carlo_portfolios,
)


def print_portfolio(
    title,
    portfolio,
    asset_symbols,
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

    for symbol, weight in zip(
        asset_symbols,
        portfolio["weights"],
    ):
        print(
            f"  {symbol}: "
            f"{weight:.2%}"
        )


def build_returns(
    symbols,
    historical_prices,
):
    returns_by_symbol = {}

    for symbol in symbols:
        returns_by_symbol[
            symbol
        ] = calculate_simple_returns(
            historical_prices[
                symbol
            ]
        )

    return returns_by_symbol


def print_latest_prices(
    symbols,
):
    print(
        "\n--- Latest Market Prices ---"
    )

    latest_prices = (
        get_latest_trade_prices(
            symbols
        )
    )

    for symbol in symbols:
        latest = (
            latest_prices[
                symbol
            ]
        )

        print(
            f"{symbol}: "
            f"${latest['price']:.2f}"
        )


def print_asset_statistics(
    asset_symbols,
    returns_by_symbol,
):
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

        daily_variance = (
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
            f"{daily_variance:.8f}"
        )


def print_market_analysis(
    asset_symbols,
    benchmark_symbol,
    returns_by_symbol,
):
    print(
        "\n--- Beta / Regression vs "
        f"{benchmark_symbol} ---"
    )

    market_returns = (
        returns_by_symbol[
            benchmark_symbol
        ]
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


def calculate_equal_weight_portfolio(
    asset_returns,
    annual_risk_free_rate,
):
    number_of_assets = len(
        asset_returns
    )

    weights = [
        1 / number_of_assets
    ] * number_of_assets

    expected_daily_returns = []

    for returns in asset_returns:
        expected_daily_returns.append(
            arithmetic_mean(
                returns
            )
        )

    daily_return = (
        portfolio_expected_return(
            weights,
            expected_daily_returns,
        )
    )

    daily_variance = (
        multi_asset_portfolio_variance(
            weights,
            asset_returns,
            sample=True,
        )
    )

    daily_volatility = (
        portfolio_volatility(
            daily_variance
        )
    )

    annual_return = (
        annualized_arithmetic_return(
            daily_return
        )
    )

    annual_volatility_value = (
        annualized_volatility(
            daily_volatility
        )
    )

    portfolio_sharpe = (
        sharpe_ratio(
            annual_return,
            annual_risk_free_rate,
            annual_volatility_value,
        )
    )

    return {
        "weights": weights,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility_value,
        "sharpe_ratio": portfolio_sharpe,
    }


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

    annual_risk_free_rate = 0.04

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

    print_latest_prices(
        all_symbols
    )

    print(
        "\nCalculating returns..."
    )

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
        for symbol in asset_symbols
    ]

    print_asset_statistics(
        asset_symbols,
        returns_by_symbol,
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

    print_market_analysis(
        asset_symbols,
        benchmark_symbol,
        returns_by_symbol,
    )

    equal_weight_portfolio = (
        calculate_equal_weight_portfolio(
            asset_returns,
            annual_risk_free_rate,
        )
    )

    print_portfolio(
        "Equal-Weight Portfolio",
        equal_weight_portfolio,
        asset_symbols,
    )

    print(
        "\n--- Monte Carlo Simulation ---"
    )

    simulation_results = (
        simulate_portfolios(
            asset_returns=asset_returns,
            number_of_portfolios=10000,
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

    monte_carlo_max_sharpe = (
        sampled_maximum_sharpe(
            simulation_results
        )
    )

    monte_carlo_min_volatility = (
        sampled_minimum_volatility(
            simulation_results
        )
    )

    print_portfolio(
        "Monte Carlo Highest-Sharpe Portfolio",
        monte_carlo_max_sharpe,
        asset_symbols,
    )

    print_portfolio(
        "Monte Carlo Lowest-Volatility Portfolio",
        monte_carlo_min_volatility,
        asset_symbols,
    )

    print(
        "\n--- Mathematical Optimization ---"
    )

    true_max_sharpe = (
        optimized_maximum_sharpe(
            asset_returns,
            annual_risk_free_rate=annual_risk_free_rate,
            periods_per_year=252,
            sample=True,
        )
    )

    true_min_volatility = (
        optimized_minimum_volatility(
            asset_returns,
            annual_risk_free_rate=annual_risk_free_rate,
            periods_per_year=252,
            sample=True,
        )
    )

    print_portfolio(
        "Optimized Maximum-Sharpe Portfolio",
        true_max_sharpe,
        asset_symbols,
    )

    print_portfolio(
        "Optimized Minimum-Volatility Portfolio",
        true_min_volatility,
        asset_symbols,
    )

    print(
        "\n--- Monte Carlo vs Optimization ---"
    )

    print(
        "Maximum Sharpe:"
    )

    print(
        f"  Monte Carlo: "
        f"{monte_carlo_max_sharpe['sharpe_ratio']:.6f}"
    )

    print(
        f"  Optimized:   "
        f"{true_max_sharpe['sharpe_ratio']:.6f}"
    )

    print(
        "\nMinimum volatility:"
    )

    print(
        f"  Monte Carlo: "
        f"{monte_carlo_min_volatility['annual_volatility']:.6%}"
    )

    print(
        f"  Optimized:   "
        f"{true_min_volatility['annual_volatility']:.6%}"
    )

    print(
        "\n--- Efficient Frontier ---"
    )

    frontier = efficient_frontier(
        asset_returns,
        annual_risk_free_rate=annual_risk_free_rate,
        periods_per_year=252,
        sample=True,
        number_of_points=60,
    )

    print(
        f"Calculated "
        f"{len(frontier)} "
        f"efficient-frontier portfolios."
    )

    print(
        f"Frontier begins at "
        f"{frontier[0]['annual_return']:.2%} "
        f"return / "
        f"{frontier[0]['annual_volatility']:.2%} "
        f"volatility."
    )

    print(
        f"Frontier ends at "
        f"{frontier[-1]['annual_return']:.2%} "
        f"return / "
        f"{frontier[-1]['annual_volatility']:.2%} "
        f"volatility."
    )

    plot_path = (
        plot_monte_carlo_portfolios(
            simulation_results,
            optimized_max_sharpe=true_max_sharpe,
            optimized_min_volatility=true_min_volatility,
            efficient_frontier_points=frontier,
        )
    )

    print(
        "\nPortfolio analysis plot saved to:"
    )

    print(
        plot_path
    )


if __name__ == "__main__":
    main()