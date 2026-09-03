from src.returns import calculate_simple_returns

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
    print(f"\n{title}")

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

    print("Weights:")

    for name, weight in zip(
        asset_names,
        portfolio["weights"],
    ):
        print(
            f"  {name}: {weight:.2%}"
        )


def main():
    print(
        "=== Quant Portfolio Modeling Engine ==="
    )

    stock_a_prices = [
        100,
        104,
        102,
        108,
        112,
        110,
        115,
        117,
        116,
        120,
    ]

    stock_b_prices = [
        50,
        51,
        53,
        52,
        55,
        56,
        54,
        57,
        59,
        60,
    ]

    stock_c_prices = [
        80,
        79,
        82,
        84,
        83,
        86,
        88,
        87,
        90,
        92,
    ]

    market_prices = [
        200,
        204,
        202,
        207,
        211,
        210,
        214,
        216,
        215,
        219,
    ]

    stock_a_returns = (
        calculate_simple_returns(
            stock_a_prices
        )
    )

    stock_b_returns = (
        calculate_simple_returns(
            stock_b_prices
        )
    )

    stock_c_returns = (
        calculate_simple_returns(
            stock_c_prices
        )
    )

    market_returns = (
        calculate_simple_returns(
            market_prices
        )
    )

    print("\n--- Returns ---")

    print(
        "Stock A:",
        stock_a_returns,
    )

    print(
        "Stock B:",
        stock_b_returns,
    )

    print(
        "Stock C:",
        stock_c_returns,
    )

    mean_a = arithmetic_mean(
        stock_a_returns
    )

    mean_b = arithmetic_mean(
        stock_b_returns
    )

    mean_c = arithmetic_mean(
        stock_c_returns
    )

    variance_a = variance(
        stock_a_returns
    )

    volatility_a = standard_deviation(
        stock_a_returns
    )

    covariance_ab = covariance(
        stock_a_returns,
        stock_b_returns,
    )

    correlation_ab = correlation(
        stock_a_returns,
        stock_b_returns,
    )

    print(
        "\n--- Asset Statistics ---"
    )

    print(
        f"Stock A mean return: "
        f"{mean_a:.4%}"
    )

    print(
        f"Stock B mean return: "
        f"{mean_b:.4%}"
    )

    print(
        f"Stock C mean return: "
        f"{mean_c:.4%}"
    )

    print(
        f"Stock A variance: "
        f"{variance_a:.8f}"
    )

    print(
        f"Stock A volatility: "
        f"{volatility_a:.4%}"
    )

    print(
        f"A/B covariance: "
        f"{covariance_ab:.8f}"
    )

    print(
        f"A/B correlation: "
        f"{correlation_ab:.4f}"
    )

    weights = [
        0.50,
        0.30,
        0.20,
    ]

    expected_returns = [
        mean_a,
        mean_b,
        mean_c,
    ]

    asset_returns = [
        stock_a_returns,
        stock_b_returns,
        stock_c_returns,
    ]

    expected_portfolio_return = (
        portfolio_expected_return(
            weights,
            expected_returns,
        )
    )

    portfolio_variance_value = (
        multi_asset_portfolio_variance(
            weights,
            asset_returns,
        )
    )

    portfolio_volatility_value = (
        portfolio_volatility(
            portfolio_variance_value
        )
    )

    annual_portfolio_return = (
        annualized_arithmetic_return(
            expected_portfolio_return
        )
    )

    annual_portfolio_volatility = (
        annualized_volatility(
            portfolio_volatility_value
        )
    )

    annual_risk_free_rate = 0.04

    portfolio_sharpe = sharpe_ratio(
        annual_portfolio_return,
        annual_risk_free_rate,
        annual_portfolio_volatility,
    )

    print(
        "\n--- Example Portfolio ---"
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
        f"{portfolio_sharpe:.4f}"
    )

    annual_return_a = (
        annualized_arithmetic_return(
            mean_a
        )
    )

    annual_volatility_a = (
        annualized_volatility(
            volatility_a
        )
    )

    cumulative_return_a = (
        cumulative_return(
            stock_a_returns
        )
    )

    compounded_annual_return_a = (
        annualized_compounded_return(
            stock_a_returns
        )
    )

    print(
        "\n--- Annualized Stock A Statistics ---"
    )

    print(
        f"Arithmetic annual return: "
        f"{annual_return_a:.2%}"
    )

    print(
        f"Annual volatility: "
        f"{annual_volatility_a:.2%}"
    )

    print(
        f"Cumulative return: "
        f"{cumulative_return_a:.2%}"
    )

    print(
        f"Compounded annual return: "
        f"{compounded_annual_return_a:.2%}"
    )

    stock_a_beta = beta(
        stock_a_returns,
        market_returns,
    )

    alpha, regression_beta = (
        linear_regression(
            market_returns,
            stock_a_returns,
        )
    )

    print(
        "\n--- Market Analysis ---"
    )

    print(
        f"Stock A beta: "
        f"{stock_a_beta:.4f}"
    )

    print(
        f"Regression alpha: "
        f"{alpha:.6f}"
    )

    print(
        f"Regression beta: "
        f"{regression_beta:.4f}"
    )

    print(
        "\n--- Monte Carlo Simulation ---"
    )

    number_of_portfolios = 5000

    simulation_results = (
        simulate_portfolios(
            asset_returns=asset_returns,
            number_of_portfolios=number_of_portfolios,
            annual_risk_free_rate=annual_risk_free_rate,
            periods_per_year=252,
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

    asset_names = [
        "Stock A",
        "Stock B",
        "Stock C",
    ]

    print_portfolio(
        "Highest-Sharpe Sampled Portfolio",
        max_sharpe,
        asset_names,
    )

    print_portfolio(
        "Lowest-Volatility Sampled Portfolio",
        min_volatility,
        asset_names,
    )

    plot_path = (
        plot_monte_carlo_portfolios(
            simulation_results
        )
    )

    print(
        "\nMonte Carlo plot saved to:"
    )

    print(plot_path)


if __name__ == "__main__":
    main()