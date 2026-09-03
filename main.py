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
    two_asset_portfolio_variance,
    portfolio_volatility,
    sharpe_ratio,
)
from src.regression import beta, linear_regression

from src.annualization import (
    annualized_arithmetic_return,
    annualized_volatility,
    cumulative_return,
    annualized_compounded_return,
)


def main():
    print("=== Quant Portfolio Modeling Engine ===")

    stock_a_prices = [100, 104, 102, 108, 112]
    stock_b_prices = [50, 51, 53, 52, 55]
    market_prices = [200, 204, 202, 207, 211]

    stock_a_returns = calculate_simple_returns(stock_a_prices)
    stock_b_returns = calculate_simple_returns(stock_b_prices)
    market_returns = calculate_simple_returns(market_prices)

    print("\n--- Returns ---")
    print("Stock A:", stock_a_returns)
    print("Stock B:", stock_b_returns)
    print("Market:", market_returns)

    mean_a = arithmetic_mean(stock_a_returns)
    mean_b = arithmetic_mean(stock_b_returns)

    variance_a = variance(stock_a_returns)
    variance_b = variance(stock_b_returns)

    volatility_a = standard_deviation(stock_a_returns)
    volatility_b = standard_deviation(stock_b_returns)

    covariance_ab = covariance(stock_a_returns, stock_b_returns)
    correlation_ab = correlation(stock_a_returns, stock_b_returns)

    annual_return_a = annualized_arithmetic_return(mean_a)
    annual_volatility_a = annualized_volatility(volatility_a)
    cumulative_return_a = cumulative_return(stock_a_returns)
    compounded_annual_return_a = annualized_compounded_return(stock_a_returns)

    print("\n--- Asset Statistics ---")
    print(f"Stock A mean return: {mean_a:.4%}")
    print(f"Stock B mean return: {mean_b:.4%}")

    print(f"Stock A variance: {variance_a:.8f}")
    print(f"Stock B variance: {variance_b:.8f}")

    print(f"Stock A volatility: {volatility_a:.4%}")
    print(f"Stock B volatility: {volatility_b:.4%}")

    print(f"Covariance: {covariance_ab:.8f}")
    print(f"Correlation: {correlation_ab:.4f}")

    weights = [0.60, 0.40]
    expected_returns = [mean_a, mean_b]

    expected_portfolio_return = portfolio_expected_return(
        weights,
        expected_returns,
    )

    portfolio_variance_value = two_asset_portfolio_variance(
        weight_a=weights[0],
        weight_b=weights[1],
        variance_a=variance_a,
        variance_b=variance_b,
        covariance_ab=covariance_ab,
    )

    portfolio_volatility_value = portfolio_volatility(
        portfolio_variance_value
    )

    risk_free_rate = 0.01

    portfolio_sharpe = sharpe_ratio(
        expected_portfolio_return,
        risk_free_rate,
        portfolio_volatility_value,
    )

    print("\n--- Portfolio ---")
    print(f"Expected return: {expected_portfolio_return:.4%}")
    print(f"Variance: {portfolio_variance_value:.8f}")
    print(f"Volatility: {portfolio_volatility_value:.4%}")
    print(f"Sharpe ratio: {portfolio_sharpe:.4f}")

    stock_a_beta = beta(stock_a_returns, market_returns)

    alpha, regression_beta = linear_regression(
        market_returns,
        stock_a_returns,
    )

    print("\n--- Market Analysis ---")
    print(f"Stock A beta: {stock_a_beta:.4f}")
    print(f"Regression alpha: {alpha:.6f}")
    print(f"Regression beta: {regression_beta:.4f}")

    print("\n--- Annualized Statistics ---")
    print(f"Stock A arithmetic annualized return: {annual_return_a:.4%}")
    print(f"Stock A annualized volatility: {annual_volatility_a:.4%}")
    print(f"Stock A cumulative return: {cumulative_return_a:.4%}")
    print(f"Stock A compounded annualized return: "f"{compounded_annual_return_a:.4%}")


if __name__ == "__main__":
    main()