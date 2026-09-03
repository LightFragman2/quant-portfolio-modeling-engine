def portfolio_expected_return(weights, expected_returns):
    if len(weights) != len(expected_returns):
        raise ValueError("Weights and expected returns must have the same length.")

    if len(weights) == 0:
        raise ValueError("Portfolio cannot be empty.")

    portfolio_return = 0

    for i in range(len(weights)):
        portfolio_return += weights[i] * expected_returns[i]

    return portfolio_return


def two_asset_portfolio_variance(
    weight_a,
    weight_b,
    variance_a,
    variance_b,
    covariance_ab,
):
    return (
        (weight_a ** 2) * variance_a
        + (weight_b ** 2) * variance_b
        + 2 * weight_a * weight_b * covariance_ab
    )


def portfolio_volatility(portfolio_variance):
    if portfolio_variance < 0:
        raise ValueError("Portfolio variance cannot be negative.")

    return portfolio_variance ** 0.5


def sharpe_ratio(
    portfolio_return,
    risk_free_rate,
    portfolio_volatility_value,
):
    if portfolio_volatility_value == 0:
        raise ValueError("Sharpe ratio is undefined when volatility is zero.")

    excess_return = portfolio_return - risk_free_rate

    return excess_return / portfolio_volatility_value