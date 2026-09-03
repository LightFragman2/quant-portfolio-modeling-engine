from src.statistics import variance, covariance


def portfolio_expected_return(weights, expected_returns):
    if len(weights) != len(expected_returns):
        raise ValueError(
            "Weights and expected returns must have the same length."
        )

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


def multi_asset_portfolio_variance(
    weights,
    asset_returns,
    sample=False,
):
    if len(weights) != len(asset_returns):
        raise ValueError(
            "There must be one weight for each asset."
        )

    if len(weights) == 0:
        raise ValueError("Portfolio cannot be empty.")

    portfolio_variance_value = 0

    # Individual asset variance contributions
    for i in range(len(weights)):
        asset_variance = variance(
            asset_returns[i],
            sample=sample,
        )

        portfolio_variance_value += (
            (weights[i] ** 2)
            * asset_variance
        )

    # Covariance contributions between every pair of assets
    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            covariance_value = covariance(
                asset_returns[i],
                asset_returns[j],
                sample=sample,
            )

            portfolio_variance_value += (
                2
                * weights[i]
                * weights[j]
                * covariance_value
            )

    # Floating-point calculations can occasionally produce
    # extremely tiny negative numbers close to zero.
    if (
        portfolio_variance_value < 0
        and abs(portfolio_variance_value) < 1e-15
    ):
        portfolio_variance_value = 0

    return portfolio_variance_value


def portfolio_volatility(portfolio_variance):
    if portfolio_variance < 0:
        raise ValueError(
            "Portfolio variance cannot be negative."
        )

    return portfolio_variance ** 0.5


def sharpe_ratio(
    portfolio_return,
    risk_free_rate,
    portfolio_volatility_value,
):
    if portfolio_volatility_value == 0:
        raise ValueError(
            "Sharpe ratio is undefined when volatility is zero."
        )

    excess_return = (
        portfolio_return
        - risk_free_rate
    )

    return (
        excess_return
        / portfolio_volatility_value
    )