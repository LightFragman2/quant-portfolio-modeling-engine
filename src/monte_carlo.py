import random

from src.statistics import arithmetic_mean

from src.portfolio import (
    portfolio_expected_return,
    multi_asset_portfolio_variance,
    portfolio_volatility,
    sharpe_ratio,
)

from src.annualization import (
    annualized_arithmetic_return,
    annualized_volatility,
)


def normalize_weights(raw_weights):
    if len(raw_weights) == 0:
        raise ValueError(
            "At least one weight is required."
        )

    for weight in raw_weights:
        if weight < 0:
            raise ValueError(
                "Raw weights cannot be negative."
            )

    total = sum(raw_weights)

    if total == 0:
        raise ValueError(
            "Raw weights cannot all be zero."
        )

    normalized_weights = []

    for weight in raw_weights:
        normalized_weights.append(
            weight / total
        )

    return normalized_weights


def generate_random_weights(
    number_of_assets,
    random_generator,
):
    if number_of_assets <= 0:
        raise ValueError(
            "Number of assets must be positive."
        )

    while True:
        raw_weights = []

        for _ in range(number_of_assets):
            raw_weights.append(
                random_generator.random()
            )

        if sum(raw_weights) > 0:
            return normalize_weights(
                raw_weights
            )


def simulate_portfolios(
    asset_returns,
    number_of_portfolios=5000,
    annual_risk_free_rate=0.04,
    periods_per_year=252,
    sample=False,
    seed=42,
):
    if len(asset_returns) == 0:
        raise ValueError(
            "At least one asset is required."
        )

    if number_of_portfolios <= 0:
        raise ValueError(
            "Number of portfolios must be positive."
        )

    number_of_observations = len(
        asset_returns[0]
    )

    if number_of_observations == 0:
        raise ValueError(
            "Asset return histories cannot be empty."
        )

    for returns in asset_returns:
        if len(returns) != number_of_observations:
            raise ValueError(
                "All assets must have the same number "
                "of return observations."
            )

    expected_returns = []

    for returns in asset_returns:
        expected_returns.append(
            arithmetic_mean(returns)
        )

    random_generator = random.Random(seed)

    simulation_results = []

    for _ in range(number_of_portfolios):
        weights = generate_random_weights(
            number_of_assets=len(asset_returns),
            random_generator=random_generator,
        )

        period_portfolio_return = (
            portfolio_expected_return(
                weights,
                expected_returns,
            )
        )

        period_portfolio_variance = (
            multi_asset_portfolio_variance(
                weights,
                asset_returns,
                sample=sample,
            )
        )

        period_portfolio_volatility = (
            portfolio_volatility(
                period_portfolio_variance
            )
        )

        annual_return = (
            annualized_arithmetic_return(
                period_portfolio_return,
                periods_per_year,
            )
        )

        annual_volatility = (
            annualized_volatility(
                period_portfolio_volatility,
                periods_per_year,
            )
        )

        portfolio_sharpe = sharpe_ratio(
            annual_return,
            annual_risk_free_rate,
            annual_volatility,
        )

        simulation_results.append(
            {
                "weights": weights,
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
                "sharpe_ratio": portfolio_sharpe,
            }
        )

    return simulation_results


def maximum_sharpe_portfolio(
    simulation_results,
):
    if len(simulation_results) == 0:
        raise ValueError(
            "Simulation results cannot be empty."
        )

    return max(
        simulation_results,
        key=lambda portfolio: portfolio["sharpe_ratio"],
    )


def minimum_volatility_portfolio(
    simulation_results,
):
    if len(simulation_results) == 0:
        raise ValueError(
            "Simulation results cannot be empty."
        )

    return min(
        simulation_results,
        key=lambda portfolio: portfolio[
            "annual_volatility"
        ],
    )