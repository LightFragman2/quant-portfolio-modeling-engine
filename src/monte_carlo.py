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


def validate_max_weight(
    number_of_assets,
    max_weight,
):
    if max_weight <= 0 or max_weight > 1:
        raise ValueError(
            "Maximum weight must be greater than 0 "
            "and no greater than 1."
        )

    if (
        number_of_assets
        * max_weight
        < 1 - 1e-12
    ):
        raise ValueError(
            "Maximum weight is too restrictive "
            "for the number of assets."
        )


def normalize_weights(
    raw_weights,
):
    if len(raw_weights) == 0:
        raise ValueError(
            "At least one weight is required."
        )

    for weight in raw_weights:
        if weight < 0:
            raise ValueError(
                "Raw weights cannot be negative."
            )

    total = sum(
        raw_weights
    )

    if total == 0:
        raise ValueError(
            "Raw weights cannot all be zero."
        )

    return [
        weight / total
        for weight in raw_weights
    ]


def construct_capped_weights(
    number_of_assets,
    random_generator,
    max_weight,
):
    asset_indices = list(
        range(
            number_of_assets
        )
    )

    random_generator.shuffle(
        asset_indices
    )

    weights = [
        0.0
    ] * number_of_assets

    remaining_weight = 1.0

    for position, asset_index in enumerate(
        asset_indices[:-1]
    ):
        assets_remaining = (
            number_of_assets
            - position
            - 1
        )

        minimum_allowed = max(
            0.0,
            remaining_weight
            - assets_remaining
            * max_weight,
        )

        maximum_allowed = min(
            max_weight,
            remaining_weight,
        )

        weight = (
            random_generator.uniform(
                minimum_allowed,
                maximum_allowed,
            )
        )

        weights[
            asset_index
        ] = weight

        remaining_weight -= (
            weight
        )

    weights[
        asset_indices[-1]
    ] = remaining_weight

    return weights


def generate_random_weights(
    number_of_assets,
    random_generator,
    max_weight=1.0,
):
    if number_of_assets <= 0:
        raise ValueError(
            "Number of assets must be positive."
        )

    validate_max_weight(
        number_of_assets,
        max_weight,
    )

    equal_weight = (
        1
        / number_of_assets
    )

    if abs(
        max_weight
        - equal_weight
    ) < 1e-12:
        return [
            equal_weight
        ] * number_of_assets

    # Exponential random variables normalized to
    # sum to 1 generate a uniform sample from the
    # long-only portfolio simplex.
    #
    # If a maximum-weight constraint exists,
    # rejection sampling keeps only valid portfolios.
    for _ in range(
        5000
    ):
        raw_weights = [
            random_generator.expovariate(
                1.0
            )
            for _ in range(
                number_of_assets
            )
        ]

        weights = (
            normalize_weights(
                raw_weights
            )
        )

        if (
            max(
                weights
            )
            <= max_weight + 1e-12
        ):
            return weights

    # Extremely tight constraints can make rejection
    # sampling inefficient, so use a guaranteed-valid
    # construction as a fallback.
    return construct_capped_weights(
        number_of_assets,
        random_generator,
        max_weight,
    )


def simulate_portfolios(
    asset_returns,
    number_of_portfolios=5000,
    annual_risk_free_rate=0.04,
    periods_per_year=252,
    sample=False,
    seed=42,
    max_weight=1.0,
):
    if len(asset_returns) == 0:
        raise ValueError(
            "At least one asset is required."
        )

    if number_of_portfolios <= 0:
        raise ValueError(
            "Number of portfolios must be positive."
        )

    number_of_assets = len(
        asset_returns
    )

    validate_max_weight(
        number_of_assets,
        max_weight,
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
            arithmetic_mean(
                returns
            )
        )

    random_generator = (
        random.Random(
            seed
        )
    )

    simulation_results = []

    for _ in range(
        number_of_portfolios
    ):
        weights = (
            generate_random_weights(
                number_of_assets,
                random_generator,
                max_weight=max_weight,
            )
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

        annual_volatility_value = (
            annualized_volatility(
                period_portfolio_volatility,
                periods_per_year,
            )
        )

        portfolio_sharpe = (
            sharpe_ratio(
                annual_return,
                annual_risk_free_rate,
                annual_volatility_value,
            )
        )

        simulation_results.append(
            {
                "weights": weights,
                "annual_return": (
                    annual_return
                ),
                "annual_volatility": (
                    annual_volatility_value
                ),
                "sharpe_ratio": (
                    portfolio_sharpe
                ),
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
        key=lambda portfolio: portfolio[
            "sharpe_ratio"
        ],
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