from scipy.optimize import minimize

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


def validate_asset_returns(asset_returns):
    if len(asset_returns) == 0:
        raise ValueError(
            "At least one asset is required."
        )

    number_of_observations = len(
        asset_returns[0]
    )

    if number_of_observations < 2:
        raise ValueError(
            "At least two return observations are required."
        )

    for returns in asset_returns:
        if len(returns) != number_of_observations:
            raise ValueError(
                "All assets must have the same number "
                "of return observations."
            )


def portfolio_metrics(
    weights,
    asset_returns,
    annual_risk_free_rate=0.04,
    periods_per_year=252,
    sample=True,
):
    validate_asset_returns(
        asset_returns
    )

    expected_period_returns = []

    for returns in asset_returns:
        expected_period_returns.append(
            arithmetic_mean(
                returns
            )
        )

    period_return = (
        portfolio_expected_return(
            weights,
            expected_period_returns,
        )
    )

    period_variance = (
        multi_asset_portfolio_variance(
            weights,
            asset_returns,
            sample=sample,
        )
    )

    period_volatility = (
        portfolio_volatility(
            period_variance
        )
    )

    annual_return = (
        annualized_arithmetic_return(
            period_return,
            periods_per_year,
        )
    )

    annual_volatility_value = (
        annualized_volatility(
            period_volatility,
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

    return {
        "weights": [
            float(weight)
            for weight in weights
        ],
        "annual_return": annual_return,
        "annual_volatility": annual_volatility_value,
        "sharpe_ratio": portfolio_sharpe,
    }


def normalize_optimized_weights(
    weights,
):
    cleaned_weights = []

    for weight in weights:
        cleaned_weight = max(
            0.0,
            min(
                1.0,
                float(weight),
            ),
        )

        cleaned_weights.append(
            cleaned_weight
        )

    total = sum(
        cleaned_weights
    )

    if total == 0:
        raise ValueError(
            "Optimized weights cannot all be zero."
        )

    return [
        weight / total
        for weight in cleaned_weights
    ]


def minimum_volatility_portfolio(
    asset_returns,
    annual_risk_free_rate=0.04,
    periods_per_year=252,
    sample=True,
):
    validate_asset_returns(
        asset_returns
    )

    number_of_assets = len(
        asset_returns
    )

    initial_weights = [
        1 / number_of_assets
    ] * number_of_assets

    bounds = [
        (0.0, 1.0)
        for _ in range(
            number_of_assets
        )
    ]

    constraints = {
        "type": "eq",
        "fun": lambda weights: (
            sum(weights) - 1
        ),
    }

    def objective(weights):
        return (
            multi_asset_portfolio_variance(
                weights,
                asset_returns,
                sample=sample,
            )
        )

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Minimum-volatility optimization failed: "
            + result.message
        )

    optimized_weights = (
        normalize_optimized_weights(
            result.x
        )
    )

    return portfolio_metrics(
        optimized_weights,
        asset_returns,
        annual_risk_free_rate,
        periods_per_year,
        sample,
    )


def maximum_sharpe_portfolio(
    asset_returns,
    annual_risk_free_rate=0.04,
    periods_per_year=252,
    sample=True,
):
    validate_asset_returns(
        asset_returns
    )

    number_of_assets = len(
        asset_returns
    )

    initial_weights = [
        1 / number_of_assets
    ] * number_of_assets

    bounds = [
        (0.0, 1.0)
        for _ in range(
            number_of_assets
        )
    ]

    constraints = {
        "type": "eq",
        "fun": lambda weights: (
            sum(weights) - 1
        ),
    }

    def objective(weights):
        metrics = (
            portfolio_metrics(
                weights,
                asset_returns,
                annual_risk_free_rate,
                periods_per_year,
                sample,
            )
        )

        return -metrics[
            "sharpe_ratio"
        ]

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Maximum-Sharpe optimization failed: "
            + result.message
        )

    optimized_weights = (
        normalize_optimized_weights(
            result.x
        )
    )

    return portfolio_metrics(
        optimized_weights,
        asset_returns,
        annual_risk_free_rate,
        periods_per_year,
        sample,
    )