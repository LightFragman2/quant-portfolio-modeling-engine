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
            "Maximum weight is too restrictive. "
            "The portfolio cannot reach weights "
            "that sum to 1."
        )


def calculate_expected_period_returns(
    asset_returns,
):
    expected_returns = []

    for returns in asset_returns:
        expected_returns.append(
            arithmetic_mean(
                returns
            )
        )

    return expected_returns


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

    expected_period_returns = (
        calculate_expected_period_returns(
            asset_returns
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
        "annual_volatility": (
            annual_volatility_value
        ),
        "sharpe_ratio": portfolio_sharpe,
    }


def clean_optimized_weights(
    weights,
    max_weight=1.0,
):
    cleaned = []

    for weight in weights:
        value = float(
            weight
        )

        if abs(value) < 1e-12:
            value = 0.0

        value = max(
            0.0,
            min(
                max_weight,
                value,
            ),
        )

        cleaned.append(
            value
        )

    total = sum(
        cleaned
    )

    if total == 0:
        raise ValueError(
            "Optimized weights cannot all be zero."
        )

    normalized = [
        weight / total
        for weight in cleaned
    ]

    for weight in normalized:
        if (
            weight
            > max_weight + 1e-6
        ):
            raise RuntimeError(
                "Optimizer returned a weight above "
                "the maximum-weight constraint."
            )

    return normalized


def create_bounds(
    number_of_assets,
    max_weight,
):
    validate_max_weight(
        number_of_assets,
        max_weight,
    )

    return [
        (
            0.0,
            max_weight,
        )
        for _ in range(
            number_of_assets
        )
    ]


def minimum_volatility_portfolio(
    asset_returns,
    annual_risk_free_rate=0.04,
    periods_per_year=252,
    sample=True,
    max_weight=1.0,
):
    validate_asset_returns(
        asset_returns
    )

    number_of_assets = len(
        asset_returns
    )

    bounds = create_bounds(
        number_of_assets,
        max_weight,
    )

    initial_weights = [
        1 / number_of_assets
    ] * number_of_assets

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
        clean_optimized_weights(
            result.x,
            max_weight,
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
    max_weight=1.0,
):
    validate_asset_returns(
        asset_returns
    )

    number_of_assets = len(
        asset_returns
    )

    bounds = create_bounds(
        number_of_assets,
        max_weight,
    )

    initial_weights = [
        1 / number_of_assets
    ] * number_of_assets

    constraints = {
        "type": "eq",
        "fun": lambda weights: (
            sum(weights) - 1
        ),
    }

    def objective(weights):
        metrics = portfolio_metrics(
            weights,
            asset_returns,
            annual_risk_free_rate,
            periods_per_year,
            sample,
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
        clean_optimized_weights(
            result.x,
            max_weight,
        )
    )

    return portfolio_metrics(
        optimized_weights,
        asset_returns,
        annual_risk_free_rate,
        periods_per_year,
        sample,
    )


def create_target_returns(
    minimum_return,
    maximum_return,
    number_of_points,
):
    if number_of_points < 2:
        raise ValueError(
            "Efficient frontier requires at least "
            "two points."
        )

    step = (
        maximum_return
        - minimum_return
    ) / (
        number_of_points - 1
    )

    return [
        minimum_return
        + i * step
        for i in range(
            number_of_points
        )
    ]


def efficient_frontier(
    asset_returns,
    annual_risk_free_rate=0.04,
    periods_per_year=252,
    sample=True,
    number_of_points=60,
    max_weight=1.0,
):
    validate_asset_returns(
        asset_returns
    )

    number_of_assets = len(
        asset_returns
    )

    bounds = create_bounds(
        number_of_assets,
        max_weight,
    )

    expected_period_returns = (
        calculate_expected_period_returns(
            asset_returns
        )
    )

    annual_asset_returns = [
        annualized_arithmetic_return(
            expected_return,
            periods_per_year,
        )
        for expected_return
        in expected_period_returns
    ]

    minimum_portfolio = (
        minimum_volatility_portfolio(
            asset_returns,
            annual_risk_free_rate,
            periods_per_year,
            sample,
            max_weight=max_weight,
        )
    )

    minimum_target_return = (
        minimum_portfolio[
            "annual_return"
        ]
    )

    maximum_target_return = max(
        annual_asset_returns
    )

    target_returns = (
        create_target_returns(
            minimum_target_return,
            maximum_target_return,
            number_of_points,
        )
    )

    frontier = []

    initial_weights = (
        minimum_portfolio[
            "weights"
        ]
    )

    def objective(weights):
        return (
            multi_asset_portfolio_variance(
                weights,
                asset_returns,
                sample=sample,
            )
        )

    for target_return in target_returns:

        def weight_constraint(
            weights,
        ):
            return (
                sum(weights) - 1
            )

        def return_constraint(
            weights,
        ):
            period_return = (
                portfolio_expected_return(
                    weights,
                    expected_period_returns,
                )
            )

            annual_return = (
                annualized_arithmetic_return(
                    period_return,
                    periods_per_year,
                )
            )

            return (
                annual_return
                - target_return
            )

        constraints = [
            {
                "type": "eq",
                "fun": weight_constraint,
            },
            {
                "type": "eq",
                "fun": return_constraint,
            },
        ]

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 2000,
                "ftol": 1e-12,
            },
        )

        if not result.success:
            continue

        optimized_weights = (
            clean_optimized_weights(
                result.x,
                max_weight,
            )
        )

        metrics = portfolio_metrics(
            optimized_weights,
            asset_returns,
            annual_risk_free_rate,
            periods_per_year,
            sample,
        )

        metrics[
            "target_return"
        ] = target_return

        frontier.append(
            metrics
        )

        initial_weights = (
            optimized_weights
        )

    if len(frontier) == 0:
        raise RuntimeError(
            "Efficient frontier optimization "
            "failed for all target returns."
        )

    return frontier