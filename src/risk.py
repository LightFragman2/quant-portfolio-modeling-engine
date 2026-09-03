from src.statistics import (
    arithmetic_mean,
    standard_deviation,
)

from src.annualization import (
    annualized_volatility,
)


def validate_returns(returns):
    if len(returns) == 0:
        raise ValueError(
            "Returns cannot be empty."
        )


def historical_quantile(
    values,
    probability,
):
    if len(values) == 0:
        raise ValueError(
            "Values cannot be empty."
        )

    if probability < 0 or probability > 1:
        raise ValueError(
            "Probability must be between 0 and 1."
        )

    sorted_values = sorted(
        values
    )

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (
        probability
        * (
            len(sorted_values)
            - 1
        )
    )

    lower_index = int(
        position
    )

    upper_index = min(
        lower_index + 1,
        len(sorted_values) - 1,
    )

    fraction = (
        position
        - lower_index
    )

    lower_value = (
        sorted_values[
            lower_index
        ]
    )

    upper_value = (
        sorted_values[
            upper_index
        ]
    )

    return (
        lower_value
        + fraction
        * (
            upper_value
            - lower_value
        )
    )


def historical_var(
    returns,
    confidence_level=0.95,
):
    validate_returns(
        returns
    )

    if (
        confidence_level <= 0
        or confidence_level >= 1
    ):
        raise ValueError(
            "Confidence level must be "
            "between 0 and 1."
        )

    tail_probability = (
        1
        - confidence_level
    )

    return_threshold = (
        historical_quantile(
            returns,
            tail_probability,
        )
    )

    return max(
        0.0,
        -return_threshold,
    )


def historical_cvar(
    returns,
    confidence_level=0.95,
):
    validate_returns(
        returns
    )

    tail_probability = (
        1
        - confidence_level
    )

    return_threshold = (
        historical_quantile(
            returns,
            tail_probability,
        )
    )

    tail_returns = [
        return_value
        for return_value in returns
        if return_value
        <= return_threshold
    ]

    if len(tail_returns) == 0:
        return 0.0

    average_tail_return = (
        arithmetic_mean(
            tail_returns
        )
    )

    return max(
        0.0,
        -average_tail_return,
    )


def growth_curve(
    returns,
    initial_value=1.0,
):
    validate_returns(
        returns
    )

    values = []

    current_value = (
        initial_value
    )

    for return_value in returns:
        current_value *= (
            1
            + return_value
        )

        values.append(
            current_value
        )

    return values


def drawdown_series(
    returns,
):
    values = growth_curve(
        returns
    )

    running_peak = (
        values[0]
    )

    drawdowns = []

    for value in values:
        if value > running_peak:
            running_peak = (
                value
            )

        drawdown = (
            value
            / running_peak
            - 1
        )

        drawdowns.append(
            drawdown
        )

    return drawdowns


def maximum_drawdown(
    returns,
):
    drawdowns = (
        drawdown_series(
            returns
        )
    )

    return min(
        drawdowns
    )


def rolling_volatility(
    returns,
    window=63,
    periods_per_year=252,
):
    validate_returns(
        returns
    )

    if window < 2:
        raise ValueError(
            "Rolling window must contain "
            "at least two observations."
        )

    if window > len(returns):
        raise ValueError(
            "Rolling window cannot exceed "
            "the return history."
        )

    results = []

    for end_index in range(
        window,
        len(returns) + 1,
    ):
        window_returns = (
            returns[
                end_index - window:
                end_index
            ]
        )

        period_volatility = (
            standard_deviation(
                window_returns,
                sample=True,
            )
        )

        results.append(
            annualized_volatility(
                period_volatility,
                periods_per_year,
            )
        )

    return results


def rolling_sharpe_ratio(
    returns,
    window=63,
    annual_risk_free_rate=0.04,
    periods_per_year=252,
):
    validate_returns(
        returns
    )

    if window < 2:
        raise ValueError(
            "Rolling window must contain "
            "at least two observations."
        )

    if window > len(returns):
        raise ValueError(
            "Rolling window cannot exceed "
            "the return history."
        )

    results = []

    for end_index in range(
        window,
        len(returns) + 1,
    ):
        window_returns = (
            returns[
                end_index - window:
                end_index
            ]
        )

        average_period_return = (
            arithmetic_mean(
                window_returns
            )
        )

        annual_return = (
            average_period_return
            * periods_per_year
        )

        period_volatility = (
            standard_deviation(
                window_returns,
                sample=True,
            )
        )

        annual_volatility_value = (
            annualized_volatility(
                period_volatility,
                periods_per_year,
            )
        )

        if annual_volatility_value == 0:
            results.append(
                0.0
            )

        else:
            results.append(
                (
                    annual_return
                    - annual_risk_free_rate
                )
                / annual_volatility_value
            )

    return results


def equal_weight_portfolio_returns(
    asset_returns,
):
    if len(asset_returns) == 0:
        raise ValueError(
            "At least one asset is required."
        )

    number_of_observations = len(
        asset_returns[0]
    )

    for returns in asset_returns:
        if len(returns) != number_of_observations:
            raise ValueError(
                "All assets must contain "
                "the same number of returns."
            )

    number_of_assets = len(
        asset_returns
    )

    portfolio_returns = []

    for observation_index in range(
        number_of_observations
    ):
        total_return = 0

        for returns in asset_returns:
            total_return += (
                returns[
                    observation_index
                ]
            )

        portfolio_returns.append(
            total_return
            / number_of_assets
        )

    return portfolio_returns