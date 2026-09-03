from src.statistics import (
    arithmetic_mean,
    standard_deviation,
)

from src.annualization import (
    annualized_arithmetic_return,
    annualized_volatility,
    annualized_compounded_return,
    cumulative_return,
)

from src.portfolio import (
    sharpe_ratio,
)

from src.optimization import (
    maximum_sharpe_portfolio,
    minimum_volatility_portfolio,
)


def validate_backtest_data(
    asset_returns,
    benchmark_returns,
    dates,
):
    if len(asset_returns) == 0:
        raise ValueError(
            "At least one asset is required."
        )

    number_of_observations = len(
        benchmark_returns
    )

    if number_of_observations == 0:
        raise ValueError(
            "Benchmark returns cannot be empty."
        )

    if len(dates) != number_of_observations:
        raise ValueError(
            "Dates and benchmark returns must "
            "have the same length."
        )

    for returns in asset_returns:
        if len(returns) != number_of_observations:
            raise ValueError(
                "All assets, benchmark returns, "
                "and dates must have matching lengths."
            )


def calculate_portfolio_return_for_day(
    weights,
    asset_returns,
    observation_index,
):
    portfolio_return = 0

    for i in range(
        len(weights)
    ):
        portfolio_return += (
            weights[i]
            * asset_returns[i][
                observation_index
            ]
        )

    return portfolio_return


def update_weights_after_returns(
    weights,
    asset_returns,
    observation_index,
):
    portfolio_return = (
        calculate_portfolio_return_for_day(
            weights,
            asset_returns,
            observation_index,
        )
    )

    portfolio_growth = (
        1 + portfolio_return
    )

    if portfolio_growth <= 0:
        raise ValueError(
            "Portfolio value fell to zero "
            "or below."
        )

    new_weights = []

    for i in range(
        len(weights)
    ):
        asset_growth = (
            1
            + asset_returns[i][
                observation_index
            ]
        )

        new_weight = (
            weights[i]
            * asset_growth
            / portfolio_growth
        )

        new_weights.append(
            new_weight
        )

    return (
        portfolio_return,
        new_weights,
    )


def calculate_turnover(
    old_weights,
    new_weights,
):
    if len(old_weights) != len(
        new_weights
    ):
        raise ValueError(
            "Weight vectors must have "
            "the same length."
        )

    total_change = 0

    for old, new in zip(
        old_weights,
        new_weights,
    ):
        total_change += abs(
            new - old
        )

    return (
        0.5
        * total_change
    )


def growth_curve(
    returns,
    initial_value=1.0,
):
    values = [
        initial_value
    ]

    current_value = (
        initial_value
    )

    for return_value in returns:
        current_value *= (
            1 + return_value
        )

        values.append(
            current_value
        )

    return values


def calculate_max_drawdown(
    returns,
):
    if len(returns) == 0:
        raise ValueError(
            "Returns cannot be empty."
        )

    values = growth_curve(
        returns
    )

    running_peak = (
        values[0]
    )

    maximum_drawdown = 0

    for value in values:
        if value > running_peak:
            running_peak = value

        drawdown = (
            value
            / running_peak
            - 1
        )

        if drawdown < maximum_drawdown:
            maximum_drawdown = (
                drawdown
            )

    return maximum_drawdown


def backtest_metrics(
    returns,
    annual_risk_free_rate=0.04,
    periods_per_year=252,
):
    if len(returns) < 2:
        raise ValueError(
            "At least two returns are required."
        )

    average_return = (
        arithmetic_mean(
            returns
        )
    )

    period_volatility = (
        standard_deviation(
            returns,
            sample=True,
        )
    )

    arithmetic_annual_return = (
        annualized_arithmetic_return(
            average_return,
            periods_per_year,
        )
    )

    compounded_annual_return = (
        annualized_compounded_return(
            returns,
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
            arithmetic_annual_return,
            annual_risk_free_rate,
            annual_volatility_value,
        )
    )

    total_return = (
        cumulative_return(
            returns
        )
    )

    maximum_drawdown = (
        calculate_max_drawdown(
            returns
        )
    )

    return {
        "total_return": total_return,
        "annual_arithmetic_return": (
            arithmetic_annual_return
        ),
        "annual_compounded_return": (
            compounded_annual_return
        ),
        "annual_volatility": (
            annual_volatility_value
        ),
        "sharpe_ratio": (
            portfolio_sharpe
        ),
        "max_drawdown": (
            maximum_drawdown
        ),
    }


def choose_target_weights(
    training_returns,
    strategy,
    annual_risk_free_rate,
    periods_per_year,
    sample,
    max_weight,
):
    number_of_assets = len(
        training_returns
    )

    if strategy == "equal_weight":
        equal_weight = (
            1
            / number_of_assets
        )

        if (
            equal_weight
            > max_weight + 1e-12
        ):
            raise ValueError(
                "Maximum weight is too low "
                "for an equal-weight portfolio."
            )

        return [
            equal_weight
        ] * number_of_assets

    if strategy == "max_sharpe":
        optimized = (
            maximum_sharpe_portfolio(
                training_returns,
                annual_risk_free_rate=(
                    annual_risk_free_rate
                ),
                periods_per_year=(
                    periods_per_year
                ),
                sample=sample,
                max_weight=max_weight,
            )
        )

        return optimized[
            "weights"
        ]

    if strategy == "min_volatility":
        optimized = (
            minimum_volatility_portfolio(
                training_returns,
                annual_risk_free_rate=(
                    annual_risk_free_rate
                ),
                periods_per_year=(
                    periods_per_year
                ),
                sample=sample,
                max_weight=max_weight,
            )
        )

        return optimized[
            "weights"
        ]

    raise ValueError(
        "Strategy must be 'max_sharpe', "
        "'min_volatility', or 'equal_weight'."
    )


def rolling_backtest(
    asset_returns,
    benchmark_returns,
    dates,
    strategy="max_sharpe",
    train_window=504,
    rebalance_frequency=63,
    annual_risk_free_rate=0.04,
    periods_per_year=252,
    sample=True,
    max_weight=1.0,
    transaction_cost_bps=0.0,
):
    validate_backtest_data(
        asset_returns,
        benchmark_returns,
        dates,
    )

    number_of_observations = len(
        benchmark_returns
    )

    if train_window < 2:
        raise ValueError(
            "Training window must contain "
            "at least two observations."
        )

    if rebalance_frequency <= 0:
        raise ValueError(
            "Rebalance frequency must be positive."
        )

    if train_window >= number_of_observations:
        raise ValueError(
            "Training window must be smaller "
            "than the available dataset."
        )

    if transaction_cost_bps < 0:
        raise ValueError(
            "Transaction cost cannot be negative."
        )

    portfolio_returns = []
    benchmark_test_returns = []
    test_dates = []
    rebalances = []

    total_turnover = 0
    total_transaction_cost_rate = 0

    current_weights = None

    test_start = (
        train_window
    )

    while (
        test_start
        < number_of_observations
    ):
        train_start = (
            test_start
            - train_window
        )

        train_end = (
            test_start
        )

        test_end = min(
            test_start
            + rebalance_frequency,
            number_of_observations,
        )

        training_returns = []

        for returns in asset_returns:
            training_returns.append(
                returns[
                    train_start:
                    train_end
                ]
            )

        target_weights = (
            choose_target_weights(
                training_returns,
                strategy,
                annual_risk_free_rate,
                periods_per_year,
                sample,
                max_weight,
            )
        )

        if current_weights is None:
            turnover = 0
        else:
            turnover = (
                calculate_turnover(
                    current_weights,
                    target_weights,
                )
            )

        transaction_cost_rate = (
            turnover
            * transaction_cost_bps
            / 10000
        )

        total_turnover += (
            turnover
        )

        total_transaction_cost_rate += (
            transaction_cost_rate
        )

        rebalances.append(
            {
                "rebalance_date": (
                    dates[
                        test_start
                    ]
                ),
                "training_start": (
                    dates[
                        train_start
                    ]
                ),
                "training_end": (
                    dates[
                        train_end - 1
                    ]
                ),
                "weights": (
                    target_weights.copy()
                ),
                "turnover": turnover,
                "transaction_cost_rate": (
                    transaction_cost_rate
                ),
            }
        )

        current_weights = (
            target_weights.copy()
        )

        for observation_index in range(
            test_start,
            test_end,
        ):
            (
                gross_portfolio_return,
                current_weights,
            ) = update_weights_after_returns(
                current_weights,
                asset_returns,
                observation_index,
            )

            if (
                observation_index
                == test_start
                and transaction_cost_rate > 0
            ):
                net_portfolio_return = (
                    (
                        1
                        - transaction_cost_rate
                    )
                    * (
                        1
                        + gross_portfolio_return
                    )
                    - 1
                )

            else:
                net_portfolio_return = (
                    gross_portfolio_return
                )

            portfolio_returns.append(
                net_portfolio_return
            )

            benchmark_test_returns.append(
                benchmark_returns[
                    observation_index
                ]
            )

            test_dates.append(
                dates[
                    observation_index
                ]
            )

        test_start = (
            test_end
        )

    portfolio_statistics = (
        backtest_metrics(
            portfolio_returns,
            annual_risk_free_rate,
            periods_per_year,
        )
    )

    benchmark_statistics = (
        backtest_metrics(
            benchmark_test_returns,
            annual_risk_free_rate,
            periods_per_year,
        )
    )

    portfolio_growth = (
        growth_curve(
            portfolio_returns
        )[1:]
    )

    benchmark_growth = (
        growth_curve(
            benchmark_test_returns
        )[1:]
    )

    number_of_costed_rebalances = max(
        0,
        len(rebalances) - 1,
    )

    if (
        number_of_costed_rebalances
        > 0
    ):
        average_turnover = (
            total_turnover
            / number_of_costed_rebalances
        )
    else:
        average_turnover = 0

    return {
        "dates": test_dates,
        "portfolio_returns": (
            portfolio_returns
        ),
        "benchmark_returns": (
            benchmark_test_returns
        ),
        "portfolio_growth": (
            portfolio_growth
        ),
        "benchmark_growth": (
            benchmark_growth
        ),
        "portfolio_metrics": (
            portfolio_statistics
        ),
        "benchmark_metrics": (
            benchmark_statistics
        ),
        "rebalances": (
            rebalances
        ),
        "strategy": strategy,
        "train_window": (
            train_window
        ),
        "rebalance_frequency": (
            rebalance_frequency
        ),
        "max_weight": (
            max_weight
        ),
        "transaction_cost_bps": (
            transaction_cost_bps
        ),
        "total_turnover": (
            total_turnover
        ),
        "average_turnover": (
            average_turnover
        ),
        "total_transaction_cost_rate": (
            total_transaction_cost_rate
        ),
    }