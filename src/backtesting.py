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
            "Dates and benchmark returns must have "
            "the same length."
        )

    for returns in asset_returns:
        if len(returns) != number_of_observations:
            raise ValueError(
                "All asset returns, benchmark returns, "
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
            1
            + return_value
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

    daily_volatility = (
        standard_deviation(
            returns,
            sample=True,
        )
    )

    annual_arithmetic_return = (
        annualized_arithmetic_return(
            average_return,
            periods_per_year,
        )
    )

    annual_compounded_return = (
        annualized_compounded_return(
            returns,
            periods_per_year,
        )
    )

    annual_volatility_value = (
        annualized_volatility(
            daily_volatility,
            periods_per_year,
        )
    )

    portfolio_sharpe = (
        sharpe_ratio(
            annual_arithmetic_return,
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
            annual_arithmetic_return
        ),
        "annual_compounded_return": (
            annual_compounded_return
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
            "Training window must be smaller than "
            "the available dataset."
        )

    if strategy not in (
        "max_sharpe",
        "min_volatility",
    ):
        raise ValueError(
            "Strategy must be 'max_sharpe' "
            "or 'min_volatility'."
        )

    portfolio_returns = []
    benchmark_test_returns = []
    test_dates = []
    rebalances = []

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
                )
            )

        else:
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
                )
            )

        weights = (
            optimized[
                "weights"
            ]
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
                    weights.copy()
                ),
            }
        )

        for observation_index in range(
            test_start,
            test_end,
        ):
            portfolio_daily_return = (
                calculate_portfolio_return_for_day(
                    weights,
                    asset_returns,
                    observation_index,
                )
            )

            portfolio_returns.append(
                portfolio_daily_return
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
    }