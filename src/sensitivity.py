from src.backtesting import (
    rolling_backtest,
)


SUPPORTED_PARAMETERS = {
    "max_weight",
    "train_window",
    "rebalance_frequency",
    "transaction_cost_bps",
}


def parse_numeric_values(
    raw_values,
):
    if not raw_values.strip():
        raise ValueError(
            "Enter at least one sensitivity value."
        )

    cleaned = (
        raw_values
        .replace(";", ",")
        .replace("\n", ",")
    )

    tokens = []

    for section in cleaned.split(","):
        tokens.extend(
            section.split()
        )

    values = []

    for token in tokens:
        try:
            value = float(
                token
            )
        except ValueError:
            raise ValueError(
                f"Invalid numeric value: {token}"
            )

        if value not in values:
            values.append(
                value
            )

    if len(values) == 0:
        raise ValueError(
            "Enter at least one sensitivity value."
        )

    return values


def validate_sensitivity_value(
    parameter_name,
    value,
    number_of_assets,
    number_of_observations,
):
    if parameter_name not in (
        SUPPORTED_PARAMETERS
    ):
        raise ValueError(
            f"Unsupported sensitivity parameter: "
            f"{parameter_name}"
        )

    if parameter_name == "max_weight":
        if value <= 0 or value > 1:
            raise ValueError(
                "Maximum weight must be greater "
                "than 0 and no greater than 1."
            )

        minimum_feasible_weight = (
            1
            / number_of_assets
        )

        if (
            value
            < minimum_feasible_weight
            - 1e-12
        ):
            raise ValueError(
                "Maximum weight is not feasible "
                "for this number of assets."
            )

    elif parameter_name == "train_window":
        if int(value) != value:
            raise ValueError(
                "Training window must be "
                "a whole number."
            )

        if value < 2:
            raise ValueError(
                "Training window must contain "
                "at least two observations."
            )

        if value >= number_of_observations:
            raise ValueError(
                "Training window must be smaller "
                "than the available dataset."
            )

    elif parameter_name == (
        "rebalance_frequency"
    ):
        if int(value) != value:
            raise ValueError(
                "Rebalance frequency must be "
                "a whole number."
            )

        if value <= 0:
            raise ValueError(
                "Rebalance frequency must "
                "be positive."
            )

    elif parameter_name == (
        "transaction_cost_bps"
    ):
        if value < 0:
            raise ValueError(
                "Transaction costs cannot "
                "be negative."
            )


def run_sensitivity_analysis(
    asset_returns,
    benchmark_returns,
    dates,
    parameter_name,
    parameter_values,
    strategy="max_sharpe",
    train_window=504,
    rebalance_frequency=63,
    annual_risk_free_rate=0.04,
    periods_per_year=252,
    sample=True,
    max_weight=1.0,
    transaction_cost_bps=10.0,
):
    if len(asset_returns) == 0:
        raise ValueError(
            "At least one asset is required."
        )

    if len(parameter_values) == 0:
        raise ValueError(
            "At least one sensitivity value "
            "is required."
        )

    number_of_assets = len(
        asset_returns
    )

    number_of_observations = len(
        benchmark_returns
    )

    results = []

    for value in parameter_values:
        validate_sensitivity_value(
            parameter_name,
            value,
            number_of_assets,
            number_of_observations,
        )

        current_train_window = (
            train_window
        )

        current_rebalance_frequency = (
            rebalance_frequency
        )

        current_max_weight = (
            max_weight
        )

        current_transaction_cost = (
            transaction_cost_bps
        )

        if parameter_name == (
            "max_weight"
        ):
            current_max_weight = (
                float(
                    value
                )
            )

        elif parameter_name == (
            "train_window"
        ):
            current_train_window = (
                int(
                    value
                )
            )

        elif parameter_name == (
            "rebalance_frequency"
        ):
            current_rebalance_frequency = (
                int(
                    value
                )
            )

        elif parameter_name == (
            "transaction_cost_bps"
        ):
            current_transaction_cost = (
                float(
                    value
                )
            )

        backtest = rolling_backtest(
            asset_returns=(
                asset_returns
            ),
            benchmark_returns=(
                benchmark_returns
            ),
            dates=dates,
            strategy=strategy,
            train_window=(
                current_train_window
            ),
            rebalance_frequency=(
                current_rebalance_frequency
            ),
            annual_risk_free_rate=(
                annual_risk_free_rate
            ),
            periods_per_year=(
                periods_per_year
            ),
            sample=sample,
            max_weight=(
                current_max_weight
            ),
            transaction_cost_bps=(
                current_transaction_cost
            ),
        )

        metrics = (
            backtest[
                "portfolio_metrics"
            ]
        )

        results.append(
            {
                "parameter": (
                    parameter_name
                ),
                "parameter_value": (
                    value
                ),
                "total_return": (
                    metrics[
                        "total_return"
                    ]
                ),
                "cagr": (
                    metrics[
                        "annual_compounded_return"
                    ]
                ),
                "annual_return": (
                    metrics[
                        "annual_arithmetic_return"
                    ]
                ),
                "annual_volatility": (
                    metrics[
                        "annual_volatility"
                    ]
                ),
                "sharpe_ratio": (
                    metrics[
                        "sharpe_ratio"
                    ]
                ),
                "max_drawdown": (
                    metrics[
                        "max_drawdown"
                    ]
                ),
                "total_turnover": (
                    backtest[
                        "total_turnover"
                    ]
                ),
                "number_of_rebalances": len(
                    backtest[
                        "rebalances"
                    ]
                ),
            }
        )

    return results