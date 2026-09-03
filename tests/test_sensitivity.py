import pytest

import src.sensitivity as sensitivity

from src.sensitivity import (
    parse_numeric_values,
    validate_sensitivity_value,
    run_sensitivity_analysis,
)


def test_parse_numeric_values():
    result = parse_numeric_values(
        "25, 40 60; 80, 100"
    )

    assert result == [
        25.0,
        40.0,
        60.0,
        80.0,
        100.0,
    ]


def test_invalid_numeric_value():
    with pytest.raises(
        ValueError
    ):
        parse_numeric_values(
            "25, hello, 50"
        )


def test_infeasible_max_weight():
    with pytest.raises(
        ValueError
    ):
        validate_sensitivity_value(
            parameter_name=(
                "max_weight"
            ),
            value=0.40,
            number_of_assets=2,
            number_of_observations=1000,
        )


def test_invalid_parameter_name():
    with pytest.raises(
        ValueError
    ):
        validate_sensitivity_value(
            parameter_name="bad_parameter",
            value=1,
            number_of_assets=4,
            number_of_observations=1000,
        )


def test_sensitivity_changes_parameter(
    monkeypatch,
):
    calls = []

    def fake_rolling_backtest(
        **kwargs,
    ):
        calls.append(
            kwargs
        )

        return {
            "portfolio_metrics": {
                "total_return": 1.0,
                "annual_compounded_return": (
                    0.20
                ),
                "annual_arithmetic_return": (
                    0.21
                ),
                "annual_volatility": (
                    0.15
                ),
                "sharpe_ratio": (
                    1.10
                ),
                "max_drawdown": (
                    -0.18
                ),
            },
            "total_turnover": 1.5,
            "rebalances": [
                {},
                {},
            ],
        }

    monkeypatch.setattr(
        sensitivity,
        "rolling_backtest",
        fake_rolling_backtest,
    )

    asset_returns = [
        [
            0.01,
            0.02,
            -0.01,
            0.01,
            0.00,
            0.02,
        ],
        [
            0.00,
            0.01,
            0.01,
            -0.01,
            0.01,
            0.00,
        ],
    ]

    benchmark_returns = [
        0.01,
        0.01,
        0.00,
        -0.01,
        0.01,
        0.01,
    ]

    dates = list(
        range(
            6
        )
    )

    results = (
        run_sensitivity_analysis(
            asset_returns=(
                asset_returns
            ),
            benchmark_returns=(
                benchmark_returns
            ),
            dates=dates,
            parameter_name=(
                "train_window"
            ),
            parameter_values=[
                3,
                4,
            ],
            train_window=3,
            rebalance_frequency=2,
            annual_risk_free_rate=0.0,
            periods_per_year=6,
            max_weight=1.0,
            transaction_cost_bps=10,
        )
    )

    assert len(
        results
    ) == 2

    assert calls[
        0
    ][
        "train_window"
    ] == 3

    assert calls[
        1
    ][
        "train_window"
    ] == 4

    assert results[
        0
    ][
        "sharpe_ratio"
    ] == pytest.approx(
        1.10
    )