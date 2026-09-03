import pytest

from src.optimization import (
    portfolio_metrics,
    minimum_volatility_portfolio,
    maximum_sharpe_portfolio,
)


def example_asset_returns():
    return [
        [
            0.010,
            -0.005,
            0.012,
            0.004,
            -0.002,
            0.009,
            0.006,
            -0.003,
        ],
        [
            0.004,
            0.003,
            -0.002,
            0.005,
            0.001,
            0.004,
            0.002,
            0.003,
        ],
        [
            0.015,
            -0.012,
            0.020,
            -0.005,
            0.018,
            0.011,
            -0.009,
            0.016,
        ],
    ]


def test_portfolio_metrics():
    asset_returns = (
        example_asset_returns()
    )

    metrics = portfolio_metrics(
        weights=[
            0.40,
            0.40,
            0.20,
        ],
        asset_returns=asset_returns,
        annual_risk_free_rate=0.04,
    )

    assert sum(
        metrics["weights"]
    ) == pytest.approx(
        1.0
    )

    assert (
        metrics[
            "annual_volatility"
        ]
        > 0
    )


def test_minimum_volatility_optimizer():
    asset_returns = (
        example_asset_returns()
    )

    equal_weight_metrics = (
        portfolio_metrics(
            weights=[
                1 / 3,
                1 / 3,
                1 / 3,
            ],
            asset_returns=asset_returns,
            annual_risk_free_rate=0.04,
        )
    )

    optimized = (
        minimum_volatility_portfolio(
            asset_returns,
            annual_risk_free_rate=0.04,
        )
    )

    assert sum(
        optimized["weights"]
    ) == pytest.approx(
        1.0
    )

    for weight in optimized[
        "weights"
    ]:
        assert 0 <= weight <= 1

    assert (
        optimized[
            "annual_volatility"
        ]
        <= equal_weight_metrics[
            "annual_volatility"
        ] + 1e-8
    )


def test_maximum_sharpe_optimizer():
    asset_returns = (
        example_asset_returns()
    )

    equal_weight_metrics = (
        portfolio_metrics(
            weights=[
                1 / 3,
                1 / 3,
                1 / 3,
            ],
            asset_returns=asset_returns,
            annual_risk_free_rate=0.04,
        )
    )

    optimized = (
        maximum_sharpe_portfolio(
            asset_returns,
            annual_risk_free_rate=0.04,
        )
    )

    assert sum(
        optimized["weights"]
    ) == pytest.approx(
        1.0
    )

    for weight in optimized[
        "weights"
    ]:
        assert 0 <= weight <= 1

    assert (
        optimized[
            "sharpe_ratio"
        ]
        >= equal_weight_metrics[
            "sharpe_ratio"
        ] - 1e-8
    )