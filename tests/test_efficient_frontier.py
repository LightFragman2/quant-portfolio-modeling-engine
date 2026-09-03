import pytest

from src.optimization import (
    minimum_volatility_portfolio,
    efficient_frontier,
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


def test_efficient_frontier():
    asset_returns = (
        example_asset_returns()
    )

    frontier = efficient_frontier(
        asset_returns,
        annual_risk_free_rate=0.04,
        periods_per_year=252,
        sample=True,
        number_of_points=15,
    )

    assert len(
        frontier
    ) == 15

    for portfolio in frontier:
        assert sum(
            portfolio["weights"]
        ) == pytest.approx(
            1.0,
            abs=1e-6,
        )

        for weight in portfolio[
            "weights"
        ]:
            assert (
                0
                <= weight
                <= 1
            )

        assert (
            portfolio[
                "annual_volatility"
            ]
            >= 0
        )


def test_frontier_starts_near_minimum_volatility():
    asset_returns = (
        example_asset_returns()
    )

    min_volatility = (
        minimum_volatility_portfolio(
            asset_returns,
            annual_risk_free_rate=0.04,
        )
    )

    frontier = efficient_frontier(
        asset_returns,
        annual_risk_free_rate=0.04,
        number_of_points=10,
    )

    assert frontier[0][
        "annual_volatility"
    ] == pytest.approx(
        min_volatility[
            "annual_volatility"
        ],
        rel=1e-4,
    )


def test_frontier_returns_increase():
    asset_returns = (
        example_asset_returns()
    )

    frontier = efficient_frontier(
        asset_returns,
        annual_risk_free_rate=0.04,
        number_of_points=12,
    )

    returns = [
        portfolio[
            "annual_return"
        ]
        for portfolio in frontier
    ]

    for i in range(
        len(returns) - 1
    ):
        assert (
            returns[i + 1]
            >= returns[i]
            - 1e-7
        )