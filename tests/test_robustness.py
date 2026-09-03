import pytest

from src.optimization import (
    maximum_sharpe_portfolio,
)

from src.backtesting import (
    calculate_turnover,
    update_weights_after_returns,
    rolling_backtest,
)


def example_returns():
    return [
        [
            0.010,
            0.004,
            -0.003,
            0.008,
            0.002,
            0.006,
            -0.002,
            0.005,
            0.004,
            0.003,
            0.007,
            -0.001,
        ],
        [
            0.003,
            0.004,
            0.002,
            0.003,
            0.004,
            0.002,
            0.003,
            0.004,
            0.002,
            0.003,
            0.004,
            0.002,
        ],
        [
            0.020,
            -0.010,
            0.018,
            -0.005,
            0.016,
            0.011,
            -0.008,
            0.015,
            0.010,
            -0.004,
            0.012,
            0.008,
        ],
    ]


def test_maximum_weight_constraint():
    asset_returns = (
        example_returns()
    )

    result = (
        maximum_sharpe_portfolio(
            asset_returns,
            annual_risk_free_rate=0.0,
            periods_per_year=12,
            max_weight=0.40,
        )
    )

    assert sum(
        result[
            "weights"
        ]
    ) == pytest.approx(
        1.0,
        abs=1e-6,
    )

    for weight in result[
        "weights"
    ]:
        assert (
            weight
            <= 0.40 + 1e-6
        )


def test_infeasible_weight_constraint():
    with pytest.raises(
        ValueError
    ):
        maximum_sharpe_portfolio(
            example_returns(),
            annual_risk_free_rate=0.0,
            periods_per_year=12,
            max_weight=0.30,
        )


def test_turnover():
    old_weights = [
        0.50,
        0.50,
    ]

    new_weights = [
        1.00,
        0.00,
    ]

    result = (
        calculate_turnover(
            old_weights,
            new_weights,
        )
    )

    assert result == pytest.approx(
        0.50
    )


def test_weights_drift():
    asset_returns = [
        [
            0.10,
        ],
        [
            0.00,
        ],
    ]

    (
        portfolio_return,
        new_weights,
    ) = update_weights_after_returns(
        weights=[
            0.50,
            0.50,
        ],
        asset_returns=(
            asset_returns
        ),
        observation_index=0,
    )

    assert portfolio_return == pytest.approx(
        0.05
    )

    assert new_weights[
        0
    ] > 0.50

    assert sum(
        new_weights
    ) == pytest.approx(
        1.0
    )


def test_capped_backtest_respects_limit():
    asset_returns = (
        example_returns()
    )

    benchmark_returns = [
        0.004,
        0.003,
        0.002,
        0.004,
        0.003,
        0.002,
        0.004,
        0.003,
        0.002,
        0.004,
        0.003,
        0.002,
    ]

    dates = list(
        range(
            12
        )
    )

    result = (
        rolling_backtest(
            asset_returns=(
                asset_returns
            ),
            benchmark_returns=(
                benchmark_returns
            ),
            dates=dates,
            strategy=(
                "max_sharpe"
            ),
            train_window=6,
            rebalance_frequency=3,
            annual_risk_free_rate=0.0,
            periods_per_year=12,
            max_weight=0.40,
        )
    )

    for rebalance in result[
        "rebalances"
    ]:
        for weight in rebalance[
            "weights"
        ]:
            assert (
                weight
                <= 0.40 + 1e-6
            )


def test_transaction_costs_reduce_return():
    asset_returns = (
        example_returns()
    )

    benchmark_returns = [
        0.004,
        0.003,
        0.002,
        0.004,
        0.003,
        0.002,
        0.004,
        0.003,
        0.002,
        0.004,
        0.003,
        0.002,
    ]

    dates = list(
        range(
            12
        )
    )

    no_cost = rolling_backtest(
        asset_returns=(
            asset_returns
        ),
        benchmark_returns=(
            benchmark_returns
        ),
        dates=dates,
        strategy=(
            "equal_weight"
        ),
        train_window=6,
        rebalance_frequency=2,
        annual_risk_free_rate=0.0,
        periods_per_year=12,
        transaction_cost_bps=0,
    )

    with_cost = rolling_backtest(
        asset_returns=(
            asset_returns
        ),
        benchmark_returns=(
            benchmark_returns
        ),
        dates=dates,
        strategy=(
            "equal_weight"
        ),
        train_window=6,
        rebalance_frequency=2,
        annual_risk_free_rate=0.0,
        periods_per_year=12,
        transaction_cost_bps=100,
    )

    assert (
        with_cost[
            "portfolio_metrics"
        ][
            "total_return"
        ]
        <= no_cost[
            "portfolio_metrics"
        ][
            "total_return"
        ]
    )