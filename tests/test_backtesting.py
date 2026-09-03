import pytest

from src.backtesting import (
    calculate_portfolio_return_for_day,
    growth_curve,
    calculate_max_drawdown,
    backtest_metrics,
    rolling_backtest,
)


def test_daily_portfolio_return():
    asset_returns = [
        [
            0.10,
            0.20,
        ],
        [
            0.04,
            -0.02,
        ],
    ]

    result = (
        calculate_portfolio_return_for_day(
            weights=[
                0.60,
                0.40,
            ],
            asset_returns=(
                asset_returns
            ),
            observation_index=0,
        )
    )

    expected = (
        0.60 * 0.10
        + 0.40 * 0.04
    )

    assert result == pytest.approx(
        expected
    )


def test_growth_curve():
    returns = [
        0.10,
        -0.05,
        0.08,
    ]

    result = growth_curve(
        returns
    )

    assert result[0] == pytest.approx(
        1.0
    )

    assert result[-1] == pytest.approx(
        1.1286
    )


def test_max_drawdown():
    returns = [
        0.10,
        -0.20,
        0.05,
    ]

    result = (
        calculate_max_drawdown(
            returns
        )
    )

    assert result == pytest.approx(
        -0.20
    )


def test_backtest_metrics():
    returns = [
        0.01,
        -0.005,
        0.007,
        0.002,
        -0.003,
    ]

    metrics = (
        backtest_metrics(
            returns,
            annual_risk_free_rate=0.04,
        )
    )

    assert (
        "total_return"
        in metrics
    )

    assert (
        "annual_compounded_return"
        in metrics
    )

    assert (
        "annual_volatility"
        in metrics
    )

    assert (
        "sharpe_ratio"
        in metrics
    )

    assert (
        "max_drawdown"
        in metrics
    )


def test_rolling_backtest_has_no_lookahead():
    asset_returns = [
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
    ]

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
            len(
                benchmark_returns
            )
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
            strategy="max_sharpe",
            train_window=6,
            rebalance_frequency=3,
            annual_risk_free_rate=0.0,
            periods_per_year=12,
            sample=True,
        )
    )

    assert (
        result[
            "rebalances"
        ][0][
            "training_end"
        ]
        < result[
            "rebalances"
        ][0][
            "rebalance_date"
        ]
    )

    assert len(
        result[
            "portfolio_returns"
        ]
    ) == 6

    assert len(
        result[
            "benchmark_returns"
        ]
    ) == 6


def test_backtest_weights_sum_to_one():
    asset_returns = [
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
    ]

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
            strategy="min_volatility",
            train_window=6,
            rebalance_frequency=3,
            annual_risk_free_rate=0.0,
            periods_per_year=12,
            sample=True,
        )
    )

    for rebalance in result[
        "rebalances"
    ]:
        assert sum(
            rebalance[
                "weights"
            ]
        ) == pytest.approx(
            1.0,
            abs=1e-6,
        )