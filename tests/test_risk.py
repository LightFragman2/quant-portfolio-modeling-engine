import pytest

from src.risk import (
    historical_quantile,
    historical_var,
    historical_cvar,
    drawdown_series,
    maximum_drawdown,
    rolling_volatility,
    rolling_sharpe_ratio,
    equal_weight_portfolio_returns,
)


def test_historical_quantile():
    values = [
        1,
        2,
        3,
        4,
        5,
    ]

    assert historical_quantile(
        values,
        0.50,
    ) == pytest.approx(
        3.0
    )


def test_historical_var():
    returns = [
        -0.10,
        -0.05,
        0.00,
        0.02,
        0.04,
    ]

    result = historical_var(
        returns,
        confidence_level=0.80,
    )

    assert result > 0


def test_cvar_is_at_least_var():
    returns = [
        -0.15,
        -0.10,
        -0.05,
        0.00,
        0.02,
        0.03,
        0.04,
    ]

    value_at_risk = (
        historical_var(
            returns,
            confidence_level=0.80,
        )
    )

    expected_shortfall = (
        historical_cvar(
            returns,
            confidence_level=0.80,
        )
    )

    assert (
        expected_shortfall
        >= value_at_risk
    )


def test_drawdown():
    returns = [
        0.10,
        -0.20,
        0.05,
    ]

    drawdowns = (
        drawdown_series(
            returns
        )
    )

    assert drawdowns[
        1
    ] == pytest.approx(
        -0.20
    )

    assert maximum_drawdown(
        returns
    ) == pytest.approx(
        -0.20
    )


def test_rolling_risk_metrics():
    returns = [
        0.01,
        -0.01,
        0.02,
        -0.005,
        0.01,
        0.003,
    ]

    volatility = (
        rolling_volatility(
            returns,
            window=3,
        )
    )

    sharpe = (
        rolling_sharpe_ratio(
            returns,
            window=3,
            annual_risk_free_rate=0.0,
        )
    )

    assert len(
        volatility
    ) == 4

    assert len(
        sharpe
    ) == 4


def test_equal_weight_returns():
    asset_returns = [
        [
            0.10,
            0.00,
        ],
        [
            0.00,
            0.10,
        ],
    ]

    result = (
        equal_weight_portfolio_returns(
            asset_returns
        )
    )

    assert result == pytest.approx(
        [
            0.05,
            0.05,
        ]
    )