import pytest

from src.returns import calculate_simple_returns
from src.statistics import (
    arithmetic_mean,
    variance,
    standard_deviation,
    covariance,
    correlation,
)
from src.portfolio import (
    portfolio_expected_return,
    two_asset_portfolio_variance,
    portfolio_volatility,
    sharpe_ratio,
)
from src.regression import beta, linear_regression

from src.annualization import (
    annualized_arithmetic_return,
    annualized_volatility,
    cumulative_return,
    annualized_compounded_return,
)

def test_simple_returns():
    prices = [100, 105, 102]

    result = calculate_simple_returns(prices)

    assert result[0] == pytest.approx(0.05)
    assert result[1] == pytest.approx(-0.0285714286)


def test_arithmetic_mean():
    values = [0.08, -0.04, 0.06, 0.02]

    assert arithmetic_mean(values) == pytest.approx(0.03)


def test_variance():
    values = [2, 4, 6]

    assert variance(values) == pytest.approx(8 / 3)


def test_standard_deviation():
    values = [2, 4, 6]

    assert standard_deviation(values) == pytest.approx((8 / 3) ** 0.5)


def test_covariance():
    a = [2, 4, 6]
    b = [8, 6, 4]

    assert covariance(a, b) == pytest.approx(-8 / 3)


def test_correlation():
    a = [1, 2, 3]
    b = [2, 4, 6]

    assert correlation(a, b) == pytest.approx(1.0)


def test_portfolio_expected_return():
    weights = [0.70, 0.30]
    expected_returns = [0.10, 0.04]

    result = portfolio_expected_return(weights, expected_returns)

    assert result == pytest.approx(0.082)


def test_two_asset_portfolio_variance():
    result = two_asset_portfolio_variance(
        weight_a=0.60,
        weight_b=0.40,
        variance_a=100,
        variance_b=25,
        covariance_ab=0,
    )

    assert result == pytest.approx(40)


def test_portfolio_volatility():
    assert portfolio_volatility(64) == pytest.approx(8)


def test_sharpe_ratio():
    result = sharpe_ratio(
        portfolio_return=0.10,
        risk_free_rate=0.04,
        portfolio_volatility_value=0.10,
    )

    assert result == pytest.approx(0.60)


def test_beta():
    market = [-2, 0, 2]
    stock = [-3, 1, 5]

    assert beta(stock, market) == pytest.approx(2)


def test_linear_regression():
    market = [-2, 0, 2]
    stock = [-3, 1, 5]

    alpha, beta_value = linear_regression(
        market,
        stock,
    )

    assert alpha == pytest.approx(1)
    assert beta_value == pytest.approx(2)

def test_annualized_arithmetic_return():
    result = annualized_arithmetic_return(
        average_period_return=0.0006,
        periods_per_year=252,
    )

    assert result == pytest.approx(0.1512)


def test_annualized_volatility():
    result = annualized_volatility(
        period_volatility=0.012,
        periods_per_year=252,
    )

    assert result == pytest.approx(
        0.012 * (252 ** 0.5)
    )


def test_cumulative_return():
    returns = [0.05, -0.02, 0.04]

    result = cumulative_return(returns)

    assert result == pytest.approx(0.07016)


def test_annualized_compounded_return():
    # 10% growth over six months.
    # Using monthly periods: 12 periods/year, 6 observed periods.
    monthly_growth_rate = (1.10 ** (1 / 6)) - 1

    returns = [monthly_growth_rate] * 6

    result = annualized_compounded_return(
        returns,
        periods_per_year=12,
    )

    assert result == pytest.approx(0.21)