from src.statistics import arithmetic_mean, covariance, variance


def beta(stock_returns, market_returns, sample=False):
    covariance_value = covariance(
        stock_returns,
        market_returns,
        sample=sample,
    )

    market_variance = variance(
        market_returns,
        sample=sample,
    )

    if market_variance == 0:
        raise ValueError("Beta is undefined when market variance is zero.")

    return covariance_value / market_variance


def linear_regression(x_values, y_values, sample=False):
    if len(x_values) != len(y_values):
        raise ValueError("X and Y datasets must have the same length.")

    beta_value = covariance(
        x_values,
        y_values,
        sample=sample,
    ) / variance(
        x_values,
        sample=sample,
    )

    mean_x = arithmetic_mean(x_values)
    mean_y = arithmetic_mean(y_values)

    alpha_value = mean_y - beta_value * mean_x

    return alpha_value, beta_value


def predict(alpha, beta_value, x):
    return alpha + beta_value * x