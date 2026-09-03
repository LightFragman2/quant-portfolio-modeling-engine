def annualized_arithmetic_return(
    average_period_return,
    periods_per_year=252,
):
    return average_period_return * periods_per_year


def annualized_volatility(
    period_volatility,
    periods_per_year=252,
):
    if period_volatility < 0:
        raise ValueError("Volatility cannot be negative.")

    return period_volatility * (periods_per_year ** 0.5)


def cumulative_return(returns):
    if len(returns) == 0:
        raise ValueError("Returns cannot be empty.")

    growth_factor = 1.0

    for return_value in returns:
        growth_factor *= 1 + return_value

    return growth_factor - 1


def annualized_compounded_return(
    returns,
    periods_per_year=252,
):
    if len(returns) == 0:
        raise ValueError("Returns cannot be empty.")

    growth_factor = 1.0

    for return_value in returns:
        growth_factor *= 1 + return_value

    number_of_periods = len(returns)

    annualization_exponent = periods_per_year / number_of_periods

    return growth_factor ** annualization_exponent - 1