def calculate_simple_returns(prices):
    if len(prices) < 2:
        raise ValueError("At least two prices are required.")

    returns = []

    for i in range(1, len(prices)):
        previous_price = prices[i - 1]
        current_price = prices[i]

        if previous_price == 0:
            raise ValueError("Price cannot be zero.")

        return_value = (current_price - previous_price) / previous_price
        returns.append(return_value)

    return returns