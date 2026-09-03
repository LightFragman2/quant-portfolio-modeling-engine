def arithmetic_mean(values):
    if len(values) == 0:
        raise ValueError("Values cannot be empty.")

    total = 0

    for value in values:
        total += value

    return total / len(values)


def variance(values, sample=False):
    if len(values) == 0:
        raise ValueError("Values cannot be empty.")

    if sample and len(values) < 2:
        raise ValueError("Sample variance requires at least two values.")

    mean_value = arithmetic_mean(values)

    squared_deviations = 0

    for value in values:
        deviation = value - mean_value
        squared_deviations += deviation ** 2

    denominator = len(values) - 1 if sample else len(values)

    return squared_deviations / denominator


def standard_deviation(values, sample=False):
    variance_value = variance(values, sample=sample)

    return variance_value ** 0.5


def covariance(values_a, values_b, sample=False):
    if len(values_a) != len(values_b):
        raise ValueError("Both datasets must have the same length.")

    if len(values_a) == 0:
        raise ValueError("Datasets cannot be empty.")

    if sample and len(values_a) < 2:
        raise ValueError("Sample covariance requires at least two observations.")

    mean_a = arithmetic_mean(values_a)
    mean_b = arithmetic_mean(values_b)

    deviation_products = 0

    for i in range(len(values_a)):
        deviation_a = values_a[i] - mean_a
        deviation_b = values_b[i] - mean_b

        deviation_products += deviation_a * deviation_b

    denominator = len(values_a) - 1 if sample else len(values_a)

    return deviation_products / denominator


def correlation(values_a, values_b, sample=False):
    covariance_value = covariance(
        values_a,
        values_b,
        sample=sample,
    )

    std_a = standard_deviation(values_a, sample=sample)
    std_b = standard_deviation(values_b, sample=sample)

    if std_a == 0 or std_b == 0:
        raise ValueError("Correlation is undefined when standard deviation is zero.")

    return covariance_value / (std_a * std_b)