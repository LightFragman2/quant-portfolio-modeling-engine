from pathlib import Path

import matplotlib.pyplot as plt

from src.monte_carlo import (
    maximum_sharpe_portfolio,
    minimum_volatility_portfolio,
)


def plot_monte_carlo_portfolios(
    simulation_results,
    output_path="outputs/monte_carlo_portfolios.png",
):
    if len(simulation_results) == 0:
        raise ValueError(
            "Simulation results cannot be empty."
        )

    volatilities = []
    returns = []
    sharpe_ratios = []

    for portfolio in simulation_results:
        volatilities.append(
            portfolio["annual_volatility"]
        )

        returns.append(
            portfolio["annual_return"]
        )

        sharpe_ratios.append(
            portfolio["sharpe_ratio"]
        )

    max_sharpe = maximum_sharpe_portfolio(
        simulation_results
    )

    min_volatility = minimum_volatility_portfolio(
        simulation_results
    )

    output_file = Path(output_path)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        volatilities,
        returns,
        c=sharpe_ratios,
        cmap="viridis",
        alpha=0.5,
        s=14,
    )

    plt.scatter(
        max_sharpe["annual_volatility"],
        max_sharpe["annual_return"],
        marker="*",
        s=250,
        label="Highest Sharpe",
    )

    plt.scatter(
        min_volatility["annual_volatility"],
        min_volatility["annual_return"],
        marker="X",
        s=180,
        label="Lowest Volatility",
    )

    plt.xlabel(
        "Annualized Volatility"
    )

    plt.ylabel(
        "Annualized Expected Return"
    )

    plt.title(
        "Monte Carlo Portfolio Simulation"
    )

    colorbar = plt.colorbar(scatter)

    colorbar.set_label(
        "Sharpe Ratio"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=180,
    )

    plt.close()

    return str(output_file)