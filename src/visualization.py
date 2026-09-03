from pathlib import Path

import matplotlib.pyplot as plt

from src.monte_carlo import (
    maximum_sharpe_portfolio as sampled_maximum_sharpe,
    minimum_volatility_portfolio as sampled_minimum_volatility,
)


def plot_monte_carlo_portfolios(
    simulation_results,
    optimized_max_sharpe=None,
    optimized_min_volatility=None,
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
            portfolio[
                "annual_volatility"
            ]
        )

        returns.append(
            portfolio[
                "annual_return"
            ]
        )

        sharpe_ratios.append(
            portfolio[
                "sharpe_ratio"
            ]
        )

    sampled_max_sharpe = (
        sampled_maximum_sharpe(
            simulation_results
        )
    )

    sampled_min_volatility = (
        sampled_minimum_volatility(
            simulation_results
        )
    )

    output_file = Path(
        output_path
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(11, 7)
    )

    scatter = plt.scatter(
        volatilities,
        returns,
        c=sharpe_ratios,
        cmap="viridis",
        alpha=0.5,
        s=14,
    )

    plt.scatter(
        sampled_max_sharpe[
            "annual_volatility"
        ],
        sampled_max_sharpe[
            "annual_return"
        ],
        marker="*",
        s=220,
        label="Monte Carlo Highest Sharpe",
    )

    plt.scatter(
        sampled_min_volatility[
            "annual_volatility"
        ],
        sampled_min_volatility[
            "annual_return"
        ],
        marker="X",
        s=160,
        label="Monte Carlo Lowest Volatility",
    )

    if optimized_max_sharpe is not None:
        plt.scatter(
            optimized_max_sharpe[
                "annual_volatility"
            ],
            optimized_max_sharpe[
                "annual_return"
            ],
            marker="P",
            s=180,
            label="Optimized Highest Sharpe",
        )

    if optimized_min_volatility is not None:
        plt.scatter(
            optimized_min_volatility[
                "annual_volatility"
            ],
            optimized_min_volatility[
                "annual_return"
            ],
            marker="D",
            s=130,
            label="Optimized Lowest Volatility",
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

    colorbar = plt.colorbar(
        scatter
    )

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

    return str(
        output_file
    )