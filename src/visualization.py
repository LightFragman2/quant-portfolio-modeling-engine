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
    efficient_frontier_points=None,
    output_path="outputs/portfolio_analysis.png",
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
        figsize=(12, 8)
    )

    scatter = plt.scatter(
        volatilities,
        returns,
        c=sharpe_ratios,
        cmap="viridis",
        alpha=0.45,
        s=14,
    )

    if (
        efficient_frontier_points
        is not None
    ):
        frontier_volatilities = []
        frontier_returns = []

        sorted_frontier = sorted(
            efficient_frontier_points,
            key=lambda portfolio: portfolio[
                "annual_volatility"
            ],
        )

        for portfolio in sorted_frontier:
            frontier_volatilities.append(
                portfolio[
                    "annual_volatility"
                ]
            )

            frontier_returns.append(
                portfolio[
                    "annual_return"
                ]
            )

        plt.plot(
            frontier_volatilities,
            frontier_returns,
            linewidth=3,
            label="Efficient Frontier",
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
        "Portfolio Optimization and Efficient Frontier"
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


def plot_backtest_results(
    backtest_results,
    benchmark_name="SPY",
    output_path="outputs/backtest_results.png",
):
    dates = (
        backtest_results[
            "dates"
        ]
    )

    portfolio_growth = (
        backtest_results[
            "portfolio_growth"
        ]
    )

    benchmark_growth = (
        backtest_results[
            "benchmark_growth"
        ]
    )

    if len(dates) == 0:
        raise ValueError(
            "Backtest contains no dates."
        )

    output_file = Path(
        output_path
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(12, 7)
    )

    plt.plot(
        dates,
        portfolio_growth,
        linewidth=2,
        label="Optimized Portfolio",
    )

    plt.plot(
        dates,
        benchmark_growth,
        linewidth=2,
        label=benchmark_name,
    )

    plt.axhline(
        y=1.0,
        linewidth=1,
        alpha=0.5,
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Growth of $1"
    )

    plt.title(
        "Out-of-Sample Portfolio Backtest"
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