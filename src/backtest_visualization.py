from pathlib import Path

import matplotlib.pyplot as plt


def plot_backtest_comparison(
    results_by_name,
    benchmark_name="SPY",
    output_path=(
        "outputs/"
        "backtest_strategy_comparison.png"
    ),
):
    if len(results_by_name) == 0:
        raise ValueError(
            "At least one backtest result "
            "is required."
        )

    output_file = Path(
        output_path
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(13, 8)
    )

    first_result = next(
        iter(
            results_by_name.values()
        )
    )

    benchmark_dates = (
        first_result[
            "dates"
        ]
    )

    benchmark_growth = (
        first_result[
            "benchmark_growth"
        ]
    )

    for name, result in (
        results_by_name.items()
    ):
        plt.plot(
            result[
                "dates"
            ],
            result[
                "portfolio_growth"
            ],
            linewidth=2,
            label=name,
        )

    plt.plot(
        benchmark_dates,
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
        "Out-of-Sample Strategy Comparison"
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