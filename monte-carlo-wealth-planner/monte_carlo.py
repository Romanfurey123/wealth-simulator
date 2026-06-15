"""Simple Monte Carlo simulation for investment growth."""

import argparse
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def simulate_growth(
    initial_investment: float,
    annual_contribution: float,
    years: int,
    mean_return: float,
    volatility: float,
    num_simulations: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate portfolio values after `years` using log-normal annual returns."""
    rng = np.random.default_rng(seed)

    # Draw one random annual return per year per simulation.
    annual_returns = rng.normal(
        loc=mean_return,
        scale=volatility,
        size=(num_simulations, years),
    )

    portfolio = np.full(num_simulations, initial_investment, dtype=float)

    for year in range(years):
        portfolio = portfolio * (1 + annual_returns[:, year]) + annual_contribution

    return portfolio


def summarize(final_values: np.ndarray) -> dict[str, float]:
    """Return median and 10th/90th percentile of final portfolio values."""
    median = float(np.median(final_values))
    p10, p90 = np.percentile(final_values, [10, 90])
    return {"median": median, "p10": float(p10), "p90": float(p90)}


def render_histogram_png(final_values: np.ndarray, initial_investment: float) -> bytes:
    """Build a histogram PNG and return the image bytes."""
    stats = summarize(final_values)
    median, p10, p90 = stats["median"], stats["p10"], stats["p90"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(final_values, bins=50, color="#2563eb", edgecolor="white", alpha=0.85)
    ax.axvline(initial_investment, color="#64748b", linestyle="--", linewidth=1.5, label="Initial investment")
    ax.axvline(median, color="#dc2626", linestyle="-", linewidth=2, label=f"Median: ${median:,.0f}")
    ax.axvline(p10, color="#16a34a", linestyle=":", linewidth=1.5, label=f"10th pct: ${p10:,.0f}")
    ax.axvline(p90, color="#16a34a", linestyle=":", linewidth=1.5, label=f"90th pct: ${p90:,.0f}")

    ax.set_title("Monte Carlo Investment Growth Simulation")
    ax.set_xlabel("Final portfolio value ($)")
    ax.set_ylabel("Number of simulations")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    return buffer.getvalue()


def plot_histogram(
    final_values: np.ndarray,
    initial_investment: float,
    output_path: str | None = None,
) -> None:
    """Display a histogram of simulated final portfolio values."""
    png_bytes = render_histogram_png(final_values, initial_investment)

    if output_path:
        with open(output_path, "wb") as file:
            file.write(png_bytes)
        print(f"Histogram saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo investment growth simulation")
    parser.add_argument("--initial", type=float, default=10_000, help="Starting investment ($)")
    parser.add_argument("--contribution", type=float, default=1_200, help="Annual contribution ($)")
    parser.add_argument("--years", type=int, default=30, help="Investment horizon (years)")
    parser.add_argument("--return", dest="mean_return", type=float, default=0.07, help="Expected annual return (e.g. 0.07 = 7%%)")
    parser.add_argument("--volatility", type=float, default=0.15, help="Annual return volatility (e.g. 0.15 = 15%%)")
    parser.add_argument("--simulations", type=int, default=10_000, help="Number of Monte Carlo runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the histogram image")
    args = parser.parse_args()

    final_values = simulate_growth(
        initial_investment=args.initial,
        annual_contribution=args.contribution,
        years=args.years,
        mean_return=args.mean_return,
        volatility=args.volatility,
        num_simulations=args.simulations,
        seed=args.seed,
    )

    stats = summarize(final_values)
    median, p10, p90 = stats["median"], stats["p10"], stats["p90"]

    print(f"Simulations: {args.simulations:,}")
    print(f"Horizon: {args.years} years")
    print(f"Initial investment: ${args.initial:,.0f}")
    print(f"Annual contribution: ${args.contribution:,.0f}")
    print(f"Expected return: {args.mean_return:.1%}, volatility: {args.volatility:.1%}")
    print()
    print("Results - final portfolio values")
    print(f"  Median:           ${median:,.0f}")
    print(f"  10th percentile:  ${p10:,.0f}")
    print(f"  90th percentile:  ${p90:,.0f}")
    print(f"  Probability of loss: {(final_values < args.initial).mean():.1%}")

    plot_histogram(final_values, args.initial, output_path=args.output)


if __name__ == "__main__":
    main()
