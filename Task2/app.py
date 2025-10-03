import argparse
from src.correlation import compute_correlation
from src.visualize import plot_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlation Analysis")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--plot", type=str, default="scatter.png", help="Output figure file")
    args = parser.parse_args()

    # Compute correlation
    r, p = compute_correlation(args.data)
    print(f"Pearson correlation coefficient: {r:.4f}")
    print(f"P-value: {p:.4f}")

    # Plot scatter
    plot_points(args.data, args.plot)
    print(f"Scatter plot saved to {args.plot}")
