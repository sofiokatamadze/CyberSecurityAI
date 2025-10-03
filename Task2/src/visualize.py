import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_points(file_path: str, fig_path: str = "scatter.png"):
    df = pd.read_csv(file_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(df['x'], df['y'], color="blue", s=80, alpha=0.7, edgecolor="k")
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Scatter Plot of Data Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
