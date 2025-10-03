import pandas as pd
from scipy.stats import pearsonr

def compute_correlation(file_path: str):
    # Load CSV
    df = pd.read_csv(file_path)

    # Extract X and Y
    x = df['x']
    y = df['y']

    # Pearson correlation
    r, p_value = pearsonr(x, y)

    return r, p_value
