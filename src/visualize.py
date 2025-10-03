import numpy as np
import matplotlib.pyplot as plt
from .utils import select_feature_columns
import pandas as pd

def normalize_columns(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df

def class_balance_plot(df, out_path):
    df = normalize_columns(df)
    y_col = None
    for c in ['is_spam','spam','label','class','target']:
        if c in df.columns:
            y_col = c; break
    if y_col is None:
        raise ValueError("Could not find target column for class balance plot.")
    counts = df[y_col].value_counts().sort_index()
    plt.figure(figsize=(5,3))
    plt.bar([0,1], [counts.get(0,0), counts.get(1,0)])
    plt.xticks([0,1], ["legit","spam"])
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Balance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def top2_scatter_by_corr(df, out_path):
    df = normalize_columns(df)
    X_cols, y_col = select_feature_columns(df)
    y = df[y_col].values
    corrs = []
    for c in X_cols:
        if df[c].std() == 0:
            continue
        r = np.corrcoef(df[c].values, y)[0,1]
        corrs.append((abs(r), c, r))
    corrs.sort(reverse=True)
    if len(corrs) >= 2:
        c1 = corrs[0][1]; c2 = corrs[1][1]
    elif len(corrs) == 1:
        c1 = corrs[0][1]; c2 = c1
    else:
        raise ValueError("No numeric feature columns to plot.")
    plt.figure(figsize=(5,4))
    m0 = df[y_col]==0
    m1 = df[y_col]==1
    plt.scatter(df.loc[m0, c1].values, df.loc[m0, c2].values, label="legit")
    plt.scatter(df.loc[m1, c1].values, df.loc[m1, c2].values, label="spam")
    plt.xlabel(c1); plt.ylabel(c2)
    plt.title("Top-2 feature scatter by |corr with target|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
