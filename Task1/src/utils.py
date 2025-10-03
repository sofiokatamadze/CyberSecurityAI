import pandas as pd
import numpy as np

def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df

def _infer_and_fix_label_polarity(df, y_col):
    """Ensure y=1 corresponds to 'spam'. If numeric labels exist, infer and flip when needed."""
    # If textual, map directly and return
    if df[y_col].dtype == object:
        df[y_col] = df[y_col].str.lower().map({"legitimate":0, "legit":0, "ham":0, "spam":1}).astype(int)
        return df

    # Numeric labels: try to infer polarity via spammy feature(s)
    candidates = [c for c in ["spam_word_count","links","capital_words","words"] if c in df.columns]
    if not candidates:
        # fall back: assume 1=spam
        return df

    # Choose the strongest spam indicator available (priority order above)
    key = candidates[0]
    means = df.groupby(y_col)[key].mean()
    # if mean(key) for class 1 is LOWER than class 0, it's probably inverted -> flip
    if 1 in means.index and 0 in means.index and means.loc[1] < means.loc[0]:
        df[y_col] = 1 - df[y_col]
    return df

def select_feature_columns(df):
    candidates = ['is_spam','spam','label','class','target']
    y_col = None
    for c in candidates:
        if c in df.columns:
            y_col = c; break
    if y_col is None:
        raise ValueError("Could not find target column. Expected one of: " + ",".join(candidates))

    df = _infer_and_fix_label_polarity(df, y_col)

    exclude = set([y_col, "id", "idx", "email_text", "text"])
    X_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return X_cols, y_col
