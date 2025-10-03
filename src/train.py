from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

def train_logreg(df, X_cols, y_col, test_size=0.30, random_state=42):
    X = df[X_cols].values
    y = df[y_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)
    info = {
        "n_samples": int(len(df)),
        "n_features": int(len(X_cols)),
        "features": X_cols,
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist()
    }
    return model, X_te, y_te, info

def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)

def load_model(path):
    return load(path)
