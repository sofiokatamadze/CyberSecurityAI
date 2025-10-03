import argparse
from pathlib import Path
from src.train import train_logreg, save_model, load_model
from src.evaluate import evaluate_model, save_confusion_matrix_plot
from src.features import EmailFeaturizer
from src.utils import load_dataset, select_feature_columns
from src.visualize import class_balance_plot, top2_scatter_by_corr

def cmd_train(args):
    df = load_dataset(args.data)
    X_cols, y_col = select_feature_columns(df)
    model, X_test, y_test, info = train_logreg(df, X_cols, y_col, test_size=0.30, random_state=42)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    model_path = Path(args.outdir) / "logreg.joblib"
    save_model(model, model_path)

    print("=== Training complete ===")
    print(f"Used features: {X_cols}")
    print(f"Target column: {y_col}")
    print(f"Coefficients: {model.coef_.tolist()}")
    print(f"Intercept: {model.intercept_.tolist()}")
    print(f"Model saved to: {model_path}")

    metrics, cm = evaluate_model(model, X_test, y_test)
    print("\n=== Holdout evaluation (30%) ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    if args.figdir:
        Path(args.figdir).mkdir(parents=True, exist_ok=True)
        save_confusion_matrix_plot(cm, Path(args.figdir)/"confusion_matrix.png", class_names=["legit","spam"])

def cmd_eval(args):
    model = load_model(args.model)
    df = load_dataset(args.data)
    X_cols, y_col = select_feature_columns(df)
    metrics, cm = evaluate_model(model, df[X_cols].values, df[y_col].values)
    print("=== Evaluation on provided dataset ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    if args.figdir:
        Path(args.figdir).mkdir(parents=True, exist_ok=True)
        save_confusion_matrix_plot(cm, Path(args.figdir)/"confusion_matrix.png", class_names=["legit","spam"])

def cmd_predict(args):
    df = load_dataset(args.data)
    X_cols, y_col = select_feature_columns(df)
    feat = EmailFeaturizer()
    text = Path(args.email).read_text(encoding="utf-8", errors="ignore")
    features = feat.from_text(text)
    x_vec = [features.get(c, 0.0) for c in X_cols]

    model = load_model(args.model)
    # Ensure we grab P(y=1) where 1 means spam (after our utils mapping)
    classes = list(getattr(model, "classes_", [0,1]))
    if 1 in classes:
        idx = classes.index(1)
    else:
        idx = -1  # fallback

    proba = model.predict_proba([x_vec])[0][idx]
    pred = int(proba >= 0.5)
    print("=== Email prediction ===")
    print(f"Probability(spam) = {proba:.4f} -> class = {'spam' if pred==1 else 'legitimate'}")
    print("Features used (aligned to dataset columns):")
    for k in X_cols:
        print(f"  {k}: {features.get(k, 0.0)}")


def cmd_viz(args):
    df = load_dataset(args.data)
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    class_balance_plot(df, out_path=Path(args.figdir)/"class_balance.png")
    top2_scatter_by_corr(df, out_path=Path(args.figdir)/"top2_scatter.png")
    print(f"Saved figures to {args.figdir}")

def main():
    p = argparse.ArgumentParser(description="Spam vs Legitimate Email Classifier (Logistic Regression)")
    sub = p.add_subparsers(required=True)

    p_train = sub.add_parser("train", help="Train logistic regression on 70% (holds out 30%)")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--outdir", default="models")
    p_train.add_argument("--figdir", default="figs")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate a saved model on a dataset")
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--data", required=True)
    p_eval.add_argument("--figdir", default="figs")
    p_eval.set_defaults(func=cmd_eval)

    p_pred = sub.add_parser("predict", help="Predict spam/legit for a raw email text file")
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--email", required=True)
    p_pred.set_defaults(func=cmd_predict)

    p_viz = sub.add_parser("viz", help="Generate two dataset visualizations")
    p_viz.add_argument("--data", required=True)
    p_viz.add_argument("--figdir", default="figs")
    p_viz.set_defaults(func=cmd_viz)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
