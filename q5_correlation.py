import argparse

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data import load_adult

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protected", default="sex", choices=["sex", "race"])
    args = parser.parse_args()

    data = load_adult(protected=args.protected)
    X, y, z = [t.numpy() for t in data["train_ds"].tensors]
    X_test, z_test = data["X_test"].cpu().numpy(), data["z_test"].numpy()

    frame = pd.DataFrame(X, columns=data["feature_names"])
    table = pd.DataFrame({
        "corr with z": frame.corrwith(pd.Series(z)),
        "corr with y": frame.corrwith(pd.Series(y)),
    }).dropna()
    table = table.reindex(table["corr with z"].abs().sort_values(ascending=False).index)

    pd.set_option("display.width", 200); pd.set_option("display.max_rows", None)
    print(f"\nFeatures most correlated with {args.protected} (z)")
    print(table.head(15).round(4))
    table.to_csv(f"q5_correlations_{args.protected}.csv")

    clf = LogisticRegression(max_iter=2000).fit(X, z)
    prob = clf.predict_proba(X_test)[:, 1]
    majority = max(z_test.mean(), 1 - z_test.mean())
    print(f"\nPredicting z from the features with logistic regression")
    print(f"test accuracy {clf.score(X_test, z_test):.4f}   majority class accuracy {majority:.4f}")
    print(f"test AUC {roc_auc_score(z_test, prob):.4f}")
