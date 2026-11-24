import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import load_adult, set_seed
from metrics import evaluate, predict_proba
from train import train

EPOCHS = 20
ALPHA = 1.0


def plot_scores(models, data, path):
    z = data["z_test"].numpy()
    names = data["group_names"]
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4))
    for ax, (title, P) in zip(axes, models):
        p = predict_proba(P, data["X_test"])
        ax.hist(p[z == 1], bins=30, range=(0, 1), alpha=0.5, density=True, label=names[0])
        ax.hist(p[z == 0], bins=30, range=(0, 1), alpha=0.5, density=True, label=names[1])
        ax.set_title(f"{title}: predicted probability")
        ax.set_xlabel("sigmoid of predictor output")
        ax.set_ylabel("density")
        ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protected", default="sex", choices=["sex", "race"])
    args = parser.parse_args()
    tag = "" if args.protected == "sex" else "_" + args.protected

    set_seed()
    data = load_adult(protected=args.protected)
    print("Using device:", data["device"])
    test = (data["X_test"], data["y_test"], data["z_test"])

    print("\nBaseline (no debiasing)")
    P_base, hist_base = train(data, debias=False, epochs=EPOCHS)

    print("\nDebiased: demographic parity")
    P_dp, hist_dp = train(data, mode="parity", alpha=ALPHA, epochs=EPOCHS)

    print("\nDebiased: equalized odds")
    P_eo, hist_eo = train(data, mode="eq_odds", alpha=ALPHA, epochs=EPOCHS)

    results = pd.DataFrame({
        "Baseline":              evaluate(P_base, *test),
        "Debiased: parity":      evaluate(P_dp, *test),
        "Debiased: equal. odds": evaluate(P_eo, *test),
    }).T
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
    print("\nResults on the test set (gaps: lower = fairer)")
    print(results.round(4))
    results.to_csv(f"results{tag}.csv")

    metrics = ["accuracy", "DP gap", "TPR gap", "FPR gap"]
    ax = results[metrics].T.plot(kind="bar", figsize=(9, 4), rot=0)
    ax.set_title("Baseline vs adversarial debiasing")
    ax.set_ylabel("value  (gaps: lower = fairer)")
    plt.tight_layout(); plt.savefig(f"comparison{tag}.png", dpi=120); plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for h, name in [(hist_base, "Baseline"), (hist_dp, "Parity"), (hist_eo, "Equal. odds")]:
        for ax, m in zip(axes, metrics):
            ax.plot(h["epoch"], h[m], label=name)
    for ax, m in zip(axes, metrics):
        ax.set_title(f"Test {m}"); ax.set_xlabel("epoch"); ax.legend()
    plt.tight_layout(); plt.savefig(f"curves{tag}.png", dpi=120); plt.close()

    plot_scores([("Baseline", P_base), ("Parity", P_dp)], data, f"scores{tag}.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist_dp["epoch"], hist_dp["loss_A"], label="Parity")
    ax.plot(hist_eo["epoch"], hist_eo["loss_A"], label="Equal. odds")
    ax.set_title("Adversary loss L_A per epoch")
    ax.set_xlabel("epoch"); ax.set_ylabel("L_A"); ax.legend()
    plt.tight_layout(); plt.savefig(f"adversary_loss{tag}.png", dpi=120); plt.close()

    print(f"\nSaved: results{tag}.csv, comparison{tag}.png, curves{tag}.png, "
          f"scores{tag}.png, adversary_loss{tag}.png")
