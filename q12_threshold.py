import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_adult
from metrics import evaluate, evaluate_scores, predict_proba
from train import train

EPOCHS = 20
ALPHA = 1.0
VAL_SIZE = 0.15

if __name__ == "__main__":
    data = load_adult(val_size=VAL_SIZE)
    names = data["group_names"]
    test = (data["X_test"], data["y_test"], data["z_test"])

    print("\nTraining: Baseline")
    P_base, _ = train(data, debias=False, epochs=EPOCHS)
    print("\nTraining: Parity")
    P_dp, _ = train(data, mode="parity", alpha=ALPHA, epochs=EPOCHS)

    p_val = predict_proba(P_base, data["X_val"])
    z_val = data["z_val"].numpy()
    target_rate = float((p_val >= 0.5).mean())
    thr_1 = float(np.quantile(p_val[z_val == 1], 1 - target_rate))
    thr_0 = float(np.quantile(p_val[z_val == 0], 1 - target_rate))

    val_before = evaluate_scores(p_val, data["y_val"], data["z_val"], 0.5)
    val_after = evaluate_scores(p_val, data["y_val"], data["z_val"],
                                np.where(z_val == 1, thr_1, thr_0))
    print(f"\nTarget positive rate on validation: {target_rate:.4f}")
    print(f"Threshold {names[0]}: {thr_1:.4f}   Threshold {names[1]}: {thr_0:.4f}")
    print(f"Validation DP gap with threshold 0.5: {val_before['DP gap']:.4f}")
    print(f"Validation DP gap with group thresholds: {val_after['DP gap']:.4f}")

    p_test = predict_proba(P_base, data["X_test"])
    z_test = data["z_test"].numpy()
    group_thr = np.where(z_test == 1, thr_1, thr_0)

    results = pd.DataFrame({
        "Baseline threshold 0.5": evaluate(P_base, *test),
        "Baseline group thresholds": evaluate_scores(p_test, data["y_test"], data["z_test"], group_thr),
        "Adversarial parity": evaluate(P_dp, *test),
    }).T
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
    print("\nResults on the test set (gaps: lower = fairer)")
    print(results.round(4))
    results.to_csv("q12_results.csv")

    metrics = ["accuracy", "DP gap", "TPR gap", "FPR gap"]
    ax = results[metrics].T.plot(kind="bar", figsize=(9, 4), rot=0)
    ax.set_title("Group thresholds vs adversarial parity")
    ax.set_ylabel("value  (gaps: lower = fairer)")
    plt.tight_layout(); plt.savefig("q12_comparison.png", dpi=120); plt.close()
    print("\nSaved: q12_results.csv, q12_comparison.png")
