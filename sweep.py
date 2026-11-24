
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import load_adult
from metrics import evaluate
from train import train

ALPHAS = [0.0, 0.03, 0.1, 0.3, 1.0]
EPOCHS = 20

if __name__ == "__main__":
    data = load_adult()
    test = (data["X_test"], data["y_test"], data["z_test"])
    rows = []
    for a in ALPHAS:
        P, _ = train(data, mode="parity", alpha=a, epochs=EPOCHS, verbose=False)
        rows.append({"alpha": a, **evaluate(P, *test)})
        print(f"alpha={a}: done")

    sweep = pd.DataFrame(rows)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
    print(sweep.round(4))
    sweep.to_csv("sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep["DP gap"], sweep["accuracy"], "o-")
    for _, r in sweep.iterrows():
        ax.annotate(f"alpha={r['alpha']}", (r["DP gap"], r["accuracy"]),
                    textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("demographic parity gap  (lower = fairer)")
    ax.set_ylabel("accuracy")
    ax.set_title("Fairness vs accuracy (parity mode)")
    plt.tight_layout(); plt.savefig("tradeoff.png", dpi=120); plt.close()
    print("Saved: sweep.csv, tradeoff.png")
