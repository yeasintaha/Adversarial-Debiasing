import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import load_adult
from metrics import evaluate
from train import train

EPOCHS = 20
ALPHA = 1.0
ADV_HIDDEN = 32
CURVES = ["accuracy", "DP gap", "TPR gap", "FPR gap", "loss_A"]

cache = {}


def run(data, name, **kwargs):
    if name not in cache:
        print(f"\nTraining: {name}")
        cache[name] = train(data, epochs=EPOCHS, **kwargs)
    return cache[name]


def baseline(data):
    return run(data, "Baseline", debias=False)


def parity(data):
    return run(data, "Parity", mode="parity", alpha=ALPHA)


def equal_odds(data):
    return run(data, "Equal odds", mode="eq_odds", alpha=ALPHA)


def stability(hist):
    return {
        "final L_A": hist["loss_A"].iloc[-1],
        "L_A std last 10": hist["loss_A"].tail(10).std(),
        "DP std last 10": hist["DP gap"].tail(10).std(),
        "DP mean abs change": hist["DP gap"].diff().abs().mean(),
        "TPR mean abs change": hist["TPR gap"].diff().abs().mean(),
    }


def report(tag, items, test):
    rows = {}
    for name, (P, hist) in items.items():
        rows[name] = {**evaluate(P, *test), **stability(hist)}
    table = pd.DataFrame(rows).T
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    print(f"\nResults for {tag} on the test set")
    print(table.round(4))
    table.to_csv(f"{tag}.csv")

    fig, axes = plt.subplots(1, len(CURVES), figsize=(22, 4))
    for name, (P, hist) in items.items():
        for ax, m in zip(axes, CURVES):
            ax.plot(hist["epoch"], hist[m], label=name)
    for ax, m in zip(axes, CURVES):
        ax.set_title(f"Test {m}" if m != "loss_A" else "Adversary loss L_A")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{tag}_curves.png", dpi=120); plt.close()
    print(f"Saved: {tag}.csv, {tag}_curves.png")


def q6(data, test):
    items = {
        "Baseline": baseline(data),
        "Parity linear adversary": parity(data),
        "Equal odds linear adversary": equal_odds(data),
        "Parity hidden adversary": run(data, "Parity hidden adversary", mode="parity",
                                       alpha=ALPHA, adv_hidden=ADV_HIDDEN),
        "Equal odds hidden adversary": run(data, "Equal odds hidden adversary", mode="eq_odds",
                                           alpha=ALPHA, adv_hidden=ADV_HIDDEN),
    }
    report("q6_hidden_adversary", items, test)


def q7(data, test):
    items = {
        "Baseline": baseline(data),
        "Parity constant alpha": parity(data),
        "Parity sqrt alpha": run(data, "Parity sqrt alpha", mode="parity", alpha=ALPHA,
                                 alpha_schedule="sqrt"),
    }
    report("q7_sqrt_alpha", items, test)


def q9(data, test):
    items = {
        "Baseline": baseline(data),
        "Equal odds": equal_odds(data),
        "Equal opportunity": run(data, "Equal opportunity", mode="eq_opp", alpha=ALPHA),
    }
    report("q9_equal_opportunity", items, test)


def q10(data, test):
    items = {
        "Baseline": baseline(data),
        "Parity projection": parity(data),
        "Parity gradient reversal": run(data, "Parity gradient reversal", mode="parity",
                                        alpha=ALPHA, grl=True),
    }
    report("q10_gradient_reversal", items, test)


def q11(data, test):
    items = {
        "Baseline": baseline(data),
        "Parity output input": parity(data),
        "Parity hidden layer input": run(data, "Parity hidden layer input", mode="parity",
                                         alpha=ALPHA, adv_input="hidden"),
    }
    report("q11_hidden_layer_input", items, test)


EXPERIMENTS = {"q6": q6, "q7": q7, "q9": q9, "q10": q10, "q11": q11}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="*")
    args = parser.parse_args()
    unknown = [w for w in args.which if w != "all" and w not in EXPERIMENTS]
    if unknown:
        parser.error(f"invalid experiment names: {unknown}, choose from all, "
                     + ", ".join(EXPERIMENTS))
    selected = list(EXPERIMENTS) if (not args.which or "all" in args.which) else args.which

    data = load_adult()
    test = (data["X_test"], data["y_test"], data["z_test"])
    for key in selected:
        EXPERIMENTS[key](data, test)
