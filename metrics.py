import numpy as np
import torch


def _to_numpy(a):
    return a.numpy() if torch.is_tensor(a) else np.asarray(a)


@torch.no_grad()
def predict_proba(predictor, X):
    predictor.eval()
    return torch.sigmoid(predictor(X)).cpu().numpy()


def evaluate_scores(p, y, z, threshold=0.5):
    y, z = _to_numpy(y), _to_numpy(z)
    yhat = (p >= threshold).astype(float)

    def rate(mask):
        return yhat[mask].mean() if mask.any() else np.nan

    return {
        "accuracy":  (yhat == y).mean(),
        "DP gap":    abs(rate(z == 1) - rate(z == 0)),
        "TPR gap":   abs(rate((z == 1) & (y == 1)) - rate((z == 0) & (y == 1))),
        "FPR gap":   abs(rate((z == 1) & (y == 0)) - rate((z == 0) & (y == 0))),
        "P(yhat=1|z=1)": rate(z == 1),
        "P(yhat=1|z=0)": rate(z == 0),
    }


def evaluate(predictor, X, y, z, threshold=0.5):
    return evaluate_scores(predict_proba(predictor, X), y, z, threshold)
