import math

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import set_seed, SEED
from models import Predictor, Adversary, grad_reverse
from metrics import evaluate


def adversary_loss(z_logit, zb, yb, mode):
    if mode == "eq_opp":
        mask = yb == 1
        if not mask.any():
            return z_logit.sum() * 0.0
        return F.binary_cross_entropy_with_logits(z_logit[mask], zb[mask])
    return F.binary_cross_entropy_with_logits(z_logit, zb)


def train(data, mode="eq_odds", alpha=1.0, debias=True, epochs=20, batch_size=128,
          lr_p=1e-3, lr_a=1e-2, seed=SEED, verbose=True,
          adv_hidden=0, alpha_schedule="const", adv_input="output", grl=False):
    assert alpha_schedule in ("const", "sqrt")
    assert adv_input in ("output", "hidden")
    set_seed(seed)
    device = data["device"]
    use_hidden = adv_input == "hidden"
    P = Predictor(data["n_features"]).to(device)
    A = Adversary(mode, n_hidden=adv_hidden, hidden_input=use_hidden).to(device)
    opt_p = torch.optim.Adam(P.parameters(), lr=lr_p)
    opt_a = torch.optim.Adam(A.parameters(), lr=lr_a)
    sched_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=epochs)
    loader = DataLoader(data["train_ds"], batch_size=batch_size, shuffle=True)
    p_params = list(P.parameters())
    history = []

    for epoch in range(1, epochs + 1):
        P.train(); A.train()
        alpha_t = alpha * math.sqrt(epoch) if alpha_schedule == "sqrt" else alpha
        tot_lp = tot_la = 0.0
        for xb, yb, zb in loader:
            xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)

            if use_hidden:
                y_logit, h = P(xb, return_hidden=True)
            else:
                y_logit, h = P(xb), None
            loss_p = F.binary_cross_entropy_with_logits(y_logit, yb)

            if debias and grl:
                h_in = grad_reverse(h, alpha_t) if use_hidden else None
                z_logit = A(grad_reverse(y_logit, alpha_t), yb, h_in)
                loss_a = adversary_loss(z_logit, zb, yb, mode)
                opt_p.zero_grad()
                opt_a.zero_grad()
                (loss_p + loss_a).backward()
                opt_p.step()
                opt_a.step()
            elif debias:
                z_logit = A(y_logit, yb, h)
                loss_a = adversary_loss(z_logit, zb, yb, mode)
                grad_p = torch.autograd.grad(loss_p, p_params, retain_graph=True)
                grad_a = torch.autograd.grad(loss_a, p_params, retain_graph=True,
                                             allow_unused=True)
                opt_p.zero_grad()
                for w, gp, ga in zip(p_params, grad_p, grad_a):
                    if ga is None:
                        ga = torch.zeros_like(w)
                    unit_ga = ga / (ga.norm() + 1e-8)
                    proj = (gp * unit_ga).sum() * unit_ga
                    w.grad = gp - proj - alpha_t * ga
                opt_p.step()

                opt_a.zero_grad()
                h_det = h.detach() if h is not None else None
                z_logit = A(y_logit.detach(), yb, h_det)
                adversary_loss(z_logit, zb, yb, mode).backward()
                opt_a.step()
            else:
                opt_p.zero_grad()
                loss_p.backward()
                opt_p.step()
                loss_a = torch.tensor(0.0)

            tot_lp += loss_p.item() * len(xb)
            tot_la += loss_a.item() * len(xb)

        sched_p.step()
        n = len(data["train_ds"])
        m = evaluate(P, data["X_test"], data["y_test"], data["z_test"])
        history.append({"epoch": epoch, "loss_P": tot_lp / n, "loss_A": tot_la / n, **m})
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"epoch {epoch:3d} | L_P {tot_lp/n:.4f} | L_A {tot_la/n:.4f} | "
                  f"acc {m['accuracy']:.4f} | DP {m['DP gap']:.3f} | "
                  f"TPR {m['TPR gap']:.3f} | FPR {m['FPR gap']:.3f}")
    return P, pd.DataFrame(history)
