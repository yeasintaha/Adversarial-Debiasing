import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, return_hidden=False):
        h = self.body(x)
        logit = self.out(h).squeeze(-1)
        if return_hidden:
            return logit, h
        return logit


class Adversary(nn.Module):
    def __init__(self, mode="eq_odds", n_hidden=0, hidden_input=False, feat_dim=64):
        super().__init__()
        assert mode in ("parity", "eq_odds", "eq_opp")
        self.mode = mode
        self.hidden_input = hidden_input
        if hidden_input:
            self.c = None
        else:
            self.c = nn.Parameter(torch.ones(1))
        uses_y = mode != "parity"
        if hidden_input:
            n_in = feat_dim + (1 if uses_y else 0)
        else:
            n_in = 4 if uses_y else 1
        if n_hidden > 0:
            self.net = nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(),
                                     nn.Linear(n_hidden, 1))
        else:
            self.net = nn.Linear(n_in, 1)

    def forward(self, y_logit, y_true, h=None):
        if self.hidden_input:
            if self.mode == "parity":
                feats = h
            else:
                feats = torch.cat([h, y_true.unsqueeze(-1)], dim=-1)
        else:
            s = torch.sigmoid((1 + self.c.abs()) * y_logit)
            if self.mode == "parity":
                feats = s.unsqueeze(-1)
            else:
                feats = torch.stack([s, s * y_true, s * (1 - y_true), y_true], dim=-1)
        return self.net(feats).squeeze(-1)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha):
    return GradReverse.apply(x, alpha)
