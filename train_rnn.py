"""
Stage 3: a recurrent model that reads the embedding trajectory across depth
(selected hidden-state indices 8, 12, 16, 20, 24, 28, and 32) and detects hallucination.

It is not just a classifier on the last state:
  * A causal GRU walks layer by layer. At each layer it sees the state x_l AND the change
    delta_l = x_l - x_{l-1} (i.e. what block l wrote into the residual stream).
  * Future prediction: from its hidden state after layer l, the GRU must predict the
    (standardized) state at layer l+1. This forces the recurrent state to model how the
    representation normally evolves. The per-layer prediction error ("surprise") is also
    fed to the classifier: a trajectory that evolves unusually is the signal.
  * Deep supervision: every step also gets a hallucination head, so you get a
    "how early can we tell" curve over depth.
  * Final decision: pooled GRU states + surprise sequence -> MLP -> hallucination logit.

Loss = BCE(final) + alpha * BCE(per-step, weighted toward later layers) + beta * MSE(next-state prediction)

Ablations:  --beta 0  (no future prediction)   --no_delta  (states only, no changes)

Usage:
  python train_rnn.py --data data/triviaqa --transfer data/nq_open --pos 1 --seeds 3
Needs data built with build_dataset.py. It reads the same seven selected layers and
three feature positions as the probe scripts. Uses the same 70/10/20 split.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

DEV = "cuda" if torch.cuda.is_available() else "cpu"
POS = ["prompt_last", "ans_mean", "ans_last"]


def auc(y, s):
    return float(roc_auc_score(y, s))


def load(d, pos):
    if pos not in range(len(POS)):
        raise ValueError(f"pos must be in 0..{len(POS) - 1}, got {pos}")
    with np.load(Path(d) / "features.npz") as features:
        y = features["y"].copy()
        layers = features["layers"].copy()
    traj = np.load(Path(d) / "traj.npy", mmap_mode="r")
    if traj.ndim != 4 or traj.shape[2] != len(POS):
        raise ValueError(f"expected traj.npy with shape [N, T, 3, d], got {traj.shape}")
    if traj.shape[0] != len(y):
        raise ValueError(f"label/trajectory count mismatch: {len(y)} labels vs {traj.shape[0]} trajectories")
    if len(layers) != traj.shape[1]:
        raise ValueError(f"layer metadata/trajectory mismatch: {len(layers)} layers vs {traj.shape[1]} steps")
    X = np.ascontiguousarray(traj[:, :, pos, :])  # [N, selected layers, d]
    return torch.from_numpy(X).to(DEV), y, layers


def moments(X, fn, chunk=256):
    """Mean/std over samples of fn(X) computed in chunks; returns [T', d] tensors."""
    s, n = 0, 0
    for i in range(0, len(X), chunk):
        v = fn(X[i:i + chunk].float()); s = s + v.sum(0); n += len(v)
    mu = s / n
    s2 = 0
    for i in range(0, len(X), chunk):
        v = fn(X[i:i + chunk].float()); s2 = s2 + ((v - mu) ** 2).sum(0)
    return mu, (s2 / n).sqrt() + 1e-4


class TrajRNN(nn.Module):
    def __init__(self, d, T, m=256, hid=256, use_err=True, use_delta=True):
        super().__init__()
        self.T, self.use_err, self.use_delta = T, use_err, use_delta
        blk = lambda: nn.Sequential(nn.Linear(d, m), nn.LayerNorm(m), nn.GELU())
        self.px, self.pd = blk(), blk()
        self.depth_emb = nn.Embedding(T, m)
        self.gru = nn.GRU(3 * m, hid, batch_first=True)
        self.pred = nn.Linear(hid, d)          # s_l -> standardized x_{l+1}
        self.step = nn.Linear(hid, 1)          # per-layer hallucination head
        self.cls = nn.Sequential(nn.Dropout(0.3), nn.Linear(3 * hid + (T - 1), 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, xz, dz):
        B = xz.shape[0]
        if not self.use_delta:
            dz = torch.zeros_like(dz)
        e = self.depth_emb.weight[None].expand(B, -1, -1)
        s, _ = self.gru(torch.cat([self.px(xz), self.pd(dz), e], -1))         # [B, T, hid]
        err = ((self.pred(s[:, :-1]) - xz[:, 1:]) ** 2).mean(-1)               # [B, T-1] surprise
        errf = torch.log(err.detach() + 1e-3) if self.use_err else torch.zeros_like(err)
        pooled = torch.cat([s[:, -1], s.mean(1), s.max(1).values, errf], -1)
        return self.cls(pooled).squeeze(-1), self.step(s).squeeze(-1), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--transfer", default=None)
    ap.add_argument("--pos", type=int, choices=range(3), default=1,
                    help="0 prompt_last, 1 ans_mean, 2 ans_last")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of per-layer BCE")
    ap.add_argument("--beta", type=float, default=1.0, help="weight of next-layer prediction loss")
    ap.add_argument("--no_delta", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results_rnn.json")
    args = ap.parse_args()
    if args.seeds < 1 or args.epochs < 1 or args.patience < 1 or args.bs < 1:
        ap.error("--seeds, --epochs, --patience, and --bs must be positive")
    if args.alpha < 0 or args.beta < 0:
        ap.error("--alpha and --beta must be non-negative")

    X, y, layers = load(args.data, args.pos)
    N, T, d = X.shape
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("training labels must contain both binary classes 0 and 1")
    idx = np.arange(N)
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
    tr, va = train_test_split(tr, test_size=0.125, stratify=y[tr], random_state=0)
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    XT, yT, transfer_layers = load(args.transfer, args.pos) if args.transfer else (None, None, None)
    if XT is not None and XT.shape[1:] != X.shape[1:]:
        raise ValueError(f"transfer trajectory shape {tuple(XT.shape[1:])} does not match training {tuple(X.shape[1:])}")
    if transfer_layers is not None and not np.array_equal(transfer_layers, layers):
        raise ValueError("transfer layer indices do not match training layer indices")
    if yT is not None and (not np.isin(yT, [0, 1]).all() or len(np.unique(yT)) != 2):
        raise ValueError("transfer labels must contain both binary classes 0 and 1")
    print(f"n={N} T={T} d={d} pos={POS[args.pos]} halluc_rate={y.mean():.3f} | {len(tr)}/{len(va)}/{len(te)}")
    transition_names = [f"{left}->{right}" for left, right in zip(layers[:-1], layers[1:])]

    # per-(layer, dim) normalization of states and of layer-to-layer changes, from train only
    mu, sd = moments(Xtr, lambda x: x)
    dmu, dsd = moments(Xtr, lambda x: x[:, 1:] - x[:, :-1])

    def prep(xb):
        x = xb.float()
        xz = ((x - mu) / sd).clamp(-10, 10)
        dz = ((x[:, 1:] - x[:, :-1] - dmu) / dsd).clamp(-10, 10)
        return xz, F.pad(dz, (0, 0, 1, 0))  # zero change at layer 0

    @torch.no_grad()
    def predict(m, Xs, bs=256):
        m.eval()
        L, S, E = [], [], []
        for i in range(0, len(Xs), bs):
            lg, st, er = m(*prep(Xs[i:i + bs]))
            L.append(lg.cpu()); S.append(st.cpu()); E.append(er.cpu())
        return torch.cat(L).numpy(), torch.cat(S).numpy(), torch.cat(E).numpy()

    pw = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), device=DEV, dtype=torch.float32)
    step_w = torch.linspace(0.1, 1.0, T, device=DEV)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)

    R = {"test_auroc": [], "test_auprc": [], "transfer_auroc": [], "depth_auroc": [], "surprise_gap": []}
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        m = TrajRNN(d, T, args.m, args.hid, use_err=args.beta > 0, use_delta=not args.no_delta).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-2)
        best, state, bad = -1, None, 0
        for ep in range(args.epochs):
            m.train()
            perm = torch.randperm(len(Xtr), device=DEV)
            for i in range(0, len(perm), args.bs):
                b = perm[i:i + args.bs]
                yb = ytr_t[b]
                logit, step, err = m(*prep(Xtr[b]))
                l_final = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pw)
                l_step = (F.binary_cross_entropy_with_logits(
                    step, yb[:, None].expand_as(step), pos_weight=pw, reduction="none") * step_w).mean()
                loss = l_final + args.alpha * l_step + args.beta * err.mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            v = auc(yva, predict(m, Xva)[0])
            if v > best:
                best, bad = v, 0
                state = {k: t.clone() for k, t in m.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        m.load_state_dict(state)

        lg, st, er = predict(m, Xte)
        R["test_auroc"].append(auc(yte, lg))
        R["test_auprc"].append(float(average_precision_score(yte, lg)))
        R["depth_auroc"].append([auc(yte, st[:, l]) for l in range(T)])
        R["surprise_gap"].append((er[yte == 1].mean(0) - er[yte == 0].mean(0)).tolist())
        if XT is not None:
            R["transfer_auroc"].append(auc(yT, predict(m, XT)[0]))
        print(f"seed {seed}: val {best:.4f} test AUROC {R['test_auroc'][-1]:.4f}"
              + (f" transfer {R['transfer_auroc'][-1]:.4f}" if XT is not None else ""))

    summ = {k: np.mean(v, 0).tolist() for k, v in R.items() if v}
    summ["test_auroc_std"] = float(np.std(R["test_auroc"]))
    summ["depth_layers"] = layers.tolist()
    summ["surprise_transitions"] = transition_names
    summ["args"] = vars(args)
    print(f"\ntest AUROC {np.mean(R['test_auroc']):.4f} +- {summ['test_auroc_std']:.4f}"
          f" | AUPRC {np.mean(R['test_auprc']):.4f}"
          + (f" | transfer AUROC {np.mean(R['transfer_auroc']):.4f}" if R["transfer_auroc"] else ""))
    print(f"per-layer AUROC (layers {', '.join(map(str, layers.tolist()))}):",
          " ".join(f"{a:.2f}" for a in summ["depth_auroc"]))
    print(f"surprise gap halluc-correct (transitions {', '.join(transition_names)}):",
          " ".join(f"{a:+.3f}" for a in summ["surprise_gap"]))
    Path(args.out).write_text(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
