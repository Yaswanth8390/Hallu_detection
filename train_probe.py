"""
Stage 2: train and evaluate hallucination probes on the features from build_dataset.py.

Probes compared (y=1 means hallucination):
  lr             : logistic regression on standardized hidden state (standard linear probe)
  ue_learned     : "hallucination unembedding" U in R^{k x d} (random init) -> GELU -> scalar head
  ue_lmhead_init : same, U initialized from top-k singular directions of the LM's unembedding matrix
  ue_lmhead_frz  : U frozen to the LM unembedding subspace, only the head trains
                   (tests whether the signal lives in the vocab-readout subspace)
Baselines from the generation itself: mean/min token logprob, first-token logprob, mean entropy.

Sweeps layer x position with LR first, then runs the unembedding probes on the best cell
and on the final layer. Optional --transfer evaluates the trained probes on another dataset
(train TriviaQA -> test NQ-Open) to check the probe isn't just learning question difficulty.

Usage:
  python train_probe.py --data data/triviaqa --transfer data/nq_open --k 64 --seeds 3
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

POS = ["prompt_last", "ans_mean", "ans_last"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(d):
    z = np.load(Path(d) / "features.npz")
    return z["feats"], z["y"], z["base"], list(z["layers"])


def auc(y, s):
    return float(roc_auc_score(y, s))


class UnembedProbe(nn.Module):
    def __init__(self, d, k, basis=None, freeze=False):
        super().__init__()
        self.U = nn.Linear(d, k, bias=False)
        if basis is not None:
            with torch.no_grad():
                self.U.weight.copy_(basis[:k])
            self.U.weight.requires_grad = not freeze
        self.head = nn.Sequential(nn.GELU(), nn.Dropout(0.2), nn.Linear(k, 1))

    def forward(self, x):
        return self.head(self.U(x)).squeeze(-1)


def fit_lr(Xtr, ytr, C=0.01):
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
    return clf.decision_function


def fit_unembed(Xtr, ytr, Xva, yva, k, basis, freeze, seed, epochs=150, lr=1e-3, wd=1e-2, patience=20):
    torch.manual_seed(seed)
    m = UnembedProbe(Xtr.shape[1], k, basis, freeze).to(DEV)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    Xt, yt = torch.tensor(Xtr, device=DEV), torch.tensor(ytr, dtype=torch.float32, device=DEV)
    Xv = torch.tensor(Xva, device=DEV)
    pos_w = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), device=DEV, dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best, best_state, bad = -1, None, 0
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            lossf(m(Xt[idx]), yt[idx]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = auc(yva, m(Xv).cpu().numpy())
        if v > best:
            best, bad = v, 0
            best_state = {k_: t.clone() for k_, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()

    @torch.no_grad()
    def predict(X):
        return m(torch.tensor(X, device=DEV)).cpu().numpy()

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--transfer", default=None)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    feats, y, base, layers = load(args.data)
    basis = torch.load(Path(args.data) / "unembed_basis.pt")
    T = load(args.transfer) if args.transfer else None

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
    tr, va = train_test_split(tr, test_size=0.125, stratify=y[tr], random_state=0)  # 70/10/20
    print(f"n={len(y)} halluc_rate={y.mean():.3f} | train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    results = {"baselines": {}, "sweep": {}, "probes": {}}

    # ---- generation-based baselines (no hidden states)
    for name, col, sign in [("neg_mean_logprob", 0, -1), ("neg_min_logprob", 1, -1),
                            ("neg_first_logprob", 2, -1), ("mean_entropy", 3, 1)]:
        r = {"test": auc(y[te], sign * base[te, col])}
        if T: r["transfer"] = auc(T[1], sign * T[2][:, col])
        results["baselines"][name] = r
        print(f"[baseline] {name:20s} {r}")

    # ---- layer x position sweep with LR
    def prep(F, li, pi, mu=None, sd=None):
        X = F[:, li, pi].astype(np.float32)
        if mu is None:
            mu, sd = X.mean(0), X.std(0) + 1e-6
        return (X - mu) / sd, mu, sd

    best = (-1, None)
    for li, L in enumerate(layers):
        for pi, pname in enumerate(POS):
            Xtr, mu, sd = prep(feats[tr], li, pi)
            Xva, _, _ = prep(feats[va], li, pi, mu, sd)
            v = auc(y[va], fit_lr(Xtr, y[tr])(Xva))
            results["sweep"][f"L{L}_{pname}"] = v
            print(f"[sweep] layer {L:2d} {pname:11s} val AUROC {v:.4f}")
            if v > best[0]:
                best = (v, (li, pi))
    li_best, pi_best = best[1]
    cells = {(li_best, pi_best), (len(layers) - 1, pi_best)}  # best cell + final layer, same position

    # ---- probes on selected cells
    for li, pi in sorted(cells):
        L, pname = layers[li], POS[pi]
        Xtr, mu, sd = prep(feats[tr], li, pi)
        Xva, _, _ = prep(feats[va], li, pi, mu, sd)
        Xte, _, _ = prep(feats[te], li, pi, mu, sd)
        Xtf = prep(T[0], li, pi, mu, sd)[0] if T else None
        cell = f"L{L}_{pname}"
        results["probes"][cell] = {}
        print(f"\n=== cell {cell} ===")

        configs = {
            "lr": lambda s: fit_lr(Xtr, y[tr]),
            "ue_learned": lambda s: fit_unembed(Xtr, y[tr], Xva, y[va], args.k, None, False, s),
            "ue_lmhead_init": lambda s: fit_unembed(Xtr, y[tr], Xva, y[va], args.k, basis, False, s),
            "ue_lmhead_frz": lambda s: fit_unembed(Xtr, y[tr], Xva, y[va], args.k, basis, True, s),
        }
        for name, make in configs.items():
            seeds = 1 if name == "lr" else args.seeds
            aucs, aps, tfs = [], [], []
            for s in range(seeds):
                pred = make(s)
                st = pred(Xte)
                aucs.append(auc(y[te], st))
                aps.append(float(average_precision_score(y[te], st)))
                if T:
                    tfs.append(auc(T[1], pred(Xtf)))
            r = {"test_auroc": float(np.mean(aucs)), "test_auroc_std": float(np.std(aucs)),
                 "test_auprc": float(np.mean(aps))}
            if T:
                r["transfer_auroc"] = float(np.mean(tfs))
            results["probes"][cell][name] = r
            print(f"{name:16s} {r}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
