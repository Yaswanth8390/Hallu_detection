"""
Stage 1: build a labeled hallucination dataset AND extract features in one pass.

For each question:
  1. Llama-3.1-8B-Instruct greedily generates a short answer.
  2. Answer is labeled against gold aliases (y=1 -> hallucination / wrong, y=0 -> correct).
  3. One teacher-forced forward pass over prompt+answer stores hidden states at
     several layers and three positions:
        0 = last prompt token   (model state *before* answering)
        1 = mean over answer tokens
        2 = last answer token
  4. Token-level logprob/entropy baselines are stored for comparison.

Also saves the top-K right singular directions of the unembedding matrix (lm_head)
so the probe script can build a probe that lives in the vocab-readout subspace.

Usage:
  python build_dataset.py --dataset triviaqa --n 6000 --out data/triviaqa
  python build_dataset.py --dataset nq_open  --n 3000 --out data/nq_open
"""
import argparse, csv, json, re, string
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYERS = [8, 12, 16, 20, 24, 28, 32]  # hidden_states indices; 32 = final (post-norm)
SYSTEM = "Answer the question with a short factual answer (a few words). Do not explain."
ABSTAIN = re.compile(r"(i don'?t know|not sure|cannot|can'?t (answer|determine)|unknown|no information)", re.I)


def norm(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_correct(answer: str, golds: list[str]) -> bool:
    a = f" {norm(answer)} "
    return any(g and f" {g} " in a for g in map(norm, golds))


def ensure_manual_labels(path: Path):
    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "question", "generated_answer", "label", "notes"])


def load_manual_labels(path: Path):
    labels = {}
    if not path.exists():
        return labels
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row.get("index")
            if idx is None or idx == "":
                continue
            try:
                labels[int(idx)] = row.get("label", "").strip().lower()
            except ValueError:
                continue
    return labels


def load_qa(name: str, n: int, seed: int):
    if name == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        get = lambda r: (r["question"], list(r["answer"]["aliases"]) + [r["answer"]["value"]])
    elif name == "nq_open":
        ds = load_dataset("google-research-datasets/nq_open", split="validation")
        get = lambda r: (r["question"], list(r["answer"]))
    else:
        raise ValueError(name)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [get(r) for r in ds]


@torch.no_grad()
def save_unembed_basis(model, out_dir: Path, k: int = 512):
    W = model.lm_head.weight.float()          # [V, d]
    G = (W.T @ W).double()                    # [d, d]
    evals, evecs = torch.linalg.eigh(G)       # ascending
    basis = evecs[:, -k:].flip(-1).T.float().cpu().contiguous()  # [k, d], descending
    torch.save(basis, out_dir / "unembed_basis.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="triviaqa")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manual-labels", default=None, help="CSV with columns: index, question, generated_answer, label, notes")
    ap.add_argument("--manual-only", action="store_true", help="Only keep rows with manual labels and skip auto labels for unlabeled rows")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manual_labels = load_manual_labels(Path(args.manual_labels)) if args.manual_labels else {}
    if args.manual_labels:
        ensure_manual_labels(Path(args.manual_labels))

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")}
    save_unembed_basis(model, out)

    qa = load_qa(args.dataset, args.n, args.seed)
    feats, labels, base, meta, traj = [], [], [], [], []

    for b in tqdm(range(0, len(qa), args.bs), desc="generate+extract"):
        batch = qa[b:b + args.bs]
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True)
            for q, _ in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id, eos_token_id=list(stop_ids))
        new = gen[:, enc.input_ids.shape[1]:].cpu().tolist()

        for i, (q, golds) in enumerate(batch):
            global_idx = b + i
            ans_ids = []
            for t in new[i]:
                if t in stop_ids:
                    break
                ans_ids.append(t)
            if not ans_ids:
                continue
            text = tok.decode(ans_ids, skip_special_tokens=True).strip()
            if ABSTAIN.search(text):
                continue

            if args.manual_labels and global_idx in manual_labels:
                label = manual_labels[global_idx]
                if label in {"correct", "corr", "c", "0", "0.0"}:
                    y = 0
                elif label in {"incorrect", "wrong", "hallucination", "hallucinated", "1", "1.0"}:
                    y = 1
                else:
                    continue
            else:
                y = 0 if is_correct(text, golds) else 1
                if args.manual_labels and args.manual_only:
                    continue

            prompt_ids = enc.input_ids[i][enc.attention_mask[i].bool()]
            P, n = len(prompt_ids), len(ans_ids)
            full = torch.cat([prompt_ids, torch.tensor(ans_ids, device="cuda")]).unsqueeze(0)
            with torch.no_grad():
                o = model(full, output_hidden_states=True)

            per_layer = []
            traj_layers = []
            for L in LAYERS:
                h = o.hidden_states[L][0].float()
                prompt_last = h[P - 1]
                ans_mean = h[P:P + n].mean(0)
                ans_last = h[P + n - 1]
                per_layer.append(torch.stack([prompt_last, ans_mean, ans_last]))
                traj_layers.append(torch.stack([prompt_last, ans_mean, ans_last]))

            feats.append(torch.stack(per_layer).half().cpu().numpy())
            traj.append(torch.stack(traj_layers).half().cpu().numpy())

            lp = torch.log_softmax(o.logits[0, P - 1:P + n - 1].float(), -1)
            tlp = lp[torch.arange(n), torch.tensor(ans_ids, device="cuda")]
            ent = -(lp.exp() * lp).sum(-1)
            base.append([tlp.mean().item(), tlp.min().item(), tlp[0].item(), ent.mean().item()])

            labels.append(y)
            meta.append({"q": q, "answer": text, "gold": golds[:3], "y": y, "manual": global_idx in manual_labels})

    if len(feats) == 0:
        raise ValueError("No labeled examples were retained. Check --manual-labels or the generated answers.")

    traj_arr = np.stack(traj)
    np.save(out / "traj.npy", traj_arr)
    np.savez_compressed(out / "features.npz",
                        feats=np.stack(feats), y=np.array(labels), base=np.array(base, dtype=np.float32),
                        layers=np.array(LAYERS))
    with open(out / "meta.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    y = np.array(labels)
    print(f"saved {len(y)} examples | hallucination rate {y.mean():.3f}")


if __name__ == "__main__":
    main()
