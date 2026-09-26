# Jacobian-based hallucination grounding — Qwen2.5-7B / TruthfulQA

Prototype for: tracing a generated token's logit back through layer-wise
hidden states via the Jacobian (vector-Jacobian product), and checking
whether the token's prediction is causally/gradient-grounded in a relevant
input entity, as a candidate interpretable hallucination signal.

## Files
- `model_utils.py` — loads Qwen2.5-7B-Instruct, greedy-generates a short answer.
- `jacobian.py` — one backward pass per generated token gives the gradient of
  its logit w.r.t. every layer's hidden state (`retain_grad()` trick — no
  need for a separate backward per layer). `trajectory_features()` turns
  that into scalar features (norm trajectory, late/early ratio, etc).
- `grounding.py` — two grounding scores against a candidate input entity:
  gradient attribution (cheap, correlational) and activation ablation
  (causal: replace the entity span, measure the logit/rank drop).
- `dataset.py` — TruthfulQA loading, a **crude regex entity extractor**, and
  a **lexical-overlap correctness label**. Both are placeholders (see
  Limitations below) — the pipeline runs without them being perfect, but
  your results are only as good as these.
- `run_pipeline.py` — ties it together. Traces **every generated token** in
  the answer (not just the first), and writes two CSVs: `--out` (one row per
  example, features mean-aggregated across its generated tokens — what
  `evaluate.py` expects) and `--per-token-out` (one row per generated token,
  raw values — for checking whether signal concentrates on specific tokens,
  e.g. the one naming a hallucinated entity, rather than being smeared
  evenly across the answer).
- `evaluate.py` — the "does it actually detect hallucinations" script: per-feature
  correlation with the hallucination label, held-out logistic-regression
  accuracy/ROC-AUC vs. a majority-class baseline, confusion matrix. Run
  `python evaluate.py --synthetic` first (no model needed) to confirm the
  evaluation logic itself is correct against a planted signal; `--in
  results.csv` for real results.
- `smoke_test.py` / `smoke_test_multitoken.py` — validate the hook/backward/shape
  logic (single-token and multi-token cases respectively) on a tiny
  random-init model (no download needed). Already run and passing.

## Setup
```
pip install torch transformers datasets accelerate bitsandbytes
```
Needs a GPU. Two options, both handled by `run_pipeline.py`'s flags:
- **Single 15GB GPU (e.g. one T4)**: default. 8-bit quantized weights
  (~7-8GB), leaves headroom for the forward+backward passes.
- **Two 15GB GPUs (e.g. Kaggle's dual T4)**: `--device auto --full-precision`.
  Splits full bf16 weights across both GPUs via accelerate (~14GB pooled
  across ~30GB), avoiding the 8-bit precision tradeoff. All the manual
  `model.lm_head(...)` / `model.get_input_embeddings()(...)` calls in
  `jacobian.py`/`grounding.py` place tensors via `input_device(model)` /
  `output_device(model)` rather than a hardcoded device string, so this
  works correctly even when different layers end up on different GPUs.

## Run
```
# single GPU, 8-bit
python run_pipeline.py --n 50 --out results.csv --per-token-out results_tokens.csv

# two GPUs, full precision
python run_pipeline.py --device auto --full-precision --n 50 --out results.csv --per-token-out results_tokens.csv

python evaluate.py --in results.csv
```
Cost scales with answer length (one forward+backward per generated token
per score type) — use `--max-new-tokens` to cap it if answers run long.

## What I could NOT test here
This sandbox has no GPU and no access to huggingface.co, so I could not
download Qwen2.5-7B weights or the TruthfulQA dataset, and could not run the
actual pipeline end-to-end on real data. What I *did* verify: the
Jacobian/hook/backward mechanics and the grounding score computations are
structurally correct, tested against a tiny randomly-initialized
Qwen2-architecture model built from a config (same code path, no download).
Run `smoke_test.py` yourself first if you change `jacobian.py` or
`grounding.py`, before burning GPU time on the real model.

## Known simplifications / things to fix before trusting results
1. **Entity extraction (`extract_candidate_entity`)** is a bare capitalized-word
   regex. It'll miss multi-word entities, lower-case entities, and pick the
   wrong noun sometimes. Swap in spaCy NER or an LLM-based entity extractor
   once you're past the plumbing stage.
2. **Correctness labels (`label_correctness`)** use lexical word-overlap
   against TruthfulQA's correct/incorrect answer lists. TruthfulQA's own
   paper uses a fine-tuned GPT-judge because lexical overlap is noisy —
   hand-check a sample of your labels before trusting downstream stats.
3. **Multi-token aggregation is a plain mean.** `run_pipeline.py` now traces every
   generated token and mean-aggregates each feature across the answer for
   `--out`. A mean can dilute a signal that's concentrated on one token (e.g.
   the token that names a hallucinated entity) — use `--per-token-out` to look
   at token-level values directly, and consider max/min aggregation or
   per-token modeling if the mean looks uninformative but per-token detail
   doesn't.
4. **Final-layer norm placement.** `hidden_states[-1]` from HF's
   `output_hidden_states=True` is the last transformer block's output; check
   whether your version applies the final RMSNorm before or after this in
   `hidden_states` vs. what actually feeds `lm_head` — if there's a
   mismatch, the final-layer logit computed manually in `jacobian.py` won't
   exactly match `model.generate()`'s own logits. Worth a numerical check
   against `out.logits` directly the first time you run on real Qwen2.5-7B.
5. **Ablation baseline token** (`" something"`) is arbitrary. Consider
   averaging over several neutral baselines, or using the entity's mean
   embedding across a corpus, to reduce baseline-choice sensitivity.
6. No SAE integration yet — the natural next step per our discussion is
   checking whether these Jacobian directions align with your HalluSAE
   layer-15 SAE features, to combine the two signals.
