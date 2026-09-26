# Token-level semantic evidence and HARP detection

This pipeline keeps the existing semantic evidence-span matcher and
counterfactual `delta_logprob = logP(original) - logP(span-removed)` score,
then adds token-level HARP features and a supervised Logistic Regression
classifier. It does not use Jacobian trajectories or a late-layer trajectory
heuristic.

## Features and labels

One CSV row is emitted per generated content word. The dataset includes the
requested `token`, `evidence_span`, `counterfactual_score`,
`semantic_similarity`, `HARP_features`, and `label` columns, plus answer,
token-offset, and label-source metadata.

HARP features follow the paper's reasoning-subspace projection formulation:
the output/unembedding weight is decomposed through its hidden-dimension Gram
matrix; the semantic rank is `k = floor(0.95 * hidden_size)`, and the remaining
lowest-singular-value right-singular vectors form `V_R`. For the causal hidden
state `h_t` that predicts each generated subtoken, the feature is
`V_R.T @ h_t`. A content word split into multiple model subtokens receives the
mean of those per-subtoken projections. The basis is saved to
`harp_basis.pt` so training and inference use identical coordinates.

Content words are selected with spaCy's POS and entity tags (open-class
adjectives, adverbs, nouns, proper nouns, numbers, and verbs, plus named-entity
tokens), not a manually maintained function-word list. The selected semantic
span and its similarity are evidence-matching features, not entailment checks.
Likewise, `counterfactual_score` measures input dependence; neither it nor
semantic similarity by itself determines whether a token is hallucinated.
The classifier learns from all of these features together with HARP features.

TruthfulQA does not supply token-level hallucination annotations. Consequently
`label` is a weak proxy: the existing answer-level lexical-overlap correctness
heuristic is copied to every content word in that answer
(`1 = answer heuristic says incorrect`, `0 = says correct`). This limitation
is recorded in `label_source`; the detector's output should not be treated as
token-ground-truth performance until trained/evaluated with genuine token
annotations.

## Setup and dataset generation

Install `requirements.txt` and the spaCy English tagger:

```sh
python -m spacy download en_core_web_sm
```

Generate features with the configured Qwen model on a GPU:

```sh
python run_pipeline.py --n 50 --max-new-tokens 32 \
  --out results_tokens.csv --harp-basis-out harp_basis.pt
```

By default, model weights are loaded in 8-bit mode. Use `--full-precision` to
disable it. Use `--content-tagger` to select a different installed spaCy model.

## Train and infer

Train Logistic Regression from the generated rows. Validation splits are
grouped by answer/example to prevent token rows from the same answer leaking
across the held-out split; the final saved estimator is then fit on all rows.

```sh
python train_detector.py --data results_tokens.csv \
  --harp-basis harp_basis.pt --out token_detector.joblib
```

Generate a full sentence and receive per-content-token hallucination
probabilities. The response text remains intact; only its individual content
tokens receive flags.

```sh
python infer.py --question "Who wrote 1984?" \
  --detector token_detector.joblib --threshold 0.5
```

The detector threshold defaults to `0.5`, matching the paper's binary
threshold convention. It can be changed with `--threshold`.

## Smoke tests

The tests use small randomly initialized models and test token alignment,
semantic matching, and counterfactual log-probabilities without downloading
the target model:

```sh
python tests/smoke_test.py
python tests/smoke_test_multitoken.py
```
