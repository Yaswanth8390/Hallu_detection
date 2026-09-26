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
lowest-singular-value right-singular vectors form `V_R`. The unembedding
Gram matrix is accumulated and diagonalized in CPU float64 to better preserve
the low-energy directions while avoiding another large GPU allocation. For the causal hidden
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

The generated feature CSV includes the old lexical-overlap label for
inspection, but **the trainer never uses it**. Instead, human answer-level
labels are provided separately. Annotators see the question and complete
answer, then label the response `supported` if its material factual claims are
correct, `hallucinated` if at least one material factual claim is false or
misleading, or `uncertain` if it cannot be judged. Lack of support in the
question alone is not proof that a claim is false; use the available factual
context or references when judging truth. Leave uncertain examples out of
training.

The trainer uses the answer labels with binary cross-entropy on the maximum
content-token logit per answer, matching HARP's answer-level max-pooling
formulation. Validation metrics are answer-level and human-label-based.
Per-token probabilities are a localization signal learned under answer-level
supervision, not human token annotations.

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

Export one annotation row per response:

```sh
python prepare_annotations.py --data results_tokens.csv --out human_labels.csv
```

Open `human_labels.csv`, review each question and complete response, and fill
`human_label` with `supported`, `hallucinated`, or `uncertain`. Save that
completed file, then train. Blank and uncertain labels are excluded; both
supported and hallucinated labels are required.

```sh
python train_detector.py --data results_tokens.csv --labels human_labels.csv \
  --harp-basis harp_basis.pt --out token_detector.joblib
```

Validation splits are stratified by answer and use only human labels. Reported
accuracy, F1, and ROC-AUC are answer-level metrics; the final saved estimator
is then fit on all human-labeled answers.

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
python tests/smoke_test_training.py
```
