# Semantic Entropy Probe

An installable package for building semantic-entropy supervision, training a
linear hidden-state probe, and estimating semantic uncertainty from one
generated answer. The repository branch contains only this SEP implementation.

## Install

Use Python 3.10+ and install the package and dependencies:

```sh
python -m pip install -e .
```

## Build an entropy-labeled dataset

The default dataset is TruthfulQA generation validation. Each question is
answered multiple times; an NLI model groups mutually entailing answers, and
the probability masses of those semantic clusters produce an entropy target.
No manual hallucination labels are needed for this training target.

```sh
sep-build-dataset --n 100 --num-samples 10 \
  --out sep_dataset.csv --device cuda --nli-device cuda
```

Dataset construction downloads the configured answer model, NLI model, and
TruthfulQA data when they are not already cached. Use `--split`, `--nli-model`,
`--temperature`, and `--top-p` to configure data generation.

## Train the probe

```sh
sep-train --data sep_dataset.csv --out sep_probe.joblib
```

Training reports held-out mean absolute and root mean squared semantic-entropy
errors, then fits the saved probe on the full dataset. The default uncertainty
alert threshold is the 75th percentile of training entropy targets.

## Single-answer inference

```sh
sep-infer --question "Who wrote 1984?" --probe sep_probe.joblib
```

Inference generates one answer and predicts its semantic entropy without
sampling multiple answers or loading the NLI model. An explicit threshold in
nats can be set with `--threshold`.

Semantic entropy is an uncertainty signal, not a factuality judgment. Validate
the probe and calibrate alert thresholds against representative data before
using alerts to make factuality decisions.

## Test

```sh
python tests/test_semantic_entropy_probe.py
```
