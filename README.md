# Hidden-State Hallucination Detection

This repository contains scripts for building answer-level datasets and training
probes to predict whether a language model's generated answer is incorrect. It
includes a standard logistic-regression probe, probes constrained to the model's
unembedding subspace, and a recurrent model over hidden-state depth.

Labels use `0 = correct` and `1 = incorrect`. The automatic labeler is a normalized
answer-alias substring matcher, so it is weak supervision rather than verified
ground truth. For reliable evaluation, review the generated answers and use the
manual-label workflow below.

## Requirements

The scripts expect Python with PyTorch, Transformers, Datasets, Accelerate,
scikit-learn, NumPy, and tqdm. Install the repository requirements and the packages
not currently listed there:

```sh
pip install -r requirements.txt
pip install scikit-learn tqdm
```

Dataset creation uses CUDA and defaults to
`Qwen/Qwen2.5-7B-Instruct`. The builder extracts hidden states at model indices
`4, 8, 12, 16, 20, 24, 28`; index `28` is the final hidden state of Qwen2.5-7B.
If you override `--model`, it must expose all of these hidden-state indices and
compatible tokenizer/chat-template and output-projection interfaces. Use the same
model and layer configuration for datasets used in transfer evaluation.

## Build a dataset and manually label it

The builder generates short answers, extracts hidden-state features and baseline
scores, and creates a CSV of candidate examples. The first run applies the automatic
alias matcher to make a provisional dataset and populates the CSV:

```sh
python build_dataset.py \
  --dataset triviaqa \
  --n 6000 \
  --seed 0 \
  --out data/triviaqa \
  --manual-labels data/triviaqa/manual_labels.csv
```

Supported dataset names are `triviaqa` and `nq_open`. Open
`data/triviaqa/manual_labels.csv` and fill `label` with `correct` or `incorrect`
(also accepted: `0` or `1`). The `notes` field is optional. Then rerun the same
command with `--manual-only` to rebuild the dataset using only reviewed labels:

```sh
python build_dataset.py \
  --dataset triviaqa \
  --n 6000 \
  --seed 0 \
  --out data/triviaqa \
  --manual-labels data/triviaqa/manual_labels.csv \
  --manual-only
```

Keep the dataset, sample count, seed, model, and generation settings unchanged
between labeling and rebuilding. Manual labels are matched against both the question
and generated answer; a stale CSV label will not be applied if either has changed.
Unlabeled examples and abstentions are omitted in manual-only mode. Include enough
examples in both classes for the stratified train/validation/test splits.

The output directory contains:

- `features.npz`: probe features, binary labels, generation baselines, and layer
  indices.
- `traj.npy`: hidden-state trajectory across indices `4, 8, 12, 16, 20, 24, 28`,
  each with prompt-last, answer-mean, and answer-last positions, used by the
  recurrent probe.
- `meta.jsonl`: question, generated answer, reference answers, and label provenance.
- `unembed_basis.pt`: leading output-projection directions for unembedding probes.
- The manually reviewed CSV when `--manual-labels` is provided.

## Train and evaluate probes

Train logistic and unembedding-subspace probes:

```sh
python train_probe.py \
  --data data/triviaqa \
  --k 64 \
  --seeds 3 \
  --out results_probe.json
```

Train the recurrent trajectory probe (`--pos` selects prompt-last `0`, answer-mean
`1`, or answer-last `2`):

```sh
python train_rnn.py \
  --data data/triviaqa \
  --pos 1 \
  --seeds 3 \
  --out results_rnn.json
```

Both trainers reserve validation data for selection and report held-out test
metrics. Optionally pass `--transfer data/nq_open` to evaluate on a second dataset;
build it with compatible model and layer settings first. Transfer data must have the
same feature dimensions and layer indices. AUROC and AUPRC measure prediction
performance against the supplied labels; they do not prove an individual answer is
factually correct. Interpret results based on automatically generated labels with
caution, and prefer human-reviewed labels for conclusions about hallucination
detection.
