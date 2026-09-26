# Token-Level Counterfactual Input Dependence

The pipeline measures how much each generated content word depends on the provided user question. It does not determine factual correctness or hallucination.

## Scoring

The current TruthfulQA input provides a question, but no supporting passage or evidence-span annotations. For each generated content word, the pipeline obtains final-layer contextual representations for the generated subtokens and candidate contiguous question spans. It mean-pools each word/span representation and selects the span with maximum cosine similarity, searching spans up to five question words. The selected span alone is removed for that word's counterfactual. This can associate a paraphrased cue such as `1984` with a representation for `Orwell`, but similarity is only a span proposal: it does not verify that the question entails or factually supports the generated claim. If no candidate span can be aligned to prompt tokens, the row is retained as `no_matching_evidence_span` without a counterfactual score.

The exact generated token IDs are held fixed in both conditions. For each subtoken, the model computes its conditional log-probability given the corresponding prompt and the same generated prefix before that subtoken. The original and counterfactual conditional log-probabilities are summed across the subtokens of each content word:

```text
delta_logprob = original_logprob - counterfactual_logprob
```

A positive delta means the matched question span increased support for that word relative to the span-removed prompt. A near-zero delta means weak dependence on that span. A negative delta means the word was more likely without that span. None of these labels establishes correctness, and low dependence on its own does not distinguish two very different situations: the model already knew the fact (parametric knowledge), or the model was confabulating regardless of what was asked.

To separate those, each word also gets an entropy and a top1-vs-top2 log-probability margin (both in nats) from the model's *original*-prompt distribution at that position, from the same forward pass used for `original_logprob`. Low entropy / high margin means the model would likely produce this word regardless of the prompt. This is layered on top of the dependence label, not used alone: words already showing `strong_input_dependence` are left as-is (`combined_classification=input_dependent`), since the dependence signal already explains the question's role there. For the remaining words, low original-prompt entropy against `--entropy-threshold` yields `parametric_knowledge`; high entropy yields `possible_hallucination`. This still is not a correctness label — it only distinguishes "confident regardless of prompt" from "not confident and not prompt-dependent either," and the entropy threshold needs the same experimental calibration as `--dependence-threshold`.

Words are grouped from tokenizer offsets so multi-subtoken words receive one row. Punctuation and a conservative list of grammatical/function words are skipped. Each row retains `generated_text` so the original generated response can be reconstructed, and includes the token, selected evidence span and its similarity, both log-probabilities, delta, dependence classification and threshold, word entropy and margin, entropy threshold, confidence classification, and the combined classification.

## Run

Install the packages in `requirements.txt`, then run on a GPU:

```sh
python run_pipeline.py --n 10 --max-new-tokens 32 --dependence-threshold 0.1 --entropy-threshold 1.0 --out results_tokens.csv --per-token-out results_tokens.csv
```

`--grounding-threshold` remains as a compatibility alias for `--dependence-threshold`. The default dependence threshold is `0.1` log-probability units and should be calibrated experimentally; low dependence is reported as `weak_input_dependence`, never as hallucination. `--entropy-threshold` (default `1.0` nats) similarly needs calibration per model/vocabulary before its `parametric_knowledge` / `possible_hallucination` split should be trusted.

The smoke tests use tiny random-initialized Qwen models and an offset-tokenizer fixture, so they validate log-probability extraction, subword aggregation, labels, and CSV formatting without downloading model weights or TruthfulQA:

```sh
python tests/smoke_test.py
python tests/smoke_test_multitoken.py
```