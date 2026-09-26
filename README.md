# Token-Level Counterfactual Input Dependence

The pipeline measures how much each generated content word depends on the provided user question. It does not determine factual correctness or hallucination.

## Scoring

The current TruthfulQA input provides a question, but no supporting passage or evidence-span annotations. The pipeline conservatively matches each generated content word to an exact or simple inflectional match in the question and removes only that matched span. The system instruction and remaining question text are unchanged. If no direct lexical match exists (for example, `Orwell` when the question only mentions `1984`), the row is retained with `no_matching_evidence_span` and is not scored against an arbitrary span. This lexical matcher does not infer semantic cues or verify that a matched phrase supports the generated claim.

The exact generated token IDs are held fixed in both conditions. For each subtoken, the model computes its conditional log-probability given the corresponding prompt and the same generated prefix before that subtoken. The original and counterfactual conditional log-probabilities are summed across the subtokens of each content word:

```text
delta_logprob = original_logprob - counterfactual_logprob
```

A positive delta means the matched question span increased support for that word relative to the span-removed prompt. A near-zero delta means weak dependence on that span. A negative delta means the word was more likely without that span. None of these labels establishes correctness or distinguishes question information from parametric model knowledge.

Words are grouped from tokenizer offsets so multi-subtoken words receive one row. Punctuation and a conservative list of grammatical/function words are skipped. Each row retains `generated_text` so the original generated response can be reconstructed, and includes the token, evidence span, both log-probabilities, delta, classification, and threshold.

## Run

Install the packages in `requirements.txt`, then run on a GPU:

```sh
python run_pipeline.py --n 10 --max-new-tokens 32 --dependence-threshold 0.1 --out results_tokens.csv --per-token-out results_tokens.csv
```

`--grounding-threshold` remains as a compatibility alias for `--dependence-threshold`. The default threshold is `0.1` log-probability units and should be calibrated experimentally; low dependence is reported as `weak_input_dependence`, never as hallucination.

The smoke tests use tiny random-initialized Qwen models and an offset-tokenizer fixture, so they validate log-probability extraction, subword aggregation, labels, and CSV formatting without downloading model weights or TruthfulQA:

```sh
python tests/smoke_test.py
python tests/smoke_test_multitoken.py
```