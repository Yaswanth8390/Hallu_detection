# Token-Level Input-Direction Grounding

This pipeline measures how directionally sensitive each generated content word is to the prompt representation. It does not classify answers as correct or hallucinated.

## Scoring

For each generated semantic word, the model's target logits for its subword pieces are summed, and the gradient of that scalar is computed with respect to the prompt input embeddings. The reported `alignment_score` is the cosine similarity between that gradient and the concatenated prompt-embedding direction. The score is signed and remains the primary output; a negative score is not treated as hallucination.

Words are reconstructed from tokenizer offsets so a multi-subword word is scored as one unit. Punctuation and a conservative set of grammatical/function words are skipped. Multiword entities are emitted as their component content words, while subword fragments within each word are grouped together.

`grounding_strength` is `strong` when the absolute cosine reaches `--grounding-threshold`, otherwise `weak`. This is only a directional-strength description, not a correctness or hallucination label. The threshold defaults to `0.1` and can be calibrated experimentally.

## Run

Install the packages in `requirements.txt`, then run on a GPU:

```sh
python run_pipeline.py --n 50 --out results.csv --grounding-threshold 0.1
```

Each CSV row represents one evaluated content word and includes its text, alignment score, weak/strong grounding status, threshold, grouped target logit, and subword count. `--per-token-out` remains available for compatibility and writes the same token-level rows as `--out`; neither output is sentence-aggregated.

The smoke tests use tiny random-initialized Qwen models and a fixed offset-tokenizer fixture, so they validate gradient mechanics and grouping without downloading model weights or TruthfulQA:

```sh
python tests/smoke_test.py
python tests/smoke_test_multitoken.py
```

`evaluate.py` and the dataset's lexical answer-label helper are legacy utilities from the previous answer-level experiment; they are not used by this token-level pipeline.