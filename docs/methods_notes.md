# Methods Notes

## Pipeline Overview

SettleIN is an RAG chatbot for international students in Australia. This document records
the system facts and decisions for the evaluation pipeline.

## Model

- **Model**: llama3.2:3b (a80c4f17acd5), 3-billion parameter LLM
- **Runtime**: Ollama (CPU inference, no GPU)
- **Parameters**: `num_ctx: 4096`, `temperature: 0`, `seed: 42`, `num_predict: 512`
- **Median latency**: 90.2s per answer (CPU, Phase 1 speed test)

## Prompt

- **Active variant**: `settlein_v4`
- **History**: v1 (initial), v2 (relevance-check instruction), v3 (passage-level IDs, strict refusal),
  v4 (v3 + inline citation placement rule)
- **Refusal string**: `I don't have information about that in my sources.`
- **Citation format**: `[S03_002]` (exact passage ID in square brackets)

## Retrieval

- **BM25**: pure-Python implementation (`src/retrieval/search.py`), k1 = 1.2, b = 0.75
- **Dense**: BAAI/bge-small-en-v1.5 (sentence-transformers), cosine similarity
- **Top-k**: 5 passages per query
- **Superseded filter**: sources listed in `data/sources_draft.csv` with
  `is_superseded=TRUE` are excluded before indexing. Currently: S18.

## Collection

- **File**: `data/collection.jsonl` — 111 passages total
- **Sources**: 15 source documents (S01–S10, S13, S15–S18; S11 and S12 excluded,
  S14 retired). S18 is superseded.
- **Search index**: 107 passages after excluding S18's 4 superseded passages

## Test Set

- **Topics**: `data/topics.csv` — 48 questions
  - 29 known, 6 inferred, 13 out_of_kb
  - 6 policy/student fairness pairs
  - 3 currency questions
- **Relevance judgments**: `data/qrels.txt` — 92 rows, TREC format
- **Gold answers**: `data/gold_answers.csv` — passage_ids column uses semicolons
- **Provenance**: questions, relevance judgments and gold answers were drafted
  with an AI assistant from the corpus, then reviewed by the team. A blind
  relevance double-judging check measures agreement with the answer key.

## Prompt Development vs Test Set

- **Phase 1 pipeline checks** ran on four test-set questions (T02Q01, T05Q01, T06Q01,
  T07Q01) for mechanical verification only (citation parsing, refusal detection, latency).
  No prompt text was changed based on those answers.
- **Prompt v4** was checked only on development questions outside the test set:
  - DEV01: "how do i top up my myki"
  - DEV02: "can i drive in victoria with my overseas licence"
  - DEV03: "what does visa condition 8534 mean"
  - DEV04: "can international students get centrelink youth allowance"

## Citation Validation

Every generation records:
- `cited_not_retrieved`: IDs cited but not among the 5 retrieved passages
- `cited_nonexistent`: IDs cited but not in the collection at all
- `prompt_eval_count`: number of prompt tokens evaluated by Ollama

## Context Window

- `num_ctx = 4096`, `num_predict = 512` → prompt budget = 3584 tokens
- Max prompt across 48 questions × 2 retrievers: 2865 tokens (T03Q01, dense)
- 719 tokens headroom — no overflow risk

## num_predict

- Original config: `num_predict: 256` (maximum generated tokens)
- Changed to `num_predict: 512` during Phase 1 because several answers were
  being silently truncated mid-sentence. The increase allows the model to
  complete multi-paragraph answers while remaining well within the context
  window budget (prompt ≤ 2865 tokens + 512 predict = 3377, under 4096).
- All generation runs (BM25, Dense, closed-book, closed_book_instructed,
  currency) use `num_predict: 512`.

## Retrieval Ablation (exploratory: run after the main evaluation)

Three methods compared, each with BM25 and dense retrievers (6 configs total).
Evaluated on 35 answerable questions (29 known + 6 inferred).

### Original (baseline)

- **BM25 index text**: `contents` field only
- **Dense embedding text**: `contents` field only
- No headings or titles included in indexed/embedded text

### Method 1: Contextual Headers

- **Index text**: `{title} > {heading}: {contents}` with `(#)` artefacts
  removed from headings
- Passage IDs unchanged; this only changes what text is indexed/embedded
- Sub-chunking for passages over 512 tokens (bge-small-en-v1.5 tokenizer):
  max 400 tokens per chunk, 50-token overlap. A passage's score is the best
  score among its sub-chunks.
- 8 passages exceeded 512 tokens: S03_002 (557), S03_003 (990), S03_008 (513),
  S05_012 (659), S06_010 (604), S08_006 (691), S09_007 (563), S10_011 (572)
- Embeddings cached separately in `data/embeddings/ctx_passage_embeddings.npy`

### Method 2: Document-First (two-stage)

- **Stage 1**: Rank source documents using a profile of title + all passage
  headings; keep top 3 documents.
- **Stage 2**: Rank passages only from those 3 documents using original passage
  text (`contents` field), return top 5.
- Uses original passage text in stage 2, not contextual headers, so each method
  is tested on its own.
- BM25 stage 1 kept a relevant document: 21/29 known, 6/6 inferred
- Dense stage 1 kept a relevant document: 28/29 known, 6/6 inferred

### Tukey HSD (α = 0.01, diff = group1 − group2)

No pairwise comparison reached significance at α = 0.01, for either retriever,
on any metric. Largest effect: BM25 ctx − docfirst nDCG@5 diff = +0.2024,
p = 0.0518.

### Generation check: settlein_v4_ctx

The `settlein_v4_ctx` prompt variant uses the same prompt text as v4 but formats
each passage as `[ID] {title} > {heading}: {contents}`. This was run on the 6
currency questions only (T01Q01, T02Q01, T03Q01, BM25 and dense, k=5,
include_superseded=true) to check whether contextual headers help the model
avoid stating the outdated 40-hour work limit. Result: contextual headers did
not remove the outdated 40-hour limit, which still appears in 2 of 6 answers
(BM25 T02Q01, dense T01Q01), and dense T03Q01 refused.

## Embedding Model Comparison (exploratory: run after the main evaluation)

Two dense embedding models compared on original passage text (no contextual
headers, no sub-chunking). Evaluated on 35 answerable questions (29 known +
6 inferred). Run file: `src/experiments/embedding_comparison.py`.

### Models

| Model | ID | Dim | Context | Prefix (query) | Prefix (doc) |
|-------|:--:|:---:|:-------:|----------------|--------------|
| bge-small-en-v1.5 | sentence-transformers | 384 | 512 | (none) | (none) |
| nomic-embed-text | 0a109f422b47 (Ollama) | 768 | 2048 | `search_query: ` | `search_document: ` |

mxbai-embed-large (468836162de7, 1024-dim, 512-token context) could not be run
reliably within our pipeline (the embedding endpoint returned context-length
errors), so it was excluded.

### Truncation

- **bge-small**: 7 passages exceed 512 tokens (S03_002: 547, S03_003: 977,
  S05_012: 643, S06_010: 591, S08_006: 669, S09_007: 544, S10_011: 553)
- **nomic**: no passages exceed 2048 tokens

### Results (combined, n=35)

| Model | nDCG@1 | nDCG@3 | nDCG@5 | Recall@5 | Hit@5 | MRR@5 |
|-------|:------:|:------:|:------:|:--------:|:-----:|:-----:|
| bge_small | 0.5429 | 0.6371 | 0.6952 | 0.7767 | 0.9429 | 0.7714 |
| nomic | 0.7000 | 0.6577 | 0.7060 | 0.7467 | 0.9714 | 0.8357 |

### Tukey HSD (α = 0.01, diff = bge_small − nomic)

| Metric | Diff | p |
|--------|:----:|:-:|
| nDCG@5 | −0.0108 | 0.8793 (ns) |
| Recall@5 | +0.0300 | 0.6831 (ns) |
| Hit@5 | −0.0286 | 0.5618 (ns) |
| MRR@5 | −0.0643 | 0.3874 (ns) |

No comparison reaches significance at α = 0.01.

### Diagnostic questions

T11Q02 and T25Q02 scored zero across all 6 retrieval ablation configs (BM25
and dense × original / ctx / docfirst). nomic-embed-text retrieves at least
one relevant passage for both: T11Q02 nDCG@5 = 0.2611, T25Q02 nDCG@5 = 0.7602.

### Fairness: nDCG@5 gaps

| Pair | bge_small gap | nomic gap |
|------|:------------:|:---------:|
| T04 | 0.0761 | 0.0000 |
| T11 | 0.6388 | 0.3777 |
| T18 | 0.0793 | 0.0361 |
| T25 | 0.9239 | 0.1900 |
| T30 | 0.0498 | 0.0000 |
| T39 | 0.0000 | 0.0000 |

Note: 6 pairs is too few for a statistical comparison across models.
nomic reduces the gap on T11 and T25 (the two hardest fairness pairs).

## Hybrid Search: Reciprocal Rank Fusion (exploratory: run after the main evaluation)

Fuses the existing BM25 and dense (bge-small-en-v1.5) rankings using RRF:
`score(p) = Σ 1/(60 + rank)` over the two full ranked lists (all 107 passages).
Run file: `src/experiments/hybrid_rrf.py`.

### Results (combined, n=35)

| Config | nDCG@1 | nDCG@3 | nDCG@5 | Recall@5 | Hit@5 | MRR@5 |
|--------|:------:|:------:|:------:|:--------:|:-----:|:-----:|
| bm25 | 0.6143 | 0.6408 | 0.6575 | 0.6586 | 0.8571 | 0.7738 |
| dense | 0.5429 | 0.6371 | 0.6952 | 0.7767 | 0.9429 | 0.7714 |
| hybrid_rrf | 0.6000 | 0.7021 | 0.7278 | 0.7676 | 0.9429 | 0.8238 |

### Tukey HSD (α = 0.01, diff = group1 − group2)

| Metric | Comparison | Diff | p |
|--------|-----------|:----:|:-:|
| nDCG@5 | bm25 − dense | −0.0377 | 0.8651 (ns) |
| nDCG@5 | bm25 − hybrid_rrf | −0.0703 | 0.6052 (ns) |
| nDCG@5 | dense − hybrid_rrf | −0.0326 | 0.8969 (ns) |
| Recall@5 | bm25 − dense | −0.1181 | 0.2800 (ns) |
| Recall@5 | bm25 − hybrid_rrf | −0.1090 | 0.3369 (ns) |
| Recall@5 | dense − hybrid_rrf | +0.0090 | 0.9924 (ns) |
| Hit@5 | bm25 − dense | −0.0857 | 0.4121 (ns) |
| Hit@5 | bm25 − hybrid_rrf | −0.0857 | 0.4121 (ns) |
| Hit@5 | dense − hybrid_rrf | −0.0000 | 1.0000 (ns) |
| MRR@5 | bm25 − dense | +0.0024 | 0.9995 (ns) |
| MRR@5 | bm25 − hybrid_rrf | −0.0500 | 0.8097 (ns) |
| MRR@5 | dense − hybrid_rrf | −0.0524 | 0.7932 (ns) |

No comparison reaches significance at α = 0.01.

### Diagnostic questions

T11Q02 and T25Q02 remain at zero for all three methods (BM25, dense, hybrid).

### Fairness: nDCG@5 gaps

| Pair | bm25 gap | dense gap | hybrid_rrf gap |
|------|:--------:|:---------:|:--------------:|
| T04 | 0.1900 | 0.0761 | 0.0498 |
| T11 | 0.5209 | 0.6388 | 0.5406 |
| T18 | 0.1597 | 0.0793 | 0.0761 |
| T25 | 1.0000 | 0.9239 | 1.0000 |
| T30 | 0.1637 | 0.0498 | 0.0000 |
| T39 | 0.0000 | 0.0000 | 0.0000 |

Note: 6 pairs is too few for a statistical comparison.

## Paired Significance Testing (evaluation correction)

The comparisons above used statsmodels `pairwise_tukeyhsd`, which treats each
configuration's 35 per-question scores as an independent sample. Every
configuration answers the same questions, so the design is paired: the large
between-question variance should cancel out, and an unpaired test hides real
differences. All retrieval comparisons were re-run with a two-sided paired
randomization (sign-flip) test, 10,000 permutations, Holm-adjusted within each
experiment and metric, α = 0.01, with 95% bootstrap CIs on the mean
difference. Code: `src/evaluation/stats.py`,
`src/evaluation/paired_retrieval_stats.py`; output:
`results/paired_retrieval_stats.csv` (Tukey p shown alongside).

Comparisons that change from not significant to significant:

| Experiment | Metric | Comparison | Diff | p (paired, Holm) | p (Tukey) |
|------------|--------|-----------|:----:|:----------------:|:---------:|
| main | Recall@5 | bm25 − dense | −0.1181 | 0.0088 | 0.1453 |
| hybrid_rrf | Recall@5 | bm25 − hybrid_rrf | −0.1090 | 0.0096 | 0.3369 |
| retrieval_ablation | Recall@5 | bm25_ctx − bm25_docfirst | +0.2076 | 0.0091 | 0.1143 |

Everything else reported as not significant above remains not significant.

## Answer Quality vs Gold Answers (evaluation addition)

`src/evaluation/answer_quality_eval.py` compares each answer with
`data/gold_answers.csv` on the 35 answerable questions (citations stripped,
refusals score 0, no LLM judge): token recall/F1 (SQuAD-style, stop words
removed), ROUGE-L recall/F1, and bge-small cosine similarity. Generated
answers are much longer than gold answers, so recall is the main lexical
number. Output: `results/answer_quality.csv`, `results/answer_quality_stats.csv`.

| Config | Token Recall | Token F1 | ROUGE-L F1 | Semantic Sim |
|--------|:-----------:|:--------:|:----------:|:------------:|
| bm25-k5-settlein_v4 | 0.4109 | 0.3933 | 0.3405 | 0.8412 |
| dense-k5-settlein_v4 | 0.4275 | 0.4228 | 0.3513 | 0.8187 |
| closed_book | 0.4067 | 0.1260 | 0.0888 | 0.8064 |
| closed_book_instructed | 0.2225 | 0.1061 | 0.0727 | 0.5538 |

Closed-book answers cover as much gold content as RAG answers (token recall
0.41 vs 0.41–0.43, not significant) but are long and generic, so their F1 is
about a third of RAG's. Lexical overlap cannot tell a correct paraphrase from
a wrong statement, so these metrics complement, not replace, the manual
faithfulness coding.

## Answerability Reporting (evaluation addition)

`answerability_eval.py` now reports 95% Wilson intervals, lenient refusal
rates (any decline phrase) and balanced accuracy, (correct refusal + 1 −
over-refusal) / 2. With 13 out-of-KB questions one question is 7.7 points:
BM25 46.2% [23.2, 70.9] vs dense 53.8% [29.1, 76.8] is a one-question
difference.

## Cross-Encoder Reranking (exploratory: run after the main evaluation)

Run file: `src/experiments/rerank_experiment.py`. First stage `hybrid_ctx`:
RRF (k = 60) of contextual-header BM25 and contextual-header dense
(bge-small, sub-chunked as in Method 1). The top 20 are reranked by a
cross-encoder reading `{title} > {heading}: {contents}` (max 512 tokens,
sigmoid score in [0, 1]). Two rerankers were tried: `BAAI/bge-reranker-base`
and `BAAI/bge-reranker-v2-m3`. Recall@20 of the pool is 0.9295, which caps
what reranking can recover.

### Results (combined, n=35)

| Config | nDCG@1 | nDCG@3 | nDCG@5 | Recall@5 | Hit@5 | MRR@5 |
|--------|:------:|:------:|:------:|:--------:|:-----:|:-----:|
| bm25 | 0.6143 | 0.6408 | 0.6575 | 0.6586 | 0.8571 | 0.7738 |
| dense | 0.5429 | 0.6371 | 0.6952 | 0.7767 | 0.9429 | 0.7714 |
| hybrid_rrf | 0.6000 | 0.7021 | 0.7278 | 0.7676 | 0.9429 | 0.8238 |
| hybrid_ctx | 0.7571 | 0.7545 | 0.7849 | 0.7900 | 0.9714 | 0.9048 |
| hybrid_ctx+rerank_base | 0.7143 | 0.7566 | 0.7740 | 0.8186 | 0.9714 | 0.8714 |
| hybrid_ctx+rerank_m3 | 0.8571 | 0.8046 | 0.8452 | 0.8633 | 0.9714 | 0.9571 |

### Paired randomization test (Holm across all 15 pairs per metric, α = 0.01)

| Metric | Comparison | Diff | p_holm |
|--------|-----------|:----:|:------:|
| nDCG@5 | dense − hybrid_ctx+rerank_m3 | −0.1500 | 0.0015 (sig) |
| nDCG@5 | hybrid_rrf − hybrid_ctx+rerank_m3 | −0.1173 | 0.0060 (sig) |
| nDCG@5 | hybrid_ctx − hybrid_ctx+rerank_m3 | −0.0603 | 0.0952 (ns) |
| Recall@5 | dense − hybrid_ctx+rerank_m3 | −0.0867 | 0.0700 (ns) |
| MRR@5 | dense − hybrid_ctx+rerank_m3 | −0.1857 | 0.0360 (ns) |

bge-reranker-base does not improve on hybrid_ctx. T11Q02 now retrieves
relevant passages (nDCG@5 0.6388 with either reranker); T25Q02 is still zero
for every configuration. Six configurations were compared on the test set,
so the best one is optimistically selected; treat its scores as an upper
estimate until confirmed on held-out questions.

### Reranker score as an answerability signal

Top-1 score, answerable (35) vs out-of-KB (13), ROC AUC: bm25 0.804, dense
0.884, hybrid_rrf 0.742, hybrid_ctx 0.800, rerank_base 0.916, rerank_m3 0.954.
For rerank_m3, 12 of 13 out-of-KB questions have top-1 < 0.025 (T07Q01: 0.251);
two answerable questions fall below 0.05 (T05Q01 0.0026, T25Q02 0.0026).
Full threshold sweep: `results/rerank_answerability.csv`.

### Generation run: hybrid_rerank-k5-settlein_v4-gate05

`python -m src.experiments.batch_run --method hybrid_rerank --variant settlein_v4
--refusal-threshold 0.05 --name hybrid_rerank-k5-settlein_v4-gate05`: hybrid_ctx
+ bge-reranker-v2-m3, top 5, same prompt (settlein_v4) and model (llama3.2:3b)
as the main runs. If the top reranker score is below 0.05 the question is
refused without calling the LLM (`refusal_mechanism = "threshold"`). The
threshold was chosen after seeing the test-set sweep above, so the refusal
numbers are optimistic. Retrieval metrics of this run match the rerank
experiment exactly (nDCG@5 0.8452).

| Config | Correct Refusal | Over-Refusal | Balanced Acc. | Citation Precision | Citation Recall | Token F1 |
|--------|:---------------:|:------------:|:-------------:|:------------------:|:---------------:|:--------:|
| bm25-k5-settlein_v4 | 0.4615 [0.23, 0.71] | 0.0000 [0.00, 0.10] | 0.7308 | 0.5732 | 0.5972 | 0.3933 |
| dense-k5-settlein_v4 | 0.5385 [0.29, 0.77] | 0.0286 [0.01, 0.14] | 0.7549 | 0.6842 | 0.5227 | 0.4228 |
| hybrid_rerank gate 0.05 | 1.0000 [0.77, 1.00] | 0.0857 [0.03, 0.22] | 0.9571 | 0.7889 | 0.4714 | 0.4257 |

- 12 of 13 out-of-KB questions were refused by the threshold, T07Q01 by the
  prompt. McNemar vs dense on correct refusal: 6 discordant pairs, all in
  favour of hybrid_rerank, p = 0.0312 (not significant at α = 0.01).
- Over-refusals: T05Q01 and T25Q02 (threshold), T03Q01 (prompt, even though
  S17_001 was ranked first). McNemar vs dense p = 0.50.
- Answer quality against gold answers does not change (all paired p > 0.5):
  with retrieval improved, the 3B generator is now the limiting factor.
- T25Q01 cites S06_001, which was not retrieved.
