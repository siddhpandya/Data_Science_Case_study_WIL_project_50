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
