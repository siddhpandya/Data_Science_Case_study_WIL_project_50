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
- **History**: v1 (initial), v2 (adaptive), v3 (passage-level IDs, strict refusal),
  v4 (v3 + inline citation placement rule)
- **Refusal string**: `I don't have information about that in my sources.`
- **Citation format**: `[S03_002]` (exact passage ID in square brackets)

## Retrieval

- **BM25**: rank_bm25 library, default parameters
- **Dense**: BAAI/bge-small-en-v1.5 (sentence-transformers), cosine similarity
- **Top-k**: 5 passages per query
- **Superseded filter**: sources listed in `data/sources_draft.csv` with
  `is_superseded=TRUE` are excluded before indexing. Currently: S18.

## Collection

- **File**: `data/collection.jsonl` (107 passages after superseded filter)
- **Sources**: 18 source documents (S01–S18), S18 superseded

## Test Set

- **Topics**: `data/topics.csv` — 48 questions
  - 29 known, 6 inferred, 13 out_of_kb
  - 6 policy/student fairness pairs
  - 3 currency questions
- **Relevance judgments**: `data/qrels.txt` — 92 rows, TREC format
- **Gold answers**: `data/gold_answers.csv` — passage_ids column uses semicolons

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

## Retrieval Ablation

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

### Tukey HSD (α = 0.01)

No pairwise comparison reached significance at α = 0.01, for either retriever,
on any metric. Largest effect: BM25 ctx vs docfirst nDCG@5 diff = −0.2024,
p = 0.0518.

### Generation check: settlein_v4_ctx

The `settlein_v4_ctx` prompt variant uses the same prompt text as v4 but formats
each passage as `[ID] {title} > {heading}: {contents}`. This was run on the 6
currency questions only (T01Q01, T02Q01, T03Q01, BM25 and dense, k=5,
include_superseded=true) to check whether contextual headers help the model
avoid stating the outdated 40-hour work limit.
