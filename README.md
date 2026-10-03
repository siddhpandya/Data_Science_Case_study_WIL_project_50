# SettleIN: Test-driven RAG for International Student Settlement Support

**RMIT COSC2669 Work Integrated Learning (WIL) Project — Group 50**

## Group Details

- **Group Number**: 50
- **Members**:
  1. Siddh Rashehsbhai Pandya — s4184225
  2. Harsh Manishkumar Patel — s4192459
  3. Viral Modi — s4185484
  4. Shrilesh Rane — S4187342
  5. Shreyas Guduri — S4178204
  6. Abhishek Mathur — S4200870

---

## Overview

SettleIN is a test-driven Retrieval-Augmented Generation (RAG) framework designed to provide accurate, attributed, and current settlement information for international students arriving in Australia (covering visas, workplace rights, tax, transport, health & Medicare, and banking).

### Evaluation Framework (v2 Scoping & Priority Tiers)

Rather than treating all metrics equally, the v2 evaluation enforces an explicit cut line:

1. **CORE (Must Deliver)**:
   - **Retrieval Quality**: Sparse (BM25) vs. Dense (`BAAI/bge-small-en-v1.5`) retrieval evaluated with `ranx` on `nDCG@{1,3,5}`, `Recall@5`, `Hit_Rate@5`, and `MRR`.
   - **Answerability & Refusal**: Correct refusal rate on out-of-KB questions vs. Over-refusal rate on known and inferred questions.
   - **Closed-Book Comparison**: No-retrieval arm against bare model to provide empirical proof that RAG adds value over parametric LLM memory.
   - **Interactive UI**: Single-page Streamlit ablation demo with evidence panel and warning badges.

2. **VALUABLE (High Priority Evidence)**:
   - **Currency**: Stale citation rate and current preference rate evaluated on `currency_sensitive` questions against real planted Wayback historical/current pairs.
   - **Manual Faithfulness & Attribution**: A rigorous 30-answer two-coder manual annotation pass with inter-annotator Cohen's kappa agreement (replacing unvalidated automated LLM-as-judge).

3. **STRETCH (Exploratory Ablations)**:
   - **Fairness Register Contrast**: Evaluation across two distinct registers: `policy` phrasing vs. `student` colloquial phrasing on matched variants.
   - **Chunking Strategy Ablation**: `section` vs. `fixed` vs. `recursive` passage chunking.

### Methodological Note: Deviation from Walert

Walert used `pyserini` (BM25 and DPR/TCT-ColBERT) with `Falcon-7B-instruct`. In SettleIN, we deliberately reproduce the evaluation methodology and experimental design, but substitute the underlying toolchain:
- **Sparse retrieval**: `rank_bm25` (pure Python, eliminating Java/Lucene dependencies)
- **Dense retrieval**: `sentence-transformers` with `BAAI/bge-small-en-v1.5`
- **Vector search**: Exact NumPy cosine similarity (brute-force search is instantaneous for ~100 passages, eliminating external vector DB complexity)
- **Generation**: Local Ollama with `llama3.2:3b` (pinned for reproducible local execution)

---

## Repository Structure

```
settlein/
├── README.md
├── SETTLEIN_BUILD_PLAN.md    # Master phased implementation plan
├── requirements.txt          # Pinned project dependencies
├── config/
│   └── config.yaml           # Central configuration
├── data/
│   ├── raw/                  # Raw fetched HTML and PDF files (immutable)
│   ├── interim/              # Extracted hierarchical sections
│   ├── embeddings/           # Cached passage embeddings
│   ├── sources.csv           # Source registry (HUMAN AUTHORED)
│   ├── sources_draft.csv     # Curated & bridged sources awaiting human confirmation
│   ├── collection.jsonl      # Passage collection (JsonCollection schema)
│   ├── topics.csv            # Questions: Known, Inferred, Out-of-KB with register & currency_sensitive (v2 Schema - HUMAN AUTHORED)
│   ├── qrels.txt             # Relevance judgments in TREC format (HUMAN AUTHORED)
│   ├── gold_answers.csv      # Ideal answers & refusal baselines (HUMAN AUTHORED)
│   └── variants_for_review.csv # Paraphrase candidates for review
├── src/
│   ├── ingest/               # Web/PDF ingestion and extraction
│   ├── retrieval/            # Chunking, BM25, dense index, TREC run generator
│   ├── generation/           # Prompts (walert & settlein), Ollama client, generation
│   ├── evaluation/           # 6 evaluation modules and summary reporter
│   ├── experiments/          # 12-configuration grid ablation runner
│   └── app/                  # Streamlit interactive UI
├── runs/                     # TREC format run files
├── results/                  # Metric tables, CSVs, and figures
└── tests/                    # Schema, ID stability, and metric unit tests
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.11 or 3.12
- [Ollama](https://ollama.com/) installed locally

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Ollama Model
Ensure the Ollama service is running, then pull the pinned model:
```bash
ollama pull llama3.2:3b
```

---

## Execution Workflow

- **Validate Ground Truth**:
  ```bash
  python validate_ground_truth.py
  ```
- **Run Retrieval + Generation** (BM25 and Dense):
  ```bash
  python -m src.experiments.batch_run --method bm25 --variant settlein_v4 --name bm25-k5-settlein_v4
  python -m src.experiments.batch_run --method dense --variant settlein_v4 --name dense-k5-settlein_v4
  # exploratory: hybrid retrieval + cross-encoder rerank + refusal gate
  python -m src.experiments.rerank_experiment
  python -m src.experiments.batch_run --method hybrid_rerank --variant settlein_v4 --refusal-threshold 0.05 --name hybrid_rerank-k5-settlein_v4-gate05
  ```
- **Run Evaluations**:
  ```bash
  python -m src.evaluation.retrieval_eval
  python -m src.evaluation.answerability_eval
  python -m src.evaluation.attribution_eval
  python -m src.evaluation.answer_quality_eval
  python -m src.evaluation.paired_retrieval_stats
  python -m src.evaluation.report
  ```
- **Interactive Streamlit Demo**:
  ```bash
  streamlit run src/app/app.py
  ```

See [walkthrough.md](walkthrough.md) for full reproduction instructions.

---

## Key Results

| Metric | BM25-k5 | Dense-k5 | Paired p (BM25 vs Dense) | Hybrid + rerank (exploratory) |
|--------|:-------:|:--------:|:------------------------:|:-----------------------------:|
| nDCG@5 | 0.6575 | 0.6952 | 0.45 (ns) | 0.8452 |
| Recall@5 | 0.6586 | 0.7767 | **0.0088** | 0.8633 |
| Hit@5 | 0.8571 | 0.9429 | 0.25 (ns) | 0.9714 |
| MRR@5 | 0.7738 | 0.7714 | 1.00 (ns) | 0.9571 |
| Correct Refusal (n=13) | 46.2% | 53.8% | — | 100% |
| Over-Refusal (n=35) | 0.0% | 2.9% | — | 8.6% |
| Citation Precision | 57.3% | 68.4% | — | 78.9% |
| Token F1 vs gold answer | 0.393 | 0.423 | — | 0.426 |

**Model**: llama3.2:3b · **Prompt**: settlein_v4 · **Context**: num_ctx=4096, num_predict=512

Significance uses a paired randomization test (n=35 answerable questions, Holm-adjusted,
α = 0.01). Dense retrieves significantly more relevant passages than BM25 (Recall@5); no
other BM25-vs-dense difference is significant. The earlier unpaired Tukey HSD ignored the
per-question pairing and reported all differences as non-significant.

**Hybrid + rerank** (`--method hybrid_rerank --refusal-threshold 0.05`) fuses
contextual-header BM25 and dense with RRF, reranks the top 20 with `BAAI/bge-reranker-v2-m3`,
and refuses without generating when the top reranker score is below 0.05. Its nDCG@5 gain
over dense is significant (paired p_holm = 0.0003). It was selected after comparing six
configurations on the test set, and its refusal threshold was chosen from a test-set
sweep, so these numbers are optimistic until confirmed on held-out questions. See
`docs/methods_notes.md` and `results/summary.md` for confidence intervals and details.
