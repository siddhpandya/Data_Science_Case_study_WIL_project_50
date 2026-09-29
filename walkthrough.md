# SettleIN Pipeline Walkthrough

## Prerequisites

- Python 3.12+
- Ollama installed and running (`ollama serve`)
- Model pulled: `ollama pull llama3.2:3b`

## Setup

```bash
pip install -r requirements.txt
```

## 1. Validate Ground Truth

```bash
python validate_ground_truth.py
```

Expected output:
```
types: {'known': 29, 'inferred': 6, 'out_of_kb': 13} | pairs: 6 | currency: 3 | qrels rows: 92
VALID: 0 errors
```

## 2. Run Retrieval + Generation

### Main runs (BM25 and Dense)

```bash
python -m src.experiments.batch_run --method bm25 --variant settlein_v4 --name bm25-k5-settlein_v4
python -m src.experiments.batch_run --method dense --variant settlein_v4 --name dense-k5-settlein_v4
```

### Baselines

```bash
# Bare closed-book (no retrieval, no refusal instruction)
python -m src.experiments.batch_run --method closed_book --variant closed_book --name closed_book

# Instructed closed-book (refusal instruction, no passages)
python -m src.experiments.batch_run --method closed_book --variant closed_book_instructed --name closed_book_instructed
```

### Currency runs (with superseded passages)

```bash
python -m src.experiments.currency_run       # Dense with superseded
python -m src.experiments.currency_bm25_run  # BM25 with superseded
```

## 3. Run Evaluations

```bash
python -m src.evaluation.retrieval_eval      # nDCG, Recall, Hit Rate, MRR
python -m src.evaluation.answerability_eval  # Correct refusal, over-refusal
python -m src.evaluation.attribution_eval    # Citation precision
python -m src.evaluation.fairness_eval       # Register gap (policy vs student)
python -m src.evaluation.currency_eval       # Corpus freshness
python -m src.evaluation.report              # Aggregated summary
```

## 4. Generate Coding Sheets

```bash
python -m src.experiments.generate_coding_sheets
```

Creates:
- `results/coding_sheets/coding_sheet_main.csv` (30 samples, blind A/B)
- `results/coding_sheets/coding_sheet_key.csv` (answer key)
- `results/coding_sheets/double_judging_sheet.csv` (10 samples for kappa)

## 5. Run the App

```bash
streamlit run src/app/app.py --server.port 8501 --server.headless true
```

## Key Files

| Path | Description |
|------|-------------|
| `data/topics.csv` | 48 test questions |
| `data/qrels.txt` | Relevance judgments (TREC format) |
| `data/gold_answers.csv` | Gold passage IDs |
| `data/collection.jsonl` | 111 passages (107 after superseded filter) |
| `data/sources_draft.csv` | Source metadata, superseded flags |
| `config/config.yaml` | Pipeline configuration |
| `results/grid_manifest.json` | Frozen experiment configuration |
| `results/summary.csv` | Aggregated results |
| `docs/methods_notes.md` | System facts and decisions |

## Configuration

Key settings in `config/config.yaml`:
- Model: `llama3.2:3b`
- Context: `num_ctx: 4096`
- Max tokens: `num_predict: 512` (changed from 256 to avoid truncation)
- Temperature: `0.0` (deterministic)
- Seed: `42`
- Prompt: `settlein_v4`
