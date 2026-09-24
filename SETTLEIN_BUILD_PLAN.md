# SettleIN Build Plan

Test-driven Retrieval-Augmented Generation for international student settlement support.
RMIT COSC2669 WIL Project, Group 50.

---

## How to use this file

Place at repo root. Work through phases in order. Each phase has a deliverable and
acceptance criteria. Do not begin a phase until the previous phase's acceptance
criteria pass.

Report progress against phase numbers. If a design decision in this document turns
out to be wrong or infeasible, stop and say so rather than silently substituting
something else, because several of these choices exist to make the evaluation valid
rather than to make the code convenient.

---

## HARD CONSTRAINT: do not generate ground truth

The following files are the academic contribution of this project and must be
written by the human team. Do not generate, infer, auto-populate or "helpfully fill
in" any of them:

- `data/topics.csv` (the Known, Inferred and Out-of-KB questions)
- `data/qrels.txt` (relevance judgments)
- `data/gold_answers.csv` (ideal answers)
- `data/sources.csv` (source registry, including last-updated dates and licences)

If an LLM writes the questions from the corpus, and an LLM answers them, and an LLM
judges the answers, the evaluation is circular and worthless. Create these files as
empty templates with correct headers and a two-row worked example only. Then stop
and wait.

The one permitted exception, which reproduces Walert's own method: generating
paraphrase variants of human-written Known questions. Those go to `data/
variants_for_review.csv` and are only promoted into `topics.csv` after a human
marks each as accepted. Build the review script, never the promotion.

---

## Target stack

| Concern | Choice | Notes |
|---|---|---|
| Python | 3.11 | Pin it |
| Sparse retrieval | `rank_bm25` | Pure Python, no Java, no Anserini |
| Dense retrieval | `sentence-transformers` | `BAAI/bge-small-en-v1.5`, CPU is fine |
| Vector search | numpy cosine, brute force | Corpus is ~100 passages, exact search is instant. Do not add FAISS |
| Generation | Ollama, `llama3.1:8b` | Local, no API cost |
| IR metrics | `ranx` | Same library Walert's `eval.py` uses |
| UI | Streamlit | Single page |
| Config | YAML, one file | No hardcoded paths or model names anywhere in `src/` |

Deviation from Walert to record in the report: Walert used pyserini (BM25 and
DPR/TCT-ColBERT) with Falcon-7B-instruct. We reproduce the methodology and the
evaluation design, not the exact toolchain, to avoid a Java dependency. State this
explicitly in the README and the final report.

---

## Repository layout

```
settlein/
├── README.md
├── SETTLEIN_BUILD_PLAN.md
├── requirements.txt
├── config/
│   └── config.yaml
├── data/
│   ├── raw/                      # fetched HTML and PDF, immutable
│   ├── sources.csv               # HUMAN AUTHORED
│   ├── collection.jsonl          # generated
│   ├── topics.csv                # HUMAN AUTHORED
│   ├── qrels.txt                 # HUMAN AUTHORED
│   ├── gold_answers.csv          # HUMAN AUTHORED
│   └── variants_for_review.csv   # generated, human accepted
├── src/
│   ├── ingest/
│   │   ├── fetch.py
│   │   ├── extract.py
│   │   └── build_collection.py
│   ├── retrieval/
│   │   ├── chunking.py
│   │   ├── bm25_index.py
│   │   ├── dense_index.py
│   │   └── search.py
│   ├── generation/
│   │   ├── prompts.py
│   │   ├── ollama_client.py
│   │   └── generate.py
│   ├── evaluation/
│   │   ├── retrieval_eval.py
│   │   ├── answerability_eval.py
│   │   ├── faithfulness_eval.py
│   │   ├── attribution_eval.py
│   │   ├── currency_eval.py
│   │   ├── fairness_eval.py
│   │   └── report.py
│   ├── experiments/
│   │   └── run_grid.py
│   └── app/
│       └── app.py
├── runs/                         # TREC format run files
├── results/                      # metric tables and figures
└── tests/
```

---

## Data schemas

Get these exactly right. Everything downstream depends on them, and two of them are
deliberately shaped to match Walert's formats so the reproduction claim is checkable.

### `data/sources.csv`

```
source_id,url,publisher,title,retrieved_at,last_updated,licence,is_superseded,superseded_by,notes
```

- `source_id`: `S01`, `S02`, ...
- `last_updated`: ISO date from the page footer, or empty if the page does not state one
- `is_superseded`: `true` or `false`. See Phase 1 note on planted stale documents
- `superseded_by`: `source_id` of the current version, else empty
- `licence`: as stated by the publisher, for example `CC BY 4.0`, `Crown copyright`, `unspecified`

### `data/collection.jsonl`

One JSON object per line. Field name `id` and `contents` are chosen to match
pyserini's `JsonCollection` format so the data is portable.

```json
{
  "id": "P001",
  "contents": "full text of the section",
  "source_id": "S03",
  "section_heading": "Work conditions for student visa holders",
  "url": "https://...",
  "last_updated": "2025-07-14",
  "retrieved_at": "2026-09-22",
  "is_superseded": false,
  "topic_area": "visa"
}
```

`topic_area` is one of: `visa`, `health`, `banking`, `tax`, `transport`,
`work_rights`, `accommodation`.

### `data/topics.csv` (v2 Schema)

```
topic_id,topic,question_id,question,question_type,variant_of,register,currency_sensitive
```

- `topic_id`: `T01`, `T02`, ...
- `topic`: topic/question title grouping a question with its variants (e.g. `student_visa_work_hours`)
- `question_id`: `T01Q01`, `T01Q02`, ...
- `question_type`: `known` | `inferred` | `out_of_kb`
- `variant_of`: the `question_id` of the original question this paraphrases, empty if original
- `register`: linguistic register: exactly `policy` or `student` (applied only to paired fairness variants; left empty on all other questions)
- `currency_sensitive`: `true` | `false` (whether the answer depends on time-sensitive or superseded rules)

### `data/qrels.txt`

TREC format, tab separated, no header:

```
T01Q01	0	P007	2
T01Q01	0	P012	1
```

Relevance labels follow Walert: `2` = highly relevant, the passage fully answers the
question. `1` = partially relevant, contributes part of an answer. Out-of-KB
questions have no rows at all.

### `data/gold_answers.csv`

```
question_id,gold_answer,passage_ids
```

For out-of-KB questions, `gold_answer` is the refusal text and `passage_ids` is empty.

---

## Phase 0: Scaffold

**Do**
- Create the directory tree above with `.gitkeep` in empty dirs
- `requirements.txt` with pinned versions
- `config/config.yaml` covering: paths, chunking params, retrieval params, model
  names, generation params, evaluation params
- `.gitignore`: `__pycache__`, `.venv`, `runs/*.txt`, `results/*`, keep `data/`
- Empty templates with headers plus one worked example row for each of the four
  human-authored files
- `README.md` stub with setup instructions

**config/config.yaml must include**

```yaml
seed: 42
paths:
  raw: data/raw
  collection: data/collection.jsonl
  topics: data/topics.csv
  qrels: data/qrels.txt
  gold: data/gold_answers.csv
  runs: runs
  results: results
chunking:
  strategy: section        # section | fixed | recursive
  max_tokens: 400
  overlap_tokens: 50
retrieval:
  bm25:
    k1: 1.2
    b: 0.75
  dense:
    model: BAAI/bge-small-en-v1.5
    normalize: true
  top_k: [1, 3, 5]
  score_threshold: null    # set in Phase 4
generation:
  provider: ollama
  model: llama3.1:8b
  temperature: 0.0
  num_predict: 256
  prompt_variant: settlein  # walert | settlein
evaluation:
  metrics: [ndcg@1, ndcg@3, ndcg@5, recall@5, hit_rate@5, mrr]
  alpha: 0.01
```

**Acceptance**: `python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"`
passes, tree matches, four template files exist with correct headers and are
otherwise empty apart from the example rows.

---

## Phase 1: Ingestion

**`src/ingest/fetch.py`**
- Reads `data/sources.csv`, fetches each URL, writes to `data/raw/{source_id}.html`
  or `.pdf`
- Sets a descriptive User-Agent identifying this as academic research
- Respects `robots.txt`. If a source disallows fetching, skip it, log it, and do not
  work around it
- Rate limit: minimum 2 seconds between requests
- Idempotent: skip if the raw file already exists unless `--force`
- Writes `data/raw/fetch_log.csv` with `source_id, status_code, bytes, fetched_at`

**`src/ingest/extract.py`**
- HTML: extract main content, drop nav, footer, cookie banners, "related links"
- PDF: `pypdf` text extraction
- Preserve heading hierarchy. Output intermediate `data/interim/{source_id}.json`
  as a list of `{heading, level, text}`
- Attempt to detect a last-updated date from the page and write it to
  `data/interim/{source_id}_meta.json` for a human to confirm into `sources.csv`.
  Do not write to `sources.csv` directly

**Note on planted superseded documents.** The corpus deliberately contains a small
number of outdated pages (an old rate, threshold or fee page) alongside their
current replacement, marked with `is_superseded=true` and `superseded_by`. This is
intentional and is the basis of the Currency metric in Phase 5. Do not filter,
deduplicate or "fix" them at any stage of the pipeline.

**Acceptance**: every row in `sources.csv` has a corresponding file in `data/raw/`
or a logged skip reason. `extract.py` produces non-empty interim JSON for each.
Spot check three extractions by eye for nav-bar contamination.

---

## Phase 2: Build the collection

**`src/retrieval/chunking.py`**

Three strategies behind one interface, selected by config:

1. `section`: one passage per heading section. This is the default and the unit that
   relevance judgments are written against
2. `fixed`: fixed token window with overlap
3. `recursive`: recursive character splitter, paragraph then sentence

Critical requirement: every chunk, under every strategy, carries a `parent_id`
pointing at its section passage ID. This lets chunking vary in later experiments
without invalidating qrels, because a chunk-level retrieval result is scored by
mapping back to `parent_id`. Do not skip this, it is the reason the testbed stays
valid when chunking is tuned.

**`src/ingest/build_collection.py`**
- Applies the configured chunking strategy
- Emits `data/collection.jsonl` with stable, deterministic IDs. `P001`, `P002`, ...
  assigned in `source_id` then document order, so re-running produces identical IDs
- Prints a summary: passage count, mean and median token length, count per
  `topic_area`, count of superseded passages

**Acceptance**: `collection.jsonl` validates against the schema above, 80 to 120
section passages from 12 to 15 sources, IDs stable across two consecutive runs
(diff the output), every chunk resolves to a `parent_id` present in the section run.

---

## Phase 3: Retrieval

**`src/retrieval/bm25_index.py`** and **`src/retrieval/dense_index.py`**

Common interface:

```python
class Retriever:
    def index(self, collection: list[dict]) -> None: ...
    def search(self, query: str, k: int) -> list[tuple[str, float]]: ...
```

- BM25: `rank_bm25.BM25Okapi`, k1 and b from config, lowercase and simple
  tokenisation, no stemming initially
- Dense: encode passages once, cache embeddings to `data/embeddings/{model}.npy`
  keyed by a hash of the collection so a changed collection invalidates the cache.
  Cosine similarity via normalised dot product with numpy

**`src/retrieval/search.py`**
- Runs every question in `topics.csv` through a retriever
- Writes TREC run files to `runs/{retriever}-{chunking}.txt`:
  `question_id Q0 passage_id rank score run_tag`
- When chunking is not `section`, map each retrieved chunk to its `parent_id` and
  deduplicate, keeping the highest scoring chunk per parent, before writing the run

**Acceptance**: both retrievers produce well-formed TREC runs for all questions.
Sanity check by hand: pick three Known questions, confirm the human-judged relevant
passage appears in the top 5 for at least one retriever. If it never does, the
problem is extraction or chunking, not retrieval. Go back to Phase 2.

---

## Phase 4: Generation

**`src/generation/prompts.py`**

Two prompt variants, both stored as constants so they can be cited verbatim in the
report.

`walert`, reproducing the original for comparability:

```
Generate an answer to be synthesized with text-to-speech for a virtual assistant,
the answer should be based on the retrieved documents for the following question.
If the retrieved documents are not related to the question, then answer NA.
```

`settlein`, the project's own, which additionally requires citation and explicit
refusal:

- answer only from the numbered passages provided
- cite the passage number after each claim, for example `[P012]`
- if the passages do not contain the answer, reply exactly
  `I don't have information about that in my sources.`
- do not give personalised visa, legal, financial or migration advice. For questions
  seeking a personal determination, state that the student should contact the
  relevant authority or their university's international student support team
- do not infer, estimate or fill gaps from general knowledge

**`src/generation/ollama_client.py`**
- HTTP client against the local Ollama endpoint
- `temperature: 0`, fixed seed, `num_predict` from config
- Retry with backoff on connection error, fail loudly if Ollama is not running
  rather than silently returning empty strings

**`src/generation/generate.py`**
- For each question, retrieve top-k, build the prompt, generate
- Optional pre-generation refusal: if `retrieval.score_threshold` is set and the top
  score falls below it, skip the LLM call entirely and return the refusal string
- Writes `results/generations/{config_name}.jsonl` with: `question_id`,
  `question_type`, `answer`, `retrieved_ids`, `retrieved_scores`, `cited_ids`,
  `refused` (bool), `refusal_mechanism` (`threshold` | `prompt` | `none`), `latency_ms`

Parse citations with a regex over `[P\d+]` and store in `cited_ids`.

**Acceptance**: a full generation pass over all questions completes. At least one
out-of-KB question produces a refusal. Manually read ten answers and confirm the
citation format is being followed.

---

## Phase 5: Evaluation

Six modules, one per dimension. Every module writes a tidy CSV to `results/` with
one row per question, so aggregation and significance testing happen in `report.py`
rather than being baked into each metric.

### 5.1 `retrieval_eval.py`

- Load qrels and runs with `ranx.Qrels` and `ranx.Run`
- Metrics: `ndcg@1`, `ndcg@3`, `ndcg@5`, `recall@5`, `hit_rate@5`, `mrr`
- **Report separately for Known and Inferred.** These are different tasks and mixing
  them hides the finding. Out-of-KB questions are excluded here, they have no qrels
- Use `ranx.compare` with `stat_test="tukey"`, `max_p=0.01`, matching Walert

### 5.2 `answerability_eval.py`

This is the metric most likely to be implemented backwards. Two separate numbers,
never one:

- **Correct refusal rate**: percentage of out-of-KB questions that were refused.
  Higher is better. This is Walert's `% Unanswered` column
- **Over-refusal rate**: percentage of Known and Inferred questions that were
  refused. Lower is better. Walert does not report this, it is our addition and it
  catches a system that games the first number by refusing everything

Also break correct refusal rate down by `refusal_mechanism` so the threshold and
prompt mechanisms can be compared.

### 5.3 `faithfulness_eval.py`

- Split each answer into sentences with `nltk`
- For each sentence, judge whether it is supported by the retrieved passages that
  were actually in the prompt
- Primary implementation: LLM-as-judge via Ollama, a separate call per sentence,
  temperature 0, returning `SUPPORTED` / `UNSUPPORTED` / `PARTIAL`
- Output: supported-claim rate per answer

**Validation requirement, do not skip.** An unvalidated LLM judge is not evidence. Emit
`results/faithfulness_sample.csv` containing a random sample of 50 sentences with
the judge's label and an empty `human_label` column. Two team members label it
independently. `report.py` computes Cohen's kappa between human and judge and
between the two humans. Report both in the final write-up. If judge-human agreement
is poor, the automated number is reported only alongside that caveat.

### 5.4 `attribution_eval.py`

For each answer with citations, against qrels:

- **Citation precision**: of the passages cited, the fraction judged relevant
- **Citation recall**: of the relevant passages that were in the prompt, the fraction cited
- **Citation validity**: fraction of cited IDs that exist in the collection at all,
  which catches invented citations
- **Unattributed answer rate**: answers making factual claims with no citation

### 5.5 `currency_eval.py`

Runs over the subset of questions flagged as currency-sensitive, meaning questions
whose answer exists in both a current and a superseded source.

- **Stale citation rate**: fraction of those answers citing a superseded passage.
  Lower is better
- **Current preference rate**: of cases where both the current and superseded
  passages are retrievable, how often the current one is ranked higher
- **Corpus freshness**: percentage of retrieved passages whose `last_updated` is
  within 12 months, reported as descriptive context only, not as a system metric

The first two are system behaviour. The third describes the corpus. Keep them
visibly separate in the output, because conflating them is the error the original
draft made.

### 5.6 `fairness_eval.py`

Uses the paraphrase variants.

- Group questions by `topic_id`
- Per topic, compute nDCG@5 for every variant
- Report: mean across variants, standard deviation across variants, and worst-variant
  performance
- Headline number: mean within-topic standard deviation. High variance means the
  system rewards students who phrase questions like the source documents, which is a
  fairness problem for exactly this user group
- Also report the gap between best-variant and worst-variant mean performance

### 5.7 `report.py`

- Aggregates every per-question CSV into `results/summary.csv`
- Produces the main results table in the shape of Walert's Table 2: rows are
  configurations, columns grouped by Known, Inferred, Out-of-KB
- Emits both Markdown and LaTeX
- Runs Tukey HSD across configurations at alpha 0.01
- Generates figures to `results/figures/`: nDCG by cutoff k, refusal rates by
  question type, faithfulness by configuration, paraphrase variance box plot

**Acceptance**: full evaluation runs end to end from committed data files and writes
a populated `results/summary.csv`. Every number in it traces back to a per-question row.

---

## Phase 6: Experiment grid

**`src/experiments/run_grid.py`**

**Primary Grid (7 core configurations)**:
- Sparse retrieval (`bm25`) with `settlein` prompt: k=1, 3, 5 (3 configurations)
- Dense retrieval (`dense`: `BAAI/bge-small-en-v1.5`) with `settlein` prompt: k=1, 3, 5 (3 configurations)
- Closed-Book baseline (`closed_book`: no-retrieval arm, bare `llama3.2:3b`) (1 configuration)

Total: **7 core configurations** (`bm25-k1-settlein`, `bm25-k3-settlein`, `bm25-k5-settlein`, `dense-k1-settlein`, `dense-k3-settlein`, `dense-k5-settlein`, `closed_book`).
The closed-book arm provides the critical empirical proof that RAG adds value over the bare model.

**Secondary Grid (Ablations on winning primary configuration)**:
- Prompt ablation: `walert` prompt variant (reproducing Walert's prompt on the best retriever/k setting)
- Chunking ablation: `fixed` vs `recursive` chunking strategies

- `--dry-run` prints the plan without executing
- Resumable: skip configurations whose output already exists
- Writes `results/grid_manifest.json` recording every config, its git commit hash,
  model name, and timestamp

**Acceptance**: `python -m src.experiments.run_grid` reproduces all 7 primary configurations
from scratch, and rerunning produces byte-identical retrieval runs.

---

## Phase 7: Streamlit app

**`src/app/app.py`**, single page:

- Question input box
- Sidebar: retriever selector, k slider, prompt variant selector, so the demo doubles
  as a live ablation
- Answer panel, with inline citation markers rendered as links to the evidence below
- Evidence panel beneath the answer, one card per retrieved passage showing: passage
  text, section heading, publisher, source URL, last-updated date, retrieval date,
  and a clear visual warning badge if `is_superseded` is true
- Refusal renders as a plain message with a pointer to the relevant official contact,
  never as an empty answer box
- Persistent footer disclaimer: this is a student research prototype, not official
  advice, and visa or migration questions should go to the Department of Home Affairs
  or the university's international student support team
- No logging of user queries. If logging is added for the demo, it is anonymous,
  disclosed in the UI, and off by default

**Acceptance**: `streamlit run src/app/app.py` serves locally. Every answer displays
its evidence. An out-of-KB question renders the refusal state correctly.

---

## Phase 8: Reproducibility

- `README.md`: setup, Ollama install and `ollama pull llama3.1:8b`, how to run each
  phase, how to regenerate every table and figure in the report
- Pin every dependency version in `requirements.txt`
- Document the deviations from Walert explicitly: retrieval toolchain, dense
  retriever model, generation model
- `tests/`: schema validation for all four data files, an ID-stability test for
  `build_collection.py`, a TREC-format validity test for run files, and a unit test
  for the metric functions against a tiny hand-computed fixture

---

## Sequencing notes

The human-authored testbed files block Phases 3 onward. Phases 0 through 2 can
proceed in parallel with the team writing questions and judgments.

Suggested order of work if time is short: Phase 0, 1, 2, then stop and wait for the
testbed. Then 3, 5.1, 5.2, which together already produce the core result. Then 4,
5.3 through 5.7, then 6. Phase 7 can be built any time after Phase 4 and should not
be left until last.
