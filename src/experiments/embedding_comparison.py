"""Embedding model comparison: bge-small-en-v1.5 vs nomic-embed-text vs mxbai-embed-large.

All three models use original passage text (contents only), no contextual headers,
no sub-chunking. nomic and mxbai run through Ollama's embedding API.
"""

import csv
import json
import math
import re
import sys
import requests
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO = Path(r"c:\Users\Abhishek\OneDrive\Documents\Data_Science_Case_study_WIL_project_50")
sys.path.insert(0, str(REPO))

from src.retrieval.search import load_collection, filter_superseded, _tokenize

OLLAMA_URL = "http://localhost:11434"


# ── Load data ────────────────────────────────────────────────────────────────

def load_topics():
    with open(REPO / "data" / "topics.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_qrels():
    qrels = {}
    with open(REPO / "data" / "qrels.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, pid, grade = parts[:4]
                qrels.setdefault(qid, {})[pid] = int(grade)
    return qrels


# ── Embedding via sentence-transformers (bge-small) ──────────────────────────

class BGESmallIndex:
    """bge-small-en-v1.5 via sentence-transformers (baseline)."""

    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    MODEL_ID = "bge-small-en-v1.5"
    MAX_TOKENS = 512

    def __init__(self, passages):
        self.passages = passages
        self.max_tokens = self.MAX_TOKENS
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.MODEL_NAME)

        # Check truncation
        tokenizer = self._model.tokenizer
        self.over_limit = []
        for p in passages:
            toks = tokenizer.encode(p["contents"], add_special_tokens=False)
            if len(toks) > self.MAX_TOKENS:
                self.over_limit.append({"id": p["id"], "tokens": len(toks)})

        texts = [p["contents"] for p in passages]
        self._embeddings = self._model.encode(texts, normalize_embeddings=True,
                                               show_progress_bar=True)
        print(f"  bge-small: {self._embeddings.shape}, {len(self.over_limit)} over {self.MAX_TOKENS} tokens")

    def search(self, query, top_k=5):
        qe = self._model.encode([query], normalize_embeddings=True)
        scores = np.dot(self._embeddings, qe.T).flatten()
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(self.passages[i], float(scores[i])) for i in ranked if scores[i] > 0]


# ── Embedding via Ollama ─────────────────────────────────────────────────────

def ollama_embed(model, texts, max_retries=3):
    """Embed texts via Ollama's /api/embed endpoint.

    Uses truncate=true so passages exceeding the model's context are
    silently truncated rather than rejected with 400.
    Retries on transient errors (model loading, context swap).
    """
    import time
    for attempt in range(max_retries):
        resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
            "model": model,
            "input": texts,
            "truncate": True,
        })
        if resp.status_code == 200:
            return np.array(resp.json()["embeddings"], dtype=np.float32)
        err_body = resp.text[:200]
        if attempt < max_retries - 1:
            wait = 2 ** attempt
            print(f"    Retry {attempt+1}: {resp.status_code} — {err_body} (waiting {wait}s)")
            time.sleep(wait)
        else:
            print(f"    FAILED after {max_retries} attempts: {resp.status_code} — {err_body}")
            resp.raise_for_status()


class OllamaEmbeddingIndex:
    """Dense retrieval using an Ollama embedding model."""

    def __init__(self, passages, model_name, model_id, max_tokens,
                 query_prefix="", doc_prefix=""):
        self.passages = passages
        self.model_name = model_name
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

        # Example request
        example_text = passages[0]["contents"][:100]
        example_input = f"{doc_prefix}{example_text}" if doc_prefix else example_text
        print(f"\n  Example request for {model_name}:")
        print(f"    POST {OLLAMA_URL}/api/embed")
        print(f"    {{\"model\": \"{model_name}\", \"input\": [\"{example_input[:80]}...\"] }}")

        # Tokenize and check truncation (approximate with whitespace)
        self.over_limit = []
        # We can't easily get exact token counts for Ollama models,
        # so we use a rough word-to-token ratio (~1.3 tokens/word)
        for p in passages:
            word_count = len(p["contents"].split())
            approx_tokens = int(word_count * 1.3)
            if approx_tokens > max_tokens:
                self.over_limit.append({
                    "id": p["id"],
                    "words": word_count,
                    "approx_tokens": approx_tokens,
                })

        # Embed all passages one at a time (Ollama sums context across batch items)
        doc_texts = [f"{doc_prefix}{p['contents']}" for p in passages]
        print(f"  Embedding {len(doc_texts)} passages with {model_name}...")

        all_embs = []
        for i, text in enumerate(doc_texts):
            emb = ollama_embed(model_name, [text])
            all_embs.append(emb)
            if (i + 1) % 20 == 0 or i == len(doc_texts) - 1:
                print(f"    {i+1}/{len(doc_texts)}")

        self._embeddings = np.vstack(all_embs)
        # Normalize
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._embeddings = self._embeddings / norms

        print(f"  {model_name}: {self._embeddings.shape}, {len(self.over_limit)} over {max_tokens} tokens (approx)")

    def search(self, query, top_k=5):
        q_text = f"{self.query_prefix}{query}"
        qe = ollama_embed(self.model_name, [q_text])
        qe = qe / np.linalg.norm(qe)
        scores = np.dot(self._embeddings, qe.T).flatten()
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(self.passages[i], float(scores[i])) for i in ranked if scores[i] > 0]


# ── Evaluation ───────────────────────────────────────────────────────────────

def ndcg_at_k(retrieved_ids, qrels_for_q, k):
    dcg = sum(qrels_for_q.get(pid, 0) / math.log2(i + 2)
              for i, pid in enumerate(retrieved_ids[:k]))
    ideal = sorted(qrels_for_q.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(retrieved_ids, qrels_for_q, k):
    relevant = {pid for pid, g in qrels_for_q.items() if g > 0}
    return len(relevant & set(retrieved_ids[:k])) / len(relevant) if relevant else 0.0

def hit_at_k(retrieved_ids, qrels_for_q, k):
    relevant = {pid for pid, g in qrels_for_q.items() if g > 0}
    return 1.0 if relevant & set(retrieved_ids[:k]) else 0.0

def mrr(retrieved_ids, qrels_for_q, k=5):
    for i, pid in enumerate(retrieved_ids[:k]):
        if qrels_for_q.get(pid, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0

def evaluate_config(name, index, questions, qrels, top_k=5):
    results = []
    for q in questions:
        qid = q["question_id"]
        if q["question_type"] == "out_of_kb":
            continue
        retrieved = index.search(q["question"], top_k=top_k)
        rids = [p["id"] for p, _ in retrieved]
        qr = qrels.get(qid, {})
        results.append({
            "config": name,
            "question_id": qid,
            "question_type": q["question_type"],
            "ndcg1": ndcg_at_k(rids, qr, 1),
            "ndcg3": ndcg_at_k(rids, qr, 3),
            "ndcg5": ndcg_at_k(rids, qr, 5),
            "recall5": recall_at_k(rids, qr, 5),
            "hit5": hit_at_k(rids, qr, 5),
            "mrr": mrr(rids, qr),
            "retrieved_ids": rids,
        })
    return results

def write_trec_run(name, index, questions, top_k=5):
    out_dir = REPO / "runs" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.txt"
    with open(path, "w") as f:
        for q in questions:
            if q["question_type"] == "out_of_kb":
                continue
            retrieved = index.search(q["question"], top_k=top_k)
            for rank, (p, score) in enumerate(retrieved, 1):
                f.write(f"{q['question_id']} Q0 {p['id']} {rank} {score:.6f} {name}\n")
    print(f"  TREC run: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    topics = load_topics()
    qrels = load_qrels()
    all_passages = load_collection()
    passages = filter_superseded(all_passages, include_superseded=False)

    answerable = [t for t in topics if t["question_type"] in ("known", "inferred")]
    print(f"Passages: {len(passages)}, Answerable questions: {len(answerable)}")

    # Model IDs from ollama list
    print("\n=== Model IDs ===")
    print("  bge-small-en-v1.5:   sentence-transformers (local)")
    print("  nomic-embed-text:    0a109f422b47 (Ollama)")
    print("  mxbai-embed-large:   468836162de7 (Ollama)")

    # Build indices
    print("\n=== Building indices ===")
    configs = {}

    print("\n  [1] bge-small-en-v1.5 (sentence-transformers, 512 tokens)")
    configs["bge_small"] = BGESmallIndex(passages)

    print("\n  [2] nomic-embed-text (Ollama, 2048 tokens)")
    configs["nomic"] = OllamaEmbeddingIndex(
        passages,
        model_name="nomic-embed-text:latest",
        model_id="0a109f422b47",
        max_tokens=2048,
        query_prefix="search_query: ",
        doc_prefix="search_document: ",
    )

    # mxbai-embed-large (468836162de7, 512 tokens) dropped:
    # Ollama returns "the input length exceeds the context length" for all passages
    # despite truncate=true and passages well within 512 tokens.
    # Standalone CLI calls work, but the /api/embed endpoint fails consistently
    # within a Python process. Likely an Ollama bug with model context initialisation.
    print("\n  [!] mxbai-embed-large: SKIPPED (Ollama context-length bug)")

    # Report truncation
    print("\n=== Truncation report ===")
    for name, idx in configs.items():
        print(f"\n  {name} (max {idx.max_tokens} tokens):")
        if idx.over_limit:
            for entry in idx.over_limit:
                if "tokens" in entry:
                    print(f"    {entry['id']}: {entry['tokens']} tokens")
                else:
                    print(f"    {entry['id']}: ~{entry['approx_tokens']} tokens ({entry['words']} words)")
        else:
            print("    No passages exceed the limit.")

    # Evaluate
    print("\n=== Evaluating ===")
    all_results = []
    for name, idx in configs.items():
        print(f"  {name}...")
        results = evaluate_config(name, idx, topics, qrels)
        all_results.extend(results)
        write_trec_run(name, idx, topics)

    # Save CSV
    csv_path = REPO / "results" / "embedding_comparison.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["config", "question_id", "question_type", "ndcg1", "ndcg3",
                      "ndcg5", "recall5", "hit5", "mrr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\n  Results: {csv_path}")

    # Summary tables
    print(f"\n{'='*80}")
    print("EMBEDDING COMPARISON RESULTS")
    print(f"{'='*80}")

    for config_name in ["bge_small", "nomic", "mxbai"]:
        rows = [r for r in all_results if r["config"] == config_name]
        known = [r for r in rows if r["question_type"] == "known"]
        inferred = [r for r in rows if r["question_type"] == "inferred"]

        for label, subset in [("Known", known), ("Inferred", inferred), ("Combined", rows)]:
            if not subset:
                continue
            n = len(subset)
            avg = lambda key: sum(r[key] for r in subset) / n
            print(f"\n  {config_name} [{label}, n={n}]")
            print(f"    nDCG@1={avg('ndcg1'):.4f}  nDCG@3={avg('ndcg3'):.4f}  nDCG@5={avg('ndcg5'):.4f}")
            print(f"    Recall@5={avg('recall5'):.4f}  Hit@5={avg('hit5'):.4f}  MRR={avg('mrr'):.4f}")

    # Diagnostic questions
    diag_qids = {"T01Q01", "T03Q01", "T04Q01", "T05Q01", "T11Q02", "T25Q02"}
    print(f"\n{'='*80}")
    print("DIAGNOSTIC QUESTIONS")
    print(f"{'='*80}")
    for qid in sorted(diag_qids):
        rows = [r for r in all_results if r["question_id"] == qid]
        if not rows:
            continue
        print(f"\n  {qid}:")
        for r in rows:
            print(f"    {r['config']:15s}  nDCG@5={r['ndcg5']:.4f}  Hit@5={r['hit5']:.0f}  "
                  f"retrieved={r['retrieved_ids']}")

    # Fairness
    print(f"\n{'='*80}")
    print("FAIRNESS: nDCG@5 for policy/student pairs")
    print(f"{'='*80}")
    pairs = defaultdict(dict)
    for t in topics:
        if t.get("register") in ("policy", "student"):
            tid = t["question_id"][:3]
            pairs[tid][t["register"]] = t["question_id"]

    for config_name in ["bge_small", "nomic", "mxbai"]:
        print(f"\n  {config_name}:")
        for tid in sorted(pairs):
            if "policy" not in pairs[tid] or "student" not in pairs[tid]:
                continue
            pol = next((r for r in all_results
                       if r["config"] == config_name and r["question_id"] == pairs[tid]["policy"]), None)
            stu = next((r for r in all_results
                       if r["config"] == config_name and r["question_id"] == pairs[tid]["student"]), None)
            if pol and stu:
                gap = abs(pol["ndcg5"] - stu["ndcg5"])
                print(f"    {tid}: policy={pol['ndcg5']:.4f}  student={stu['ndcg5']:.4f}  gap={gap:.4f}")

    # Tukey HSD — diff = group1 minus group2
    print(f"\n{'='*80}")
    print("TUKEY HSD (alpha = 0.01) — diff = group1 minus group2")
    print(f"{'='*80}")
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        models = ["bge_small", "nomic"]
        for metric in ["ndcg5", "recall5", "hit5", "mrr"]:
            groups = {}
            for m in models:
                groups[m] = [r[metric] for r in all_results if r["config"] == m]

            if all(len(v) > 0 for v in groups.values()):
                data = []
                labels = []
                for m in models:
                    data.extend(groups[m])
                    labels.extend([m] * len(groups[m]))

                tukey = pairwise_tukeyhsd(data, labels, alpha=0.01)
                print(f"\n  {metric}:")
                for row in tukey.summary().data[1:]:
                    g1, g2, meandiff, p_adj, lower, upper, reject = row
                    diff = -meandiff  # negate: first-named minus second-named
                    sig = "sig" if reject else "ns"
                    print(f"    {g1:15s} − {g2:15s}: diff={diff:+.4f}  p={p_adj:.4f}  ({sig})")
    except ImportError as e:
        print(f"  Could not run Tukey HSD: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
