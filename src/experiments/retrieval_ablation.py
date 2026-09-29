"""Retrieval ablation: Original vs Contextual Headers vs Document-First.

Three methods × two retrievers = six configurations.
No generation except currency check (item 9).
"""

import csv
import json
import math
import re
import sys
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO = Path(r"c:\Users\Abhishek\OneDrive\Documents\Data_Science_Case_study_WIL_project_50")
sys.path.insert(0, str(REPO))

from src.retrieval.search import (
    load_collection, load_superseded_sources, filter_superseded,
    _tokenize, BM25Index, DenseIndex
)


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

def load_gold():
    with open(REPO / "data" / "gold_answers.csv", encoding="utf-8") as f:
        return {r["question_id"].strip(): r for r in csv.DictReader(f)}


# ── Helpers ──────────────────────────────────────────────────────────────────

def clean_heading(h):
    """Remove (#) artefacts from headings."""
    return re.sub(r'\(#[^)]*\)', '', h).strip()

def build_ctx_text(passage):
    """Build contextual text: {title} > {heading}: {contents}"""
    title = passage.get("title", "")
    heading = clean_heading(passage.get("heading", ""))
    contents = passage.get("contents", "")
    parts = []
    if title:
        parts.append(title)
    if heading:
        parts.append(heading)
    prefix = " > ".join(parts)
    if prefix:
        return f"{prefix}: {contents}"
    return contents


# ── Method 1: Contextual Headers ────────────────────────────────────────────

class ContextualBM25Index:
    """BM25 over contextual text: {title} > {heading}: {contents}."""

    def __init__(self, passages, k1=1.2, b=0.75):
        self.passages = passages
        self.k1 = k1
        self.b = b

        self.doc_tokens = [_tokenize(build_ctx_text(p)) for p in passages]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.n_docs = len(passages)

        self.inverted = {}
        for doc_idx, tokens in enumerate(self.doc_tokens):
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            for term, freq in tf.items():
                if term not in self.inverted:
                    self.inverted[term] = []
                self.inverted[term].append((doc_idx, freq))

        self.idf = {}
        for term, postings in self.inverted.items():
            df = len(postings)
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query, top_k=5):
        query_tokens = _tokenize(query)
        scores = [0.0] * self.n_docs
        for token in query_tokens:
            if token not in self.inverted:
                continue
            idf = self.idf[token]
            for doc_idx, tf in self.inverted[token]:
                dl = self.doc_lens[doc_idx]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] += idf * num / den
        ranked = sorted(range(self.n_docs), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.passages[i], scores[i]) for i in ranked if scores[i] > 0]


class ContextualDenseIndex:
    """Dense retrieval with contextual text and sub-chunking for >512 tokens."""

    def __init__(self, passages, model_name="BAAI/bge-small-en-v1.5",
                 max_tokens=512, chunk_size=400, overlap=50):
        self.passages = passages
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._model = None
        self._chunk_embeddings = None
        self._chunk_to_passage = None
        self.over_512_report = []

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def _build(self):
        if self._chunk_embeddings is not None:
            return

        self._load_model()
        tokenizer = self._model.tokenizer

        texts = []
        chunk_to_passage = []

        for pidx, p in enumerate(self.passages):
            ctx = build_ctx_text(p)
            tokens = tokenizer.encode(ctx, add_special_tokens=False)

            if len(tokens) > self.max_tokens:
                self.over_512_report.append({
                    "id": p["id"], "tokens": len(tokens),
                    "heading": p.get("heading", "")[:60]
                })
                # Sub-chunk
                start = 0
                while start < len(tokens):
                    end = min(start + self.chunk_size, len(tokens))
                    chunk_text = tokenizer.decode(tokens[start:end])
                    texts.append(chunk_text)
                    chunk_to_passage.append(pidx)
                    start += self.chunk_size - self.overlap
            else:
                texts.append(ctx)
                chunk_to_passage.append(pidx)

        print(f"  Contextual dense: {len(texts)} chunks from {len(self.passages)} passages")
        if self.over_512_report:
            print(f"  {len(self.over_512_report)} passages over 512 tokens, sub-chunked")

        cache_path = REPO / "data" / "embeddings" / "ctx_passage_embeddings.npy"
        ids_path = REPO / "data" / "embeddings" / "ctx_chunk_map.json"

        current_key = [p["id"] for p in self.passages]
        if cache_path.exists() and ids_path.exists():
            with open(ids_path) as f:
                cached = json.load(f)
            if cached.get("passage_ids") == current_key:
                self._chunk_embeddings = np.load(str(cache_path))
                self._chunk_to_passage = cached["chunk_to_passage"]
                print(f"  Loaded cached ctx embeddings: {self._chunk_embeddings.shape}")
                return

        embeddings = self._model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

        np.save(str(cache_path), embeddings)
        with open(ids_path, "w") as f:
            json.dump({"passage_ids": current_key, "chunk_to_passage": chunk_to_passage}, f)

        self._chunk_embeddings = embeddings
        self._chunk_to_passage = chunk_to_passage

    def search(self, query, top_k=5):
        self._build()
        self._load_model()
        qe = self._model.encode([query], normalize_embeddings=True)
        scores = np.dot(self._chunk_embeddings, qe.T).flatten()

        # Best score per passage
        passage_scores = {}
        for cidx, sc in enumerate(scores):
            pidx = self._chunk_to_passage[cidx]
            if pidx not in passage_scores or sc > passage_scores[pidx]:
                passage_scores[pidx] = float(sc)

        ranked = sorted(passage_scores, key=passage_scores.get, reverse=True)[:top_k]
        return [(self.passages[pidx], passage_scores[pidx]) for pidx in ranked
                if passage_scores[pidx] > 0]


# ── Method 2: Document-First ────────────────────────────────────────────────

class DocumentFirstBM25:
    """Two-stage: rank documents first, then passages within top docs."""

    def __init__(self, passages, k1=1.2, b=0.75, top_docs=3):
        self.passages = passages
        self.top_docs = top_docs

        # Build document profiles: title + all headings
        doc_profiles = {}
        doc_passages = defaultdict(list)
        for pidx, p in enumerate(passages):
            sid = p.get("source_id", "")
            doc_passages[sid].append(pidx)
            if sid not in doc_profiles:
                doc_profiles[sid] = p.get("title", "")
            heading = p.get("heading", "")
            if heading:
                doc_profiles[sid] += " " + heading

        self.doc_ids = sorted(doc_profiles.keys())
        self.doc_texts = [doc_profiles[d] for d in self.doc_ids]
        self.doc_passages = doc_passages

        # Build BM25 for documents
        self.doc_tokens = [_tokenize(t) for t in self.doc_texts]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl_doc = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.n_docs = len(self.doc_ids)

        self.inv_doc = {}
        for didx, tokens in enumerate(self.doc_tokens):
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            for term, freq in tf.items():
                self.inv_doc.setdefault(term, []).append((didx, freq))

        self.idf_doc = {}
        for term, postings in self.inv_doc.items():
            df = len(postings)
            self.idf_doc[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

        # BM25 for passages (original contents only)
        self.passage_bm25 = BM25Index(passages, k1, b)

    def search(self, query, top_k=5):
        # Stage 1: rank documents
        query_tokens = _tokenize(query)
        doc_scores = [0.0] * self.n_docs
        for token in query_tokens:
            if token not in self.inv_doc:
                continue
            idf = self.idf_doc[token]
            for didx, tf in self.inv_doc[token]:
                dl = self.doc_lens[didx]
                num = tf * (1.2 + 1)
                den = tf + 1.2 * (1 - 0.75 + 0.75 * dl / self.avgdl_doc)
                doc_scores[didx] += idf * num / den

        top_doc_ids = sorted(range(self.n_docs), key=lambda i: doc_scores[i],
                            reverse=True)[:self.top_docs]
        kept_source_ids = {self.doc_ids[i] for i in top_doc_ids}

        # Stage 2: rank passages from kept documents only
        kept_passage_idxs = set()
        for sid in kept_source_ids:
            kept_passage_idxs.update(self.doc_passages[sid])

        # Use the passage BM25 but filter
        all_results = self.passage_bm25.search(query, top_k=len(self.passages))
        filtered = [(p, s) for p, s in all_results
                    if self.passages.index(p) if p.get("source_id", "") in kept_source_ids]

        # Re-rank properly
        query_tokens_set = set(query_tokens)
        passage_scores = []
        for pidx in kept_passage_idxs:
            p = self.passages[pidx]
            tokens = _tokenize(p["contents"])
            tf_map = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            dl = len(tokens)
            sc = 0.0
            for qt in query_tokens:
                if qt in tf_map:
                    idf = self.passage_bm25.idf.get(qt, 0)
                    tf = tf_map[qt]
                    num = tf * 2.2
                    den = tf + 1.2 * (1 - 0.75 + 0.75 * dl / self.passage_bm25.avgdl)
                    sc += idf * num / den
            passage_scores.append((pidx, sc))

        passage_scores.sort(key=lambda x: x[1], reverse=True)
        results = [(self.passages[pidx], sc) for pidx, sc in passage_scores[:top_k] if sc > 0]
        return results

    def stage1_kept(self, query):
        """Return which source_ids were kept in stage 1."""
        query_tokens = _tokenize(query)
        doc_scores = [0.0] * self.n_docs
        for token in query_tokens:
            if token not in self.inv_doc:
                continue
            idf = self.idf_doc[token]
            for didx, tf in self.inv_doc[token]:
                dl = self.doc_lens[didx]
                num = tf * 2.2
                den = tf + 1.2 * (1 - 0.75 + 0.75 * dl / self.avgdl_doc)
                doc_scores[didx] += idf * num / den
        top_doc_ids = sorted(range(self.n_docs), key=lambda i: doc_scores[i],
                            reverse=True)[:self.top_docs]
        return {self.doc_ids[i] for i in top_doc_ids}


class DocumentFirstDense:
    """Two-stage dense: rank documents, then passages within top docs."""

    def __init__(self, passages, model_name="BAAI/bge-small-en-v1.5", top_docs=3):
        self.passages = passages
        self.top_docs = top_docs
        self.model_name = model_name

        # Build document profiles
        doc_profiles = {}
        self.doc_passages = defaultdict(list)
        for pidx, p in enumerate(passages):
            sid = p.get("source_id", "")
            self.doc_passages[sid].append(pidx)
            if sid not in doc_profiles:
                doc_profiles[sid] = p.get("title", "")
            heading = p.get("heading", "")
            if heading:
                doc_profiles[sid] += " " + heading

        self.doc_ids = sorted(doc_profiles.keys())
        self.doc_texts = [doc_profiles[d] for d in self.doc_ids]

        # Pre-encode everything once
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._doc_embs = self._model.encode(self.doc_texts, normalize_embeddings=True)
        self._passage_embs = self._model.encode(
            [p["contents"] for p in passages], normalize_embeddings=True)

    def search(self, query, top_k=5):
        q_emb = self._model.encode([query], normalize_embeddings=True)

        # Stage 1: rank docs
        doc_scores = np.dot(self._doc_embs, q_emb.T).flatten()
        top_idxs = np.argsort(doc_scores)[::-1][:self.top_docs]
        kept_sids = {self.doc_ids[i] for i in top_idxs}

        # Stage 2: rank passages from kept docs using pre-computed embeddings
        kept_pidxs = []
        for sid in kept_sids:
            kept_pidxs.extend(self.doc_passages[sid])

        if not kept_pidxs:
            return []

        kept_embs = self._passage_embs[kept_pidxs]
        p_scores = np.dot(kept_embs, q_emb.T).flatten()

        ranked = np.argsort(p_scores)[::-1][:top_k]
        return [(self.passages[kept_pidxs[i]], float(p_scores[i])) for i in ranked
                if p_scores[i] > 0]

    def stage1_kept(self, query):
        q_emb = self._model.encode([query], normalize_embeddings=True)
        doc_scores = np.dot(self._doc_embs, q_emb.T).flatten()
        top_idxs = np.argsort(doc_scores)[::-1][:self.top_docs]
        return {self.doc_ids[i] for i in top_idxs}


# ── Evaluation ───────────────────────────────────────────────────────────────

def ndcg_at_k(retrieved_ids, qrels_for_q, k):
    """Compute nDCG@k."""
    dcg = 0.0
    for i, pid in enumerate(retrieved_ids[:k]):
        rel = qrels_for_q.get(pid, 0)
        dcg += rel / math.log2(i + 2)

    ideal = sorted(qrels_for_q.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(retrieved_ids, qrels_for_q, k):
    relevant = {pid for pid, g in qrels_for_q.items() if g > 0}
    if not relevant:
        return 0.0
    found = relevant & set(retrieved_ids[:k])
    return len(found) / len(relevant)

def hit_at_k(retrieved_ids, qrels_for_q, k):
    relevant = {pid for pid, g in qrels_for_q.items() if g > 0}
    return 1.0 if relevant & set(retrieved_ids[:k]) else 0.0

def mrr(retrieved_ids, qrels_for_q):
    for i, pid in enumerate(retrieved_ids):
        if qrels_for_q.get(pid, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_config(name, index, questions, qrels, top_k=5):
    """Evaluate a retrieval config. Returns per-question results."""
    results = []
    for q in questions:
        qid = q["question_id"]
        qtext = q["question"]
        qtype = q["question_type"]

        if qtype == "out_of_kb":
            continue

        retrieved = index.search(qtext, top_k=top_k)
        rids = [p["id"] for p, _ in retrieved]
        qr = qrels.get(qid, {})

        results.append({
            "config": name,
            "question_id": qid,
            "question_type": qtype,
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
    """Write TREC run file."""
    out_dir = REPO / "runs" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.txt"
    with open(path, "w") as f:
        for q in questions:
            qid = q["question_id"]
            if q["question_type"] == "out_of_kb":
                continue
            retrieved = index.search(q["question"], top_k=top_k)
            for rank, (p, score) in enumerate(retrieved, 1):
                f.write(f"{qid} Q0 {p['id']} {rank} {score:.6f} {name}\n")
    print(f"  TREC run: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    topics = load_topics()
    qrels = load_qrels()
    all_passages = load_collection()
    passages = filter_superseded(all_passages, include_superseded=False)

    answerable = [t for t in topics if t["question_type"] in ("known", "inferred")]
    print(f"Passages: {len(passages)}, Answerable questions: {len(answerable)}")

    # Build all 6 indices
    print("\n=== Building indices ===")
    configs = {}

    # Original
    print("  Building BM25 original...")
    configs["bm25_original"] = BM25Index(passages)
    print("  Building Dense original...")
    configs["dense_original"] = DenseIndex(passages)

    # Method 1: Contextual headers
    print("  Building BM25 contextual...")
    configs["bm25_ctx"] = ContextualBM25Index(passages)
    print("  Building Dense contextual...")
    ctx_dense = ContextualDenseIndex(passages)
    configs["dense_ctx"] = ctx_dense

    # Method 2: Document-first
    print("  Building BM25 docfirst...")
    configs["bm25_docfirst"] = DocumentFirstBM25(passages)
    print("  Building Dense docfirst...")
    configs["dense_docfirst"] = DocumentFirstDense(passages)

    # Report over-512 token passages
    # Need to build ctx embeddings to get the report
    ctx_dense._build()
    if ctx_dense.over_512_report:
        print(f"\n=== Passages over 512 tokens (contextual) ===")
        for r in ctx_dense.over_512_report:
            print(f"  {r['id']}: {r['tokens']} tokens — {r['heading']}")
    else:
        print("\n  No passages over 512 tokens with contextual headers.")

    # Evaluate all configs
    print("\n=== Evaluating ===")
    all_results = []
    for name, idx in configs.items():
        print(f"  {name}...")
        results = evaluate_config(name, idx, topics, qrels)
        all_results.extend(results)
        write_trec_run(name, idx, topics)

    # Document-first stage1 diagnostic
    print("\n=== Document-first stage 1 diagnostic ===")
    for retriever_name, docfirst in [("bm25", configs["bm25_docfirst"]),
                                      ("dense", configs["dense_docfirst"])]:
        for qtype in ["known", "inferred"]:
            qs = [t for t in answerable if t["question_type"] == qtype]
            kept_count = 0
            for q in qs:
                kept_sids = docfirst.stage1_kept(q["question"])
                relevant_pids = {pid for pid, g in qrels.get(q["question_id"], {}).items() if g > 0}
                # Check if any kept doc contains a relevant passage
                relevant_sids = set()
                for p in passages:
                    if p["id"] in relevant_pids:
                        relevant_sids.add(p["source_id"])
                if kept_sids & relevant_sids:
                    kept_count += 1
            print(f"  {retriever_name} docfirst stage1 ({qtype}): {kept_count}/{len(qs)} kept at least one relevant doc")

    # Save results CSV
    csv_path = REPO / "results" / "retrieval_ablation.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["config", "question_id", "question_type", "ndcg1", "ndcg3",
                      "ndcg5", "recall5", "hit5", "mrr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {k: r[k] for k in fieldnames}
            writer.writerow(row)
    print(f"\n  Results: {csv_path}")

    # Print summary tables
    print("\n" + "=" * 80)
    print("RETRIEVAL ABLATION RESULTS")
    print("=" * 80)

    for config_name in sorted(set(r["config"] for r in all_results)):
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
            print(f"    {r['config']:25s}  nDCG@5={r['ndcg5']:.4f}  Hit@5={r['hit5']:.0f}  retrieved={r['retrieved_ids']}")

    # Fairness
    print(f"\n{'='*80}")
    print("FAIRNESS: nDCG@5 for policy/student pairs")
    print(f"{'='*80}")
    fair_topics = load_topics()
    pairs = defaultdict(dict)
    for t in fair_topics:
        if t.get("register") in ("policy", "student"):
            tid = t["topic_id"] if "topic_id" in t else t["question_id"][:3]
            pairs[tid][t["register"]] = t["question_id"]

    for config_name in sorted(set(r["config"] for r in all_results)):
        print(f"\n  {config_name}:")
        for tid in sorted(pairs):
            if "policy" not in pairs[tid] or "student" not in pairs[tid]:
                continue
            policy_r = next((r for r in all_results
                            if r["config"] == config_name and r["question_id"] == pairs[tid]["policy"]), None)
            student_r = next((r for r in all_results
                             if r["config"] == config_name and r["question_id"] == pairs[tid]["student"]), None)
            if policy_r and student_r:
                gap = abs(policy_r["ndcg5"] - student_r["ndcg5"])
                print(f"    {tid}: policy={policy_r['ndcg5']:.4f}  student={student_r['ndcg5']:.4f}  gap={gap:.4f}")

    # Tukey HSD — diff convention: first-named minus second-named
    print(f"\n{'='*80}")
    print("TUKEY HSD (alpha = 0.01) — diff = group1 minus group2")
    print(f"{'='*80}")
    try:
        from scipy.stats import f_oneway
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        for retriever in ["bm25", "dense"]:
            print(f"\n  Retriever: {retriever}")
            methods = ["original", "ctx", "docfirst"]
            for metric in ["ndcg5", "recall5", "hit5", "mrr"]:
                groups = {}
                for m in methods:
                    cname = f"{retriever}_{m}"
                    groups[m] = [r[metric] for r in all_results if r["config"] == cname]

                if all(len(v) > 0 for v in groups.values()):
                    data = []
                    labels = []
                    for m in methods:
                        data.extend(groups[m])
                        labels.extend([m] * len(groups[m]))

                    tukey = pairwise_tukeyhsd(data, labels, alpha=0.01)
                    print(f"\n    {metric}:")
                    for row in tukey.summary().data[1:]:
                        g1, g2, meandiff, p_adj, lower, upper, reject = row
                        # statsmodels reports meandiff as group2 - group1;
                        # negate to get group1 - group2 (first-named minus second-named)
                        diff = -meandiff
                        sig = "sig" if reject else "ns"
                        print(f"      {g1:10s} − {g2:10s}: diff={diff:+.4f}  p={p_adj:.4f}  ({sig})")
    except ImportError as e:
        print(f"  Could not run Tukey HSD: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
