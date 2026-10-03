"""Answer quality against the human-reviewed gold answers.

The other evaluation modules score retrieval, refusal and citations, but none
compares the generated answer with data/gold_answers.csv. This module adds
three reference-based metrics for the answerable questions (known +
inferred). No LLM judge is used.

  - token_recall / token_f1: SQuAD-style bag-of-words overlap with the gold
    answer (lower-cased, punctuation and stop words removed). Generated
    answers are much longer than the gold answers, so recall (how much of the
    gold content the answer covers) is the more meaningful number; F1 is
    reported for completeness.
  - rougeL_recall / rougeL_f1: longest-common-subsequence overlap, which also
    rewards keeping facts in order.
  - semantic_sim: cosine similarity between bge-small-en-v1.5 embeddings of
    answer and gold.

Citations ([S03_002]) are stripped before scoring. A refusal on an answerable
question scores 0 on every metric. Out-of-KB questions are handled by
answerability_eval.py and are not scored here.

Outputs:
  results/answer_quality.csv         per question per config
  results/answer_quality_stats.csv   paired tests between configs
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.stats import bootstrap_ci, compare_configs

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
STOP_WORDS = set("""a an and are as at be been but by can do does for from has have
if in into is it its of on or so than that the their them then there these they this
to was were will with you your i my me we our""".split())
METRICS = ["token_recall", "token_f1", "rougeL_recall", "rougeL_f1", "semantic_sim"]


def strip_citations(text: str) -> str:
    return re.sub(r"\[S\d{2}_\d{3}\]", "", text)


def tokens(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", strip_citations(text).lower())
    return [w for w in text.split() if w not in STOP_WORDS]


def prf(overlap: float, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    if overlap == 0:
        return 0.0, 0.0, 0.0
    p, r = overlap / n_pred, overlap / n_gold
    return p, r, 2 * p * r / (p + r)


def token_overlap(pred: list[str], gold: list[str]) -> tuple[float, float, float]:
    common = sum((Counter(pred) & Counter(gold)).values())
    return prf(common, len(pred), len(gold))


def lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: list[str], gold: list[str]) -> tuple[float, float, float]:
    return prf(lcs_length(pred, gold), len(pred), len(gold))


def load_gold() -> dict[str, str]:
    with open(REPO_ROOT / "data" / "gold_answers.csv", encoding="utf-8") as f:
        return {r["question_id"].strip(): r["gold_answer"] for r in csv.DictReader(f)}


def load_generations(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def evaluate_all_configs() -> list[dict]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    gold = load_gold()

    rows = []
    gen_dir = REPO_ROOT / "results" / "generations"
    for gen_file in sorted(gen_dir.glob("*.jsonl")):
        if "currency" in gen_file.stem:
            continue
        gens = [g for g in load_generations(gen_file)
                if g.get("question_type") in ("known", "inferred") and g["question_id"] in gold]
        if not gens:
            continue
        answers = [strip_citations(g["answer"]) for g in gens]
        golds = [gold[g["question_id"]] for g in gens]
        emb_a = model.encode(answers, normalize_embeddings=True)
        emb_g = model.encode(golds, normalize_embeddings=True)

        for g, a_emb, g_emb in zip(gens, emb_a, emb_g):
            refused = g.get("refused_strict", g.get("refused", False))
            pred, ref = tokens(g["answer"]), tokens(gold[g["question_id"]])
            _, t_r, t_f = token_overlap(pred, ref)
            _, l_r, l_f = rouge_l(pred, ref)
            sim = float(np.dot(a_emb, g_emb))
            scores = dict(token_recall=t_r, token_f1=t_f, rougeL_recall=l_r,
                          rougeL_f1=l_f, semantic_sim=sim)
            if refused:
                scores = {k: 0.0 for k in scores}
            rows.append({"config": gen_file.stem, "question_id": g["question_id"],
                         "question_type": g["question_type"], "refused": refused,
                         **{k: round(v, 4) for k, v in scores.items()}})
    return rows


def main():
    rows = evaluate_all_configs()
    out = REPO_ROOT / "results" / "answer_quality.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    configs = list(dict.fromkeys(r["config"] for r in rows))
    print(f"{'Config':<28}{'n':>4}" + "".join(f"{m:>22}" for m in METRICS))
    for cfg in configs:
        cr = [r for r in rows if r["config"] == cfg]
        line = f"{cfg:<28}{len(cr):>4}"
        for m in METRICS:
            vals = [r[m] for r in cr]
            lo, hi = bootstrap_ci(vals)
            line += f"   {np.mean(vals):.3f} [{lo:.2f},{hi:.2f}]"
        print(line)

    stats_rows = []
    for m in METRICS:
        per_q = {cfg: {r["question_id"]: r[m] for r in rows if r["config"] == cfg}
                 for cfg in configs}
        stats_rows.extend(compare_configs(per_q, m))
    stats_out = REPO_ROOT / "results" / "answer_quality_stats.csv"
    with open(stats_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)
    print(f"\nWritten: {out}\nWritten: {stats_out}")


if __name__ == "__main__":
    main()
