"""McNemar's exact test for refusal comparisons."""
import json
import sys
from pathlib import Path
from scipy.stats import binomtest

REPO = Path(r"c:\Users\Abhishek\OneDrive\Documents\Data_Science_Case_study_WIL_project_50")


def load_gen(name):
    recs = [json.loads(l) for l in open(REPO / "results" / "generations" / f"{name}.jsonl")]
    return {r["question_id"]: r for r in recs}


def mcnemar_exact(a_refused, b_refused, qids):
    """McNemar's exact test: only discordant pairs matter."""
    b_only = sum(1 for q in qids if not a_refused[q] and b_refused[q])
    a_only = sum(1 for q in qids if a_refused[q] and not b_refused[q])
    n = b_only + a_only
    if n == 0:
        return 1.0, b_only, a_only
    result = binomtest(min(b_only, a_only), n, 0.5)
    return result.pvalue, b_only, a_only


def main():
    bm25 = load_gen("bm25-k5-settlein_v4")
    dense = load_gen("dense-k5-settlein_v4")
    cb_i = load_gen("closed_book_instructed")

    out_of_kb = {q for q, r in bm25.items() if r["question_type"] == "out_of_kb"}
    answerable = {q for q, r in bm25.items() if r["question_type"] in ("known", "inferred")}

    print("=" * 70)
    print("McNEMAR'S EXACT TEST")
    print("=" * 70)

    for name_a, gen_a, name_b, gen_b in [
        ("BM25", bm25, "CB_instructed", cb_i),
        ("Dense", dense, "CB_instructed", cb_i),
    ]:
        ref_a = {q: gen_a[q]["refused_strict"] for q in gen_a}
        ref_b = {q: gen_b[q]["refused_strict"] for q in gen_b}

        # Correct refusal (out_of_kb only)
        p_cr, b_only_cr, a_only_cr = mcnemar_exact(ref_a, ref_b, out_of_kb)
        print(f"\n  {name_a} vs {name_b} — Correct Refusal (out_of_kb, n={len(out_of_kb)})")
        print(f"    {name_a} refuses, {name_b} doesn't: {a_only_cr}")
        print(f"    {name_b} refuses, {name_a} doesn't: {b_only_cr}")
        print(f"    McNemar p = {p_cr:.4f}")

        # Over-refusal (answerable only)
        p_or, b_only_or, a_only_or = mcnemar_exact(ref_a, ref_b, answerable)
        print(f"\n  {name_a} vs {name_b} — Over-Refusal (answerable, n={len(answerable)})")
        print(f"    {name_a} over-refuses, {name_b} doesn't: {a_only_or}")
        print(f"    {name_b} over-refuses, {name_a} doesn't: {b_only_or}")
        print(f"    McNemar p = {p_or:.4f}")


if __name__ == "__main__":
    main()
