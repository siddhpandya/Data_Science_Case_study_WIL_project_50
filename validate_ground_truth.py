"""Validate data/topics.csv, data/qrels.txt and data/gold_answers.csv against data/collection.jsonl.
Run from the repo root: python validate_ground_truth.py   (exits 1 on any error)"""
import csv, json, sys, collections
REFUSAL = "I don't have information about that in my sources."
coll = {json.loads(l)["id"] for l in open("data/collection.jsonl", encoding="utf-8")}
topics = list(csv.DictReader(open("data/topics.csv", encoding="utf-8")))
gold = list(csv.DictReader(open("data/gold_answers.csv", encoding="utf-8")))
qrels = collections.defaultdict(dict)
for line in open("data/qrels.txt", encoding="utf-8"):
    qid, z, pid, lab = line.rstrip("\n").split("\t")
    qrels[qid][pid] = int(lab)
errs = []
qids = [t["question_id"] for t in topics]
if len(qids) != len(set(qids)): errs.append("duplicate question_id")
byq = {t["question_id"]: t for t in topics}
for qid, d in qrels.items():
    if qid not in byq: errs.append(f"qrels for unknown {qid}")
    for pid, lab in d.items():
        if pid not in coll: errs.append(f"{qid}: passage {pid} not in collection")
        if lab not in (1, 2): errs.append(f"{qid}: bad label {lab}")
        if pid.startswith("S18"): errs.append(f"{qid}: superseded passage {pid} judged relevant")
for t in topics:
    q, ty = t["question_id"], t["question_type"]
    rel = qrels.get(q, {})
    if ty == "known" and 2 not in rel.values(): errs.append(f"{q}: known without a label-2 passage")
    if ty == "inferred" and len(rel) < 2: errs.append(f"{q}: inferred with <2 relevant passages")
    if ty == "out_of_kb" and rel: errs.append(f"{q}: out_of_kb has qrels")
    if t["register"] not in ("", "policy", "student"): errs.append(f"{q}: bad register")
    if t["register"] == "student":
        v = t["variant_of"]
        if v not in byq or byq[v]["register"] != "policy": errs.append(f"{q}: student twin without policy original")
        elif qrels.get(v) != rel: errs.append(f"{q}: twin qrels differ from {v}")
    if t["register"] == "policy" and not any(x["variant_of"] == q for x in topics): errs.append(f"{q}: policy row without twin")
    if t["currency_sensitive"] not in ("true", "false"): errs.append(f"{q}: bad currency flag")
g = {r["question_id"]: r for r in gold}
if set(g) != set(qids): errs.append("gold answers do not match questions one to one")
for q, r in g.items():
    ty = byq[q]["question_type"]
    if ty == "out_of_kb":
        if r["gold_answer"] != REFUSAL or r["passage_ids"]: errs.append(f"{q}: out_of_kb gold must be exact refusal, no passages")
    else:
        for pid in [p.strip() for p in r["passage_ids"].split(";") if p.strip()]:
            if pid not in qrels.get(q, {}): errs.append(f"{q}: gold cites {pid} not in its qrels")
        if not r["passage_ids"]: errs.append(f"{q}: gold without passage ids")
c = collections.Counter(t["question_type"] for t in topics)
print("types:", dict(c), "| pairs:", sum(t["register"]=="policy" for t in topics), "| currency:", sum(t["currency_sensitive"]=="true" for t in topics), "| qrels rows:", sum(len(v) for v in qrels.values()))
print("ERRORS:" if errs else "VALID: 0 errors", *errs, sep="\n  ")
sys.exit(1 if errs else 0)
