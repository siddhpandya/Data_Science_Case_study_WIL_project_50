# SettleIN Evaluation Results

## Retrieval Metrics

| Config | nDCG@1 | nDCG@3 | nDCG@5 | Recall@5 | Hit@5 | MRR |
|--------|--------|--------|--------|----------|-------|-----|
| bm25-k5-settlein_v4 | 0.6143 | 0.6408 | 0.6575 | 0.6586 | 0.8571 | 0.7738 |
| dense-k5-settlein_v4 | 0.5429 | 0.6371 | 0.6952 | 0.7767 | 0.9429 | 0.7714 |
| hybrid_rerank-k5-settlein_v4-gate05 | 0.8571 | 0.8046 | 0.8452 | 0.8633 | 0.9714 | 0.9571 |

## Answerability

95% Wilson intervals; 13 out-of-KB and 35 answerable questions.

| Config | Correct Refusal | Over-Refusal | Balanced Acc. |
|--------|-----------------|--------------|---------------|
| bm25-k5-settlein_v4 | 0.4615 [0.232, 0.709] | 0.0 [0.000, 0.099] | 0.7308 |
| closed_book | 0.0 [0.000, 0.228] | 0.0 [0.000, 0.099] | 0.5 |
| closed_book_instructed | 0.1538 [0.043, 0.422] | 0.2571 [0.142, 0.421] | 0.4484 |
| dense-k5-settlein_v4 | 0.5385 [0.291, 0.768] | 0.0286 [0.005, 0.145] | 0.7549 |
| hybrid_rerank-k5-settlein_v4-gate05 | 1.0 [0.772, 1.000] | 0.0857 [0.030, 0.224] | 0.9571 |

## Faithfulness

| Config | Supported-Claim Rate |
|--------|---------------------|

## Attribution

| Config | Citation Precision | Citation Recall | Unattributed Rate |
|--------|-------------------|-----------------|-------------------|
| bm25-k5-settlein_v4 | 0.5732 | 0.5972 | 0.0208 |
| closed_book | None | None | 1.0 |
| closed_book_instructed | None | None | 0.6875 |
| dense-k5-settlein_v4 | 0.6842 | 0.5227 | 0.0417 |
| hybrid_rerank-k5-settlein_v4-gate05 | 0.7889 | 0.4714 | 0.0417 |

## Answer Quality vs Gold Answers

Answerable questions only (n=35); refusals score 0. No LLM judge.

| Config | Token Recall | Token F1 | ROUGE-L F1 | Semantic Sim |
|--------|--------------|----------|------------|--------------|
| bm25-k5-settlein_v4 | 0.4109 | 0.3933 | 0.3405 | 0.8412 |
| closed_book | 0.4067 | 0.126 | 0.0888 | 0.8064 |
| closed_book_instructed | 0.2225 | 0.1061 | 0.0727 | 0.5538 |
| dense-k5-settlein_v4 | 0.4275 | 0.4228 | 0.3513 | 0.8187 |
| hybrid_rerank-k5-settlein_v4-gate05 | 0.4233 | 0.4257 | 0.3597 | 0.7968 |

## Significance (retrieval, paired randomization test)

diff = group1 - group2; p_holm adjusted per metric; alpha = 0.01. Unpaired Tukey HSD shown for comparison. Family `main` is the planned BM25-vs-dense comparison; `main_vs_exploratory` adds later exploratory runs as a separate Holm family.

| Family | Metric | Comparison | Diff | 95% CI | p (paired) | p_holm | p (Tukey, unpaired) |
|--------|--------|-----------|------|--------|------------|--------|---------------------|
| main | ndcg5 | bm25-k5-settlein_v4 - dense-k5-settlein_v4 | -0.0377 | [-0.1387, 0.0538] | 0.4517 | 0.4517 | 0.6258 |
| main | recall5 | bm25-k5-settlein_v4 - dense-k5-settlein_v4 | -0.1181 | [-0.2029, -0.039] | 0.0088 | 0.0088 * | 0.1453 |
| main | hit5 | bm25-k5-settlein_v4 - dense-k5-settlein_v4 | -0.0857 | [-0.2, 0.0] | 0.2528 | 0.2528 | 0.2381 |
| main | mrr | bm25-k5-settlein_v4 - dense-k5-settlein_v4 | 0.0024 | [-0.1262, 0.1238] | 1.0 | 1.0 | 0.9776 |
| main_vs_exploratory | ndcg5 | bm25-k5-settlein_v4 - dense-k5-settlein_v4 | -0.0377 | [-0.1387, 0.0538] | 0.4517 | 0.4517 | 0.8479 |
| main_vs_exploratory | ndcg5 | bm25-k5-settlein_v4 - hybrid_rerank-k5-settlein_v4-gate05 | -0.1877 | [-0.2874, -0.0981] | 0.0002 | 0.0004 * | 0.0204 |
| main_vs_exploratory | ndcg5 | dense-k5-settlein_v4 - hybrid_rerank-k5-settlein_v4-gate05 | -0.15 | [-0.2132, -0.0913] | 0.0001 | 0.0003 * | 0.0796 |
| main_vs_exploratory | recall5 | bm25-k5-settlein_v4 - dense-k5-settlein_v4 | -0.1181 | [-0.2029, -0.039] | 0.0088 | 0.014 | 0.2487 |
| main_vs_exploratory | recall5 | bm25-k5-settlein_v4 - hybrid_rerank-k5-settlein_v4-gate05 | -0.2048 | [-0.3014, -0.1186] | 0.0001 | 0.0003 * | 0.0176 |
| main_vs_exploratory | recall5 | dense-k5-settlein_v4 - hybrid_rerank-k5-settlein_v4-gate05 | -0.0867 | [-0.1438, -0.0343] | 0.007 | 0.014 | 0.4696 |
| main_vs_exploratory | hit5 | bm25-k5-settlein_v4 - dense-k5-settlein_v4 | -0.0857 | [-0.2, 0.0] | 0.2528 | 0.5055 | 0.3684 |
| main_vs_exploratory | hit5 | bm25-k5-settlein_v4 - hybrid_rerank-k5-settlein_v4-gate05 | -0.1143 | [-0.2286, -0.0286] | 0.1246 | 0.3738 | 0.1725 |
| main_vs_exploratory | hit5 | dense-k5-settlein_v4 - hybrid_rerank-k5-settlein_v4-gate05 | -0.0286 | [-0.0857, 0.0] | 1.0 | 1.0 | 0.8938 |
| main_vs_exploratory | mrr | bm25-k5-settlein_v4 - dense-k5-settlein_v4 | 0.0024 | [-0.1262, 0.1238] | 1.0 | 1.0 | 0.9994 |
| main_vs_exploratory | mrr | bm25-k5-settlein_v4 - hybrid_rerank-k5-settlein_v4-gate05 | -0.1833 | [-0.3143, -0.0643] | 0.0093 | 0.0186 | 0.0382 |
| main_vs_exploratory | mrr | dense-k5-settlein_v4 - hybrid_rerank-k5-settlein_v4-gate05 | -0.1857 | [-0.2929, -0.0857] | 0.0024 | 0.0072 * | 0.0352 |
