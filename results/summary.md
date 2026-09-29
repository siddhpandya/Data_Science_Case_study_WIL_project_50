# SettleIN Evaluation Results

## Retrieval Metrics

| Config | nDCG@1 | nDCG@3 | nDCG@5 | Recall@5 | Hit@5 | MRR |
|--------|--------|--------|--------|----------|-------|-----|
| bm25-k5-settlein_v4 | 0.6143 | 0.6408 | 0.6575 | 0.6586 | 0.8571 | 0.7738 |
| dense-k5-settlein_v4 | 0.5429 | 0.6371 | 0.6952 | 0.7767 | 0.9429 | 0.7714 |

## Answerability

| Config | Correct Refusal | Over-Refusal |
|--------|-----------------|---------------|
| bm25-k5-settlein_v4 | 0.4615 | 0.0 |
| closed_book | 0.0 | 0.0 |
| closed_book_instructed | 0.1538 | 0.2571 |
| dense-k5-settlein_v4 | 0.5385 | 0.0286 |

## Faithfulness

| Config | Supported-Claim Rate |
|--------|---------------------|

## Attribution

| Config | Citation Precision | Unattributed Rate |
|--------|-------------------|-------------------|
| bm25-k5-settlein_v4 | 0.5732 | 0.0208 |
| closed_book | None | 1.0 |
| dense-k5-settlein_v4 | 0.6842 | 0.0417 |
