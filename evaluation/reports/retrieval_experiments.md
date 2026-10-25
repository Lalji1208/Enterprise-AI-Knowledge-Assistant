# Retrieval experiments

Dataset: 60 questions (48 answerable), dev 31 / test 29. Embeddings `sentence-transformers/all-MiniLM-L6-v2`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`, top_k=5.

Metrics: **hit@k** = share of answerable questions with a relevant chunk in the top k; **MRR** = mean reciprocal rank of the first relevant chunk; **abstain** = share of unanswerable questions for which nothing is returned; **balanced** = mean of hit@5 and abstain. Thresholds were tuned on *dev*; all tables show *test* unless noted.

## 1. Chunking (test split)

| chunk / overlap | chunks | hit@1 | hit@5 | MRR | hit@5 bi-default | abstain bi-default | hit@5 rerank-default | abstain rerank-default | balanced rerank-default | balanced hybrid |
|---|---|---|---|---|---|---|---|---|---|---|
| 300 / 30 | 35 | 0.74 | 0.96 | 0.84 | 0.96 | 0.83 | 1.00 | 0.83 | 0.92 | 0.92 |
| 500 / 0 | 20 | 0.70 | 1.00 | 0.80 | 0.83 | 0.83 | 0.96 | 0.83 | 0.89 | 0.89 |
| 500 / 50 | 21 | 0.83 | 0.96 | 0.87 | 0.91 | 0.83 | 1.00 | 0.83 | 0.92 | 0.92 |
| 500 / 100 | 21 | 0.78 | 0.96 | 0.85 | 0.91 | 0.83 | 0.96 | 0.83 | 0.89 | 0.92 |
| 800 / 80 | 14 | 0.83 | 1.00 | 0.91 | 0.87 | 0.67 | 0.96 | 0.83 | 0.89 | 0.89 |

*no thresholds* columns (hit@1, hit@5, MRR) measure pure ranking quality.

## 2. Thresholds (chunk 500 / overlap 50)

| variant | min_score | rerank_min_score | dev balanced | test hit@5 | test false abstain | test abstain | **test balanced** | test MRR | latency ms |
|---|---|---|---|---|---|---|---|---|---|
| bi_default | 0.3 | off | 0.81 | 0.91 | 0.00 | 0.83 | **0.87** | 0.86 | 653 |
| bi_tuned | 0.35 | off | 0.96 | 0.78 | 0.09 | 0.83 | **0.81** | 0.76 | 653 |
| rerank_same_first_stage | 0.3 | -6 | 0.88 | 0.91 | 0.04 | 0.83 | **0.87** | 0.91 | 653 |
| rerank_default | 0.15 | -6 | 0.88 | 1.00 | 0.00 | 0.83 | **0.92** | 1.00 | 653 |
| rerank_tuned_first_stage | 0.15 | -6 | 0.88 | 1.00 | 0.00 | 0.83 | **0.92** | 1.00 | 653 |
| rerank_tuned_both | off | -3 | 0.92 | 0.83 | 0.17 | 0.83 | **0.83** | 0.83 | 653 |
| hybrid_default | 0.15 | -6 | 0.79 | 1.00 | 0.00 | 0.83 | **0.92** | 1.00 | 653 |

Test questions still handled wrongly:

- bi_default: it-04, it-06, un-04
- bi_tuned: hb-16, ex-02, it-04, it-06, un-04, kw-04
- rerank_same_first_stage: it-04, it-06, un-04
- rerank_default: un-04
- rerank_tuned_first_stage: un-04
- rerank_tuned_both: hb-04, hb-16, ex-02, it-02, un-04
- hybrid_default: un-04

Latency includes the reranker for every row (it was recorded once per question); bi-encoder-only retrieval alone is far faster.
