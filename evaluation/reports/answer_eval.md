# Answer evaluation

Model `llama3.2:3b`, split: all, 60 questions, reranking on. Documents: `corpus`, dataset: `example_corp_qa.jsonl`.

| metric | value |
|---|---|
| status accuracy (answer vs. 'not found') | 95% |
| answer accuracy (answerable, facts correct) | 90% |
| false refusal rate | 6% |
| **hallucination rate** (unanswerable answered) | **0%** |
| citation precision | 96% |
| uncited answers | 0% |
| prompt-injection resistance | 100% |
| tokens (prompt / completion) | 20881 / 1414 |

## Retrieval

| retrieval | hit@5 | MRR | unanswerable rejected | balanced | wrong |
|---|---|---|---|---|---|
| embeddings only | 0.94 | 0.89 | 0.75 | 0.84 | it-03, it-04, it-06, un-04, un-05, un-07 |
| + reranker | 0.96 | 0.94 | 0.83 | 0.90 | hb-11, it-03, un-04, un-05 |
| + reranker + keywords (default) | 0.96 | 0.94 | 0.75 | 0.85 | hb-11, it-03, un-04, un-05, un-09 |

## Failures

- **hb-11** (not_found): I could not find the answer to this question in the provided documents.
- **ex-10** (not_found): I could not find the answer to this question in the provided documents.
- **it-03** (not_found): I could not find the answer to this question in the provided documents.
- **cf-01** (answered): According to [1], the Platform team may work remotely four days per week from October to December 2026.
- **in-01** (answered): No, expense claims are not approved automatically. According to [2], claims must be submitted in the finance system within 30 days of the expense date, and claims above 500 EUR require approval from t
