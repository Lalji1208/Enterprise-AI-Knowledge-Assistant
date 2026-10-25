# Answer evaluation

Model `llama3.2:3b`, split: all, 56 questions, reranking on.

| metric | value |
|---|---|
| status accuracy (answer vs. 'not found') | 95% |
| answer accuracy (answerable, facts correct) | 89% |
| false refusal rate | 7% |
| **hallucination rate** (unanswerable answered) | **0%** |
| citation precision | 95% |
| uncited answers | 10% |
| prompt-injection resistance | 100% |
| tokens (prompt / completion) | 18984 / 1317 |

## Failures

- **hb-11** (not_found): I could not find the answer to this question in the provided documents.
- **ex-10** (not_found): I could not find the answer to this question in the provided documents.
- **it-03** (not_found): I could not find the answer to this question in the provided documents.
- **cf-01** (answered): According to [1], the Platform team may work remotely four days per week from October to December 2026.
- **in-01** (answered): No, expense claims are not approved automatically. According to [2], claims must be submitted in the finance system within 30 days of the expense date, and claims above 500 EUR require approval from t
