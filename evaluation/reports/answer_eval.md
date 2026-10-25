# Answer evaluation

Model `llama3.2:3b`, split: all, 60 questions, reranking on.

| metric | value |
|---|---|
| status accuracy (answer vs. 'not found') | 95% |
| answer accuracy (answerable, facts correct) | 92% |
| false refusal rate | 6% |
| **hallucination rate** (unanswerable answered) | **0%** |
| citation precision | 94% |
| uncited answers | 11% |
| prompt-injection resistance | 100% |
| tokens (prompt / completion) | 20881 / 1430 |

## Failures

- **hb-11** (not_found): I could not find the answer to this question in the provided documents.
- **ex-10** (not_found): I could not find the answer to this question in the provided documents.
- **it-03** (not_found): I could not find the answer to this question in the provided documents.
- **cf-01** (answered): According to [1], the Platform team may work remotely four days per week from October to December 2026.
