# Answer evaluation

Model `llama3.2:3b`, split: all, 34 questions, reranking on. Documents: `documents`, dataset: `research_paper_qa.jsonl`.

| metric | value |
|---|---|
| status accuracy (answer vs. 'not found') | 85% |
| answer accuracy (answerable, facts correct) | 77% |
| false refusal rate | 19% |
| **hallucination rate** (unanswerable answered) | **0%** |
| citation precision | 68% |
| uncited answers | 0% |
| prompt-injection resistance | n/a |
| tokens (prompt / completion) | 14072 / 1124 |

## Retrieval

| retrieval | hit@5 | MRR | unanswerable rejected | balanced | wrong |
|---|---|---|---|---|---|
| embeddings only | 0.54 | 0.42 | 0.62 | 0.58 | rp-01, rp-02, rp-03, rp-06, rp-09, rp-12, rp-19, rp-20, rp-22, rp-23, rp-25, rp-26, rp-u1, rp-u2, rp-u6 |
| + reranker | 0.73 | 0.62 | 0.62 | 0.68 | rp-01, rp-02, rp-04, rp-09, rp-22, rp-25, rp-26, rp-u2, rp-u3, rp-u6 |
| + reranker + keywords (default) | 0.73 | 0.62 | 0.62 | 0.68 | rp-01, rp-02, rp-04, rp-09, rp-22, rp-25, rp-26, rp-u2, rp-u3, rp-u6 |

## Failures

- **rp-01** (not_found): I could not find the answer to this question in the provided documents.
- **rp-02** (not_found): I could not find the answer to this question in the provided documents.
- **rp-09** (not_found): I could not find the answer to this question in the provided documents.
- **rp-22** (not_found): I could not find the answer to this question in the provided documents.
- **rp-25** (not_found): I could not find the answer to this question in the provided documents.
- **rp-26** (answered): According to studies, ATS can increase uniformity [1].
