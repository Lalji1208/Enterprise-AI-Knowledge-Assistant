# Answer evaluation

Model `llama3.2:3b`, split: all, 12 questions, reranking on. Documents: `documents`, dataset: `research_paper_followups.jsonl`.

| metric | value |
|---|---|
| status accuracy (answer vs. 'not found') | 83% |
| answer accuracy (answerable, facts correct) | 70% |
| false refusal rate | 20% |
| **hallucination rate** (unanswerable answered) | **0%** |
| citation precision | 64% |
| uncited answers | 25% |
| prompt-injection resistance | n/a |
| tokens (prompt / completion) | 6230 / 458 |

## Failures

- **fu-02** (not_found): I could not find the answer to this question in the provided documents.
- **fu-05** (answered): According to [1] (page 2), the NLTK tokenizer is described as the tool used to divide the cleaned data into small tokens.
- **fu-06** (not_found): I could not find the answer to this question in the provided documents.
