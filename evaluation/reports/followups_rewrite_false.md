# Answer evaluation

Model `llama3.2:3b`, split: all, 12 questions, reranking on. Documents: `documents`, dataset: `research_paper_followups.jsonl`.

| metric | value |
|---|---|
| status accuracy (answer vs. 'not found') | 75% |
| answer accuracy (answerable, facts correct) | 50% |
| false refusal rate | 30% |
| **hallucination rate** (unanswerable answered) | **0%** |
| citation precision | 71% |
| uncited answers | 0% |
| prompt-injection resistance | n/a |
| tokens (prompt / completion) | 3967 / 217 |

## Failures

- **fu-01** (not_found): I could not find the answer to this question in the provided documents.
- **fu-02** (answered): According to [4], the proposed system uses 80% of the total resumes from the Resume_csv file, and the remaining 20% is used for testing. However, the exact number of resumes is not explicitly stated.
- **fu-04** (answered): The SVM algorithm learns a decision boundary between the relevant and irrelevant resumes when screening resumes [2]. It seeks out the hyperplane that maximizes the margin between the two classes to en
- **fu-05** (not_found): I could not find the answer to this question in the provided documents.
- **fu-06** (not_found): I could not find the answer to this question in the provided documents.
