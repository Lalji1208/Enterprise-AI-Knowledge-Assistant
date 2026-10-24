"""Phase 13 - Evaluation dataset: labelled questions about a document collection.

One JSON object per line (JSONL). Relevance is defined by short evidence phrases
rather than chunk ids, so the same labels stay valid for any chunk size:
a retrieved chunk is relevant to a question if it comes from an evidence
`source` and contains that evidence `text` (case/whitespace-insensitive).

    {"id": "hb-01", "split": "dev", "category": "handbook",
     "question": "How many vacation days do full-time staff get each year?",
     "answerable": true,
     "evidence": [{"source": "employee_handbook.pdf", "text": "25 days of paid annual leave"}],
     "answer_keywords": [["25"]],          # every group must match (any alternative)
     "must_not_contain": []}               # phrases that must never appear in an answer
"""

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from models.chunk import Chunk
from models.document import Document

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
DEFAULT_DATASET = DATASETS_DIR / "example_corp_qa.jsonl"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


class Evidence(BaseModel):
    source: str
    text: str = Field(min_length=3)


class EvalItem(BaseModel):
    id: str
    split: Literal["dev", "test"]
    category: str
    question: str = Field(min_length=3)
    answerable: bool
    evidence: list[Evidence] = Field(default_factory=list)
    answer_keywords: list[list[str]] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> "EvalItem":
        if self.answerable and not self.evidence:
            raise ValueError(f"{self.id}: answerable questions need evidence")
        if not self.answerable and self.evidence:
            raise ValueError(f"{self.id}: unanswerable questions must not have evidence")
        return self

    def is_relevant(self, chunk: Chunk) -> bool:
        text = normalize(chunk.text)
        return any(e.source == chunk.source and normalize(e.text) in text for e in self.evidence)


def load_dataset(path: str | Path = DEFAULT_DATASET, split: str | None = None) -> list[EvalItem]:
    items = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                items.append(EvalItem.model_validate(json.loads(line)))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    ids = [i.id for i in items]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate ids in {path}: {sorted(duplicates)}")
    return [i for i in items if split is None or i.split == split]


def check_evidence(items: Iterable[EvalItem], documents: Iterable[Document]) -> list[str]:
    """Problems with the labels: evidence that does not occur in the (cleaned)
    documents would make a question impossible to score correctly."""
    texts = {d.source: normalize(d.text) for d in documents}
    problems = []
    for item in items:
        for ev in item.evidence:
            if ev.source not in texts:
                problems.append(f"{item.id}: unknown source '{ev.source}'")
            elif normalize(ev.text) not in texts[ev.source]:
                problems.append(f"{item.id}: evidence not found in {ev.source}: '{ev.text}'")
    return problems
