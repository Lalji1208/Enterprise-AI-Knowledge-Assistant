"""Shared test helpers: a fake embedding model (so unit tests never download one)
and a tiny PDF writer (so tests need no binary fixture files)."""

from pathlib import Path

import numpy as np

DIM = 8


class FakeModel:
    """Deterministic stand-in: a text's vector depends only on its words."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], dict]] = []

    def get_embedding_dimension(self) -> int:
        return DIM

    def _vectors(self, texts: list[str], normalize: bool) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float64)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                out[i, sum(map(ord, word)) % DIM] += 1.0
        if normalize:
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        return out

    def encode_document(self, texts: list[str], **kwargs) -> np.ndarray:
        self.calls.append(("document", texts, kwargs))
        return self._vectors(texts, kwargs["normalize_embeddings"])

    def encode_query(self, texts: list[str], **kwargs) -> np.ndarray:
        self.calls.append(("query", texts, kwargs))
        return self._vectors(texts, kwargs["normalize_embeddings"])


def make_pdf(path: Path, page_texts: list[str]) -> Path:
    """Write a minimal valid PDF with one line of Helvetica text per page."""
    n = len(page_texts)
    font_id = 3 + 2 * n
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [{}] /Count {} >>".format(
            " ".join(f"{3 + 2 * i} 0 R" for i in range(n)), n
        ),
    ]
    for i, text in enumerate(page_texts):
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {4 + 2 * i} 0 R >>"
        )
        objects.append(f"<< /Length {len(content)} >>\nstream\n{content}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = b"%PDF-1.4\n"
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    path.write_bytes(out)
    return path
