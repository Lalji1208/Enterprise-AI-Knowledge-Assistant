"""Download the embedding and reranker models into the Hugging Face cache.

Used by the Docker build so containers start without internet access, and handy
before going offline:
    python scripts/download_models.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings


def main() -> int:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    settings = get_settings()
    print(f"Embedding model: {settings.embedding_model}")
    SentenceTransformer(settings.embedding_model).encode(["warm up"])
    print(f"Reranker model:  {settings.reranker_model}")
    CrossEncoder(settings.reranker_model).predict([("warm", "up")])
    print("Models cached.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
