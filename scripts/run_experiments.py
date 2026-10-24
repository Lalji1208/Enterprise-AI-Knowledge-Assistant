"""Run the retrieval experiments (Phase 15) on the evaluation corpus.

Run from the project root (takes a few minutes on CPU):
    python scripts/run_experiments.py
Writes evaluation/reports/retrieval_experiments.md and .json. Indexes are built
in data/processed/eval/ and never touch the real index.
"""

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from evaluation.experiments import run, write_reports


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    for noisy in ("pypdf", "httpx", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    start = time.perf_counter()
    result = run(get_settings().processed_dir / "eval")
    md, js = write_reports(result)
    print(f"\nDone in {time.perf_counter() - start:.0f}s")
    print(f"Report: {md}\nData:   {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
