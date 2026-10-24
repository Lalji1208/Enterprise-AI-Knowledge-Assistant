"""Verify that the development environment is ready for the project.

Run from the project root (with the virtual environment activated):

    python scripts/check_environment.py

It checks the Python version, imports every key dependency (an actual import,
so Windows DLL problems with PyTorch or FAISS show up here and not later),
and confirms that a .env file exists.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

MIN_PYTHON = (3, 11)

# (import name, pip package name)
REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("dotenv", "python-dotenv"),
    ("pydantic", "pydantic"),
    ("pydantic_settings", "pydantic-settings"),
    ("pypdf", "pypdf"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("torch", "torch"),
    ("sentence_transformers", "sentence-transformers"),
    ("faiss", "faiss-cpu"),
    ("openai", "openai"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("streamlit", "streamlit"),
    ("pytest", "pytest"),
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_python() -> bool:
    """Return True if the running interpreter is new enough."""
    version = sys.version_info
    ok = version >= MIN_PYTHON
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] Python {version.major}.{version.minor}.{version.micro} "
          f"(need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")
    print(f"       interpreter: {sys.executable}")
    return ok


def check_packages() -> bool:
    """Import each required package and print its version."""
    all_ok = True
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK  ] {pip_name:<24} {version}")
        except Exception as exc:  # noqa: BLE001 - we want to report any failure
            all_ok = False
            print(f"[FAIL] {pip_name:<24} {type(exc).__name__}: {exc}")
    return all_ok


def check_env_file() -> bool:
    """Warn (not fail) if .env has not been created yet."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        print("[OK  ] .env file found")
    else:
        print("[WARN] .env not found - run: copy .env.example .env")
    return True


def main() -> int:
    print("=== Enterprise AI Knowledge Assistant: environment check ===\n")
    results = [check_python(), check_packages(), check_env_file()]
    print()
    if all(results):
        print("Environment ready. Phase 0 complete.")
        return 0
    print("Some checks failed - see [FAIL] lines above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
