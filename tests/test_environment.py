"""Smoke tests for Phase 0: interpreter version and project layout."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_PACKAGES = [
    "app", "api", "rag", "evaluation", "models",
    "database", "services", "config",
]


def test_python_version_is_supported() -> None:
    assert sys.version_info >= (3, 11)


def test_project_packages_exist() -> None:
    for name in EXPECTED_PACKAGES:
        assert (PROJECT_ROOT / name / "__init__.py").is_file(), f"missing package: {name}"


def test_data_directories_exist() -> None:
    assert (PROJECT_ROOT / "data" / "documents").is_dir()
    assert (PROJECT_ROOT / "data" / "processed").is_dir()


def test_env_example_has_no_real_key() -> None:
    """Guard against accidentally committing a secret into the template."""
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=\n" in text
