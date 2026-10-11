import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway project root with a copy of configs/, so tests never touch real results."""
    shutil.copytree(REPO / "configs", tmp_path / "configs", ignore=shutil.ignore_patterns("split.json"))
    monkeypatch.setenv("SSP_HOME", str(tmp_path))
    monkeypatch.setenv("SSP_RUNS", str(tmp_path / "runs"))
    monkeypatch.delenv("MNMS_ROOT", raising=False)
    return tmp_path
