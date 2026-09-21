"""Walking the project, pruning exclusions, and building the outline."""
import os
from unittest.mock import MagicMock, patch

import pytest


def test_scan_project_files_skips_venv(tmp_path):
    from rag_backend import scan_project_files

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    venv = tmp_path / "venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "bad.py").write_text("y = 2\n", encoding="utf-8")

    files = scan_project_files(str(tmp_path))
    assert "main.py" in files
    assert not any("venv" in f for f in files)

def test_iter_project_files_prunes_virtualenvs(tmp_path):
    """Regression: collection used glob('**/*.py') and filtered afterwards, so it
    walked the whole .venv tree before discarding it."""
    from rag_backend import _iter_project_files

    (tmp_path / "main.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# hi", encoding="utf-8")
    buried = tmp_path / ".venv" / "Lib" / "site-packages"
    buried.mkdir(parents=True)
    (buried / "dep.py").write_text("y = 2", encoding="utf-8")

    found = [os.path.basename(f) for f in _iter_project_files(str(tmp_path), {".py"})]
    assert found == ["main.py"]

    both = sorted(os.path.basename(f) for f in _iter_project_files(str(tmp_path), {".py", ".md"}))
    assert both == ["main.py", "notes.md"]

def test_get_embeddings_is_cached():
    import rag_backend

    rag_backend._EMBEDDINGS_CACHE.clear()
    with patch("rag_backend.HuggingFaceEmbeddings") as mock_emb:
        rag_backend.get_embeddings()
        rag_backend.get_embeddings()
    assert mock_emb.call_count == 1
    rag_backend._EMBEDDINGS_CACHE.clear()

def test_build_project_outline_covers_every_file(tmp_path):
    from rag_backend import build_project_outline

    (tmp_path / "a.py").write_text('"""Module A."""\nclass Foo:\n    def bar(self, x):\n        pass\n', encoding="utf-8")
    (tmp_path / "b.py").write_text("def helper(y):\n    return y\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("x = 1\n", encoding="utf-8")
    meta = {"root": str(tmp_path), "all_files": ["a.py", "b.py", "c.py"]}

    outline = build_project_outline(str(tmp_path), meta)
    for name in ("a.py", "b.py", "c.py"):
        assert f"### FILE: {name}" in outline
    assert "class Foo" in outline
    assert "def bar(self, x)" in outline
    assert "def helper(y)" in outline
    assert "Module A." in outline

def test_outline_survives_a_file_that_cannot_be_parsed(tmp_path):
    from rag_backend import build_project_outline

    (tmp_path / "ok.py").write_text("def f():\n    pass\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("def f(:\n", encoding="utf-8")
    meta = {"root": str(tmp_path), "all_files": ["ok.py", "bad.py"]}

    outline = build_project_outline(str(tmp_path), meta)
    assert "### FILE: ok.py" in outline
    assert "### FILE: bad.py" in outline
    assert "cannot parse" in outline
