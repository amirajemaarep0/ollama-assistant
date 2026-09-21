"""pyflakes: undefined names and cleanup findings in code that parses."""
import os
from unittest.mock import MagicMock, patch

import pytest


def test_check_python_quality_finds_undefined_name():
    from rag_backend import check_python_quality

    findings = check_python_quality("def f(n):\n    return n + nmae\n", "x.py")
    bugs = [f for f in findings if f["is_bug"]]
    assert any("nmae" in f["msg"] for f in bugs)
    assert bugs[0]["line"] == 2

def test_check_python_quality_classifies_cleanup_separately():
    from rag_backend import check_python_quality

    findings = check_python_quality("import sys\nx = 1\n", "x.py")
    assert findings and all(not f["is_bug"] for f in findings)
    assert "sys" in findings[0]["msg"]

def test_check_python_quality_is_silent_on_unparseable_source():
    """The syntax error is the caller's story; pyflakes must not double-report."""
    from rag_backend import check_python_quality

    assert check_python_quality("def f(:\n", "x.py") == []

def test_syntax_question_reports_bug_in_valid_file(tmp_path):
    from rag_backend import try_answer_syntax_question

    (tmp_path / "typo.py").write_text("def f(n):\n    return nmae\n", encoding="utf-8")
    meta = {"root": str(tmp_path), "all_files": ["typo.py"]}

    answer = try_answer_syntax_question("What is wrong with typo.py?", str(tmp_path), meta)
    assert answer is not None
    assert "nmae" in answer

def test_cleanup_only_findings_do_not_hijack_a_soft_question(tmp_path):
    """An unused import is not what 'what is wrong with x.py' is asking about."""
    from rag_backend import try_answer_syntax_question

    (tmp_path / "tidy.py").write_text("import sys\nx = 1\n", encoding="utf-8")
    meta = {"root": str(tmp_path), "all_files": ["tidy.py"]}

    assert try_answer_syntax_question("What is wrong with tidy.py?", str(tmp_path), meta) is None
    explicit = try_answer_syntax_question("Check tidy.py for syntax errors", str(tmp_path), meta)
    assert explicit is not None and "sys" in explicit
