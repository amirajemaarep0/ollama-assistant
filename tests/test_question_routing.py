"""Intent classification: which handler a question reaches, and which it must not."""
import os
from unittest.mock import MagicMock, patch

import pytest


def test_rag_backend_directory_listing_short_circuit():
    from rag_backend import is_directory_listing_question, query_llm, build_index_meta

    assert is_directory_listing_question("List all files in the current directory")
    assert is_directory_listing_question("liste des fichiers du projet")
    assert not is_directory_listing_question("Explain what app.py does")

    meta = build_index_meta(
        ".",
        "python",
        ["app.py", "rag_backend.py"],
    )
    meta["root"] = "C:/proj"
    meta["all_files"] = ["app.py", "rag_backend.py", "readme.md"]
    meta["not_indexed_files"] = ["readme.md"]

    response, docs = query_llm(None, "List files in this folder", index_meta=meta)
    assert "app.py" in response
    assert "readme.md" in response
    assert len(docs) == 0

def test_extract_mentioned_files():
    from rag_backend import extract_mentioned_files

    assert "test.py" in extract_mentioned_files("What does test.py do?")
    assert "app.py" in extract_mentioned_files("Explain app.py and rag_backend.py")

def test_syntax_question_triggers_only_when_it_should():
    from rag_backend import is_syntax_check_question

    assert is_syntax_check_question("What is wrong with app.py?")
    assert is_syntax_check_question("Fix broken.py")
    assert is_syntax_check_question("Are there syntax errors in the project?")
    # No file named and no "syntax"/"compile" wording: this is a normal question.
    assert not is_syntax_check_question("How does the app handle errors?")
    assert not is_syntax_check_question("What does app.py do?")

def test_project_overview_question_triggers_only_for_whole_project():
    from rag_backend import is_project_overview_question as Q

    assert Q("Scan the whole project")
    assert Q("What does this project do?")
    assert Q("Give me an overview of the codebase")
    assert Q("Explain the architecture")
    assert not Q("What does app.py do?")
    assert not Q("How does record_audio work?")

def test_optimization_question_detection():
    from rag_backend import is_optimization_question as Q

    assert Q("Optimize count_duplicates in slow.py")
    assert Q("make build_report faster in slow.py")
    assert Q("refactor the lookup method")
    assert not Q("What does build_report do?")
    assert not Q("Are there syntax errors in slow.py?")

def test_scan_intent_reaches_the_checker_without_the_word_syntax():
    """Regression: "scan for errors in the project" required the literal word
    "syntax", so it fell through to retrieval and answered from 3 chunks."""
    from rag_backend import is_syntax_check_question as Q

    for question in (
        "scan for errors in the project",
        "check the project for errors",
        "find all errors",
        "find any issues in the code",
        "any bugs?",
        "list all the errors",
        "scanne le projet pour des erreurs",
    ):
        assert Q(question), question

def test_questions_about_error_handling_are_not_scans():
    """A question about how code behaves must not trigger a project scan."""
    from rag_backend import is_syntax_check_question as Q

    for question in (
        "how does the project handle errors?",
        "How does app.py handle errors",
        "explain the error handling in rag_backend.py",
        "why does this fail at runtime",
        "describe the error messages",
    ):
        assert not Q(question), question

def test_scan_request_on_a_clean_project_reports_the_all_clear(tmp_path):
    """Regression: "scan for errors" on a healthy project found nothing and fell
    through to the model instead of saying so."""
    from rag_backend import try_answer_syntax_question

    (tmp_path / "a.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("VALUE = 1\n", encoding="utf-8")
    meta = {"root": str(tmp_path), "all_files": ["a.py", "b.py"]}

    answer = try_answer_syntax_question("scan for errors in the project", str(tmp_path), meta)
    assert answer is not None, "an explicit scan must answer, even when clean"
    assert "No syntax errors" in answer

    # A vague question about a healthy file still reaches the model.
    assert try_answer_syntax_question("What is wrong with a.py?", str(tmp_path), meta) is None
