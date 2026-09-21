"""query_llm end to end: prompt assembly, grounding, and short circuits."""
import os
from unittest.mock import MagicMock, patch

import pytest


def test_rag_backend_query_llm_no_context():
    with patch("rag_backend.OllamaBackend") as MockOllama:
        mock_llm = MagicMock(spec=["invoke"])
        mock_llm.invoke.return_value = "This is a mocked answer."
        MockOllama.return_value = mock_llm

        from rag_backend import query_llm

        response, docs = query_llm(None, "What is this?")

        assert response == "This is a mocked answer."
        assert len(docs) == 0
        mock_llm.invoke.assert_called_once()

def test_load_mentioned_files_context(tmp_path):
    from rag_backend import load_mentioned_files_context

    (tmp_path / "test.py").write_text("print('hello')\n", encoding="utf-8")
    meta = {"all_files": ["test.py"]}
    ctx = load_mentioned_files_context(str(tmp_path), ["test.py"], meta)
    assert "hello" in ctx
    assert "test.py" in ctx

def test_try_answer_file_content_question(tmp_path):
    from rag_backend import try_answer_file_content_question

    (tmp_path / "test.py").write_text("print('hello, World!')\n", encoding="utf-8")
    meta = {"root": str(tmp_path), "all_files": ["test.py"]}
    answer = try_answer_file_content_question(
        "What is the content of test.py?", str(tmp_path), meta
    )
    assert answer is not None
    assert "hello, World!" in answer
    assert "test.py" in answer

def test_try_answer_file_content_question_uses_backend_for_description(tmp_path):
    from rag_backend import try_answer_file_content_question

    (tmp_path / "test.py").write_text("print('hello, World!')\n", encoding="utf-8")
    meta = {"root": str(tmp_path), "all_files": ["test.py"]}

    mock_backend = MagicMock(spec=["query"])
    mock_backend.query.return_value = "This file prints a greeting message."

    answer = try_answer_file_content_question(
        "What does test.py do?",
        str(tmp_path),
        meta,
        backend=mock_backend,
        num_predict=128,
    )

    assert answer == "This file prints a greeting message."
    mock_backend.query.assert_called_once()

def test_rag_backend_query_llm_with_backend_object():
    mock_backend = MagicMock(spec=["query"])
    mock_backend.query.return_value = "Backend-specific answer."

    from rag_backend import query_llm

    response, docs = query_llm(None, "What is this?", backend=mock_backend)

    assert response == "Backend-specific answer."
    assert len(docs) == 0
    mock_backend.query.assert_called_once()

def test_query_llm_applies_system_prompt_to_backend_object():
    """Regression: the system prompt was only set when the backend already had a
    `system` attribute, which no llm_backends object did."""
    from llm_backends import OllamaBackend
    from rag_backend import query_llm, _system_prompt_for_mode

    backend = OllamaBackend("llama3")
    backend.query = MagicMock(return_value="ok")

    query_llm(None, "What is this?", backend=backend, index_mode="documents")

    assert backend.system == _system_prompt_for_mode("documents")

def test_query_llm_survives_backend_without_system_attribute():
    from rag_backend import query_llm

    backend = MagicMock(spec=["query"])
    backend.query.return_value = "fine"
    response, _docs = query_llm(None, "hi", backend=backend)
    assert response == "fine"

def test_query_llm_answers_syntax_question_without_the_model(tmp_path):
    from rag_backend import query_llm

    (tmp_path / "broken.py").write_text("print('oops'\n", encoding="utf-8")
    meta = {"root": str(tmp_path), "all_files": ["broken.py"]}
    backend = MagicMock(spec=["query"])

    response, docs = query_llm(
        None, "What is wrong with broken.py?", backend=backend,
        index_meta=meta, project_root=str(tmp_path),
    )

    assert "broken.py" in response
    assert docs == []
    backend.query.assert_not_called()

def test_query_llm_sends_every_file_for_a_whole_project_question(tmp_path):
    """Regression: retrieval returned only k chunks, so most files never reached
    the model on a 'scan the whole project' question."""
    from rag_backend import query_llm

    for name in ("one.py", "two.py", "three.py", "four.py"):
        (tmp_path / name).write_text(f"def fn_{name[:-3]}():\n    pass\n", encoding="utf-8")
    meta = {"root": str(tmp_path), "all_files": ["one.py", "two.py", "three.py", "four.py"]}

    captured = {}

    class Spy:
        system = None

        def query(self, prompt, max_tokens=512):
            captured["prompt"] = prompt
            return "ok"

    query_llm(None, "Scan the whole project", backend=Spy(),
              index_meta=meta, project_root=str(tmp_path))

    for name in ("one.py", "two.py", "three.py", "four.py"):
        assert f"### FILE: {name}" in captured["prompt"]
