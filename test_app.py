import pytest
from unittest.mock import patch, MagicMock
import os

def test_voice_stt_transcribe_mock():
    # Mock whisper and OS functions to test the transcription logic
    import voice_stt

    voice_stt._MODEL_CACHE.clear()  # get_model() memoises; start from a clean slate
    with patch("voice_stt.whisper.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Hello world"}
        mock_load.return_value = mock_model
        
        from voice_stt import transcribe_audio
        
        with patch("voice_stt.os.path.exists", return_value=True):
            with patch("voice_stt.os.remove") as mock_remove:
                result = transcribe_audio("dummy.wav")
                assert result == "Hello world"
                mock_model.transcribe.assert_called_once_with("dummy.wav", fp16=False)
                mock_remove.assert_called_once_with("dummy.wav")

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


def test_scan_project_files_skips_venv(tmp_path):
    from rag_backend import scan_project_files

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    venv = tmp_path / "venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "bad.py").write_text("y = 2\n", encoding="utf-8")

    files = scan_project_files(str(tmp_path))
    assert "main.py" in files
    assert not any("venv" in f for f in files)


def test_extract_mentioned_files():
    from rag_backend import extract_mentioned_files

    assert "test.py" in extract_mentioned_files("What does test.py do?")
    assert "app.py" in extract_mentioned_files("Explain app.py and rag_backend.py")


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


def test_ollama_backend_honours_base_url():
    """Regression: the client used to be the bare `ollama` module, so a custom
    base_url was silently ignored and every call went to localhost:11434."""
    from llm_backends import OllamaBackend

    backend = OllamaBackend("llama3", base_url="http://example.invalid:9999/")
    with patch("ollama.Client") as mock_client:
        backend._init_client()
    mock_client.assert_called_once_with(host="http://example.invalid:9999")


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


def test_whisper_model_is_cached():
    import voice_stt

    voice_stt._MODEL_CACHE.clear()
    with patch("voice_stt.whisper.load_model") as mock_load:
        mock_load.return_value = MagicMock()
        voice_stt.get_model("base")
        voice_stt.get_model("base")
    assert mock_load.call_count == 1
    voice_stt._MODEL_CACHE.clear()


def test_check_python_syntax_reports_location():
    from rag_backend import check_python_syntax

    assert check_python_syntax("x = 1\n", "ok.py") is None

    problem = check_python_syntax("def f(name):\n    print('hi' + name\n", "bad.py")
    assert problem is not None
    assert problem["line"] == 2
    assert "never closed" in problem["msg"]


def test_syntax_question_triggers_only_when_it_should():
    from rag_backend import is_syntax_check_question

    assert is_syntax_check_question("What is wrong with app.py?")
    assert is_syntax_check_question("Fix broken.py")
    assert is_syntax_check_question("Are there syntax errors in the project?")
    # No file named and no "syntax"/"compile" wording: this is a normal question.
    assert not is_syntax_check_question("How does the app handle errors?")
    assert not is_syntax_check_question("What does app.py do?")


def test_try_answer_syntax_question_finds_the_error(tmp_path):
    from rag_backend import try_answer_syntax_question

    (tmp_path / "broken.py").write_text("def greet(name):\n    print('hi' + name\n", encoding="utf-8")
    meta = {"root": str(tmp_path), "all_files": ["broken.py"]}

    answer = try_answer_syntax_question("What is wrong with broken.py?", str(tmp_path), meta)
    assert answer is not None
    assert "broken.py" in answer
    assert "line 2" in answer


def test_try_answer_syntax_question_falls_through_when_file_is_clean(tmp_path):
    """A clean file plus a soft trigger must reach the LLM, not answer 'no errors'."""
    from rag_backend import try_answer_syntax_question

    (tmp_path / "fine.py").write_text("x = 1\n", encoding="utf-8")
    meta = {"root": str(tmp_path), "all_files": ["fine.py"]}

    assert try_answer_syntax_question("What is wrong with fine.py?", str(tmp_path), meta) is None
    # Explicit "syntax" wording still gets the all-clear.
    explicit = try_answer_syntax_question("Any syntax errors in fine.py?", str(tmp_path), meta)
    assert explicit is not None and "no syntax errors" in explicit.lower()


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


def test_project_overview_question_triggers_only_for_whole_project():
    from rag_backend import is_project_overview_question as Q

    assert Q("Scan the whole project")
    assert Q("What does this project do?")
    assert Q("Give me an overview of the codebase")
    assert Q("Explain the architecture")
    assert not Q("What does app.py do?")
    assert not Q("How does record_audio work?")


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
