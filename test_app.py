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
