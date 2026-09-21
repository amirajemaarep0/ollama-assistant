"""The Ollama client wrapper."""
import os
from unittest.mock import MagicMock, patch

import pytest


def test_ollama_backend_honours_base_url():
    """Regression: the client used to be the bare `ollama` module, so a custom
    base_url was silently ignored and every call went to localhost:11434."""
    from llm_backends import OllamaBackend

    backend = OllamaBackend("llama3", base_url="http://example.invalid:9999/")
    with patch("ollama.Client") as mock_client:
        backend._init_client()
    mock_client.assert_called_once_with(host="http://example.invalid:9999")
