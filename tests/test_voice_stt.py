"""Speech-to-text: transcription and model caching."""
import os
from unittest.mock import MagicMock, patch

import pytest


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

def test_whisper_model_is_cached():
    import voice_stt

    voice_stt._MODEL_CACHE.clear()
    with patch("voice_stt.whisper.load_model") as mock_load:
        mock_load.return_value = MagicMock()
        voice_stt.get_model("base")
        voice_stt.get_model("base")
    assert mock_load.call_count == 1
    voice_stt._MODEL_CACHE.clear()
