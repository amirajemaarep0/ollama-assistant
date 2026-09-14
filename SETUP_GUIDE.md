# Local Voice AI Dev Assistant - Setup Guide

Everything runs **on your machine**. No code or documents leave the computer.

The stack follows the project specification:

| Component | Tool |
|---|---|
| Language | Python |
| Interface | Streamlit |
| AI Engine (Local) | Ollama |
| Transcription | OpenAI Whisper (local) |
| Context Handling | LangChain |

---

## 🚀 Quick Start

### 1. Install the Python requirements

```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama

1. Download Ollama: https://ollama.com/download
2. Start the Ollama app. It listens on `http://localhost:11434`.
3. Pull a model:

```bash
ollama pull llama3
```

The app offers the two models named in the project specification. Pull whichever
you plan to use:

```bash
ollama pull llama3          # ~4.7 GB, best answers
ollama pull mistral         # ~4.1 GB, lighter and faster
```

### 3. Run the app

```bash
streamlit run app.py
```

Then open `http://localhost:8501`.

---

## 📝 Using the App

1. **Test Ollama** with the sidebar button. It confirms the server is reachable
   *and* that the model you named is actually pulled.
2. **Set the project folder path** in the sidebar.
3. **Choose what to index**: Python only, documents only, or both.
4. Click **Load & Index Directory**.
5. **Ask a question** by typing, or click **Record** to ask out loud.
6. Expand **Retrieved snippets** under an answer to see which files were used.

The first recording downloads the Whisper model (~140 MB for `base`) and the
first indexing run downloads the embedding model (~90 MB). Both are cached
afterwards.

---

## 🔧 Troubleshooting

### "Cannot reach Ollama"
1. Make sure the Ollama app is running.
2. Check the **Ollama API URL** in the sidebar (default `http://localhost:11434`).
3. Verify from a terminal: `ollama list`.

### "Model is not pulled"
Run `ollama pull <model>` with the name shown in the sidebar.

### Out of memory when loading the model
- Lower **Context window** to 2048.
- Switch the **Ollama model** to `mistral` (lighter than `llama3`).
- Reduce **Snippets to retrieve** and **Max characters per snippet**.
- Close other applications and restart Ollama.

### Answers are slow
Ollama runs an 8B model locally; the first query also loads it into memory.
A smaller model is the biggest single speedup.

### Voice input does nothing
Check that a microphone is available and permitted. Whisper decodes the WAV
directly, so FFmpeg is not required.

---

## 🧪 Tests

```bash
pytest test_app.py
```
