# Local Voice AI Dev Assistant - Setup Guide

This app supports **5 different LLM backends** for local inference. Choose one based on your preferences.

---

## 🚀 Quick Start

### 1. Install Base Requirements
```bash
pip install -r requirements.txt
```

### 2. Choose & Setup Your LLM Backend

---

## 🤖 LLM Backend Options

### **Option 1: Ollama** ⭐ (Recommended - Easiest)
**Best for:** Beginners, want a simple setup

1. Install Ollama: https://ollama.com/download
2. Start the Ollama app (runs on `localhost:11434`)
3. Pull a model:
   ```bash
   ollama pull llama3          # Full-size (~4GB)
   ollama pull mistral         # Medium (~5GB, faster)
   ollama pull neural-chat     # Small (~4GB)
   ollama pull qwen:1.5b       # Tiny (~1GB, very fast on CPU)
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```
5. In app, select **Ollama** backend (should auto-detect)

---

### **Option 2: LLaMA.cpp** ⚡ (Fastest on CPU)
**Best for:** Best performance, CPU-only machines

1. Install:
   ```bash
   pip install llama-cpp-python
   ```

2. Download a GGUF model (quantized):
   - From [Hugging Face](https://huggingface.co/search/full-text?q=gguf):
     - `TheBloke/Mistral-7B-Instruct-v0.1-GGUF`
     - `TheBloke/Phi-2-GGUF`
     - `TheBloke/qwen1_5-1_8b-chat-gguf`
   - Extract `.gguf` file to a known path

3. In app, select **LLaMA.cpp** and point to `.gguf` file

---

### **Option 3: GPT4All** 🎯 (Simplest)
**Best for:** Absolute beginners, auto-download models

1. Install:
   ```bash
   pip install gpt4all
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

3. Select **GPT4All** backend, models auto-download on first use

---

### **Option 4: Hugging Face Transformers** 🔬 (Most Control)
**Best for:** Power users, custom models, quantization

1. Install:
   ```bash
   pip install transformers torch bitsandbytes
   ```

2. Run the app and select **Hugging Face**

3. Paste a model ID (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)

4. Optional: Enable 4-bit quantization for lower VRAM

---

### **Option 5: LM Studio** 🖥️ (GUI + API)
**Best for:** Visual model management, easy switching

1. Install [LM Studio](https://lmstudio.ai/)
2. Download models via the GUI
3. Start local server (listens on `localhost:1234`)
4. In app, select **LM Studio** backend

---

## 💡 Quick Comparison

| Feature | Ollama | LLaMA.cpp | GPT4All | HF | LM Studio |
|---------|--------|-----------|---------|-----|-----------|
| **Setup Time** | 5 min | 10 min | 2 min | 15 min | 10 min |
| **Speed** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **CPU Only** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **GPU Support** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Model Count** | High | High | Limited | Huge | Medium |
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 🎯 Recommendations by Use Case

- **Just want it to work:** → **GPT4All** or **Ollama**
- **Best performance:** → **LLaMA.cpp**
- **Learning/experimenting:** → **Hugging Face**
- **GUI + easy switching:** → **LM Studio**
- **Production ready:** → **Ollama** or **LLaMA.cpp**

---

## 🔧 Troubleshooting

### Backend not connecting?
1. Check "Test Backend Connection" in sidebar
2. Ensure service is running (Ollama app, LM Studio, etc.)
3. Try restarting the app: `streamlit run app.py`

### Model running very slow?
- Switch to smaller model (qwen, phi, neural-chat)
- Reduce "Snippets to retrieve" slider
- Enable 4-bit quantization (HF backend)
- Use LLaMA.cpp with GPU acceleration

### Out of memory?
- Use 4-bit quantization
- Try a smaller model
- Reduce max context window
- Enable GPU acceleration

---

## 📝 Using the App

1. **Index your code/docs** in the sidebar ("Project folder path")
2. **Select what to index** (Python only, docs, or both)
3. **Test your backend** (button in sidebar)
4. **Ask a question** via text or voice
5. **View retrieved snippets** in expanders below answers

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

---

Enjoy! 🎉
