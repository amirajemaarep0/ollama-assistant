import os
import tempfile
import time
from pathlib import Path

import streamlit as st

from llm_backends import get_backend

# voice_stt pulls in torch + whisper (~4s) and rag_backend pulls in transformers via
# langchain (~13s). Importing them at module scope blocked the first paint for ~18s
# even when the user never touched voice input. They are imported on first use instead;
# sys.modules keeps them warm for later reruns.

st.set_page_config(
    page_title="Local Voice AI Dev Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎙️ Local Voice AI Dev Assistant")
st.markdown(
    "Ask questions grounded in **files you index locally** (choose Python only, documents only, or "
    "both in the sidebar). Answers are not pulled from the public web."
)

# Session state initialization
session_defaults = {
    "vs": None,
    "chat_history": [],
    "repo_path": "",
    "ollama_base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    "ollama_model": os.environ.get("OLLAMA_MODEL", "llama3"),  # must be one of OLLAMA_MODELS
    "ollama_num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "4096")),
    "indexed_mode": None,
    "index_meta": None,
    "indexed_files_count": 0,
    "last_indexed_path": None,
    "indexing_timestamp": None,
    "open_file_path": "",
    "llm_backend_client": None,
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Models named in the project specification ("running models like Llama 3 or Mistral").
OLLAMA_MODELS = ("llama3", "mistral")

INDEX_LABELS = (
    "Python (.py) only",
    "Documents only (.pdf, .md, .txt, .rst)",
    "Python + documents",
)
INDEX_LABEL_TO_MODE = {
    INDEX_LABELS[0]: "python",
    INDEX_LABELS[1]: "documents",
    INDEX_LABELS[2]: "both",
}

def backend_config() -> dict:
    """Current backend settings, as kwargs for get_backend()."""
    return {
        "backend_type": "ollama",
        "model_name": st.session_state.ollama_model,
        "base_url": st.session_state.ollama_base_url,
        "num_ctx": st.session_state.ollama_num_ctx,
    }


def get_active_backend():
    """Return the configured backend, rebuilding it whenever the settings change.

    The client used to be created once and kept until the user pressed
    "Test Backend Connection" again, so switching model or backend in the sidebar
    silently kept querying the old one.
    """
    config = backend_config()
    signature = tuple(sorted((k, str(v)) for k, v in config.items()))
    if (
        st.session_state.llm_backend_client is None
        or st.session_state.get("llm_backend_signature") != signature
    ):
        kwargs = dict(config)
        backend_type = kwargs.pop("backend_type")
        model_name = kwargs.pop("model_name")
        st.session_state.llm_backend_client = get_backend(backend_type, model_name, **kwargs)
        st.session_state.llm_backend_signature = signature
    return st.session_state.llm_backend_client


@st.cache_resource
def get_vectorstore(path: str, mode: str) -> tuple:
    """Cache vectorstore to avoid re-indexing on reruns."""
    from rag_backend import build_vectorstore

    try:
        vs, meta = build_vectorstore(path, index_mode=mode)
        return vs, meta, True
    except Exception as e:
        return None, None, str(e)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Ollama is the local inference engine named in the project specification.
    st.subheader("🤖 Ollama")
    st.session_state.ollama_base_url = st.text_input(
        "Ollama API URL",
        value=st.session_state.ollama_base_url,
        help="Default: http://localhost:11434"
    ).strip().rstrip("/")
    # The specification names Llama 3 and Mistral as the models to run on Ollama,
    # so those are the only two offered.
    model_index = OLLAMA_MODELS.index(st.session_state.ollama_model) if (
        st.session_state.ollama_model in OLLAMA_MODELS
    ) else 0
    st.session_state.ollama_model = st.selectbox(
        "Ollama model",
        OLLAMA_MODELS,
        index=model_index,
        help="Pull it first with `ollama pull <model>`.",
    )
    st.session_state.ollama_num_ctx = st.slider(
        "Context window (tokens)",
        1024,
        8192,
        st.session_state.ollama_num_ctx,
        step=512,
        help="Lower this (2048–4096) if you see memory errors with llama3 on a 4 GB GPU.",
    )

    # Test backend connection
    if st.button("🔍 Test Ollama Connection"):
        try:
            backend = get_active_backend()
            is_healthy, msg = backend.health_check()
            if is_healthy:
                st.success(msg)
            else:
                st.error(msg)
        except Exception as e:
            st.error(f"❌ Error: {str(e)[:150]}")
    
    st.markdown("---")
    
    # Existing configuration
    index_choice = st.radio(
        "What to index",
        INDEX_LABELS,
        key="rag_index_choice",
        help="PDFs in the folder are ignored unless you pick documents or both. "
        "After changing this, click **Load & Index Directory** again.",
    )
    index_mode = INDEX_LABEL_TO_MODE[index_choice]
    st.caption("**Speed:** smaller model, fewer chunks, lower max tokens. Ollama uses your GPU when one is available.")
    retriever_k = st.slider(
        "Snippets to retrieve",
        1,
        8,
        3,
        help="How many chunks from the indexed files to send to the model.",
    )
    max_answer_tokens = st.slider("Max answer tokens", 128, 2048, 512, step=64, help="Caps generation length.")
    max_chars_per_doc = st.slider(
        "Max characters per snippet",
        400,
        2000,
        900,
        step=100,
        help="Truncates each retrieved passage sent to the model.",
    )
    use_mmr = st.checkbox("Use MMR retrieval", value=False, help="Slightly better diversity, slower than plain similarity.")
    path_input = st.text_input(
        "Project folder path",
        value=st.session_state.repo_path,
        help="Folder to scan. What gets indexed depends on **What to index** above.",
    )
    path_input = path_input.strip().strip('"').strip("'")

    open_file_input = st.text_input(
        "Open code file path (optional)",
        value=st.session_state.open_file_path,
        help="A specific file whose contents should be injected into the prompt, e.g. the current editor file.",
    )
    st.session_state.open_file_path = open_file_input.strip().strip('"').strip("'")

    index_col, clear_col = st.columns([3, 1])
    with index_col:
        if st.button("Load & Index Directory", use_container_width=True):
            if not path_input:
                st.error("⚠️ Please enter a project folder path.")
            elif not os.path.exists(path_input):
                st.error(f"❌ Directory does not exist: {path_input}")
            elif not os.path.isdir(path_input):
                st.error(f"❌ Path is not a directory: {path_input}")
            else:
                st.session_state.repo_path = path_input
                with st.spinner("📊 Indexing codebase... This might take a minute."):
                    try:
                        vs, meta, result = get_vectorstore(path_input, index_mode)
                        if isinstance(result, bool) and result:
                            st.session_state.vs = vs
                            st.session_state.index_meta = meta
                            st.session_state.indexed_mode = index_mode
                            st.session_state.last_indexed_path = path_input
                            st.session_state.indexing_timestamp = time.time()
                            if meta:
                                st.session_state.indexed_files_count = len(meta.get("indexed_files", []))
                            st.success(
                                f"✅ Indexed {st.session_state.indexed_files_count} file(s) for search "
                                f"({len(meta.get('all_files', [])) if meta else 0} total in folder)."
                            )
                        else:
                            st.error(f"❌ Error indexing: {result}")
                    except Exception as e:
                        st.error(f"❌ Error indexing: {str(e)[:200]}")
    
    with clear_col:
        if st.button("🗑️ Clear", help="Clear chat history and index"):
            st.session_state.chat_history = []
            st.session_state.vs = None
            st.session_state.index_meta = None
            st.rerun()
    
    if st.session_state.vs is not None:
        st.success(f"✅ Indexed: {st.session_state.last_indexed_path}")
        meta = st.session_state.index_meta
        if meta:
            st.caption(
                f"{len(meta.get('indexed_files', []))} searchable · "
                f"{len(meta.get('all_files', []))} files in folder · mode: {meta.get('index_mode')}"
            )
            with st.expander("📁 All files in project"):
                indexed_set = set(meta.get("indexed_files", []))
                for path in meta.get("all_files", []):
                    mark = "✓" if path in indexed_set else "○"
                    st.text(f"{mark} {path}")
        if st.session_state.indexing_timestamp:
            elapsed = time.time() - st.session_state.indexing_timestamp
            st.caption(f"Last indexed {elapsed:.0f}s ago")
        if st.button("Refresh Index", help="Re-index the current directory"):
            st.session_state.vs = None
            st.cache_resource.clear()
            st.rerun()
            
# Open by default until something is indexed, then get out of the way.
with st.expander("ℹ️ What this app can do", expanded=st.session_state.vs is None):
    what_col, ask_col = st.columns(2)

    with what_col:
        st.markdown(
            """
**📂 Indexes a project folder**

Python files, documents (`.pdf`, `.md`, `.txt`, `.rst`), or both.
`venv`, `.git`, `node_modules` and caches are skipped automatically.

**🎙️ Takes voice or typed questions**

Press the 🎙 mic in the chat bar below, next to the send arrow, and speak.
Whisper transcribes it on this machine. No microphone? Just type.

**🩺 Checks your Python**

Python's own parser finds syntax errors; pyflakes finds undefined names
and unused imports. No model involved, so it is instant and exact.
Neither runs your code, so logic errors are not covered.

**🔍 Shows its sources**

Each answer names the files it used. Expand *Retrieved snippets* under an
answer to read the exact passages it was given. Ordinary questions retrieve
only the closest few chunks — ask about the *whole project* to cover them all.

**🔒 Runs fully offline**

Ollama, Whisper and the embeddings all run locally.
No code or document leaves this computer.
"""
        )

    with ask_col:
        st.markdown(
            """
**Five kinds of question it recognises**

| Ask it | What happens |
| --- | --- |
| *List all files in the project* | Instant file inventory, no model needed |
| *Scan the whole project* | Outlines **every** file, not just the closest matches |
| *What is wrong with app.py?* | Parses it, points at syntax errors and undefined names, no model needed |
| *What does app.py do?* | Reads that file and explains it |
| *How does the indexing work?* | Searches the index, answers from the best matches |

**Also**

- Works in English and French (*quels fichiers*, *contenu de…*).
- **Open code file path** injects the file you are editing into every question.
- **Export Chat** saves the whole conversation as a text file.
- Read-only: it explains and diagnoses, it never edits your files.
"""
        )


if st.session_state.vs is None:
    st.info("👈 **Step 1:** Index a directory in the sidebar | **Step 2:** Ask a question below")

# accept_audio puts a microphone button inside the chat bar, right beside the
# send arrow. Recording happens in the browser, so no microphone access is
# needed on the server and there is no fixed recording length any more.
user_input = st.chat_input(
    "Type your question, or press the mic to ask out loud...",
    accept_audio=True,
    audio_sample_rate=16000,
)

if user_input is not None:
    typed = (getattr(user_input, "text", None) or "").strip()
    recording = getattr(user_input, "audio", None)
    spoken = ""

    if recording is not None:
        try:
            with st.spinner("⏳ Transcribing your recording..."):
                from voice_stt import transcribe_audio

                clip = os.path.join(tempfile.gettempdir(), "chat_recording.wav")
                with open(clip, "wb") as handle:
                    handle.write(recording.getvalue())
                spoken = (transcribe_audio(clip) or "").strip()
            if spoken:
                st.success(f"🎙️ Heard: {spoken}")
            else:
                st.warning("⚠️ Could not make out any speech. Try recording again.")
        except Exception as e:
            st.error(f"❌ Transcription failed: {str(e)[:200]}")

    question = " ".join(part for part in (typed, spoken) if part)
    if question:
        st.session_state.current_prompt = question

# Main Chat Interface
if "current_prompt" in st.session_state and st.session_state.current_prompt:
    prompt = st.session_state.current_prompt
    
    if st.session_state.vs is None:
        st.error("❌ No index loaded. Please index a directory in the sidebar first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.spinner("🤔 Thinking..."):
            try:
                if st.session_state.vs is not None and st.session_state.indexed_mode != index_mode:
                    raise ValueError(
                        "⚠️ Sidebar **What to index** does not match the loaded index. "
                        "Click **Load & Index Directory** again (or switch the radio back to match the last load)."
                    )
                prompt_index_mode = (
                    st.session_state.indexed_mode
                    if st.session_state.vs is not None
                    else index_mode
                )
                
                from rag_backend import query_llm

                backend_client = get_active_backend()

                open_file_path = st.session_state.open_file_path
                open_file_text = ""
                if open_file_path:
                    try:
                        if os.path.exists(open_file_path) and os.path.isfile(open_file_path):
                            with open(open_file_path, "r", encoding="utf-8", errors="ignore") as file:
                                open_file_text = file.read()
                        else:
                            st.warning(f"Open file path not found or not a file: {open_file_path}")
                            open_file_path = None
                    except Exception as e:
                        st.warning(f"Could not read open file: {str(e)[:150]}")
                        open_file_path = None
                        open_file_text = ""

                # Query with RAG
                response, docs = query_llm(
                    st.session_state.vs,
                    prompt,
                    backend=backend_client,
                    retriever_k=retriever_k,
                    use_mmr=use_mmr,
                    max_chars_per_doc=max_chars_per_doc,
                    num_predict=max_answer_tokens,
                    num_ctx=st.session_state.ollama_num_ctx,
                    project_root=st.session_state.repo_path or None,
                    index_mode=prompt_index_mode,
                    open_file_path=open_file_path,
                    open_file_text=open_file_text,
                    index_meta=st.session_state.index_meta,
                )
                st.session_state.chat_history.append({"role": "assistant", "content": response, "docs": docs})
            except Exception as e:
                err = str(e).lower()
                if any(
                    x in err
                    for x in (
                        "actively refused",
                        "10061",
                        "failed to establish",
                        "connection refused",
                        "name or service not known",
                    )
                ):
                    st.error(
                        "🔴 **Cannot reach Ollama.**\n\n"
                        "1. Install it from ollama.com and start the Ollama app\n"
                        "2. Pull the model: `ollama pull llama3`\n"
                        "3. Check the **Ollama API URL** in the sidebar "
                        "(default `http://localhost:11434`)"
                    )
                elif any(
                    x in err
                    for x in (
                        "mem_buffer",
                        "runner process has terminated",
                        "out of memory",
                        "failed to allocate",
                    )
                ):
                    st.error(
                        "🔴 **Ollama ran out of memory loading the model.**\n\n"
                        "Your GPU/RAM cannot fit this model at the current context size.\n\n"
                        "**Try:**\n"
                        "1. Lower **Context window** in the sidebar to **2048** or **4096**\n"
                        "2. Switch the **Ollama model** to `mistral`\n"
                        "3. Close other apps, then restart Ollama\n"
                        "4. Reduce **Snippets to retrieve** and **Max characters per snippet**"
                    )
                else:
                    st.error(f"❌ Error querying model: {str(e)[:200]}")
            
    if "current_prompt" in st.session_state:
        del st.session_state.current_prompt

st.markdown("---")
st.subheader("💬 Chat History")

if not st.session_state.chat_history:
    st.info("No messages yet. Start by asking a question!")
else:
    export_col, clear_hist_col, count_col = st.columns(3)
    with export_col:
        # download_button nested inside st.button needed two clicks and disappeared
        # on the next rerun; render it directly instead.
        chat_text = "\n".join(
            f"**{msg['role'].upper()}:** {msg['content']}" for msg in st.session_state.chat_history
        )
        st.download_button(
            label="📥 Export Chat",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with clear_hist_col:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with count_col:
        st.metric("Messages", len(st.session_state.chat_history))
    
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "docs" in msg and msg["docs"]:
                with st.expander(f"📄 Retrieved {len(msg['docs'])} snippets (from indexed files)"):
                    for j, doc in enumerate(msg["docs"], 1):
                        source = doc.metadata.get('source', 'Unknown')
                        st.markdown(f"**Chunk {j}** • `{Path(source).name}`")
                        st.code(doc.page_content, language="python" if source.endswith(".py") else "text")
            if msg["role"] == "user" and i > 0:
                st.divider()
