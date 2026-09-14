import os
import re
from typing import Any, Literal

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

IndexMode = Literal["python", "documents", "both"]


def _path_should_skip(file_path: str) -> bool:
    norm = file_path.replace("/", "\\").lower()
    for bad in (
        "\\venv\\",
        "\\.venv\\",
        "\\node_modules\\",
        "\\.git\\",
        "\\__pycache__\\",
    ):
        if bad in norm:
            return True
    return False


def scan_project_files(repo_path: str) -> list[str]:
    """Return relative paths of all files under repo_path (skipping venv/.git/etc.)."""
    root = os.path.abspath(repo_path)
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if not _path_should_skip(os.path.join(dirpath, d) + os.sep)
        ]
        for name in filenames:
            full = os.path.join(dirpath, name)
            if _path_should_skip(full):
                continue
            found.append(os.path.relpath(full, root).replace("\\", "/"))
    return sorted(found)


_LISTING_PATTERNS = re.compile(
    r"\b("
    r"list\s+(all\s+)?files|files\s+in|directory\s+contents|what\s+files|"
    r"show\s+(me\s+)?(all\s+)?files|project\s+files|folder\s+contents|"
    r"liste(r|\s)?\s*(des\s+)?fichiers|fichiers\s+(dans|du)|contenu\s+du\s+dossier|"
    r"quels\s+fichiers|arborescence"
    r")\b",
    re.IGNORECASE,
)


def is_directory_listing_question(question: str) -> bool:
    return bool(_LISTING_PATTERNS.search(question))


_FILE_NAME_PATTERN = re.compile(
    r"\b([\w.-]+\.(?:py|md|txt|rst|pdf))\b",
    re.IGNORECASE,
)


def extract_mentioned_files(question: str) -> list[str]:
    """Filenames explicitly named in the question (e.g. test.py)."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _FILE_NAME_PATTERN.finditer(question):
        name = match.group(1)
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out


_FILE_CONTENT_PATTERNS = re.compile(
    r"\b("
    r"content|contents|inside|show\s+me|what\s+is\s+in|what\s+does|"
    r"read|open|display|contenu|contient|affiche|montre"
    r")\b",
    re.IGNORECASE,
)


def is_file_content_question(question: str) -> bool:
    return bool(_FILE_CONTENT_PATTERNS.search(question)) and bool(extract_mentioned_files(question))


def read_project_file(project_root: str, filename: str, index_meta: dict[str, Any] | None) -> tuple[str, str] | None:
    """Return (relative_path, text) for a named file under project_root, or None."""
    root = os.path.abspath(project_root)
    catalog = [f.replace("\\", "/") for f in (index_meta or {}).get("all_files", [])]
    name_lower = filename.lower()
    basename = os.path.basename(name_lower)
    matches = [
        rel
        for rel in catalog
        if rel.lower() == name_lower or os.path.basename(rel).lower() == basename
    ]
    if not matches:
        candidate = os.path.join(root, filename)
        if os.path.isfile(candidate):
            matches = [filename.replace("\\", "/")]

    for rel in matches[:1]:
        full = os.path.join(root, rel.replace("/", os.sep))
        if not os.path.isfile(full) or full.lower().endswith(".pdf"):
            continue
        try:
            with open(full, encoding="utf-8", errors="ignore") as handle:
                return rel, handle.read()
        except OSError:
            continue
    return None


def format_file_content_answer(rel_path: str, text: str) -> str:
    ext = os.path.splitext(rel_path)[1].lower()
    lang = "python" if ext == ".py" else "text"
    body = text if text.strip() else "(empty file)"
    summary = (
        f"**File:** `{rel_path}`\n\n"
        f"**Content:**\n\n```{lang}\n{body}\n```\n\n"
    )
    lines = body.strip().splitlines()
    if ext == ".py" and lines:
        summary += (
            f"This Python file has **{len(lines)}** line(s). "
            "Describe what it does based only on the content above."
        )
    return summary


def _build_file_content_prompt(rel_path: str, text: str, question: str) -> str:
    ext = os.path.splitext(rel_path)[1].lower()
    lang = "python" if ext == ".py" else "text"
    body = text if text.strip() else "(empty file)"
    return (
        "You are a helpful code assistant. Use only the file content below to answer the user question. "
        "Do not invent behavior not present in the file.\n\n"
        f"File: {rel_path}\n"
        f"Content:\n```{lang}\n{body}\n```\n\n"
        f"Question: {question}\n\n"
        "Answer directly and concisely. If the user asked what the file does, describe its purpose and main behavior."
    )


def try_answer_file_content_question(
    question: str,
    project_root: str | None,
    index_meta: dict[str, Any] | None,
    backend: Any | None = None,
    num_predict: int = 512,
) -> str | None:
    if not is_file_content_question(question):
        return None
    root = (project_root or (index_meta or {}).get("root") or "").strip()
    if not root:
        return None

    mentioned = extract_mentioned_files(question)
    blocks: list[str] = []
    for name in mentioned:
        found = read_project_file(root, name, index_meta)
        if found:
            rel, text = found
            if backend is not None:
                blocks.append(_build_file_content_prompt(rel, text, question))
            else:
                blocks.append(format_file_content_answer(rel, text))

    if not blocks:
        return (
            f"I could not read **{', '.join(mentioned)}** under `{root}`. "
            "Check the filename, re-index the project, or set the full path in "
            "**Open code file path** in the sidebar."
        )

    if backend is None:
        return "\n\n".join(blocks)

    prompt_text = "\n\n".join(blocks)
    if hasattr(backend, "invoke") and callable(getattr(backend, "invoke")):
        response = backend.invoke(prompt_text)
    elif hasattr(backend, "query") and callable(getattr(backend, "query")):
        response = backend.query(prompt_text, max_tokens=num_predict)
    else:
        raise AttributeError("Backend object must expose query() or invoke()")

    if not isinstance(response, str):
        response = getattr(response, "text", None) or str(response)
    return response.strip()


def load_mentioned_files_context(
    project_root: str,
    filenames: list[str],
    index_meta: dict[str, Any] | None,
    max_chars: int = 4000,
) -> str:
    """Read files the user named directly so small files are not missed by vector search."""
    if not project_root or not filenames:
        return ""

    root = os.path.abspath(project_root)
    catalog = [f.replace("\\", "/") for f in (index_meta or {}).get("all_files", [])]
    parts: list[str] = []

    for name in filenames:
        name_lower = name.lower()
        basename = os.path.basename(name_lower)
        matches = [
            rel
            for rel in catalog
            if rel.lower() == name_lower or os.path.basename(rel).lower() == basename
        ]
        if not matches:
            candidate = os.path.join(root, name)
            if os.path.isfile(candidate):
                matches = [name.replace("\\", "/")]

        for rel in matches[:2]:
            full = os.path.join(root, rel.replace("/", os.sep))
            if not os.path.isfile(full) or full.lower().endswith(".pdf"):
                continue
            try:
                with open(full, encoding="utf-8", errors="ignore") as handle:
                    text = handle.read()
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n... [truncated]"
                parts.append(f"### FILE (user asked about this file): {rel}\n{text}")
            except OSError as exc:
                parts.append(f"### FILE (unreadable): {rel}\nError: {exc}")

    return "\n\n".join(parts)


def build_index_meta(
    repo_path: str,
    index_mode: IndexMode,
    indexed_files: list[str],
) -> dict[str, Any]:
    all_files = scan_project_files(repo_path)
    indexed_norm = {p.replace("\\", "/") for p in indexed_files}
    not_indexed = [f for f in all_files if f not in indexed_norm]
    return {
        "root": os.path.abspath(repo_path),
        "index_mode": index_mode,
        "indexed_files": sorted(indexed_norm),
        "all_files": all_files,
        "not_indexed_files": not_indexed,
    }


def format_file_inventory_answer(meta: dict[str, Any]) -> str:
    root = meta["root"]
    mode = meta["index_mode"]
    indexed = meta["indexed_files"]
    all_files = meta["all_files"]
    not_indexed = meta["not_indexed_files"]

    lines = [
        f"**Project root:** `{root}`",
        f"**Index mode:** {mode}",
        f"**All files in folder** (excluding venv, .git, node_modules, __pycache__): **{len(all_files)}**",
        f"**Files indexed for Q&A:** **{len(indexed)}**",
        "",
    ]
    if mode == "python":
        lines.append(
            "_Note: only `.py` files are indexed in this mode. Other files appear below but are not searchable by content._"
        )
    elif mode == "documents":
        lines.append(
            "_Note: only `.pdf`, `.md`, `.txt`, `.rst` are indexed. Other files are listed but not searchable._"
        )
    lines.append("")

    if all_files:
        lines.append("### All files")
        for path in all_files:
            tag = " *(indexed)*" if path in set(indexed) else ""
            lines.append(f"- `{path}`{tag}")
    else:
        lines.append("_No files found under this path._")

    if not_indexed and len(not_indexed) <= 30:
        lines.append("")
        lines.append(f"### Not indexed ({len(not_indexed)})")
        for path in not_indexed[:30]:
            lines.append(f"- `{path}`")

    return "\n".join(lines)


def inventory_context_for_prompt(meta: dict[str, Any] | None, max_listed: int = 120) -> str:
    if not meta:
        return ""
    indexed = meta["indexed_files"]
    all_files = meta["all_files"]
    lines = [
        f"Project file inventory: {len(all_files)} file(s) total, {len(indexed)} indexed (mode={meta['index_mode']}).",
        "Indexed paths:",
    ]
    for path in indexed[:max_listed]:
        lines.append(f"  - {path}")
    if len(indexed) > max_listed:
        lines.append(f"  ... and {len(indexed) - max_listed} more indexed file(s).")
    not_indexed = meta.get("not_indexed_files") or []
    if not_indexed:
        lines.append("Present but NOT indexed (content not in vector store):")
        for path in not_indexed[:40]:
            lines.append(f"  - {path}")
        if len(not_indexed) > 40:
            lines.append(f"  ... and {len(not_indexed) - 40} more.")
    return "\n".join(lines)


def build_vectorstore(repo_path: str, index_mode: IndexMode = "python"):
    """
    index_mode:
      - python: only .py (PDFs and other docs in the folder are never indexed).
      - documents: .pdf, .md, .txt, .rst only.
      - both: all of the above.
    """
    if not os.path.exists(repo_path) or not os.path.isdir(repo_path):
        raise ValueError(f"Invalid directory path: {repo_path}")

    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    import glob

    documents = []
    indexed_sources: set[str] = set()
    root_abs = os.path.abspath(repo_path)

    def _rel(path: str) -> str:
        return os.path.relpath(path, root_abs).replace("\\", "/")

    def load_text_file(file_path: str) -> None:
        try:
            loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
            loaded = loader.load()
            if loaded:
                documents.extend(loaded)
                indexed_sources.add(_rel(file_path))
        except Exception as e:
            print(f"Warning: could not read {file_path}. Skipping. Error: {e}")

    def load_pdf_file(file_path: str) -> None:
        try:
            loaded = PyPDFLoader(file_path).load()
            if loaded:
                documents.extend(loaded)
                indexed_sources.add(_rel(file_path))
        except Exception as e:
            print(f"Warning: could not read PDF {file_path}. Skipping. Error: {e}")

    want_py = index_mode in ("python", "both")
    want_docs = index_mode in ("documents", "both")

    if want_docs:
        doc_patterns = ("**/*.md", "**/*.txt", "**/*.rst", "**/*.pdf")
        print(f"Loading documents from {repo_path} ({', '.join(doc_patterns)})...")
        for pattern in doc_patterns:
            for file_path in glob.glob(os.path.join(repo_path, pattern), recursive=True):
                if _path_should_skip(file_path):
                    continue
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".pdf":
                    load_pdf_file(file_path)
                else:
                    load_text_file(file_path)

    if want_py:
        print(f"Loading Python files from {repo_path}...")
        for file_path in glob.glob(os.path.join(repo_path, "**", "*.py"), recursive=True):
            if _path_should_skip(file_path):
                continue
            load_text_file(file_path)

    if not documents:
        hint = {
            "python": "No .py files found (check path and that files are not only under venv/.git).",
            "documents": "No .pdf/.md/.txt/.rst files found.",
            "both": "No matching .py or document files found.",
        }[index_mode]
        raise ValueError(hint)

    print(f"Loaded {len(documents)} file segment(s). Splitting (mode={index_mode})...")
    if index_mode == "python":
        splitter = RecursiveCharacterTextSplitter.from_language(
            language="python", chunk_size=1000, chunk_overlap=100
        )
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    texts = splitter.split_documents(documents)

    print(f"Split into {len(texts)} chunks. Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    db = Chroma.from_documents(texts, embeddings)
    print("Vector store created successfully.")
    meta = build_index_meta(repo_path, index_mode, sorted(indexed_sources))
    print(
        f"Inventory: {len(meta['all_files'])} file(s) in tree, "
        f"{len(meta['indexed_files'])} indexed for search."
    )
    return db, meta


def _docs_to_context(docs: list, max_chars_per_doc: int) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown_file"
        text = d.page_content
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "\n... [truncated]"
        parts.append(f"### FILE: {src}\n{text}")
    return "\n\n".join(parts)


def _active_file_context(file_path: str | None, file_text: str | None, max_chars: int = 1200) -> str:
    if not file_path or not file_text:
        return ""
    excerpt = file_text.strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars] + "\n... [truncated]"
    return f"### ACTIVE FILE: {file_path}\n{excerpt}\n"


def _system_prompt_for_mode(index_mode: IndexMode) -> str:
    base = (
        "You use ONLY the excerpts below (each starts with ### FILE). "
        "Cite only file paths that actually appear in those excerpts. "
        "Do not contradict yourself (for example, do not say you skipped PDFs if a .pdf path appears "
        "in the excerpts). If the excerpts are insufficient, say what is missing."
    )
    if index_mode == "python":
        return (
            base
            + " The excerpts are Python source. Answer about what this project does and how it is "
            "structured in code (modules, functions, flow) using those files. "
            "Do not invent content from PDFs or other files not shown in the excerpts."
        )
    if index_mode == "documents":
        return (
            base
            + " The excerpts are from project documents (reports, notes). Summarize substance: "
            "findings, comparisons, conclusions. Do not narrate programming mechanics unless the user "
            "asks how the app is implemented."
        )
    return (
        base
        + " The excerpts may mix code and documents; stay grounded in them and name the files you use."
    )


def _user_source_rules(index_mode: IndexMode) -> str:
    if index_mode == "python":
        return (
            "All excerpts below are from indexed .py files only. Do not mention PDFs or other paths "
            "unless they appear explicitly in an excerpt."
        )
    if index_mode == "documents":
        return (
            "All excerpts below are from indexed documents only (.pdf, .md, .txt, .rst). "
            "Do not claim answers come from .py unless a .py path appears in an excerpt."
        )
    return "Excerpts may be code and/or documents; only use what appears below."


def query_llm(
    vectorstore,
    question: str,
    backend=None,
    model_name: str | None = None,
    base_url: str | None = None,
    *,
    retriever_k: int = 3,
    use_mmr: bool = False,
    max_chars_per_doc: int = 900,
    num_predict: int = 512,
    num_ctx: int = 4096,
    temperature: float = 0.2,
    project_root: str | None = None,
    index_mode: IndexMode = "python",
    open_file_path: str | None = None,
    open_file_text: str | None = None,
    index_meta: dict[str, Any] | None = None,
):
    if index_meta and is_directory_listing_question(question):
        return format_file_inventory_answer(index_meta), []

    effective_root = (project_root or (index_meta or {}).get("root") or "").strip()

    backend_obj = backend
    if backend_obj is None:
        resolved_base = (base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        resolved_model = model_name or os.environ.get("OLLAMA_MODEL", "llama3")
        backend_obj = Ollama(
            model=resolved_model,
            base_url=resolved_base,
            num_predict=num_predict,
            num_ctx=num_ctx,
            temperature=temperature,
            keep_alive="5m",
            system=_system_prompt_for_mode(index_mode),
        )
    else:
        system_prompt = _system_prompt_for_mode(index_mode)
        if hasattr(backend_obj, "system"):
            try:
                backend_obj.system = system_prompt
            except Exception:
                pass

    direct_file_answer = try_answer_file_content_question(
        question,
        project_root,
        index_meta,
        backend=backend_obj,
        num_predict=num_predict,
    )
    if direct_file_answer is not None:
        return direct_file_answer, []

    if backend is None:
        backend = backend_obj

    if vectorstore is not None:
        if use_mmr:
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": retriever_k, "fetch_k": max(retriever_k * 3, 8)},
            )
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
        docs = retriever.invoke(question)
        context = _docs_to_context(docs, max_chars_per_doc)
    else:
        docs = []
        context = "No indexed project excerpts available."

    mentioned = extract_mentioned_files(question)
    if effective_root:
        named_files_context = load_mentioned_files_context(
            effective_root, mentioned, index_meta, max_chars=max(max_chars_per_doc * 4, 2000)
        )
        if named_files_context:
            context = named_files_context + "\n\n" + context

    root = effective_root
    project_line = (
        f"Indexed project root: {root}"
        if root
        else "Indexed project root: (not set—use ### FILE labels in the excerpts.)"
    )

    inventory_block = inventory_context_for_prompt(index_meta)
    if inventory_block:
        project_line = project_line + "\n\n" + inventory_block

    open_file_context = _active_file_context(open_file_path, open_file_text, max_chars_per_doc)
    rules = _user_source_rules(index_mode)
    template = (
        "{project_line}\n\n"
        + rules
        + "\n\n"
        + ("Active file content:\n---\n{active_file_context}\n---\n\n" if open_file_context else "")
        + "Project excerpts:\n---\n{context}\n---\n\nQuestion: {question}\n\n"
        "Write a direct answer grounded in the excerpts. Name the files you used. "
        "Avoid generic web tutorials."
    )
    prompt = PromptTemplate.from_template(template)
    prompt_text = prompt.format(
        project_line=project_line,
        active_file_context=open_file_context,
        context=context,
        question=question,
    )

    if hasattr(backend, "invoke") and callable(getattr(backend, "invoke")):
        response = backend.invoke(prompt_text)
    elif hasattr(backend, "query") and callable(getattr(backend, "query")):
        response = backend.query(prompt_text, max_tokens=num_predict)
    else:
        raise AttributeError("Backend object must expose query() or invoke()")

    if not isinstance(response, str):
        response = getattr(response, "text", None) or str(response)
    return response.strip(), docs
