import ast
import json
import os
import re
import subprocess
import sys
from typing import Any, Literal

from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from llm_backends import OllamaBackend

IndexMode = Literal["python", "documents", "both"]


def _path_should_skip(file_path: str) -> bool:
    norm = file_path.replace("/", "\\").lower()
    for bad in (
        "\\venv\\",
        "\\.venv\\",
        "\\env\\",
        "\\node_modules\\",
        "\\.git\\",
        "\\__pycache__\\",
        "\\.pytest_cache\\",
        "\\.mypy_cache\\",
        "\\.ruff_cache\\",
        "\\site-packages\\",
        "\\.idea\\",
        "\\.vscode\\",
        "\\.cursor\\",
        "\\build\\",
        "\\dist\\",
        "\\.ipynb_checkpoints\\",
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


def _iter_project_files(root: str, extensions: set[str]):
    """Yield absolute paths under root whose extension is in `extensions`.

    Prunes venv/.git/node_modules/__pycache__ *during* the walk. The previous
    implementation globbed `**/*.ext` per pattern and filtered afterwards, which
    descended into .venv (~20k files here) once per pattern before discarding it.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if not _path_should_skip(os.path.join(dirpath, d) + os.sep)
        ]
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() in extensions:
                yield os.path.join(dirpath, name)


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


def try_answer_optimization_question(
    question: str,
    project_root: str | None,
    index_meta: dict[str, Any] | None,
    backend: Any | None = None,
    num_predict: int = 512,
) -> str | None:
    """Rewrite one named function, giving the model its exact source.

    Retrieval would hand over a few 900-character chunks, which is not enough to
    rewrite a function correctly. The whole function is extracted with ast
    instead, and the reply is checked for parseability afterwards.
    """
    if not is_optimization_question(question) or backend is None:
        return None
    root = (project_root or (index_meta or {}).get("root") or "").strip()
    if not root:
        return None

    named = [n for n in extract_mentioned_files(question) if n.lower().endswith(".py")]
    if not named:
        return None

    for rel, source in _resolve_python_files(root, named, index_meta):
        name = find_requested_function(question, source, rel)
        if name is None:
            available = ", ".join(
                sorted(q.rsplit(".", 1)[-1] for q, _ in iter_function_defs(source, rel))
            )
            return (
                f"Name the function you want optimised in `{rel}`. "
                + (f"It defines: {available}." if available else "It defines no functions.")
            )

        found = extract_function_source(source, name, rel)
        if found is None:
            return None
        qualified, function_source, start_line = found

        prompt = build_optimization_prompt(rel, qualified, function_source, question)
        if hasattr(backend, "invoke") and callable(getattr(backend, "invoke")):
            answer = backend.invoke(prompt)
        elif hasattr(backend, "query") and callable(getattr(backend, "query")):
            answer = backend.query(prompt, max_tokens=num_predict)
        else:
            return None
        if not isinstance(answer, str):
            answer = getattr(answer, "text", None) or str(answer)
        answer = answer.strip()

        header = f"**Optimising `{qualified}`** in `{rel}` (line {start_line})"
        return header + chr(10) + chr(10) + answer + chr(10) + chr(10) + verify_optimized_function(answer, qualified, function_source)

    return None


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


_SYNTAX_PATTERNS = re.compile(
    r"\b("
    r"syntax|syntaxe|compiles?|parses?|"
    r"wrong|bugs?|errors?|broken|fails?|failing|fix|fixes|check|verify|invalid|"
    r"erreurs?|probl[eè]mes?|corrige[rz]?|v[ée]rifie[rz]?|cass[ée]"
    r")\b",
    re.IGNORECASE,
)

# "syntax"/"compile" wording is what licenses a whole-project scan; a bare
# "how does the project handle errors?" must not trigger one.
_SYNTAX_EXPLICIT = re.compile(r"\b(syntax|syntaxe|compiles?|parses?)\b", re.IGNORECASE)

_PROJECT_SCOPE = re.compile(
    r"\b(project|projet|codebase|all\s+(?:the\s+)?files|every\s+file|tout\s+le\s+code)\b",
    re.IGNORECASE,
)


_ERROR_NOUNS = r"(?:errors?|bugs?|problems?|issues?|mistakes?|erreurs?|probl[eè]mes?|fautes?)"

# A request to go and look: "scan for errors", "check the project for bugs",
# "find all errors", "any problems?". The verb is what separates these from a
# question about behaviour.
_SCAN_INTENT = re.compile(
    r"\b(?:scan|check|find|look|search|list|report|detect|show|spot|"
    r"any|are\s+there|is\s+there|"
    r"scanne[rz]?|v[ée]rifie[rz]?|cherche[rz]?|liste[rz]?|trouve[rz]?)\b"
    r"[^.?!]{0,60}?\b" + _ERROR_NOUNS,
    re.IGNORECASE,
)

# "how does X handle errors", "explain the error handling" - these ask about the
# code's behaviour and must not kick off a scan.
_EXPLANATORY = re.compile(
    r"\b(?:how\s+(?:does|do|is|are|should)|explain|describe|why\s+(?:does|do|is|are)|"
    r"what\s+happens|handle[sd]?\s+" + _ERROR_NOUNS + r"|"
    r"comment\s+(?:est|sont|fait))\b",
    re.IGNORECASE,
)


def is_syntax_check_question(question: str) -> bool:
    """True for 'what is wrong with app.py', 'scan for errors in the project'.

    False for 'how does the project handle errors?', which asks about behaviour
    rather than requesting a scan.
    """
    if _EXPLANATORY.search(question):
        return False

    # "scan for errors", "find any issues" - a verb plus a problem noun is a
    # request to look, and stands on its own without any other keyword.
    if _SCAN_INTENT.search(question):
        return True

    if not _SYNTAX_PATTERNS.search(question):
        return False

    if any(n.lower().endswith(".py") for n in extract_mentioned_files(question)):
        return True

    # No file named and no scan verb: only explicit syntax wording counts.
    return bool(_PROJECT_SCOPE.search(question) and _SYNTAX_EXPLICIT.search(question))


def check_python_syntax(source: str, rel_path: str) -> dict[str, Any] | None:
    """Parse `source`; return error details, or None when it is valid Python."""
    try:
        ast.parse(source, filename=rel_path)
        return None
    except SyntaxError as exc:
        return {
            "path": rel_path,
            "line": exc.lineno,
            "offset": exc.offset,
            "msg": exc.msg,
            "text": (exc.text or "").rstrip("\n"),
        }
    except ValueError as exc:  # e.g. source containing null bytes
        return {"path": rel_path, "line": None, "offset": None, "msg": str(exc), "text": ""}


_CLOSERS = {"(": ")", "[": "]", "{": "}"}


def _unclosed_bracket_line(lines: list[str], before_line: int) -> tuple[int, str] | None:
    """Find a bracket opened before `before_line` and never closed.

    CPython reports a generic "invalid syntax" at the point where an unclosed
    bracket finally becomes impossible, which is often several lines below the
    real mistake. Scanning the bracket depth finds where it actually opened.
    """
    stack: list[tuple[str, int]] = []
    for number, line in enumerate(lines[: max(before_line, 0)], start=1):
        in_quote = ""
        index = 0
        while index < len(line):
            char = line[index]
            if in_quote:
                if char == "\\":
                    index += 2
                    continue
                if char == in_quote:
                    in_quote = ""
            elif char in "\"'":
                in_quote = char
            elif char == "#":
                break
            elif char in _CLOSERS:
                stack.append((char, number))
            elif char in ")]}":
                if stack:
                    stack.pop()
            index += 1
    if not stack:
        return None
    opener, number = stack[0]
    return number, _CLOSERS[opener]


def _repair_reported_line(lines: list[str], problem: dict[str, Any]) -> bool:
    """Apply the minimal fix for one reported error, in place.

    Returns True when the line was changed. The result is never shown to the
    user; it exists only so CPython's parser can get past this error and reveal
    the next genuine one.
    """
    lineno = problem.get("line")
    message = problem.get("msg") or ""
    if not lineno or lineno > len(lines):
        return False

    index = lineno - 1
    line = lines[index]
    stripped = line.rstrip()

    if "expected ':'" in message:
        lines[index] = stripped + ":"
        return True

    if "was never closed" in message:
        opener = message.split("'")[1] if "'" in message else ""
        closer = _CLOSERS.get(opener)
        if closer:
            lines[index] = stripped + closer
            return True
        return False

    if "unterminated string literal" in message:
        quote = '"' if stripped.count('"') % 2 else ("'" if stripped.count("'") % 2 else "")
        if quote:
            lines[index] = stripped + quote
            return True
        return False

    if "Missing parentheses in call to" in message:
        match = re.match(r"^(\s*)(print|exec)\s+(.*)$", line)
        if match:
            indent, name, rest = match.groups()
            lines[index] = f"{indent}{name}({rest.rstrip()})"
            return True
        return False

    if "expected an indented block" in message:
        indent = len(line) - len(line.lstrip())
        lines[index] = " " * (indent + 4) + line.lstrip()
        return True

    # Generic "invalid syntax" is usually a bracket opened further up.
    found = _unclosed_bracket_line(lines, index)
    if found:
        number, closer = found
        lines[number - 1] = lines[number - 1].rstrip() + closer
        return True

    return False


def check_python_syntax_all(
    source: str, rel_path: str, max_errors: int = 10
) -> list[dict[str, Any]]:
    """Every syntax error in one file, each with CPython's own message.

    ast.parse stops at the first error and cannot resume, so this repairs the
    reported line on a throwaway copy of the source and parses again. Every
    error therefore comes from CPython itself rather than from a recovering
    parser guessing, which is what keeps cascades out of the list.
    """
    first = check_python_syntax(source, rel_path)
    if first is None:
        return []

    results = [first]
    lines = source.splitlines()
    current = first

    for _ in range(max_errors - 1):
        if not _repair_reported_line(lines, current):
            break
        nxt = check_python_syntax(chr(10).join(lines), rel_path)
        if nxt is None:
            break
        if nxt["line"] == current["line"] and nxt["msg"] == current["msg"]:
            break  # repair did not move us on; stop rather than loop
        nxt["recovered"] = True
        results.append(nxt)
        current = nxt

    return results


_OPTIMISE_PATTERNS = re.compile(
    r"\b("
    r"optimi[sz]e|optimi[sz]ed|optimi[sz]ation|"
    r"improve|refactor|rewrite|clean\s+up|simplify|"
    r"faster|quicker|speed\s+\w*\s*up|"
    r"more\s+efficient|performance|"
    r"optimise[rz]?|am[ée]liore[rz]?|refactorise[rz]?|simplifie[rz]?|plus\s+rapide"
    r")\b",
    re.IGNORECASE,
)


def is_optimization_question(question: str) -> bool:
    return bool(_OPTIMISE_PATTERNS.search(question))


def iter_function_defs(source: str, rel_path: str = "<unknown>"):
    """Yield (qualified_name, node) for every function and method in a module."""
    try:
        tree = ast.parse(source, filename=rel_path)
    except (SyntaxError, ValueError):
        return

    def walk(node, prefix=""):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield prefix + child.name, child
            elif isinstance(child, ast.ClassDef):
                yield from walk(child, prefix + child.name + ".")

    yield from walk(tree)


def extract_function_source(source: str, name: str, rel_path: str = "<unknown>"):
    """Exact source of one function, decorators included, or None.

    `name` may be a bare function name or Class.method.
    """
    lines = source.splitlines()
    wanted = name.lower()
    for qualified, node in iter_function_defs(source, rel_path):
        if qualified.lower() != wanted and qualified.lower().rsplit(".", 1)[-1] != wanted:
            continue
        start = node.lineno
        for decorator in getattr(node, "decorator_list", []):
            start = min(start, decorator.lineno)
        end = getattr(node, "end_lineno", node.lineno)
        return qualified, chr(10).join(lines[start - 1 : end]), start
    return None


def find_requested_function(question: str, source: str, rel_path: str) -> str | None:
    """Which function in this file the question is about.

    Matches the question against the names actually defined in the file rather
    than trying to parse the sentence, so "speed up summarise" and
    "optimize the summarise() method" both work.
    """
    lowered = question.lower()
    best = None
    for qualified, _node in iter_function_defs(source, rel_path):
        bare = qualified.rsplit(".", 1)[-1]
        if re.search(r"\b" + re.escape(bare.lower()) + r"\b", lowered):
            if best is None or len(bare) > len(best.rsplit(".", 1)[-1]):
                best = qualified
    return best


def build_optimization_prompt(rel_path: str, name: str, function_source: str, question: str) -> str:
    return (
        "You are a Python performance engineer. Rewrite the function below so it is "
        "faster and clearer, without changing what it returns.\n\n"
        f"File: {rel_path}\n"
        f"Function: {name}\n\n"
        "```python\n"
        f"{function_source}\n"
        "```\n\n"
        f"Request: {question}\n\n"
        "Rules:\n"
        "- Keep the same function name, parameters and return value.\n"
        "- Use only the standard library.\n"
        "- Reply with ONE complete ```python code block containing the whole "
        "rewritten function, then a short bullet list of what you changed and why.\n"
        "- If the function is already efficient, say so plainly and return it unchanged."
    )


def extract_code_block(answer: str) -> str | None:
    """The first fenced code block in a model reply."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", answer, re.DOTALL)
    return match.group(1).rstrip() if match else None


_DIFF_HARNESS = '''
import itertools, json, sys

ORIGINAL = {original!r}
REWRITTEN = {rewritten!r}
NAME = {name!r}

def load(src):
    namespace = {{}}
    exec(compile(src, "<candidate>", "exec"), namespace)
    return namespace[NAME]

try:
    before, after = load(ORIGINAL), load(REWRITTEN)
except Exception as exc:
    print(json.dumps({{"status": "load-failed", "detail": str(exc)[:200]}}))
    sys.exit()

CANDIDATES = [
    [], [1], [1, 1], [1, 1, 1], [1, 2, 2, 3, 3, 3], [5, 4, 3, 2, 1],
    list("aabbcc"), list(range(8)), [0, 0, 0, 1], [True, False, True],
]

mismatches, ran = [], 0
for value in CANDIDATES:
    try:
        expected = before(list(value))
    except Exception:
        continue
    try:
        actual = after(list(value))
    except Exception as exc:
        ran += 1
        mismatches.append({{"input": repr(value), "expected": repr(expected),
                           "actual": "raised " + type(exc).__name__}})
        continue
    ran += 1
    if expected != actual:
        mismatches.append({{"input": repr(value), "expected": repr(expected),
                           "actual": repr(actual)}})

print(json.dumps({{"status": "ok", "ran": ran, "mismatches": mismatches[:4]}}))
'''


def differential_test(
    original_source: str, rewritten_source: str, name: str, timeout: int = 15
) -> dict[str, Any]:
    """Run both versions on sample inputs and compare, in a separate process.

    A rewrite can parse, keep its name, and still return different answers -- an
    LLM cannot run what it writes. This is the only check that catches that.
    A subprocess keeps a hang or a crash in generated code away from the app.
    """
    bare = name.rsplit(".", 1)[-1]
    if "." in name:
        return {"status": "skipped", "detail": "methods are not tested automatically"}

    script = _DIFF_HARNESS.format(
        original=original_source, rewritten=rewritten_source, name=bare
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "detail": f"did not finish within {timeout}s"}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)[:200]}

    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception:
        return {"status": "error", "detail": (completed.stderr or "no output")[:200]}


def format_differential_note(result: dict[str, Any], name: str) -> str:
    status = result.get("status")
    if status == "ok" and result.get("ran"):
        mismatches = result.get("mismatches") or []
        if not mismatches:
            return (
                f"✅ Behaviour check: `{name}` returned the same result as the "
                f"original on all {result['ran']} sample input(s)."
            )
        lines = [
            f"❌ **Behaviour changed.** The rewrite disagrees with the original "
            f"on {len(mismatches)} of {result['ran']} sample input(s). Do not use it as is:",
            "",
        ]
        for m in mismatches:
            lines.append(f"- `{m['input']}` → original `{m['expected']}`, rewrite `{m['actual']}`")
        return chr(10).join(lines)
    if status == "timeout":
        return f"⚠️ Behaviour check timed out ({result.get('detail')}). The rewrite may not terminate."
    return f"⚠️ Behaviour could not be checked automatically ({result.get('detail', status)}). Test it yourself."


def verify_optimized_function(
    answer: str, name: str, original_source: str | None = None
) -> str:
    """Check the returned code parses and still defines the function.

    The model cannot run anything, so this is the one guarantee worth making
    automatically before the user pastes the result into their file.
    """
    code = extract_code_block(answer)
    if code is None:
        return "⚠️ No code block found in the reply, so nothing could be checked."

    problem = check_python_syntax(code, name + " (rewritten)")
    if problem:
        return (
            f"❌ The rewritten code does not parse: line {problem['line']}: "
            f"{problem['msg']}. Do not paste it in as is."
        )

    bare = name.rsplit(".", 1)[-1]
    defined = {q.rsplit(".", 1)[-1] for q, _ in iter_function_defs(code)}
    if bare not in defined:
        found = ", ".join(sorted(defined)) or "none"
        return f"⚠️ The reply defines {found}, not `{bare}`. Check it before using it."

    parts = [f"✅ The rewritten `{bare}` parses and keeps its name."]
    quality = [f for f in check_python_quality(code, "rewritten.py") if f["is_bug"]]
    if quality:
        parts.append(
            "⚠️ " + "; ".join(f"line {f['line']}: {f['msg']}" for f in quality)
        )
    if original_source is not None:
        parts.append(
            format_differential_note(
                differential_test(original_source, code, name), bare
            )
        )
    return chr(10).join(parts)


def _resolve_python_files(project_root: str, names: list[str], index_meta: dict[str, Any] | None):
    """Yield (rel_path, source) for the named .py files, or for every indexed .py."""
    root = os.path.abspath(project_root)
    catalog = [f.replace("\\", "/") for f in (index_meta or {}).get("all_files", [])]

    if names:
        wanted = []
        for name in names:
            if not name.lower().endswith(".py"):
                continue
            basename = os.path.basename(name.lower())
            matches = [
                rel for rel in catalog
                if rel.lower() == name.lower() or os.path.basename(rel).lower() == basename
            ]
            if not matches and os.path.isfile(os.path.join(root, name)):
                matches = [name.replace("\\", "/")]
            wanted.extend(matches[:1] or [])
    else:
        wanted = [rel for rel in catalog if rel.lower().endswith(".py")]

    seen = set()
    for rel in wanted:
        if rel in seen:
            continue
        seen.add(rel)
        full = os.path.join(root, rel.replace("/", os.sep))
        if not os.path.isfile(full):
            continue
        try:
            with open(full, encoding="utf-8", errors="ignore") as handle:
                yield rel, handle.read()
        except OSError:
            continue


def format_syntax_report(problems: list[dict[str, Any]], checked: list[str]) -> str:
    if not problems:
        if len(checked) == 1:
            return f"✅ `{checked[0]}` has no syntax errors. It parses cleanly."
        return f"✅ No syntax errors. All **{len(checked)}** Python file(s) parse cleanly."

    broken_files = {p["path"] for p in problems}
    extra = sum(1 for p in problems if p.get("recovered"))
    headline = (
        f"❌ Found syntax errors in **{len(broken_files)}** of {len(checked)} "
        "Python file(s) checked."
    )
    if extra:
        headline += f" {len(problems)} problem area(s) in total."
    lines = [headline, ""]
    for p in problems:
        where = f"line {p['line']}" if p["line"] else "unknown line"
        marker = "↳ also" if p.get("recovered") else "**`" + p["path"] + "`** —"
        lines.append(f"{marker} {where}: {p['msg']}" if p.get("recovered")
                     else f"**`{p['path']}`** — {where}: {p['msg']}")
        if p["text"]:
            lines.append("")
            lines.append("```python")
            lines.append(p["text"])
            if p["offset"] and p["offset"] > 0:
                lines.append(" " * (p["offset"] - 1) + "^")
            lines.append("```")
        lines.append("")

    clean = len(checked) - len(broken_files)
    if clean:
        lines.append(f"_The other {clean} file(s) parse cleanly._")
    if extra:
        lines.append("")
        lines.append(
            "_Python reports one error at a time, so the ones marked with an arrow "
            "were found by provisionally fixing the error above them. Fix them from "
            "the top down: correcting the first may change the lines reported after it._"
        )
    return "\n".join(lines).rstrip()


# Pyflakes message classes that are tidiness rather than likely bugs. Anything
# not in this set is reported as a probable bug, so message types added by a
# future pyflakes surface instead of being buried in the cleanup section.
_QUALITY_CLEANUP = frozenset(
    {
        "UnusedImport",
        "UnusedVariable",
        "UnusedAnnotation",
        "UnusedIndirectAssignment",
        "RedefinedWhileUnused",
        "ImportStarUsed",
        "FStringMissingPlaceholders",
        "TStringMissingPlaceholders",
    }
)


def check_python_quality(source: str, rel_path: str) -> list[dict[str, Any]]:
    """Undefined names, unused imports and similar, using pyflakes.

    Returns [] when pyflakes is not installed, so the syntax check keeps working
    on its own and pyflakes stays an optional extra rather than a hard
    dependency. Also returns [] for unparseable source: the caller reports the
    syntax error for that file instead.
    """
    try:
        from pyflakes.checker import Checker
    except ImportError:
        return []

    try:
        tree = ast.parse(source, filename=rel_path)
    except (SyntaxError, ValueError):
        return []

    findings: list[dict[str, Any]] = []
    for message in Checker(tree, filename=rel_path).messages:
        kind = type(message).__name__
        try:
            text = message.message % message.message_args
        except Exception:
            text = str(message)
        findings.append(
            {
                "path": rel_path,
                "line": message.lineno,
                "kind": kind,
                "msg": text,
                "is_bug": kind not in _QUALITY_CLEANUP,
            }
        )
    findings.sort(key=lambda f: (not f["is_bug"], f["path"], f["line"]))
    return findings


def format_quality_report(findings: list[dict[str, Any]], checked: list[str]) -> str:
    bugs = [f for f in findings if f["is_bug"]]
    tidy = [f for f in findings if not f["is_bug"]]
    scope = f"`{checked[0]}`" if len(checked) == 1 else f"{len(checked)} Python file(s)"

    lines: list[str] = []
    if bugs:
        lines.append(f"⚠️ No syntax errors in {scope}, but found **{len(bugs)}** likely bug(s):")
        lines.append("")
        for f in bugs:
            lines.append(f"- **`{f['path']}`** line {f['line']}: {f['msg']}")
        lines.append("")
    else:
        lines.append(f"✅ No syntax errors and no likely bugs in {scope}.")
        lines.append("")

    if tidy:
        # Plain markdown only: st.write escapes HTML, so a <details> block would
        # render as literal tags in the chat.
        lines.append(f"**Also {len(tidy)} cleanup suggestion(s):**")
        lines.append("")
        for f in tidy:
            lines.append(f"- `{f['path']}` line {f['line']}: {f['msg']}")

    lines.append("")
    lines.append(
        "_Checked with Python's parser and pyflakes. Neither runs your code, so "
        "logic errors (wrong result, off-by-one, division by zero) are not covered._"
    )
    return "\n".join(lines).rstrip()


def try_answer_syntax_question(
    question: str,
    project_root: str | None,
    index_meta: dict[str, Any] | None,
) -> str | None:
    """Answer syntax questions with Python's own parser - no model call.

    Returns None to fall through to the LLM when nothing is broken and the user
    did not explicitly ask about syntax (so "what is wrong with app.py" on a
    clean file still gets a real discussion of its logic).
    """
    if not is_syntax_check_question(question):
        return None
    root = (project_root or (index_meta or {}).get("root") or "").strip()
    if not root:
        return None

    named = extract_mentioned_files(question)
    checked: list[str] = []
    problems: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    for rel, source in _resolve_python_files(root, named, index_meta):
        checked.append(rel)
        found = check_python_syntax_all(source, rel)
        if found:
            problems.extend(found)
        else:
            quality.extend(check_python_quality(source, rel))

    if not checked:
        return None
    if problems:
        return format_syntax_report(problems, checked)

    # Nothing unparseable. An undefined name is still the answer to "what is
    # wrong with x.py", so report bugs; stay quiet about mere tidiness unless
    # the user explicitly asked for a check.
    if any(f["is_bug"] for f in quality) or _SYNTAX_EXPLICIT.search(question):
        return format_quality_report(quality, checked)
    return None


_OVERVIEW_PATTERNS = re.compile(
    r"\b("
    r"whole\s+project|entire\s+project|all\s+(?:the\s+)?(?:files|code|modules)|"
    r"scan\s+(?:the\s+)?(?:whole\s+|entire\s+)?(?:project|codebase|repo)|"
    r"overview|outline|architecture|structure|high[-\s]?level|"
    r"what\s+does\s+(?:this|the)\s+(?:project|codebase|app|application)\s+do|"
    r"summar(?:y|ise|ize)\s+(?:of\s+)?(?:the\s+)?(?:whole\s+|entire\s+)?(?:project|codebase)|"
    r"tout\s+le\s+projet|vue\s+d.ensemble|architecture\s+du\s+projet|structure\s+du\s+projet"
    r")\b",
    re.IGNORECASE,
)


def is_project_overview_question(question: str) -> bool:
    """True for 'scan the whole project', 'what does this project do', etc."""
    return bool(_OVERVIEW_PATTERNS.search(question))


def _outline_one_file(source: str, rel_path: str, max_members: int = 40) -> str:
    """A compact signature-level outline of one module."""
    try:
        tree = ast.parse(source, filename=rel_path)
    except (SyntaxError, ValueError) as exc:
        return f"### FILE: {rel_path}\n(cannot parse: {exc})"

    lines = [f"### FILE: {rel_path}"]
    doc = ast.get_docstring(tree)
    if doc:
        lines.append(f'"""{doc.strip().splitlines()[0]}"""')

    def signature(node) -> str:
        args = [a.arg for a in node.args.args]
        if node.args.vararg:
            args.append("*" + node.args.vararg.arg)
        if node.args.kwarg:
            args.append("**" + node.args.kwarg.arg)
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(args)})"

    count = 0
    for node in tree.body:
        if count >= max_members:
            lines.append("    ... (truncated)")
            break
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(b.id for b in node.bases if isinstance(b, ast.Name))
            lines.append(f"class {node.name}({bases})" if bases else f"class {node.name}")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    lines.append("    " + signature(sub))
                    count += 1
            count += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.append(signature(node))
            count += 1

    if len(lines) == 1:
        lines.append("(no classes or functions)")
    return "\n".join(lines)


def build_project_outline(
    project_root: str,
    index_meta: dict[str, Any] | None,
    max_files: int = 60,
) -> str:
    """Signature-level outline of EVERY Python file, not just retrieved chunks.

    Vector search returns the k chunks nearest the question, so a whole-project
    question saw only a few files. This covers all of them at a fraction of the
    size of their contents.
    """
    files = list(_resolve_python_files(project_root, [], index_meta))
    if not files:
        return ""

    blocks = [_outline_one_file(source, rel) for rel, source in files[:max_files]]
    if len(files) > max_files:
        blocks.append(f"... and {len(files) - max_files} more Python file(s) not outlined.")
    return "\n\n".join(blocks)


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
    root = meta.get("root", "(unknown)")
    mode = meta.get("index_mode", "unknown")
    indexed = meta.get("indexed_files") or []
    all_files = meta.get("all_files") or []
    not_indexed = meta.get("not_indexed_files") or []

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
        indexed_set = set(indexed)
        for path in all_files:
            tag = " *(indexed)*" if path in indexed_set else ""
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
    # Tolerate a partial meta: a missing key must not sink the whole answer.
    indexed = meta.get("indexed_files") or []
    all_files = meta.get("all_files") or []
    mode = meta.get("index_mode", "unknown")
    lines = [
        f"Project file inventory: {len(all_files)} file(s) total, {len(indexed)} indexed (mode={mode}).",
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


_EMBEDDINGS_CACHE: dict[tuple[str, str], Any] = {}


def get_embeddings(model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
    """Load the embedding model once per process; it costs seconds to construct."""
    key = (model_name, device)
    if key not in _EMBEDDINGS_CACHE:
        _EMBEDDINGS_CACHE[key] = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs={"device": device}
        )
    return _EMBEDDINGS_CACHE[key]


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

    wanted_exts: set[str] = set()
    if want_py:
        wanted_exts.add(".py")
    if want_docs:
        wanted_exts.update((".md", ".txt", ".rst", ".pdf"))

    print(f"Scanning {repo_path} for {', '.join(sorted(wanted_exts))}...")
    for file_path in _iter_project_files(root_abs, wanted_exts):
        if os.path.splitext(file_path)[1].lower() == ".pdf":
            load_pdf_file(file_path)
        else:
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
    embeddings = get_embeddings()
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

    # Python's own parser is exact and instant, so try it before any model call.
    syntax_answer = try_answer_syntax_question(question, effective_root, index_meta)
    if syntax_answer is not None:
        return syntax_answer, []

    system_prompt = _system_prompt_for_mode(index_mode)

    backend_obj = backend
    if backend_obj is None:
        resolved_base = (base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        resolved_model = model_name or os.environ.get("OLLAMA_MODEL", "llama3")
        backend_obj = OllamaBackend(
            resolved_model,
            base_url=resolved_base,
            num_ctx=num_ctx,
            temperature=temperature,
            system=system_prompt,
        )
    else:
        # Previously guarded by hasattr(backend_obj, "system"), which is False for every
        # llm_backends object -- so the mode-specific grounding rules were silently
        # dropped on the path the app actually uses.
        try:
            backend_obj.system = system_prompt
        except (AttributeError, TypeError):
            pass

    optimization_answer = try_answer_optimization_question(
        question, effective_root, index_meta, backend=backend_obj, num_predict=num_predict
    )
    if optimization_answer is not None:
        return optimization_answer, []

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

    # A whole-project question must not be answered from the k nearest chunks.
    if effective_root and is_project_overview_question(question):
        outline = build_project_outline(effective_root, index_meta)
        if outline:
            context = (
                "Signature-level outline of EVERY Python file in the project:"
                + chr(10) + chr(10) + outline
            )
            docs = []

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
