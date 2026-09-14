"""Phase 1 proof of concept: ask a local Ollama model about a source file."""
import sys

from llm_backends import OllamaBackend


def test_rag(path: str = "rag_poc.py", model: str = "llama3"):
    with open(path, encoding="utf-8") as f:
        code_content = f.read()

    prompt = (
        "You are an AI coding assistant. Explain what the following Python code does:\n\n"
        f"{code_content}"
    )

    backend = OllamaBackend(model)
    healthy, message = backend.health_check()
    print(message)
    if not healthy:
        return

    print(f"Sending query to {model}...")
    print("\n--- Response ---")
    print(backend.query(prompt))


if __name__ == "__main__":
    test_rag(*sys.argv[1:])
