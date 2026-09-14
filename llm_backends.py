"""Local LLM backend.

The project specification fixes the local inference engine as Ollama
("AI Engine (Local): Ollama"), so that is the only backend implemented here.
The LLMBackend base class is kept so an additional engine could be slotted in
without touching the RAG layer.
"""
from typing import Optional, Tuple


class LLMBackend:
    """Base class for LLM backends."""

    def __init__(self, model_name: str, system: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.system = system
        self.kwargs = kwargs
        self.model = None

    def query(self, prompt: str, max_tokens: int = 512) -> str:
        """Query the model and return response."""
        raise NotImplementedError

    def health_check(self) -> Tuple[bool, str]:
        """Check if backend is available. Returns (is_healthy, message)."""
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    """Ollama backend using the local REST API (default http://localhost:11434)."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        num_ctx: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.client = None

    def _init_client(self):
        try:
            import ollama
        except ImportError:
            raise ImportError("ollama package not installed. Install with: pip install ollama")
        # ollama exposes Client(host=...); there is no `ollama.Ollama`. Using the bare
        # module would silently ignore base_url and always hit localhost:11434.
        self.client = ollama.Client(host=self.base_url)

    def query(self, prompt: str, max_tokens: int = 512) -> str:
        if self.client is None:
            self._init_client()

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                system=self.system or None,
                stream=False,
                options={
                    "num_predict": max_tokens,
                    "num_ctx": self.num_ctx,
                    "temperature": self.temperature,
                },
            )
            return (response.get("response") or "").strip()
        except Exception as e:
            raise ConnectionError(f"Failed to query Ollama at {self.base_url}: {e}")

    def health_check(self) -> Tuple[bool, str]:
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code != 200:
                return False, f"❌ Ollama returned HTTP {response.status_code} at {self.base_url}"
            tags = response.json().get("models", [])
            names = {m.get("name", "") for m in tags}
            if self.model_name and not any(
                n == self.model_name or n.split(":")[0] == self.model_name.split(":")[0]
                for n in names
            ):
                available = ", ".join(sorted(n for n in names if n)) or "none"
                return False, (
                    f"⚠️ Ollama is running but model '{self.model_name}' is not pulled. "
                    f"Run `ollama pull {self.model_name}`. Available: {available}"
                )
            return True, f"✅ Ollama is running at {self.base_url} with '{self.model_name}'"
        except Exception as e:
            return False, f"❌ Ollama unreachable at {self.base_url}: {e}"


def get_backend(backend_type: str, model_name: str, **kwargs) -> LLMBackend:
    """Factory function to get LLM backend."""
    backends = {
        "ollama": OllamaBackend,
    }

    if backend_type not in backends:
        raise ValueError(f"Unknown backend: {backend_type}. Choose from: {list(backends.keys())}")

    return backends[backend_type](model_name, **kwargs)
