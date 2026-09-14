"""
LLM Backend abstraction layer supporting multiple local inference engines.
"""
from typing import Optional, Tuple
import os


class LLMBackend:
    """Base class for LLM backends."""

    def __init__(self, model_name: str, system: str | None = None, **kwargs):
        self.model_name = model_name
        self.system = system
        self.kwargs = kwargs
        self.model = None

    def _apply_system(self, prompt: str) -> str:
        """Prepend the system prompt for backends with no dedicated system field."""
        if self.system:
            return self.system + "\n\n" + prompt
        return prompt
    
    def query(self, prompt: str, max_tokens: int = 512) -> str:
        """Query the model and return response."""
        raise NotImplementedError
    
    def health_check(self) -> Tuple[bool, str]:
        """Check if backend is available. Returns (is_healthy, message)."""
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    """Ollama backend using REST API."""
    
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


class LlamaCppBackend(LLMBackend):
    """LLaMA.cpp backend (GGUF models on CPU/GPU)."""
    
    def __init__(self, model_name: str, n_ctx: int = 4096, **kwargs):
        super().__init__(model_name, **kwargs)
        self.n_gpu_layers = kwargs.get("n_gpu_layers", -1)  # -1 = all on GPU if available
        self.n_ctx = n_ctx
    
    def _init_model(self):
        try:
            from llama_cpp import Llama
            
            if not os.path.exists(self.model_name):
                raise FileNotFoundError(f"Model file not found: {self.model_name}")
            
            self.model = Llama(
                model_path=self.model_name,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=False
            )
        except ImportError:
            raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        except Exception as e:
            raise RuntimeError(f"Failed to load GGUF model: {e}")
    
    def query(self, prompt: str, max_tokens: int = 512) -> str:
        if self.model is None:
            self._init_model()
        
        try:
            response = self.model(
                self._apply_system(prompt),
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            raise RuntimeError(f"LLaMA.cpp query failed: {e}")
    
    def health_check(self) -> Tuple[bool, str]:
        try:
            if not os.path.exists(self.model_name):
                return False, f"❌ Model not found: {self.model_name}"
            if self.model is None:
                self._init_model()
            return True, "✅ LLaMA.cpp ready"
        except Exception as e:
            return False, f"❌ LLaMA.cpp error: {e}"


class GPT4AllBackend(LLMBackend):
    """GPT4All backend (easy local inference)."""
    
    def _init_model(self):
        try:
            from gpt4all import GPT4All
            self.model = GPT4All(self.model_name, allow_download=False, verbose=False)
        except ImportError:
            raise ImportError("gpt4all not installed. Install with: pip install gpt4all")
        except Exception as e:
            raise RuntimeError(f"Failed to load GPT4All model: {e}")
    
    def query(self, prompt: str, max_tokens: int = 512) -> str:
        if self.model is None:
            self._init_model()
        
        try:
            response = self.model.generate(self._apply_system(prompt), max_tokens=max_tokens, temp=0.7)
            return response.strip()
        except Exception as e:
            raise RuntimeError(f"GPT4All query failed: {e}")
    
    def health_check(self) -> Tuple[bool, str]:
        try:
            if self.model is None:
                self._init_model()
            return True, "✅ GPT4All ready"
        except Exception as e:
            return False, f"❌ GPT4All error: {e}"


class HuggingFaceBackend(LLMBackend):
    """Hugging Face Transformers backend (full control, quantization support)."""
    
    def __init__(self, model_name: str, use_4bit: bool = True, device: str = "auto", **kwargs):
        super().__init__(model_name, **kwargs)
        self.use_4bit = use_4bit
        self.device = device
    
    def _init_model(self):
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            
            if self.use_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype="float16",
                        bnb_4bit_use_double_quant=True
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=bnb_config,
                        device_map="auto"
                    )
                except ImportError:
                    print("⚠️ bitsandbytes not available, loading in full precision")
                    model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.device)
            else:
                model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.device)
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = pipeline("text-generation", model=model, tokenizer=tokenizer)
        except ImportError:
            raise ImportError("transformers not installed. Install with: pip install transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load HF model: {e}")
    
    def query(self, prompt: str, max_tokens: int = 512) -> str:
        if self.model is None:
            self._init_model()
        
        try:
            full_prompt = self._apply_system(prompt)
            # max_new_tokens, not max_length: max_length counts the prompt too, so a long
            # RAG context would leave zero budget for the answer.
            output = self.model(
                full_prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                return_full_text=False,
            )
            return output[0]["generated_text"].strip()
        except Exception as e:
            raise RuntimeError(f"HF query failed: {e}")
    
    def health_check(self) -> Tuple[bool, str]:
        try:
            if self.model is None:
                self._init_model()
            return True, "✅ Hugging Face ready"
        except Exception as e:
            return False, f"❌ HF error: {e}"


class LMStudioBackend(LLMBackend):
    """LM Studio backend (REST API on localhost:1234)."""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:1234", **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip("/")
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def query(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            payload = {
                "prompt": self._apply_system(prompt),
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.95,
            }
            if self.model_name and self.model_name != "default":
                payload["model"] = self.model_name

            response = self._get_session().post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"].strip()
        except ImportError:
            raise ImportError("requests not installed. Install with: pip install requests")
        except Exception as e:
            raise RuntimeError(f"LM Studio query failed: {e}")
    
    def health_check(self) -> Tuple[bool, str]:
        try:
            import requests
            response = requests.get(f"{self.base_url}/v1/models", timeout=2)
            return response.status_code == 200, "✅ LM Studio is running"
        except Exception as e:
            return False, f"❌ LM Studio unreachable at {self.base_url}: {e}"


def get_backend(backend_type: str, model_name: str, **kwargs) -> LLMBackend:
    # `system` is accepted by every backend via LLMBackend.__init__.
    """Factory function to get LLM backend."""
    backends = {
        "ollama": OllamaBackend,
        "llama_cpp": LlamaCppBackend,
        "gpt4all": GPT4AllBackend,
        "huggingface": HuggingFaceBackend,
        "lm_studio": LMStudioBackend,
    }
    
    if backend_type not in backends:
        raise ValueError(f"Unknown backend: {backend_type}. Choose from: {list(backends.keys())}")
    
    return backends[backend_type](model_name, **kwargs)
