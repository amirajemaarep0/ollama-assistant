"""
LLM Backend abstraction layer supporting multiple local inference engines.
"""
from typing import Optional, Tuple
import os


class LLMBackend:
    """Base class for LLM backends."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.model = None
    
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
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url
        self.num_ctx = num_ctx
        self.client = None
    
    def _init_client(self):
        try:
            import ollama
            if hasattr(ollama, "Ollama"):
                self.client = ollama.Ollama(base_url=self.base_url)
            else:
                self.client = ollama
        except ImportError:
            raise ImportError("ollama package not installed. Install with: pip install ollama")
    
    def query(self, prompt: str, max_tokens: int = 512) -> str:
        if self.client is None:
            self._init_client()
        
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                options={
                    "num_predict": max_tokens,
                    "num_ctx": self.num_ctx,
                    "temperature": 0.7,
                },
            )
            return response.get("response", "").strip()
        except Exception as e:
            raise ConnectionError(f"Failed to query Ollama at {self.base_url}: {e}")
    
    def health_check(self) -> Tuple[bool, str]:
        if self.client is None:
            self._init_client()
        
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200, "✅ Ollama is running"
        except Exception as e:
            return False, f"❌ Ollama unreachable: {e}"


class LlamaCppBackend(LLMBackend):
    """LLaMA.cpp backend (GGUF models on CPU/GPU)."""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.n_gpu_layers = kwargs.get("n_gpu_layers", -1)  # -1 = all on GPU if available
    
    def _init_model(self):
        try:
            from llama_cpp import Llama
            
            if not os.path.exists(self.model_name):
                raise FileNotFoundError(f"Model file not found: {self.model_name}")
            
            self.model = Llama(
                model_path=self.model_name,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=2048,
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
                prompt,
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
            response = self.model.generate(prompt, max_tokens=max_tokens, temp=0.7)
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
            output = self.model(prompt, max_length=max_tokens, do_sample=True, temperature=0.7)
            return output[0]["generated_text"][len(prompt):].strip()
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
        self.base_url = base_url
    
    def query(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            import requests
            
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.95
                },
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
