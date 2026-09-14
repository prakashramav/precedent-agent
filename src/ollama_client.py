import sys
from pathlib import Path
import time
from typing import List, Dict, Any, Optional
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config


class OllamaClient:
    """
    Lightweight client for Ollama REST API using plain `requests`.
    No LangChain, LlamaIndex, or heavy frameworks.
    """

    def __init__(
        self,
        base_url: str = config.OLLAMA_BASE_URL,
        llm_model: str = config.OLLAMA_LLM_MODEL,
        embed_model: str = config.OLLAMA_EMBED_MODEL,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.timeout = timeout

    def get_embedding(self, text: str) -> List[float]:
        """
        Get 768-dim embedding for a single text using Ollama REST API (/api/embed or /api/embeddings).
        """
        res = self.get_embeddings_batch([text])
        return res[0] if res else []

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """
        Get embeddings for a list of texts in efficient batches using /api/embed.
        """
        if not texts:
            return []

        all_embeddings = []
        embed_url = f"{self.base_url}/api/embed"
        legacy_url = f"{self.base_url}/api/embeddings"

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            try:
                # Modern Ollama batch API
                resp = requests.post(
                    embed_url,
                    json={"model": self.embed_model, "input": chunk},
                    timeout=self.timeout
                )
                if resp.status_code == 200:
                    data = resp.json()
                    embs = data.get("embeddings", [])
                    all_embeddings.extend(embs)
                else:
                    # Fallback to single calls
                    for t in chunk:
                        r = requests.post(legacy_url, json={"model": self.embed_model, "prompt": t}, timeout=self.timeout)
                        all_embeddings.append(r.json().get("embedding", []))
            except Exception as e:
                print(f"Embedding error: {e}, retrying individually...", flush=True)
                for t in chunk:
                    try:
                        r = requests.post(legacy_url, json={"model": self.embed_model, "prompt": t}, timeout=self.timeout)
                        all_embeddings.append(r.json().get("embedding", []))
                    except Exception:
                        all_embeddings.append([0.0] * 768)

            print(f"Computed embeddings for {min(i + batch_size, len(texts))}/{len(texts)} texts...", flush=True)

        return all_embeddings

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 400,
        stream: bool = False,
    ) -> str:
        """
        Call Ollama LLM generation endpoint (/api/generate).
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()


# Singleton instance for convenience
ollama_client = OllamaClient()
