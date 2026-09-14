import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.ollama_client import ollama_client


class HistoricalResolutionIndex:
    """
    In-memory vector retrieval index using NumPy cosine similarity.
    No heavy or hosted vector DB.
    Indexes (customer issue + brand resolution) pairs from threads
    that satisfied the noisy resolution proxy.
    """

    def __init__(
        self,
        vectors_path: Path = config.INDEX_VECTORS_PATH,
        metadata_path: Path = config.INDEX_METADATA_PATH,
    ):
        self.vectors_path = vectors_path
        self.metadata_path = metadata_path
        self.vectors: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []

    def is_built(self) -> bool:
        return self.vectors_path.exists() and self.metadata_path.exists()

    def build_index(
        self,
        pairs_path: Path = config.RESOLUTION_PAIRS_PATH,
        max_samples: int = 400,
        batch_size: int = 20,
    ):
        """
        Load resolution pairs, embed each pair, normalize vectors,
        and save vectors and metadata.
        """
        if not pairs_path.exists():
            raise FileNotFoundError(f"Resolution pairs file not found: {pairs_path}")

        print(f"Loading resolution pairs from {pairs_path}...")
        raw_pairs = []
        with open(pairs_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_pairs.append(json.loads(line))
                if len(raw_pairs) >= max_samples:
                    break

        print(f"Indexing {len(raw_pairs)} resolution pairs for RAG grounding...")
        
        # Build combined texts: customer problem + historical resolution
        texts_to_embed = []
        metadata_records = []
        
        for p in raw_pairs:
            cust_text = p.get("customer_text", "").strip()
            reply_text = p.get("brand_reply", "").strip()
            # Embed customer problem description paired with resolution
            combined = f"Customer Query: {cust_text}\nBrand Resolution: {reply_text}"
            texts_to_embed.append(combined)
            
            metadata_records.append({
                "thread_id": p.get("thread_id", ""),
                "customer_text": cust_text,
                "customer_raw": p.get("customer_raw", ""),
                "brand_reply": reply_text,
                "brand_raw": p.get("brand_raw", ""),
                "combined_text": combined,
            })

        # Compute embeddings in batches via Ollama REST API
        print(f"Computing embeddings via {config.OLLAMA_EMBED_MODEL}...")
        embeddings = ollama_client.get_embeddings_batch(texts_to_embed, batch_size=batch_size)
        vecs = np.array(embeddings, dtype=np.float32)

        # Normalize vectors for fast cosine similarity dot products
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized_vecs = vecs / norms

        # Save to disk
        self.vectors_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.vectors_path, normalized_vecs)
        
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            for rec in metadata_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        self.vectors = normalized_vecs
        self.metadata = metadata_records
        print(f"Successfully saved index ({len(metadata_records)} vectors) to {self.vectors_path}")

    def load_index(self):
        """Load pre-built vectors and metadata into memory."""
        if not self.is_built():
            raise FileNotFoundError(
                f"Index files not found at {self.vectors_path} and {self.metadata_path}. "
                f"Call build_index() first."
            )
        self.vectors = np.load(self.vectors_path)
        self.metadata = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))
        print(f"Loaded index with {len(self.metadata)} precedents from {self.vectors_path}.")

    def search(
        self,
        query: str,
        top_k: int = config.TOP_K_RETRIEVAL,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Given a new customer message, return top-k similar past cases
        with their similarity scores AND what the brand actually replied.
        """
        if self.vectors is None or not self.metadata:
            self.load_index()

        q_emb = np.array(ollama_client.get_embedding(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            return []
        q_emb = q_emb / q_norm

        # Compute cosine similarity via dot product against normalized vectors
        scores = np.dot(self.vectors, q_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score < threshold:
                continue
            item = self.metadata[idx]
            results.append({
                "rank": rank + 1,
                "similarity_score": round(score, 4),
                "thread_id": item["thread_id"],
                "customer_text": item["customer_text"],
                "brand_reply": item["brand_reply"],
                "customer_raw": item.get("customer_raw", ""),
                "brand_raw": item.get("brand_raw", ""),
            })

        return results


# Global singleton instance
resolution_index = HistoricalResolutionIndex()


if __name__ == "__main__":
    idx = HistoricalResolutionIndex()
    idx.build_index(max_samples=250, batch_size=20)
    
    # Test retrieval
    test_query = "My iPhone battery is draining so fast after the new iOS update! Can I fix this?"
    print(f"\nTesting retrieval for: '{test_query}'")
    hits = idx.search(test_query, top_k=3)
    for h in hits:
        print(f"[{h['rank']}] Score: {h['similarity_score']:.4f}")
        print(f"  Customer: {h['customer_text']}")
        print(f"  Brand Reply: {h['brand_reply']}\n")
