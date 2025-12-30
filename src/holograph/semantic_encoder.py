"""
Semantic Encoder for HOLOGRAPH.

Converts text (paper excerpts, causal claims) into dense embeddings
using Qwen3-Embedding model via Together AI API.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np
import requests

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    provider: str = "together"  # together, openai, local
    model: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"  # Qwen embedding model
    api_key: Optional[str] = None
    api_base: str = "https://api.together.xyz/v1"
    batch_size: int = 32
    max_length: int = 8192
    cache_dir: Optional[str] = "experiments/embedding_cache"
    timeout: int = 60


class EmbeddingCache:
    """File-based cache for embeddings."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, np.ndarray] = {}

    def _hash_key(self, text: str, model: str) -> str:
        """Generate hash key for cache lookup."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """Get cached embedding if exists."""
        key = self._hash_key(text, model)

        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Check file cache
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            embedding = np.load(cache_file)
            self.memory_cache[key] = embedding
            return embedding

        return None

    def set(self, text: str, model: str, embedding: np.ndarray):
        """Cache an embedding."""
        key = self._hash_key(text, model)
        self.memory_cache[key] = embedding

        cache_file = self.cache_dir / f"{key}.npy"
        np.save(cache_file, embedding)

    def get_batch(self, texts: List[str], model: str) -> Dict[int, np.ndarray]:
        """Get cached embeddings for batch, return dict of index -> embedding."""
        cached = {}
        for i, text in enumerate(texts):
            emb = self.get(text, model)
            if emb is not None:
                cached[i] = emb
        return cached


class SemanticEncoder:
    """
    Encodes text into semantic embeddings for causal reasoning.

    Uses Qwen3-Embedding for high-quality semantic representations
    that capture causal relationships in text.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.api_key = self.config.api_key or os.environ.get("TOGETHER_API_KEY")

        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not set")

        self.cache = EmbeddingCache(self.config.cache_dir) if self.config.cache_dir else None
        self.embedding_dim: Optional[int] = None
        self.total_tokens = 0

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text or list of texts

        Returns:
            Embeddings array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Check cache for all texts
        embeddings = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        if self.cache:
            cached = self.cache.get_batch(texts, self.config.model)
            for i, text in enumerate(texts):
                if i in cached:
                    embeddings[i] = cached[i]
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        # Fetch uncached embeddings in batches
        if uncached_texts:
            new_embeddings = self._fetch_embeddings(uncached_texts)
            for i, idx in enumerate(uncached_indices):
                embeddings[idx] = new_embeddings[i]

                # Cache the result
                if self.cache:
                    self.cache.set(texts[idx], self.config.model, new_embeddings[i])

        result = np.stack(embeddings)
        if self.embedding_dim is None:
            self.embedding_dim = result.shape[1]

        return result

    def _fetch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Fetch embeddings from API in batches."""
        all_embeddings = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_embeddings = self._api_call(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _api_call(self, texts: List[str]) -> List[np.ndarray]:
        """Make API call to get embeddings."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model,
            "input": texts
        }

        try:
            response = requests.post(
                f"{self.config.api_base}/embeddings",
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            data = response.json()

            # Track usage
            if "usage" in data:
                self.total_tokens += data["usage"].get("total_tokens", 0)

            # Extract embeddings in order
            embeddings = []
            for item in sorted(data["data"], key=lambda x: x["index"]):
                embeddings.append(np.array(item["embedding"], dtype=np.float32))

            return embeddings

        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding API call failed: {e}")
            raise

    def encode_causal_claim(self, claim: Dict) -> np.ndarray:
        """
        Encode a causal claim with structured context.

        Args:
            claim: Dict with keys like 'cause', 'effect', 'direction', 'evidence'

        Returns:
            Embedding vector
        """
        # Format claim as structured text
        text = self._format_claim(claim)
        return self.encode(text)[0]

    def encode_causal_claims(self, claims: List[Dict]) -> np.ndarray:
        """Encode multiple causal claims."""
        texts = [self._format_claim(c) for c in claims]
        return self.encode(texts)

    def _format_claim(self, claim: Dict) -> str:
        """Format claim dict as text for embedding."""
        cause = claim.get("cause", "unknown")
        effect = claim.get("effect", "unknown")
        direction = claim.get("direction", "?")
        evidence = claim.get("evidence", "")

        direction_text = {
            "+": "increases",
            "-": "decreases",
            "?": "affects"
        }.get(direction, "affects")

        text = f"{cause} {direction_text} {effect}"
        if evidence:
            text += f". Evidence: {evidence}"

        return text

    def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between embedding sets.

        Args:
            embeddings1: Shape (n1, dim)
            embeddings2: Shape (n2, dim)

        Returns:
            Similarity matrix of shape (n1, n2)
        """
        # Normalize embeddings
        norm1 = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
        norm2 = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)

        # Compute dot product
        return norm1 @ norm2.T

    def find_similar_claims(
        self,
        query_claim: Dict,
        candidate_claims: List[Dict],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar claims to a query.

        Returns:
            List of (index, similarity_score, claim) tuples
        """
        query_emb = self.encode_causal_claim(query_claim)
        candidate_embs = self.encode_causal_claims(candidate_claims)

        similarities = self.compute_similarity(
            query_emb.reshape(1, -1),
            candidate_embs
        )[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (int(idx), float(similarities[idx]), candidate_claims[idx])
            for idx in top_indices
        ]

    def cluster_claims(
        self,
        claims: List[Dict],
        n_clusters: int = 5
    ) -> List[int]:
        """
        Cluster claims by semantic similarity.

        Args:
            claims: List of claim dicts
            n_clusters: Number of clusters

        Returns:
            Cluster assignments for each claim
        """
        from sklearn.cluster import KMeans

        embeddings = self.encode_causal_claims(claims)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(embeddings).tolist()

    def get_usage_stats(self) -> Dict:
        """Get token usage statistics."""
        return {
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.total_tokens * 0.01 / 1_000_000  # Qwen embedding pricing
        }


class CausalSemanticSpace:
    """
    Semantic space specialized for causal reasoning.

    Maintains embeddings for variables, edges, and claims
    to support causal graph operations.
    """

    def __init__(self, encoder: SemanticEncoder):
        self.encoder = encoder
        self.variable_embeddings: Dict[str, np.ndarray] = {}
        self.edge_embeddings: Dict[tuple, np.ndarray] = {}
        self.claim_embeddings: List[tuple] = []  # (claim, embedding)

    def add_variable(self, name: str, description: str = ""):
        """Add a variable to the semantic space."""
        text = f"{name}: {description}" if description else name
        self.variable_embeddings[name] = self.encoder.encode(text)[0]

    def add_variables(self, variables: Dict[str, str]):
        """Add multiple variables with descriptions."""
        texts = [f"{name}: {desc}" if desc else name for name, desc in variables.items()]
        embeddings = self.encoder.encode(texts)
        for i, (name, _) in enumerate(variables.items()):
            self.variable_embeddings[name] = embeddings[i]

    def add_edge(self, source: str, target: str, claim: Dict):
        """Add an edge with its supporting claim."""
        edge_key = (source, target)
        self.edge_embeddings[edge_key] = self.encoder.encode_causal_claim(claim)
        self.claim_embeddings.append((claim, self.edge_embeddings[edge_key]))

    def find_variable_matches(
        self,
        query: str,
        top_k: int = 3
    ) -> List[tuple]:
        """Find variables matching a query string."""
        if not self.variable_embeddings:
            return []

        query_emb = self.encoder.encode(query)[0]
        var_names = list(self.variable_embeddings.keys())
        var_embs = np.stack(list(self.variable_embeddings.values()))

        similarities = self.encoder.compute_similarity(
            query_emb.reshape(1, -1),
            var_embs
        )[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(var_names[i], float(similarities[i])) for i in top_indices]

    def compute_edge_consistency(
        self,
        source: str,
        target: str,
        new_claim: Dict
    ) -> float:
        """
        Check if a new claim is consistent with existing edge.

        Returns:
            Similarity score (higher = more consistent)
        """
        edge_key = (source, target)
        if edge_key not in self.edge_embeddings:
            return 0.0

        new_emb = self.encoder.encode_causal_claim(new_claim)
        existing_emb = self.edge_embeddings[edge_key]

        similarity = self.encoder.compute_similarity(
            new_emb.reshape(1, -1),
            existing_emb.reshape(1, -1)
        )[0, 0]

        return float(similarity)


def create_semantic_encoder(config: Optional[EmbeddingConfig] = None) -> SemanticEncoder:
    """Factory function to create semantic encoder."""
    return SemanticEncoder(config)
