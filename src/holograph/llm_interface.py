"""
LLM Interface for HOLOGRAPH.

Handles API calls to LLMs for:
- Causal claim extraction
- Query answering
- Contradiction resolution (discriminator queries)

Supports multiple providers via unified API gateway.
"""

import os
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import requests

from .errors import BudgetExceededError

logger = logging.getLogger(__name__)


# =============================================================================
# Model Registry - Justification for ICML 2026
# =============================================================================
MODEL_REGISTRY = {
    # Primary Model: Extended thinking for complex causal reasoning
    "primary": {
        "model_id": "deepseek-ai/DeepSeek-V3.2-Exp-thinking-on",
        "description": "DeepSeek V3.2 with extended thinking - SOTA reasoning capability",
        "use_case": "E1-E5 core experiments, complex causal inference",
        "justification": "Extended thinking enables multi-step causal reasoning required for contradiction detection and latent variable proposal"
    },
    # Validation Models: Different architectures for robustness
    "validation_gemini": {
        "model_id": "google/gemini-2.5-pro-thinking-on",
        "description": "Google Gemini 2.5 Pro with thinking",
        "use_case": "V1-V5 model robustness validation",
        "justification": "Different model family (Google) to verify method generalization"
    },
    "validation_qwen": {
        "model_id": "togetherai/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
        "description": "Qwen3 235B with thinking capability",
        "use_case": "V1-V5 model robustness validation",
        "justification": "Different model family (Alibaba) with comparable reasoning capability"
    },
    "validation_r1": {
        "model_id": "togetherai/deepseek-ai/DeepSeek-R1-0528",
        "description": "DeepSeek R1 - reasoning specialist",
        "use_case": "V1-V5 model robustness validation",
        "justification": "Reasoning-specialized architecture for comparison"
    },
    # Fast model for bulk queries
    "fast": {
        "model_id": "togetherai/deepseek-ai/DeepSeek-V3-0324",
        "description": "DeepSeek V3 without extended thinking",
        "use_case": "High-throughput extraction, ablation baselines",
        "justification": "Faster inference for scalability experiments"
    }
}


@dataclass
class LLMConfig:
    """Configuration for LLM API."""
    provider: str = "sglang"  # sglang (unified gateway), deepseek, qwen, openai
    model: str = "deepseek-ai/DeepSeek-V3.2-Exp-thinking-on"  # Model name for API
    model_role: str = "primary"  # primary, validation_gemini, validation_qwen, validation_r1, fast
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.1  # Low for deterministic causal reasoning
    max_tokens: int = 4096  # Increased for thinking models
    timeout: int = 120  # Increased for thinking models
    max_retries: int = 3
    cache_dir: Optional[str] = "experiments/llm_cache"

    # Thinking model specific
    enable_thinking: bool = True  # For models with thinking capability

    # P0: Budget limits for operational safety
    max_total_tokens: int = 500000  # Hard limit on total tokens (input + output)
    max_total_queries: int = 100  # Hard limit on total queries

    def __post_init__(self):
        # Set model from registry if role specified
        if self.model_role in MODEL_REGISTRY:
            self.model = MODEL_REGISTRY[self.model_role]["model_id"]


# API configuration from environment variables
# Required: SGLANG_API_BASE and SGLANG_API_KEY must be set
def _get_api_config():
    """Get API configuration from environment variables."""
    api_base = os.environ.get("SGLANG_API_BASE")
    api_key = os.environ.get("SGLANG_API_KEY")

    if not api_base:
        raise ValueError(
            "SGLANG_API_BASE environment variable not set. "
            "Set it to your SGLang gateway URL (e.g., http://your-server:10000/v1)"
        )
    if not api_key:
        raise ValueError(
            "SGLANG_API_KEY environment variable not set. "
            "Set it to your SGLang API key."
        )
    return api_base, api_key


class LLMCache:
    """Simple file-based cache for LLM responses."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, prompt: str, model: str) -> str:
        """Generate hash key for cache lookup."""
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, prompt: str, model: str) -> Optional[str]:
        """Get cached response if exists."""
        key = self._hash_key(prompt, model)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get('response')
        return None

    def set(self, prompt: str, model: str, response: str):
        """Cache a response."""
        key = self._hash_key(prompt, model)
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, 'w') as f:
            json.dump({
                'prompt': prompt[:500],  # Truncate for readability
                'model': model,
                'response': response,
                'timestamp': time.time()
            }, f)


class BaseLLMInterface(ABC):
    """Abstract base class for LLM interfaces."""

    @abstractmethod
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a query to the LLM and get response."""
        pass

    @abstractmethod
    def extract_causal_claims(self, text: str, context: str) -> List[Dict]:
        """Extract causal claims from text."""
        pass


class UnifiedLLMInterface(BaseLLMInterface):
    """
    Unified interface for SGLang API gateway.

    Supports all models through a single endpoint with OpenAI-compatible API.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

        # Get API config from environment (raises if not set)
        if config.api_key and config.api_base:
            self.api_key = config.api_key
            self.api_base = config.api_base
        else:
            api_base, api_key = _get_api_config()
            self.api_key = config.api_key or api_key
            self.api_base = config.api_base or api_base

        self.cache = LLMCache(config.cache_dir) if config.cache_dir else None
        self.total_tokens = {"input": 0, "output": 0}
        self.query_count = 0
        self.query_log: List[Dict] = []

        logger.info(f"Initialized UnifiedLLMInterface with model: {config.model}")

    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send query to LLM via unified API."""
        # P0: Check budget limits before making API call
        current_total_tokens = self.total_tokens["input"] + self.total_tokens["output"]
        if current_total_tokens >= self.config.max_total_tokens:
            raise BudgetExceededError(
                f"Token limit {self.config.max_total_tokens} exceeded",
                budget_type="tokens",
                current=current_total_tokens,
                limit=self.config.max_total_tokens
            )

        if self.query_count >= self.config.max_total_queries:
            raise BudgetExceededError(
                f"Query limit {self.config.max_total_queries} exceeded",
                budget_type="queries",
                current=self.query_count,
                limit=self.config.max_total_queries
            )

        # Check cache first
        if self.cache:
            cached = self.cache.get(prompt, self.config.model)
            if cached:
                logger.debug("Cache hit")
                return cached

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout
                )
                response.raise_for_status()

                data = response.json()
                result = data["choices"][0]["message"]["content"]
                elapsed = time.time() - start_time

                # Track usage
                if "usage" in data:
                    self.total_tokens["input"] += data["usage"].get("prompt_tokens", 0)
                    self.total_tokens["output"] += data["usage"].get("completion_tokens", 0)

                self.query_count += 1
                self.query_log.append({
                    "timestamp": time.time(),
                    "model": self.config.model,
                    "prompt_preview": prompt[:100],
                    "response_preview": result[:100],
                    "elapsed_seconds": elapsed
                })

                # Cache result
                if self.cache:
                    self.cache.set(prompt, self.config.model, result)

                return result

            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

        return ""

    def extract_causal_claims(self, text: str, context: str) -> List[Dict]:
        """Extract causal claims from text using LLM."""
        system_prompt = """You are a causal reasoning expert. Extract all causal claims from the given text.
For each claim, identify:
1. cause: The cause variable (use concise variable names)
2. effect: The effect variable (use concise variable names)
3. direction: positive (+), negative (-), or unknown (?)
4. confidence: high, medium, or low
5. evidence: Brief quote supporting the claim

Return as JSON array. Example:
[{"cause": "smoking", "effect": "lung_cancer", "direction": "+", "confidence": "high", "evidence": "smoking causes lung cancer"}]

Be thorough and extract ALL causal relationships mentioned, including implicit ones."""

        prompt = f"""Context: {context}

Text to analyze:
{text}

Extract all causal claims as JSON array:"""

        response = self.query(prompt, system_prompt)

        try:
            # Parse JSON from response
            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            claims = json.loads(response.strip())
            return claims if isinstance(claims, list) else []
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse claims JSON: {response[:200]}")
            return []

    def answer_causal_query(
        self,
        query: str,
        context: str,
        current_graph: Optional[Dict] = None
    ) -> Dict:
        """Answer a causal query about relationship between variables."""
        system_prompt = """You are a causal reasoning expert. Answer causal queries based on domain knowledge.
Provide structured response with:
1. answer: yes/no/unknown
2. confidence: 0.0-1.0
3. reasoning: Brief explanation of your causal reasoning
4. edge_weight: Suggested causal strength (-1.0 to 1.0, 0 if no effect)

Think carefully about:
- Direct vs indirect causation
- Confounding factors
- Temporal precedence
- Mechanism plausibility"""

        graph_info = ""
        if current_graph:
            graph_info = f"\nCurrent graph beliefs: {json.dumps(current_graph)}"

        prompt = f"""Context: {context}
{graph_info}

Query: {query}

Respond as JSON:"""

        response = self.query(prompt, system_prompt)

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {
                "answer": "unknown",
                "confidence": 0.0,
                "reasoning": response[:200],
                "edge_weight": 0.0
            }

    def resolve_contradiction(
        self,
        claim_a: Dict,
        claim_b: Dict,
        context: str
    ) -> Dict:
        """Resolve contradiction between two claims by proposing latent variable."""
        system_prompt = """You are a causal reasoning expert resolving contradictions in causal graphs.
Given two contradictory causal claims, analyze whether there's a latent (hidden) variable that explains both.

This is crucial for causal discovery: apparent contradictions often arise because:
1. A hidden confounder affects both variables
2. The relationship is context-dependent (moderated by an unobserved variable)
3. There's measurement error in the observed variables

Respond with:
1. resolution_type: "latent_variable" | "context_dependent" | "measurement_error" | "unresolvable"
2. latent_variable: Name of proposed latent variable (if applicable) - be specific and domain-appropriate
3. explanation: How this latent variable resolves the contradiction
4. new_structure: Proposed causal structure as edges list [{"from": "A", "to": "B", "sign": "+/-"}]
5. confidence: 0.0-1.0 in this resolution"""

        prompt = f"""Context: {context}

Claim A: {json.dumps(claim_a)}
Claim B: {json.dumps(claim_b)}

These claims appear contradictory. Analyze and propose resolution:"""

        response = self.query(prompt, system_prompt)

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {
                "resolution_type": "unresolvable",
                "explanation": response[:200]
            }

    def get_usage_stats(self) -> Dict:
        """Get token usage statistics."""
        return {
            "total_input_tokens": self.total_tokens["input"],
            "total_output_tokens": self.total_tokens["output"],
            "total_queries": self.query_count,
            "model": self.config.model,
            "model_role": self.config.model_role
        }


# Legacy interfaces for backward compatibility
class DeepSeekInterface(UnifiedLLMInterface):
    """Interface for DeepSeek API (via unified gateway)."""

    def __init__(self, config: LLMConfig):
        if "deepseek" not in config.model.lower():
            config.model = MODEL_REGISTRY["primary"]["model_id"]
        super().__init__(config)


class QwenInterface(UnifiedLLMInterface):
    """Interface for Qwen API (via unified gateway)."""

    def __init__(self, config: LLMConfig):
        if "qwen" not in config.model.lower():
            config.model = MODEL_REGISTRY["validation_qwen"]["model_id"]
        super().__init__(config)


def create_llm_interface(config: LLMConfig) -> BaseLLMInterface:
    """Factory function to create appropriate LLM interface."""
    # All models go through unified interface
    return UnifiedLLMInterface(config)


def get_default_llm(model_role: str = "primary") -> BaseLLMInterface:
    """Get default LLM interface for specified role."""
    config = LLMConfig(
        provider="sglang",
        model_role=model_role,
        cache_dir="experiments/llm_cache"
    )
    return create_llm_interface(config)


def get_model_info(model_role: str) -> Dict:
    """Get information about a model role for paper documentation."""
    if model_role in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_role]
    return {"error": f"Unknown model role: {model_role}"}


def list_available_models() -> List[Dict]:
    """List all available models with their justifications."""
    return [
        {"role": role, **info}
        for role, info in MODEL_REGISTRY.items()
    ]
