"""
HOLOGRAPH Agent - Active Inference via Natural Gradient Descent.

Implements the main optimization loop combining:
- Natural Gradient Descent with Tikhonov regularization
- Expected Free Energy minimization for query selection
- Topological obstruction resolution
- Real LLM integration for causal queries
"""

import torch
from torch import nn
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging
import numpy as np

# Import centralized constants
import sys
from pathlib import Path
# Add project root to path for constants import
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments.config.constants import (
    SPECTRAL_MARGIN,
    CONVERGENCE_THRESHOLD,
    QUERY_UNCERTAINTY_THRESHOLD,
    FISHER_MIN_VALUE,
    DEFAULT_LLM_CONFIDENCE,
    OBSTRUCTION_LOSS_THRESHOLD,
    LATENT_VAR_INIT_SCALE,
    DISCRETIZATION_THRESHOLD,
)

from .causal_state import CausalState
from .sheaf_engine import SheafEngine, ContextOverlap
from .llm_interface import BaseLLMInterface, LLMConfig, create_llm_interface
from .semantic_encoder import SemanticEncoder, EmbeddingConfig, create_semantic_encoder
from .query_generator import (
    QueryGenerator, QueryConfig, Query,
    ActiveQueryAgent, create_query_generator
)

logger = logging.getLogger(__name__)


@dataclass
class HolographConfig:
    """Configuration for HOLOGRAPH agent."""
    n_vars: int
    learning_rate: float = 0.01
    lambda_descent: float = 1.0
    lambda_spec: float = 0.1
    lambda_acyclic: float = 1.0
    lambda_reg: float = 1e-4  # Tikhonov damping
    spectral_margin: float = SPECTRAL_MARGIN  # From constants
    max_steps: int = 1000
    convergence_threshold: float = CONVERGENCE_THRESHOLD  # From constants
    use_natural_gradient: bool = True
    device: str = 'cpu'

    # LLM configuration (via SGLang unified gateway)
    llm_provider: str = "sglang"
    llm_model_role: str = "primary"  # primary, validation_gemini, validation_qwen, validation_r1, fast
    llm_api_key: Optional[str] = None

    # Embedding configuration
    embedding_model: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    embedding_api_key: Optional[str] = None

    # Query configuration
    max_queries_per_step: int = 3
    query_uncertainty_threshold: float = QUERY_UNCERTAINTY_THRESHOLD  # From constants

    # Active learning
    use_active_queries: bool = True
    query_interval: int = 50  # Query every N steps

    # Budget limits (P0: Operational Safety)
    max_total_queries: int = 100  # Hard limit on total queries
    max_total_tokens: int = 500000  # Hard limit on total tokens (input + output)


class HolographAgent:
    """
    Active Inference agent for causal discovery.

    Optimizes a continuous causal state θ = (W, L) using:
    - Semantic energy from LLM responses
    - Frobenius descent loss for sheaf consistency
    - Spectral regularization for numerical stability
    - Acyclicity constraint (NOTEARS)
    """

    def __init__(
        self,
        config: HolographConfig,
        var_names: Optional[List[str]] = None,
        domain_context: str = "",
        llm_interface: Optional[BaseLLMInterface] = None,
        semantic_encoder: Optional[SemanticEncoder] = None
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.sheaf = SheafEngine(device=self.device)

        # Variable names
        self.var_names = var_names or [f"X_{i}" for i in range(config.n_vars)]
        self.domain_context = domain_context

        # Initialize state
        self.theta = CausalState.random_init(
            n_vars=config.n_vars,
            device=self.device,
            variable_names=self.var_names
        ).requires_grad_(True)

        # Context overlaps (populated during training)
        self.context_overlaps: List[ContextOverlap] = []

        # Initialize LLM interface
        if llm_interface is not None:
            self.llm = llm_interface
        else:
            try:
                from .llm_interface import UnifiedLLMInterface
                llm_config = LLMConfig(
                    provider=config.llm_provider,
                    model_role=config.llm_model_role,
                    api_key=config.llm_api_key
                )
                if config.llm_provider == "sglang":
                    self.llm = UnifiedLLMInterface(llm_config)
                else:
                    self.llm = create_llm_interface(llm_config)
            except ValueError as e:
                logger.warning(f"LLM interface not available: {e}")
                self.llm = None

        # Initialize semantic encoder
        if semantic_encoder is not None:
            self.encoder = semantic_encoder
        else:
            try:
                emb_config = EmbeddingConfig(
                    model=config.embedding_model,
                    api_key=config.embedding_api_key
                )
                self.encoder = create_semantic_encoder(emb_config)
            except ValueError as e:
                logger.warning(f"Semantic encoder not available: {e}")
                self.encoder = None

        # Initialize query generator
        query_config = QueryConfig(
            max_queries_per_step=config.max_queries_per_step,
            uncertainty_threshold=config.query_uncertainty_threshold
        )
        self.query_generator = create_query_generator(
            self.var_names, domain_context, query_config
        )

        # Graph embedding projection (learned during training)
        self.graph_projection: Optional[nn.Linear] = None

        # Training history
        self.history = {
            'loss_total': [],
            'loss_semantic': [],
            'loss_descent': [],
            'loss_acyclic': [],
            'loss_spectral': [],
            'spectral_radius': [],
            'num_queries': 0,
            'query_log': []
        }

    def initialize_from_llm(self, initial_queries: Optional[int] = None) -> Dict:
        """
        Initialize causal beliefs from LLM queries.

        Args:
            initial_queries: Number of initial queries to make.
                             Defaults to config.max_queries_per_step.

        Returns:
            Dictionary with initialization statistics
        """
        if self.llm is None:
            logger.warning("LLM not available, using random initialization")
            return {"status": "no_llm", "queries": 0}

        # P0: Use config value, respecting budget
        if initial_queries is None:
            initial_queries = self.config.max_queries_per_step

        # P0: Cap by remaining budget
        remaining_budget = self.config.max_total_queries - self.history['num_queries']
        initial_queries = min(initial_queries, remaining_budget)

        if initial_queries <= 0:
            logger.warning("Query budget already exhausted, skipping initialization")
            return {"status": "budget_exhausted", "queries": 0}

        logger.info(f"Initializing from LLM with {initial_queries} queries (budget: {remaining_budget})")

        # Select initial queries
        queries = self.query_generator.select_queries(self.theta, initial_queries)

        # Process each query
        for query in queries:
            # P0: Check budget before each query
            if self.history['num_queries'] >= self.config.max_total_queries:
                logger.warning(
                    f"Query budget exhausted ({self.history['num_queries']}/{self.config.max_total_queries})"
                )
                break

            try:
                response = self.llm.answer_causal_query(
                    query.text,
                    self.domain_context
                )

                query = self.query_generator.process_llm_response(query, response)

                # Update edge weights based on response
                update = self.query_generator.get_edge_update(query)
                if update:
                    i, j, weight = update
                    with torch.no_grad():
                        self.theta.W[i, j] = weight

                self.history['num_queries'] += 1
                self.history['query_log'].append({
                    'query': query.text,
                    'response': response,
                    'update': update
                })

            except Exception as e:
                logger.warning(f"Query failed: {e}")

        return {
            "status": "success",
            "queries": self.history['num_queries'],
            "edges_updated": len([q for q in queries if q.answered])
        }

    def encode_observation(self, text: str) -> Optional[torch.Tensor]:
        """
        Encode text observation to embedding.

        Args:
            text: Text to encode

        Returns:
            Embedding tensor or None if encoder unavailable
        """
        if self.encoder is None:
            return None

        embedding = self.encoder.encode(text)
        return torch.tensor(embedding, dtype=torch.float32, device=self.device)

    def compute_semantic_energy(
        self,
        observation_embedding: torch.Tensor,
        graph_embedding_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Compute semantic energy between observation and graph.

        E = ||φ(y) - Ψ(W, M)||²

        Args:
            observation_embedding: Text embedding φ(y)
            graph_embedding_fn: Optional function to embed graph structure

        Returns:
            Semantic energy (scalar tensor)
        """
        if graph_embedding_fn is None:
            # Default: use flattened W as graph embedding
            graph_embedding = self.theta.W.flatten()
        else:
            graph_embedding = graph_embedding_fn(self.theta)

        # Use learned projection if available
        if self.graph_projection is not None:
            graph_embedding = self.graph_projection(graph_embedding)

        # Match dimensions if needed
        if observation_embedding.shape != graph_embedding.shape:
            min_dim = min(len(observation_embedding), len(graph_embedding))
            observation_embedding = observation_embedding[:min_dim]
            graph_embedding = graph_embedding[:min_dim]

        return torch.norm(observation_embedding - graph_embedding) ** 2

    def compute_total_loss(
        self,
        observation_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with all components.

        L = L_sem + λ_d * L_descent + λ_a * L_acyclic + λ_s * L_spec

        Returns:
            Tuple of (total_loss, component_losses_dict)
        """
        losses = {}

        # Semantic energy
        if observation_embedding is not None:
            L_sem = self.compute_semantic_energy(observation_embedding)
        else:
            L_sem = torch.tensor(0.0, device=self.device)
        losses['semantic'] = L_sem.item()

        # Frobenius descent loss
        L_descent = self.sheaf.compute_descent_loss(self.theta, self.context_overlaps)
        losses['descent'] = L_descent.item()

        # Acyclicity constraint
        L_acyclic = self.theta.acyclicity_penalty()
        losses['acyclic'] = L_acyclic.item()

        # Spectral regularization
        L_spec = self.sheaf.compute_spectral_penalty(self.theta, self.config.spectral_margin)
        losses['spectral'] = L_spec.item()

        # Total loss
        total = (
            L_sem +
            self.config.lambda_descent * L_descent +
            self.config.lambda_acyclic * L_acyclic +
            self.config.lambda_spec * L_spec
        )
        losses['total'] = total.item()

        return total, losses

    def compute_fisher_information(self) -> torch.Tensor:
        """
        Compute (approximate) Fisher Information Matrix.

        G(θ) = Σ_X E[∇log P(y|do(X);θ) ∇log P(y|do(X);θ)^T] + λ_reg I

        Uses empirical approximation with gradient outer products.
        """
        n_params = self.theta.W.numel() + self.theta.L.numel()

        # Compute empirical Fisher using recent gradients
        if hasattr(self, '_gradient_history') and len(self._gradient_history) > 0:
            # Outer product approximation
            grads = torch.stack(self._gradient_history[-10:])  # Last 10 gradients
            fisher_diag = (grads ** 2).mean(dim=0)
            # Ensure minimum value for numerical stability
            fisher_diag = torch.clamp(fisher_diag, min=FISHER_MIN_VALUE)
        else:
            # Initialize with ones
            fisher_diag = torch.ones(n_params, device=self.device)

        # Add Tikhonov regularization
        fisher_diag = fisher_diag + self.config.lambda_reg

        return fisher_diag

    def natural_gradient_step(
        self,
        observation_embedding: Optional[torch.Tensor] = None,
        lr: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Perform one natural gradient descent step.

        θ_new = θ - η * G(θ)^{-1} * ∇L

        Args:
            observation_embedding: Current observation embedding
            lr: Learning rate override

        Returns:
            Dictionary of loss components
        """
        lr = lr or self.config.learning_rate

        # Zero gradients
        if self.theta.W.grad is not None:
            self.theta.W.grad.zero_()
        if self.theta.L.grad is not None:
            self.theta.L.grad.zero_()

        # Compute loss and backward
        total_loss, losses = self.compute_total_loss(observation_embedding)
        total_loss.backward()

        # Store gradients for Fisher estimation
        if not hasattr(self, '_gradient_history'):
            self._gradient_history = []

        if self.theta.W.grad is not None and self.theta.L.grad is not None:
            current_grad = torch.cat([
                self.theta.W.grad.flatten(),
                self.theta.L.grad.flatten()
            ])
            self._gradient_history.append(current_grad.detach().clone())
            if len(self._gradient_history) > 20:
                self._gradient_history.pop(0)

        with torch.no_grad():
            if self.config.use_natural_gradient:
                # Natural gradient with diagonal Fisher approximation
                fisher_diag = self.compute_fisher_information()

                # Split for W and L
                n_W = self.theta.W.numel()
                fisher_W = fisher_diag[:n_W].reshape_as(self.theta.W)
                fisher_L = fisher_diag[n_W:].reshape_as(self.theta.L)

                # Natural gradient: g_nat = F^{-1} @ g
                if self.theta.W.grad is not None:
                    self.theta.W -= lr * self.theta.W.grad / fisher_W
                if self.theta.L.grad is not None:
                    self.theta.L -= lr * self.theta.L.grad / fisher_L
            else:
                # Standard gradient descent
                if self.theta.W.grad is not None:
                    self.theta.W -= lr * self.theta.W.grad
                if self.theta.L.grad is not None:
                    self.theta.L -= lr * self.theta.L.grad

            # Ensure L stays lower-triangular
            self.theta.L.tril_()

        # Record history
        self.history['loss_total'].append(losses['total'])
        self.history['loss_semantic'].append(losses['semantic'])
        self.history['loss_descent'].append(losses['descent'])
        self.history['loss_acyclic'].append(losses['acyclic'])
        self.history['loss_spectral'].append(losses['spectral'])
        self.history['spectral_radius'].append(self.theta.spectral_radius())

        return losses

    def active_query_step(self, step: int) -> List[Query]:
        """
        Perform active learning query step.

        Args:
            step: Current training step

        Returns:
            List of answered queries
        """
        if not self.config.use_active_queries:
            return []

        if self.llm is None:
            return []

        # Only query at intervals
        if step % self.config.query_interval != 0:
            return []

        # P0: Check budget before making queries
        remaining_budget = self.config.max_total_queries - self.history['num_queries']
        if remaining_budget <= 0:
            logger.info(
                f"Query budget exhausted ({self.history['num_queries']}/{self.config.max_total_queries})"
            )
            return []

        # P0: Cap queries to remaining budget
        n_queries = min(self.config.max_queries_per_step, remaining_budget)

        # Select and execute queries
        queries = self.query_generator.select_queries(
            self.theta,
            n_queries
        )

        answered = []
        for query in queries:
            # P0: Re-check budget before each query
            if self.history['num_queries'] >= self.config.max_total_queries:
                logger.info(
                    f"Query budget exhausted ({self.history['num_queries']}/{self.config.max_total_queries})"
                )
                break

            try:
                response = self.llm.answer_causal_query(
                    query.text,
                    self.domain_context
                )

                query = self.query_generator.process_llm_response(query, response)
                answered.append(query)

                # Apply update
                update = self.query_generator.get_edge_update(query)
                if update:
                    i, j, weight = update
                    with torch.no_grad():
                        # Soft update: blend with current belief
                        alpha = response.get("confidence", DEFAULT_LLM_CONFIDENCE)
                        self.theta.W[i, j] = (1 - alpha) * self.theta.W[i, j] + alpha * weight

                self.history['num_queries'] += 1
                self.history['query_log'].append({
                    'step': step,
                    'query': query.text,
                    'response': response,
                    'update': update
                })

            except Exception as e:
                logger.warning(f"Active query failed: {e}")

        return answered

    def compute_expected_free_energy(
        self,
        candidate_query: str,
        uncertainty_estimator: Optional[Callable] = None
    ) -> float:
        """
        Compute Expected Free Energy for a query.

        G(a) = Ambiguity + Risk

        Args:
            candidate_query: Query to evaluate
            uncertainty_estimator: Function to estimate posterior uncertainty

        Returns:
            EFE value (lower is better)
        """
        if uncertainty_estimator is not None:
            epistemic_value = uncertainty_estimator(candidate_query, self.theta)
        else:
            # Heuristic: variance of edge weights as uncertainty proxy
            epistemic_value = torch.var(self.theta.W).item()

        # Pragmatic value: how much would this query help descent loss?
        pragmatic_value = self.sheaf.compute_descent_loss(
            self.theta, self.context_overlaps
        ).item()

        # Minimize EFE = - epistemic - pragmatic
        return -epistemic_value - pragmatic_value

    def detect_topological_obstruction(
        self,
        loss_window: int = 10,
        loss_threshold: float = OBSTRUCTION_LOSS_THRESHOLD
    ) -> bool:
        """
        Detect if optimization is stuck due to topological obstruction.

        An obstruction is detected when descent loss plateaus above threshold.
        """
        if len(self.history['loss_descent']) < loss_window:
            return False

        recent = self.history['loss_descent'][-loss_window:]
        delta = abs(recent[-1] - recent[0])

        # Check if stuck at high value
        is_stuck = delta < loss_threshold
        is_high = recent[-1] > self.config.convergence_threshold * 10

        return is_stuck and is_high

    def resolve_obstruction(
        self,
        use_llm: bool = True
    ) -> Optional[str]:
        """
        Resolve topological obstruction by proposing latent variable.

        This implements the "Manifold Surgery" from Control Flow spec.

        Args:
            use_llm: Whether to use LLM for discriminator query

        Returns:
            Name of proposed latent variable, or None
        """
        # Find worst residual pair
        (ctx_i, ctx_j), residual = self.sheaf.find_max_residual_pair(
            self.theta, self.context_overlaps
        )

        if ctx_i is None:
            return None

        logger.info(f"Obstruction detected between {ctx_i} and {ctx_j} (residual={residual:.4f})")

        # Query LLM for resolution if available
        new_var_name = None
        if use_llm and self.llm is not None:
            try:
                # Get the contradicting claims from contexts
                claim_a = {"context": ctx_i, "variables": list(ctx_i)}
                claim_b = {"context": ctx_j, "variables": list(ctx_j)}

                resolution = self.llm.resolve_contradiction(
                    claim_a, claim_b, self.domain_context
                )

                if resolution.get("resolution_type") == "latent_variable":
                    new_var_name = resolution.get("latent_variable", f"Z_{self.theta.n_vars}")
                    logger.info(f"LLM proposed latent variable: {new_var_name}")
                    logger.info(f"Explanation: {resolution.get('explanation', '')}")
            except Exception as e:
                logger.warning(f"LLM resolution failed: {e}")

        # Default name if LLM didn't provide one
        if new_var_name is None:
            new_var_name = f"Z_{self.theta.n_vars}"

        # Expand state space
        self._expand_state_space(new_var_name)

        return new_var_name

    def _expand_state_space(self, new_var_name: str):
        """Add a new latent variable to the state space."""
        n = self.theta.n_vars
        device = self.device

        # Expand W
        new_W = torch.zeros(n + 1, n + 1, device=device)
        new_W[:n, :n] = self.theta.W.detach()
        # Initialize new edges with small random values
        new_W[n, :n] = torch.randn(n, device=device) * LATENT_VAR_INIT_SCALE
        new_W[:n, n] = torch.randn(n, device=device) * LATENT_VAR_INIT_SCALE

        # Expand L
        new_L = torch.zeros(n + 1, n + 1, device=device)
        new_L[:n, :n] = self.theta.L.detach()
        new_L[n, n] = 1.0  # Identity covariance for new variable

        # Update variable names
        var_names = list(self.var_names)
        var_names.append(new_var_name)
        self.var_names = var_names

        self.theta = CausalState(
            W=new_W,
            L=new_L,
            variable_names=var_names
        ).requires_grad_(True)

        # Update query generator with new variables
        self.query_generator = create_query_generator(
            var_names, self.domain_context,
            QueryConfig(
                max_queries_per_step=self.config.max_queries_per_step,
                uncertainty_threshold=self.config.query_uncertainty_threshold
            )
        )

        logger.info(f"Expanded state space to {self.theta.n_vars} variables")

    def train_step(
        self,
        step: int,
        observation_embedding: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform one training step of the active causal loop.

        Args:
            step: Current step number
            observation_embedding: Current observation embedding

        Returns:
            Dictionary of loss components
        """
        # Natural gradient update
        losses = self.natural_gradient_step(observation_embedding)

        # Active querying
        queries = self.active_query_step(step)
        losses['queries_this_step'] = len(queries)

        # Check for obstruction
        if self.detect_topological_obstruction():
            new_var = self.resolve_obstruction()
            if new_var:
                losses['latent_added'] = new_var

        return losses

    def train(
        self,
        observations: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
        callback: Optional[Callable] = None
    ) -> Dict:
        """
        Full training loop.

        Args:
            observations: List of text observations to encode
            max_steps: Maximum training steps
            callback: Optional callback(step, losses) after each step

        Returns:
            Training result dictionary
        """
        max_steps = max_steps or self.config.max_steps

        # Encode observations if provided
        observation_embeddings = None
        if observations and self.encoder is not None:
            observation_embeddings = [
                self.encode_observation(obs) for obs in observations
            ]

        # Initialize from LLM if available
        if self.llm is not None:
            init_result = self.initialize_from_llm()
            logger.info(f"Initialization: {init_result}")

        # Training loop
        for step in range(max_steps):
            # P0: Check query budget before proceeding
            if self.history['num_queries'] >= self.config.max_total_queries:
                logger.warning(
                    f"Query budget exhausted ({self.history['num_queries']}/{self.config.max_total_queries}). "
                    "Stopping active inference."
                )
                break

            # P0: Check token budget if LLM is available
            if self.llm is not None and hasattr(self.llm, 'total_tokens'):
                current_tokens = (
                    self.llm.total_tokens.get("input", 0) +
                    self.llm.total_tokens.get("output", 0)
                )
                if current_tokens >= self.config.max_total_tokens:
                    logger.warning(
                        f"Token budget exhausted ({current_tokens}/{self.config.max_total_tokens}). "
                        "Stopping active inference."
                    )
                    break

            # Get current observation embedding
            obs_emb = None
            if observation_embeddings:
                idx = step % len(observation_embeddings)
                obs_emb = observation_embeddings[idx]

            # Training step
            losses = self.train_step(step, obs_emb)

            # Callback
            if callback:
                callback(step, losses)

            # Check convergence
            if self.is_converged():
                logger.info(f"Converged at step {step}")
                break

            # Logging
            if step % 100 == 0:
                logger.info(
                    f"Step {step}: loss={losses['total']:.6f}, "
                    f"acyclic={losses['acyclic']:.6f}, "
                    f"queries={self.history['num_queries']}"
                )

        return self.get_result()

    def is_converged(self) -> bool:
        """Check if optimization has converged."""
        if len(self.history['loss_total']) < 2:
            return False

        recent_loss = self.history['loss_total'][-1]
        return recent_loss < self.config.convergence_threshold

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'theta_W': self.theta.W,
            'theta_L': self.theta.L,
            'variable_names': self.var_names,
            'history': self.history,
            'config': self.config.__dict__
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        data = torch.load(path, map_location=self.device)

        self.var_names = data.get('variable_names', self.var_names)
        self.theta = CausalState(
            W=data['theta_W'],
            L=data['theta_L'],
            variable_names=self.var_names
        ).requires_grad_(True)

        self.history = data['history']
        logger.info(f"Checkpoint loaded from {path}")

    def get_result(self, threshold: float = DISCRETIZATION_THRESHOLD) -> Dict:
        """
        Get final result after training.

        Args:
            threshold: Edge threshold for discretization

        Returns:
            Dictionary with learned graph and metadata
        """
        # Get LLM usage stats if available
        llm_stats = {}
        if self.llm is not None and hasattr(self.llm, 'get_usage_stats'):
            llm_stats = self.llm.get_usage_stats()

        return {
            'W_continuous': self.theta.W.detach().cpu().numpy(),
            'M_continuous': self.theta.M.detach().cpu().numpy(),
            'W_discrete': self.theta.discretize(threshold),
            'variable_names': self.var_names,
            'final_loss': self.history['loss_total'][-1] if self.history['loss_total'] else None,
            'final_spectral_radius': self.theta.spectral_radius(),
            'final_acyclicity': self.theta.acyclicity_penalty().item(),
            'num_queries': self.history['num_queries'],
            'training_steps': len(self.history['loss_total']),
            'llm_usage': llm_stats
        }
