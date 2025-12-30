"""
Experiment runner for HOLOGRAPH with checkpointing and self-documenting results.

Implements Phase 5 requirements:
- Checkpoint & Resume support
- Self-documenting results (JSON with metadata)
- Paper-ready output pipeline
- Real LLM and dataset integration
"""

import json
import os
import time
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
import numpy as np
import torch

from .causal_state import CausalState
from .sheaf_engine import SheafEngine, ContextOverlap
from .agent import HolographAgent, HolographConfig
from .metrics import compute_all_metrics, RashomonMetrics, SheafExactnessMetrics
from .dataset_loader import DatasetLoader, CausalDataset, create_dataset_loader
from .llm_interface import LLMConfig, BaseLLMInterface, create_llm_interface
from .semantic_encoder import EmbeddingConfig, SemanticEncoder, create_semantic_encoder

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    experiment_id: str
    description: str
    hypothesis: str
    claims_supported: List[str]

    # Model config
    method: str = "holograph"
    n_vars: int = 20
    learning_rate: float = 0.01
    lambda_descent: float = 1.0
    lambda_spec: float = 0.1
    lambda_reg: float = 1e-4
    max_steps: int = 1000
    use_natural_gradient: bool = True  # For ablation A1

    # Data config
    dataset: str = "synthetic"
    dataset_kwargs: Dict = field(default_factory=dict)
    seed: int = 42

    # LLM config (via SGLang unified gateway)
    llm_provider: str = "sglang"
    llm_model_role: str = "primary"  # Model role from MODEL_REGISTRY
    llm_api_key: Optional[str] = None
    use_llm: bool = True

    # Embedding config
    embedding_model: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    embedding_api_key: Optional[str] = None
    use_embeddings: bool = True

    # Query config
    use_active_queries: bool = True
    max_queries_per_step: int = 3
    query_interval: int = 50

    # Budget limits (P0: Operational Safety)
    max_total_queries: int = 100  # Hard limit on total queries
    max_total_tokens: int = 500000  # Hard limit on total tokens

    # Output config
    output_dir: str = "experiments/outputs"
    save_checkpoints: bool = True
    checkpoint_interval: int = 100


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        return result.stdout.strip()[:7] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_slurm_job_id() -> str:
    """Get SLURM job ID if running under SLURM."""
    return os.environ.get('SLURM_JOB_ID', 'local')


class ExperimentRunner:
    """
    Runner for HOLOGRAPH experiments with checkpointing.

    Features:
    - Automatic checkpoint saving/loading
    - Self-documenting JSON results
    - Training history logging
    - Real LLM and dataset integration
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / config.experiment_id / f"seed_{config.seed}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Set random seeds
        self._set_seeds(config.seed)

        # Load dataset
        self.dataset: Optional[CausalDataset] = None
        self.ground_truth: Optional[np.ndarray] = None
        self._load_dataset()

        # Initialize LLM interface
        self.llm: Optional[BaseLLMInterface] = None
        if config.use_llm:
            self._init_llm()

        # Initialize semantic encoder
        self.encoder: Optional[SemanticEncoder] = None
        if config.use_embeddings:
            self._init_encoder()

        # Initialize agent
        self._init_agent()

        # Tracking
        self.start_time = None
        self.current_step = 0

    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _setup_logging(self):
        """Setup logging to file and console."""
        log_file = self.output_dir / "training.log"

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        )

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)

    def _load_dataset(self):
        """Load dataset based on configuration."""
        loader = create_dataset_loader()

        # Build kwargs from config (n_vars, seed) merged with dataset_kwargs
        # dataset_kwargs can override these if explicitly provided
        kwargs = {
            'n_vars': self.config.n_vars,
            'seed': self.config.seed,
        }
        kwargs.update(self.config.dataset_kwargs)

        try:
            if self.config.dataset == "synthetic":
                # Load synthetic dataset based on experiment type
                self.dataset = loader.load_for_experiment(
                    self.config.experiment_id,
                    **kwargs
                )
            else:
                # Load named dataset
                self.dataset = loader.load(
                    self.config.dataset,
                    **kwargs
                )

            self.ground_truth = self.dataset.ground_truth_W

            # Log if dataset size differs from requested (e.g., SACHS is fixed at 11)
            if self.dataset.n_vars != self.config.n_vars:
                logger.info(
                    f"Dataset '{self.config.dataset}' has fixed size: "
                    f"{self.dataset.n_vars} vars (config requested {self.config.n_vars})"
                )
            self.config.n_vars = self.dataset.n_vars

            logger.info(
                f"Loaded dataset: {self.dataset.name} "
                f"({self.dataset.n_vars} variables)"
            )

        except Exception as e:
            logger.warning(f"Failed to load dataset: {e}")
            logger.info("Using random initialization")

    def _init_llm(self):
        """Initialize LLM interface."""
        try:
            # Import MODEL_REGISTRY to resolve model_role
            from .llm_interface import MODEL_REGISTRY, UnifiedLLMInterface

            # Use unified interface with model role
            if self.config.llm_provider == "sglang":
                model_role = self.config.llm_model_role
                # Create LLMConfig with model_role
                llm_config = LLMConfig(
                    provider="sglang",
                    model_role=model_role,
                    api_key=self.config.llm_api_key
                )
                self.llm = UnifiedLLMInterface(llm_config)
                model_info = MODEL_REGISTRY.get(model_role, {})
                model_id = model_info.get('model_id', model_role)
                logger.info(f"Initialized LLM: {model_id} (role: {model_role})")
            else:
                # Fallback to legacy config
                llm_config = LLMConfig(
                    provider=self.config.llm_provider,
                    model=self.config.llm_model,
                    api_key=self.config.llm_api_key
                )
                self.llm = create_llm_interface(llm_config)
                logger.info(f"Initialized LLM: {self.config.llm_provider}/{self.config.llm_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")
            self.llm = None

    def _init_encoder(self):
        """Initialize semantic encoder."""
        try:
            emb_config = EmbeddingConfig(
                model=self.config.embedding_model,
                api_key=self.config.embedding_api_key
            )
            self.encoder = create_semantic_encoder(emb_config)
            logger.info(f"Initialized encoder: {self.config.embedding_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize encoder: {e}")
            self.encoder = None

    def _init_agent(self):
        """Initialize HOLOGRAPH agent."""
        # Get variable names and domain context from dataset
        var_names = None
        domain_context = ""
        if self.dataset:
            var_names = self.dataset.var_names
            domain_context = self.dataset.domain_context

        agent_config = HolographConfig(
            n_vars=self.config.n_vars,
            learning_rate=self.config.learning_rate,
            lambda_descent=self.config.lambda_descent,
            lambda_spec=self.config.lambda_spec,
            lambda_reg=self.config.lambda_reg,
            max_steps=self.config.max_steps,
            use_natural_gradient=self.config.use_natural_gradient,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            llm_provider=self.config.llm_provider,
            llm_model_role=self.config.llm_model_role,
            llm_api_key=self.config.llm_api_key,
            embedding_model=self.config.embedding_model,
            embedding_api_key=self.config.embedding_api_key,
            use_active_queries=self.config.use_active_queries,
            max_queries_per_step=self.config.max_queries_per_step,
            query_interval=self.config.query_interval,
            # P0: Budget limits
            max_total_queries=self.config.max_total_queries,
            max_total_tokens=self.config.max_total_tokens
        )

        self.agent = HolographAgent(
            config=agent_config,
            var_names=var_names,
            domain_context=domain_context,
            llm_interface=self.llm,
            semantic_encoder=self.encoder
        )

        logger.info(f"Initialized agent with {self.config.n_vars} variables")

    def _checkpoint_path(self) -> Path:
        return self.output_dir / "checkpoint.pt"

    def _results_path(self) -> Path:
        return self.output_dir / "results.json"

    def _training_log_path(self) -> Path:
        return self.output_dir / "training.csv"

    def load_checkpoint_if_exists(self) -> bool:
        """Load checkpoint if it exists. Returns True if loaded."""
        checkpoint_path = self._checkpoint_path()
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            data = torch.load(checkpoint_path)
            self.agent.load_checkpoint(str(checkpoint_path))
            self.current_step = data.get('step', 0)
            logger.info(f"Resumed from step {self.current_step}")
            return True
        return False

    def save_checkpoint(self):
        """Save current training state."""
        checkpoint_path = self._checkpoint_path()
        torch.save({
            'step': self.current_step,
            'theta_W': self.agent.theta.W,
            'theta_L': self.agent.theta.L,
            'variable_names': self.agent.var_names,
            'history': self.agent.history,
            'config': asdict(self.config)
        }, checkpoint_path)
        logger.debug(f"Checkpoint saved at step {self.current_step}")

    def save_training_log(self):
        """Save training history to CSV."""
        log_path = self._training_log_path()
        history = self.agent.history

        n_steps = len(history['loss_total'])
        with open(log_path, 'w') as f:
            f.write("step,loss_total,loss_semantic,loss_descent,loss_acyclic,loss_spectral,spectral_radius,num_queries\n")
            for i in range(n_steps):
                f.write(
                    f"{i},{history['loss_total'][i]:.6f},"
                    f"{history['loss_semantic'][i]:.6f},"
                    f"{history['loss_descent'][i]:.6f},"
                    f"{history['loss_acyclic'][i]:.6f},"
                    f"{history['loss_spectral'][i]:.6f},"
                    f"{history['spectral_radius'][i]:.6f},"
                    f"{history['num_queries']}\n"
                )

    def save_results(self, metrics: Dict[str, Any]):
        """Save self-documenting results JSON."""
        result = self.agent.get_result()

        results_json = {
            "_metadata": {
                "experiment_id": self.config.experiment_id,
                "description": self.config.description,
                "hypothesis": self.config.hypothesis,
                "claims_supported": self.config.claims_supported,
                "config": asdict(self.config),
                "git_commit": get_git_commit(),
                "slurm_job_id": get_slurm_job_id(),
                "seed": self.config.seed,
                "timestamp": datetime.now().isoformat(),
                "wall_time_seconds": time.time() - self.start_time if self.start_time else 0,
                "dataset": self.dataset.name if self.dataset else "none"
            },
            "results": {
                **metrics,
                "num_queries": result['num_queries'],
                "training_steps": result['training_steps'],
                "final_loss": {
                    "total": result['final_loss'],
                    "spectral_radius": result['final_spectral_radius'],
                    "acyclicity": result['final_acyclicity']
                },
                "llm_usage": result.get('llm_usage', {})
            },
            "artifacts": {
                "learned_graph": str(self.output_dir / "graph.npz"),
                "training_log": str(self._training_log_path()),
                "checkpoint": str(self._checkpoint_path())
            }
        }

        with open(self._results_path(), 'w') as f:
            json.dump(results_json, f, indent=2)

        # Save graph as numpy
        np.savez(
            self.output_dir / "graph.npz",
            W_continuous=result['W_continuous'],
            M_continuous=result['M_continuous'],
            W_discrete=result['W_discrete']
        )

        # Save query log if available
        if result.get('num_queries', 0) > 0:
            query_log_path = self.output_dir / "query_log.json"
            with open(query_log_path, 'w') as f:
                json.dump(self.agent.history.get('query_log', []), f, indent=2)

        logger.info(f"Results saved to {self._results_path()}")

    def set_ground_truth(self, W_true: np.ndarray):
        """Set ground truth for evaluation."""
        self.ground_truth = W_true

    def run(
        self,
        resume_if_exists: bool = True,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run the experiment.

        Args:
            resume_if_exists: Resume from checkpoint if available
            callback: Optional callback(step, losses) after each step

        Returns:
            Dictionary of final metrics
        """
        self.start_time = time.time()

        # Resume from checkpoint if exists
        if resume_if_exists:
            self.load_checkpoint_if_exists()

        logger.info(f"Starting experiment: {self.config.experiment_id}")
        logger.info(f"Dataset: {self.dataset.name if self.dataset else 'random'}")
        logger.info(f"Device: {self.agent.device}")

        # Training loop
        for step in range(self.current_step, self.config.max_steps):
            self.current_step = step

            # Training step
            losses = self.agent.train_step(step)

            # Callback
            if callback:
                callback(step, losses)

            # Logging
            if step % 10 == 0:
                logger.info(
                    f"Step {step}: loss={losses['total']:.6f}, "
                    f"descent={losses['descent']:.6f}, "
                    f"acyclic={losses['acyclic']:.6f}, "
                    f"queries={self.agent.history['num_queries']}"
                )

            # Checkpointing
            if self.config.save_checkpoints and step % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
                self.save_training_log()

            # Convergence check
            if self.agent.is_converged():
                logger.info(f"Converged at step {step}")
                break

        # Final checkpoint
        self.save_checkpoint()
        self.save_training_log()

        # Compute metrics
        result = self.agent.get_result()
        if self.ground_truth is not None:
            metrics = compute_all_metrics(result['W_discrete'], self.ground_truth)
        else:
            metrics = {'note': 'No ground truth provided'}

        # Add final losses to metrics
        metrics['final_loss_total'] = result['final_loss']
        metrics['final_spectral_radius'] = result['final_spectral_radius']
        metrics['final_acyclicity'] = result['final_acyclicity']
        metrics['wall_time_seconds'] = time.time() - self.start_time
        metrics['num_queries'] = result['num_queries']

        # Save results
        self.save_results(metrics)

        logger.info(f"Experiment complete. Metrics: {metrics}")
        return metrics


class RashomonExperimentRunner(ExperimentRunner):
    """
    Specialized runner for Rashomon Stress Test (E5).

    Tests contradiction detection and resolution using real LLM.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.rashomon_metrics = RashomonMetrics()

    def inject_contradiction(
        self,
        var_i: int,
        var_j: int,
        strength: float = 1.0
    ) -> Dict:
        """
        Inject a contradictory context.

        Creates two contexts with opposite edge directions.
        """
        # Create context A: i -> j
        context_a = ContextOverlap(
            context_i=f"A_{var_i}_{var_j}",
            context_j="global",
            indices_i=[var_i, var_j],
            indices_j=list(range(self.agent.theta.n_vars)),
            intersection=[var_i, var_j]
        )

        # Create context B: j -> i
        context_b = ContextOverlap(
            context_i=f"B_{var_i}_{var_j}",
            context_j="global",
            indices_i=[var_j, var_i],
            indices_j=list(range(self.agent.theta.n_vars)),
            intersection=[var_j, var_i]
        )

        self.agent.context_overlaps.extend([context_a, context_b])

        # Set conflicting edge weights
        with torch.no_grad():
            self.agent.theta.W[var_i, var_j] = strength
            self.agent.theta.W[var_j, var_i] = -strength

        # Record baseline descent loss
        baseline_loss = self.agent.sheaf.compute_descent_loss(
            self.agent.theta, self.agent.context_overlaps
        ).item()

        return {
            'var_i': var_i,
            'var_j': var_j,
            'strength': strength,
            'baseline_loss': baseline_loss
        }

    def run_rashomon_test(
        self,
        n_scenarios: int = 5,
        include_negative: bool = True
    ) -> Dict[str, float]:
        """
        Run full Rashomon stress test.

        Args:
            n_scenarios: Number of contradiction scenarios
            include_negative: Include scenarios without contradictions

        Returns:
            Rashomon metrics dictionary
        """
        self.start_time = time.time()
        logger.info("Starting Rashomon Stress Test")

        n_vars = self.agent.theta.n_vars
        var_names = self.agent.var_names

        # Test with contradictions
        for i in range(n_scenarios):
            var_i = i % n_vars
            var_j = (i + 1) % n_vars

            # Inject contradiction
            injection = self.inject_contradiction(var_i, var_j)
            logger.info(f"Injected contradiction: {var_names[var_i]} <-> {var_names[var_j]}")

            # Run training steps to detect
            initial_loss = injection['baseline_loss']
            losses = []
            for step in range(50):
                step_losses = self.agent.train_step(step)
                losses.append(step_losses['descent'])

            # Measure detection (loss spike)
            max_loss = max(losses)
            loss_std = np.std(losses) if len(losses) > 1 else 1.0
            loss_delta = (max_loss - initial_loss) / (loss_std + 1e-8)

            self.rashomon_metrics.record_detection(
                contradiction_injected=True,
                descent_loss_delta=loss_delta
            )

            # Check resolution
            obstruction_detected = self.agent.detect_topological_obstruction()
            if obstruction_detected:
                new_var = self.agent.resolve_obstruction(use_llm=self.llm is not None)
                self.rashomon_metrics.record_resolution(
                    contradiction_resolved=new_var is not None,
                    latent_proposed=new_var is not None,
                    shd_improvement=0.0  # Would need ground truth
                )
                logger.info(f"Resolution proposed: {new_var}")

        # Test without contradictions (false positive check)
        if include_negative:
            for i in range(n_scenarios):
                # Reset state
                self.agent.theta = CausalState.random_init(
                    n_vars=n_vars,
                    device=self.agent.device,
                    variable_names=var_names
                ).requires_grad_(True)

                initial_loss = self.agent.sheaf.compute_descent_loss(
                    self.agent.theta, []  # Empty overlaps
                ).item()

                losses = []
                for step in range(50):
                    step_losses = self.agent.train_step(step)
                    losses.append(step_losses['descent'])

                max_loss = max(losses)
                loss_std = np.std(losses) if len(losses) > 1 else 1.0
                loss_delta = (max_loss - initial_loss) / (loss_std + 1e-8)

                self.rashomon_metrics.record_detection(
                    contradiction_injected=False,
                    descent_loss_delta=loss_delta
                )

        # Compute final metrics
        metrics = self.rashomon_metrics.compute_metrics()
        metrics['wall_time_seconds'] = time.time() - self.start_time
        metrics['num_queries'] = self.agent.history['num_queries']

        # Save results
        self.save_results(metrics)

        logger.info(f"Rashomon test complete: {metrics}")
        return metrics


class SheafExactnessRunner(ExperimentRunner):
    """
    Specialized runner for Sheaf Exactness Validation (X1-X4).

    Tests presheaf axiom satisfaction.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.sheaf_metrics = SheafExactnessMetrics()

    def run_axiom_tests(
        self,
        context_depths: List[int] = [3, 5, 10]
    ) -> Dict[str, float]:
        """
        Run sheaf exactness validation.

        Args:
            context_depths: Different nesting depths to test

        Returns:
            Sheaf exactness metrics dictionary
        """
        self.start_time = time.time()
        logger.info("Starting Sheaf Exactness Validation")

        n_vars = self.agent.theta.n_vars

        for depth in context_depths:
            logger.info(f"Testing depth {depth}")

            # X1: Identity axiom
            all_indices = list(range(n_vars))
            identity_error = self.agent.sheaf.verify_identity_axiom(
                self.agent.theta, all_indices
            )
            self.sheaf_metrics.record_identity(identity_error)
            logger.info(f"X1 Identity error: {identity_error:.2e}")

            # X2: Transitivity axiom
            for start in range(0, n_vars - depth, depth // 2):
                indices_U = list(range(n_vars))
                indices_V = list(range(start, min(start + depth, n_vars)))
                indices_Z = list(range(start, min(start + depth // 2, n_vars)))

                if len(indices_Z) >= 2:
                    transitivity_error = self.agent.sheaf.verify_transitivity_axiom(
                        self.agent.theta, indices_U, indices_V, indices_Z
                    )
                    self.sheaf_metrics.record_transitivity(transitivity_error)

            # X3: Locality on overlaps
            for i in range(0, n_vars - depth, depth // 2):
                overlap = ContextOverlap(
                    context_i=f"ctx_{i}",
                    context_j=f"ctx_{i+depth//2}",
                    indices_i=list(range(i, min(i + depth, n_vars))),
                    indices_j=list(range(i + depth // 2, min(i + depth + depth // 2, n_vars))),
                    intersection=list(range(i + depth // 2, min(i + depth, n_vars)))
                )

                if overlap.intersection:
                    locality_loss = self.agent.sheaf.compute_descent_loss(
                        self.agent.theta, [overlap]
                    ).item()
                    self.sheaf_metrics.record_locality(locality_loss)

            # X4: Gluing - train and measure final descent loss
            for step in range(100):
                self.agent.train_step(step)

            final_descent = self.agent.sheaf.compute_descent_loss(
                self.agent.theta, self.agent.context_overlaps
            ).item()
            self.sheaf_metrics.record_gluing(final_descent)

        # Compute final metrics
        metrics = self.sheaf_metrics.compute_metrics()
        passes = self.sheaf_metrics.passes_thresholds()
        metrics.update({f"passes_{k}": v for k, v in passes.items()})
        metrics['wall_time_seconds'] = time.time() - self.start_time

        # Save results
        self.save_results(metrics)

        logger.info(f"Sheaf exactness test complete: {metrics}")
        return metrics


def load_config_from_yaml(yaml_path: str) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    import yaml

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(**data)


def run_experiment_from_config(
    config_path: str,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run experiment from YAML config file.

    Args:
        config_path: Path to YAML config
        seed: Optional seed override

    Returns:
        Experiment metrics
    """
    config = load_config_from_yaml(config_path)
    if seed is not None:
        config.seed = seed

    # Choose appropriate runner based on experiment type
    exp_id = config.experiment_id
    if exp_id == "E5" or "rashomon" in exp_id.lower():
        runner = RashomonExperimentRunner(config)
        return runner.run_rashomon_test()
    elif exp_id.startswith("X") or "sheaf" in exp_id.lower():
        runner = SheafExactnessRunner(config)
        return runner.run_axiom_tests()
    else:
        runner = ExperimentRunner(config)
        return runner.run()
