"""
HOLOGRAPH: Active Causal Discovery via Continuous Sheaf Alignment and Natural Gradient Descent

A rigorous framework for extracting causal structures from LLMs using:
- Continuous relaxation of causal graphs (ADMGs)
- Sheaf-theoretic consistency via Algebraic Latent Projections
- Natural Gradient Descent for geometry-aware optimization
- Active query selection via Expected Free Energy minimization
"""

# Core components
from .causal_state import CausalState
from .sheaf_engine import SheafEngine, ContextOverlap
from .agent import HolographAgent, HolographConfig

# Metrics
from .metrics import (
    compute_shd,
    compute_f1,
    compute_sid,
    compute_all_metrics,
    RashomonMetrics,
    SheafExactnessMetrics
)

# LLM and encoding
from .llm_interface import (
    LLMConfig,
    BaseLLMInterface,
    DeepSeekInterface,
    QwenInterface,
    create_llm_interface,
    get_default_llm
)

from .semantic_encoder import (
    EmbeddingConfig,
    SemanticEncoder,
    CausalSemanticSpace,
    create_semantic_encoder
)

from .graph_encoder import (
    GraphEncoderConfig,
    GraphEncoder,
    GraphSimilarity,
    StructuralFeatureExtractor,
    create_graph_encoder
)

# Data loading
from .dataset_loader import (
    CausalDataset,
    DatasetConfig,
    DatasetLoader,
    SACHSDataset,
    DREAM4Dataset,
    SyntheticDataset,
    CLadderDataset,
    create_dataset_loader
)

# Query generation
from .query_generator import (
    Query,
    QueryConfig,
    QueryGenerator,
    ActiveQueryAgent,
    create_query_generator,
    create_active_query_agent
)

# Experiment running
from .experiment import (
    ExperimentConfig,
    ExperimentRunner,
    RashomonExperimentRunner,
    SheafExactnessRunner,
    load_config_from_yaml,
    run_experiment_from_config
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "CausalState",
    "SheafEngine",
    "ContextOverlap",
    "HolographAgent",
    "HolographConfig",
    # Metrics
    "compute_shd",
    "compute_f1",
    "compute_sid",
    "compute_all_metrics",
    "RashomonMetrics",
    "SheafExactnessMetrics",
    # LLM
    "LLMConfig",
    "BaseLLMInterface",
    "DeepSeekInterface",
    "QwenInterface",
    "create_llm_interface",
    "get_default_llm",
    # Encoding
    "EmbeddingConfig",
    "SemanticEncoder",
    "CausalSemanticSpace",
    "create_semantic_encoder",
    "GraphEncoderConfig",
    "GraphEncoder",
    "GraphSimilarity",
    "StructuralFeatureExtractor",
    "create_graph_encoder",
    # Data
    "CausalDataset",
    "DatasetConfig",
    "DatasetLoader",
    "SACHSDataset",
    "DREAM4Dataset",
    "SyntheticDataset",
    "CLadderDataset",
    "create_dataset_loader",
    # Queries
    "Query",
    "QueryConfig",
    "QueryGenerator",
    "ActiveQueryAgent",
    "create_query_generator",
    "create_active_query_agent",
    # Experiments
    "ExperimentConfig",
    "ExperimentRunner",
    "RashomonExperimentRunner",
    "SheafExactnessRunner",
    "load_config_from_yaml",
    "run_experiment_from_config",
]
