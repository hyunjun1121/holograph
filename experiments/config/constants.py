"""
HOLOGRAPH Centralized Constants

This file contains all magic numbers and thresholds used throughout the codebase.
Each constant is documented with its meaning and justification.

Usage:
    from experiments.config.constants import EDGE_THRESHOLD, INIT_SCALE
"""

# =============================================================================
# Edge Detection Thresholds
# =============================================================================

# Threshold for considering an edge to exist in the continuous graph
# Below this value, the edge is considered zero
# Justification: Standard choice for DAG learning (Zheng et al., 2018)
EDGE_THRESHOLD = 0.01

# Threshold for discretizing edges to binary adjacency matrix
# Used in evaluation metrics and final graph output
# Justification: Balances precision/recall based on empirical tuning
DISCRETIZATION_THRESHOLD = 0.3

# Threshold for bidirected edge (latent confounder indicator)
BIDIRECTED_EDGE_THRESHOLD = 0.01

# Threshold for feature extraction from graph structure
# Used in StructuralFeatureExtractor for counting edges and computing statistics
# Justification: Higher than EDGE_THRESHOLD (0.01) to focus on significant edges
# Note: This is intentionally different from EDGE_THRESHOLD for robustness
FEATURE_EXTRACTION_THRESHOLD = 0.1

# =============================================================================
# Initialization Parameters
# =============================================================================

# Scale for random parameter initialization
# Justification: Small initialization helps with gradient stability
PARAM_INIT_SCALE = 0.1

# Scale for initializing edges to newly added latent variables
LATENT_VAR_INIT_SCALE = 0.1

# Edge density for random initialization
# Justification: Sparse graphs are typical in causal discovery
RANDOM_INIT_EDGE_DENSITY = 0.3

# =============================================================================
# Numerical Stability
# =============================================================================

# Small epsilon for numerical stability in matrix inversions
MATRIX_EPSILON = 1e-6

# Minimum Fisher information diagonal for natural gradient
# Justification: Prevents division by near-zero values
FISHER_MIN_VALUE = 0.01

# Spectral radius margin for DAG constraint
# Graph is considered DAG if spectral_radius < 1 - margin
SPECTRAL_MARGIN = 0.1

# =============================================================================
# Convergence Criteria
# =============================================================================

# Loss change threshold for convergence detection
CONVERGENCE_THRESHOLD = 1e-4

# Gradient norm threshold for convergence
GRADIENT_NORM_THRESHOLD = 1e-6

# =============================================================================
# Query Generation Parameters
# =============================================================================

# Uncertainty threshold for generating queries
# Edges with uncertainty > threshold are query candidates
QUERY_UNCERTAINTY_THRESHOLD = 0.3

# Minimum edge weight to consider for queries
QUERY_MIN_EDGE_WEIGHT = 0.1

# Default LLM confidence when response doesn't include confidence
DEFAULT_LLM_CONFIDENCE = 0.5

# =============================================================================
# Instrumental Value Weights
# =============================================================================

# Weights for different query types in Expected Free Energy calculation
# Justification: Empirically tuned on validation set
DIRECTION_QUERY_WEIGHT = 0.5
MECHANISM_QUERY_WEIGHT = 0.3
CONFOUNDER_QUERY_WEIGHT = 0.7

# Edge weight contribution to bidirected features
BIDIRECTED_EDGE_WEIGHT_FACTOR = 0.5

# =============================================================================
# Sheaf/Contradiction Detection
# =============================================================================

# Threshold for detecting contradiction via descent loss
CONTRADICTION_DETECTION_THRESHOLD = 0.5

# Normalized loss delta (in std units) for obstruction detection
OBSTRUCTION_LOSS_THRESHOLD = 0.01

# Thresholds for sheaf axiom satisfaction (X1-X4 experiments)
IDENTITY_AXIOM_THRESHOLD = 1e-6
TRANSITIVITY_AXIOM_THRESHOLD = 0.01
LOCALITY_AXIOM_THRESHOLD = 0.01
GLUING_AXIOM_THRESHOLD = 0.01

# =============================================================================
# Rashomon Test (E5) Targets
# =============================================================================

# Target detection rate for contradictions
RASHOMON_DETECTION_TARGET = 0.95

# Target resolution rate for contradictions
RASHOMON_RESOLUTION_TARGET = 0.70

# Maximum acceptable false positive rate
RASHOMON_MAX_FPR = 0.05

# Detection threshold in std units
RASHOMON_DETECTION_STD_THRESHOLD = 2.0

# =============================================================================
# Dataset Generation
# =============================================================================

# Edge weight ranges for synthetic datasets
SYNTHETIC_EDGE_WEIGHT_MIN = 0.5
SYNTHETIC_EDGE_WEIGHT_MAX = 1.0

# Edge weight ranges for latent confounder datasets
LATENT_EDGE_WEIGHT_MIN = 0.3
LATENT_EDGE_WEIGHT_MAX = 1.0

# Default edge probability for ER graphs
SYNTHETIC_DEFAULT_EDGE_PROB = 0.2

# Default average degree for scale-free graphs
SYNTHETIC_DEFAULT_AVG_DEGREE = 2.0

# Alias for backwards compatibility
EDGE_WEIGHT_MIN = SYNTHETIC_EDGE_WEIGHT_MIN
EDGE_WEIGHT_MAX = SYNTHETIC_EDGE_WEIGHT_MAX

# =============================================================================
# API Configuration
# =============================================================================

# Default temperature for LLM queries (low for deterministic reasoning)
LLM_DEFAULT_TEMPERATURE = 0.1

# Maximum tokens for LLM responses
LLM_DEFAULT_MAX_TOKENS = 4096

# API timeout in seconds
LLM_DEFAULT_TIMEOUT = 120
EMBEDDING_DEFAULT_TIMEOUT = 60

# Embedding batch size
EMBEDDING_DEFAULT_BATCH_SIZE = 32

# =============================================================================
# Output/Checkpointing
# =============================================================================

# Default checkpoint interval (steps between saves)
DEFAULT_CHECKPOINT_INTERVAL = 100

# Logging interval (steps between log messages)
DEFAULT_LOG_INTERVAL = 10

# =============================================================================
# Overridable Hyperparameters
# =============================================================================
# These constants should be treated as defaults that can be overridden
# by experiment-specific YAML configs. Use get_param() to access them.

OVERRIDABLE_HYPERPARAMETERS = {
    # Discretization is task-dependent and may need tuning
    'DISCRETIZATION_THRESHOLD': DISCRETIZATION_THRESHOLD,
    # Query strategy should be adjustable per experiment
    'QUERY_UNCERTAINTY_THRESHOLD': QUERY_UNCERTAINTY_THRESHOLD,
    # Fisher regularization affects optimization stability
    # NOTE: Consider 1e-3 ~ 1e-4 for better numerical stability
    'FISHER_MIN_VALUE': FISHER_MIN_VALUE,
    # LLM-related parameters
    'DEFAULT_LLM_CONFIDENCE': DEFAULT_LLM_CONFIDENCE,
    'LLM_DEFAULT_TEMPERATURE': LLM_DEFAULT_TEMPERATURE,
    # Learning parameters that may need per-experiment tuning
    'CONVERGENCE_THRESHOLD': CONVERGENCE_THRESHOLD,
    'SPECTRAL_MARGIN': SPECTRAL_MARGIN,
}

# WARNING: CONTRADICTION_DETECTION_THRESHOLD = 0.5 may be too loose.
# Frobenius norm scales with sqrt(n_vars), so this threshold should be
# scale-dependent. Consider using normalized thresholds:
#   threshold = base_threshold * sqrt(n_overlap_vars)
# or a relative threshold based on edge magnitude statistics.

# =============================================================================
# Helper Functions
# =============================================================================

def get_param(config: dict, name: str, default=None):
    """
    Get parameter with config override.

    Priority order:
    1. config[name] if present and not None
    2. OVERRIDABLE_HYPERPARAMETERS[name] if name is overridable
    3. Explicit default if provided
    4. Constants module global if name exists

    Args:
        config: Configuration dictionary (from YAML or experiment config)
        name: Parameter name (e.g., 'DISCRETIZATION_THRESHOLD')
        default: Fallback default value

    Returns:
        Parameter value with appropriate priority

    Example:
        >>> config = {'DISCRETIZATION_THRESHOLD': 0.25}
        >>> threshold = get_param(config, 'DISCRETIZATION_THRESHOLD')
        0.25  # From config

        >>> config = {}
        >>> threshold = get_param(config, 'DISCRETIZATION_THRESHOLD')
        0.3   # From constants default
    """
    # Priority 1: Config override
    if config and name in config and config[name] is not None:
        return config[name]

    # Priority 2: Overridable hyperparameter default
    if name in OVERRIDABLE_HYPERPARAMETERS:
        return OVERRIDABLE_HYPERPARAMETERS[name]

    # Priority 3: Explicit default
    if default is not None:
        return default

    # Priority 4: Module global
    if name in globals():
        return globals()[name]

    raise KeyError(f"Unknown parameter: {name}")


def get_all_constants() -> dict:
    """Return all constants as a dictionary for logging/saving."""
    return {k: v for k, v in globals().items()
            if k.isupper() and not k.startswith('_')}


def get_scale_dependent_threshold(
    base_threshold: float,
    n_vars: int,
    method: str = 'sqrt'
) -> float:
    """
    Compute scale-dependent threshold for matrix comparisons.

    Frobenius norm scales with matrix size, so thresholds should be
    adjusted accordingly for fair comparison across different graph sizes.

    Args:
        base_threshold: Base threshold value (for n_vars=1)
        n_vars: Number of variables in the overlap
        method: Scaling method
            - 'sqrt': threshold = base * sqrt(n_vars)
            - 'linear': threshold = base * n_vars
            - 'log': threshold = base * log(n_vars + 1)

    Returns:
        Scale-adjusted threshold

    Example:
        >>> get_scale_dependent_threshold(0.1, n_vars=25, method='sqrt')
        0.5  # 0.1 * sqrt(25) = 0.5
    """
    import math

    if method == 'sqrt':
        return base_threshold * math.sqrt(n_vars)
    elif method == 'linear':
        return base_threshold * n_vars
    elif method == 'log':
        return base_threshold * math.log(n_vars + 1)
    else:
        raise ValueError(f"Unknown scaling method: {method}")
