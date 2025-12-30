"""
Main entry point for running HOLOGRAPH experiments.

Usage:
    python -m holograph.run --config experiments/configs/E1_benchmark.yaml
    python -m holograph.run --experiment E1 --seed 42 --model_role primary
"""

import argparse
import yaml
import sys
import logging
from pathlib import Path

# Add project root to path for constants import
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments.config.constants import (
    get_param,
    OVERRIDABLE_HYPERPARAMETERS,
    DISCRETIZATION_THRESHOLD,
    QUERY_UNCERTAINTY_THRESHOLD,
    FISHER_MIN_VALUE,
    CONVERGENCE_THRESHOLD,
    SPECTRAL_MARGIN,
)

from .experiment import (
    ExperimentConfig,
    ExperimentRunner,
    RashomonExperimentRunner,
    SheafExactnessRunner
)
from .llm_interface import MODEL_REGISTRY

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str, overrides: dict = None) -> ExperimentConfig:
    """
    Load experiment configuration from YAML file with proper priority handling.

    Priority order (implemented via get_param):
    1. Command-line overrides (highest)
    2. YAML config values
    3. OVERRIDABLE_HYPERPARAMETERS defaults
    4. Module global constants
    """
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    # Apply command-line overrides first (highest priority)
    if overrides:
        data.update(overrides)

    # Get valid field names from ExperimentConfig
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(ExperimentConfig)}

    # Filter data to only include valid ExperimentConfig fields
    filtered_data = {k: v for k, v in data.items() if k in valid_fields}

    return ExperimentConfig(**filtered_data)


# =============================================================================
# Experiment Definitions
# =============================================================================
EXPERIMENT_CONFIGS = {
    # Core Experiments (E1-E5)
    'E1': {
        'description': 'Single paper causal extraction',
        'hypothesis': 'HOLOGRAPH extracts accurate causal graphs from text',
        'claims_supported': ['C1', 'C2'],
        'dataset': 'sachs',
        'n_vars': 20,
        'max_steps': 1000,
    },
    'E2': {
        'description': 'Multi-paper fusion with sheaf consistency',
        'hypothesis': 'Sheaf-based fusion outperforms naive concatenation',
        'claims_supported': ['C3', 'T1', 'T2'],
        'dataset': 'synthetic',
        'n_vars': 30,
        'max_steps': 1500,
    },
    'E3': {
        'description': '5-paper corpus integration',
        'hypothesis': 'Larger corpus improves accuracy via sheaf constraints',
        'claims_supported': ['C3', 'C4', 'T2'],
        'dataset': 'sf',
        'n_vars': 50,
        'max_steps': 2000,
    },
    'E4': {
        'description': '20-paper corpus integration',
        'hypothesis': 'HOLOGRAPH scales to large corpora',
        'claims_supported': ['C3', 'C4', 'T2', 'S1'],
        'dataset': 'sf',
        'n_vars': 100,
        'max_steps': 3000,
    },
    'E5': {
        'description': 'Rashomon Stress Test - contradiction detection & resolution',
        'hypothesis': 'Detection ≥95%, Resolution ≥70%',
        'claims_supported': ['C1', 'C5', 'T3', 'T4'],
        'dataset': 'latent',
        'n_vars': 30,
        'max_steps': 500,
    },

    # Ablation Studies (A1-A6)
    'A1': {
        'description': 'Ablation: Without Natural Gradient',
        'hypothesis': 'Natural gradient improves convergence',
        'claims_supported': ['T5'],
        'use_natural_gradient': False,
    },
    'A2': {
        'description': 'Ablation: Without Sheaf Consistency',
        'hypothesis': 'Sheaf consistency is essential for multi-source',
        'claims_supported': ['T1', 'T2'],
        'lambda_descent': 0.0,
    },
    'A3': {
        'description': 'Ablation: Without Spectral Regularization',
        'hypothesis': 'Spectral regularization prevents instability',
        'claims_supported': ['T6'],
        'lambda_spec': 0.0,
    },
    'A4': {
        'description': 'Ablation: Without Active Queries',
        'hypothesis': 'EFE-based selection improves query efficiency',
        'claims_supported': ['T7'],
        'use_active_queries': False,
    },
    'A5': {
        'description': 'Ablation: Without Extended Thinking',
        'hypothesis': 'Extended thinking improves causal reasoning',
        'claims_supported': ['M1'],
        'llm_model_role': 'fast',  # Uses non-thinking model
    },
    'A6': {
        'description': 'Ablation: Without LLM (pure optimization)',
        'hypothesis': 'LLM guidance is essential for semantic grounding',
        'claims_supported': ['C1', 'C2'],
        'use_llm': False,
        'use_active_queries': False,
    },

    # Sheaf Exactness (X1-X4)
    'X1': {
        'description': 'Sheaf identity axiom: ρ_UU = I',
        'hypothesis': 'Identity projection error < 1e-6',
        'claims_supported': ['T1'],
    },
    'X2': {
        'description': 'Sheaf transitivity: ρ_ZU = ρ_ZV ∘ ρ_VU',
        'hypothesis': 'Composition error < epsilon',
        'claims_supported': ['T2'],
    },
    'X3': {
        'description': 'Sheaf locality: restrictions agree on overlaps',
        'hypothesis': 'Descent loss < threshold',
        'claims_supported': ['T3'],
    },
    'X4': {
        'description': 'Sheaf gluing: global reconstruction is exact',
        'hypothesis': 'Gluing residual < threshold',
        'claims_supported': ['T4'],
    },

    # Model Robustness (V1-V5)
    'V1': {
        'description': 'Model Robustness: Google Gemini 2.5 Pro',
        'hypothesis': 'Comparable performance to primary model',
        'claims_supported': ['R1'],
        'llm_model_role': 'validation_gemini',
    },
    'V2': {
        'description': 'Model Robustness: Alibaba Qwen3-235B',
        'hypothesis': 'Comparable performance to primary model',
        'claims_supported': ['R1'],
        'llm_model_role': 'validation_qwen',
    },
    'V3': {
        'description': 'Model Robustness: DeepSeek R1',
        'hypothesis': 'Reasoning specialist maintains or improves',
        'claims_supported': ['R1', 'R2'],
        'llm_model_role': 'validation_r1',
    },
    'V4': {
        'description': 'Model Robustness: Rashomon across models',
        'hypothesis': 'Contradiction detection works for all models',
        'claims_supported': ['R1', 'C5'],
    },
    'V5': {
        'description': 'Model Robustness: Cross-model consistency',
        'hypothesis': 'Different models produce similar structures',
        'claims_supported': ['R1', 'R3'],
    },

    # Component Interactions (I1-I6)
    'I1': {
        'description': 'Interaction: NGD + Sheaf Consistency',
        'hypothesis': 'Components synergize positively',
        'claims_supported': ['I1'],
    },
    'I2': {
        'description': 'Interaction: NGD + Spectral Reg',
        'hypothesis': 'Components synergize positively',
        'claims_supported': ['I2'],
    },
    'I3': {
        'description': 'Interaction: Sheaf + Active Queries',
        'hypothesis': 'Components synergize positively',
        'claims_supported': ['I3'],
    },
    'I4': {
        'description': 'Interaction: Full - NGD',
        'hypothesis': 'NGD contributes incrementally',
        'claims_supported': ['I4'],
    },
    'I5': {
        'description': 'Interaction: Full - Sheaf',
        'hypothesis': 'Sheaf contributes incrementally',
        'claims_supported': ['I5'],
    },
    'I6': {
        'description': 'Interaction: Full - Spectral',
        'hypothesis': 'Spectral reg contributes incrementally',
        'claims_supported': ['I6'],
    },

    # Scalability (S1-S3)
    'S1': {
        'description': 'Scalability: 100 variables',
        'hypothesis': 'HOLOGRAPH scales to 100 vars',
        'claims_supported': ['S1'],
        'n_vars': 100,
        'llm_model_role': 'fast',
    },
    'S2': {
        'description': 'Scalability: 500 variables',
        'hypothesis': 'HOLOGRAPH scales to 500 vars',
        'claims_supported': ['S2'],
        'n_vars': 500,
        'llm_model_role': 'fast',
    },
    'S3': {
        'description': 'Scalability: Real-time streaming',
        'hypothesis': 'HOLOGRAPH handles streaming input',
        'claims_supported': ['S3'],
        'llm_model_role': 'fast',
    },
}


def get_config_for_experiment(
    experiment_id: str,
    seed: int = 42,
    model_role: str = "primary",
    output_dir: str = "experiments/outputs",
    **overrides
) -> ExperimentConfig:
    """
    Create experiment configuration from experiment ID.

    Uses get_param pattern for overridable hyperparameters to ensure
    proper priority: overrides → exp_info → constants.py defaults
    """
    if experiment_id not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment_id}")

    exp_info = EXPERIMENT_CONFIGS[experiment_id]

    # Base configuration
    config_dict = {
        'experiment_id': experiment_id,
        'description': exp_info['description'],
        'hypothesis': exp_info['hypothesis'],
        'claims_supported': exp_info['claims_supported'],
        'seed': seed,
        'output_dir': output_dir,

        # Defaults
        'method': 'holograph',
        'n_vars': exp_info.get('n_vars', 30),
        'learning_rate': 0.01,
        'lambda_descent': 1.0,
        'lambda_spec': 0.1,
        'lambda_reg': 0.0001,
        'max_steps': exp_info.get('max_steps', 1000),
        'dataset': exp_info.get('dataset', 'sachs'),

        # LLM configuration
        'llm_provider': 'sglang',
        'llm_model_role': exp_info.get('llm_model_role', model_role),
        'use_llm': exp_info.get('use_llm', True),
        'use_embeddings': True,

        # Query configuration
        'use_active_queries': exp_info.get('use_active_queries', True),
        'max_queries_per_step': 3,
        'query_interval': 50,

        # Output
        'save_checkpoints': True,
        'checkpoint_interval': 100,

        # P1: Use get_param for overridable hyperparameters (constants.py fallback)
        'convergence_threshold': get_param(overrides, 'CONVERGENCE_THRESHOLD'),
        'spectral_margin': get_param(overrides, 'SPECTRAL_MARGIN'),
        'query_uncertainty_threshold': get_param(overrides, 'QUERY_UNCERTAINTY_THRESHOLD'),
        'discretization_threshold': get_param(overrides, 'DISCRETIZATION_THRESHOLD'),
        'fisher_min_value': get_param(overrides, 'FISHER_MIN_VALUE'),
    }

    # Apply experiment-specific overrides
    for key in ['use_natural_gradient', 'lambda_descent', 'lambda_spec',
                'use_llm', 'use_active_queries', 'llm_model_role']:
        if key in exp_info:
            config_dict[key] = exp_info[key]

    # Apply command-line overrides (highest priority)
    config_dict.update(overrides)

    # Override model role if specified
    if model_role != "primary":
        config_dict['llm_model_role'] = model_role

    return ExperimentConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(description="Run HOLOGRAPH experiment")

    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help='Experiment ID'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--model_role',
        type=str,
        default='primary',
        choices=list(MODEL_REGISTRY.keys()),
        help='LLM model role (primary, validation_gemini, validation_qwen, validation_r1, fast)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--resume_if_exists',
        action='store_true',
        default=True,
        help='Resume from checkpoint if exists'
    )
    parser.add_argument(
        '--n_vars',
        type=int,
        help='Number of variables (override)'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        help='Maximum training steps (override)'
    )

    args = parser.parse_args()

    # Build overrides
    overrides = {}
    if args.n_vars:
        overrides['n_vars'] = args.n_vars
    if args.max_steps:
        overrides['max_steps'] = args.max_steps

    # Load config
    if args.config:
        config = load_config(args.config, {'seed': args.seed, **overrides})
    elif args.experiment:
        config = get_config_for_experiment(
            args.experiment,
            seed=args.seed,
            model_role=args.model_role,
            output_dir=args.output_dir,
            **overrides
        )
    else:
        parser.error("Either --config or --experiment must be specified")

    # Log configuration
    logger.info(f"Experiment: {config.experiment_id}")
    logger.info(f"Model Role: {getattr(config, 'llm_model_role', 'primary')}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Output: {config.output_dir}")

    # Select appropriate runner
    if config.experiment_id == 'E5' or config.experiment_id.startswith('V4'):
        runner = RashomonExperimentRunner(config)
        metrics = runner.run_rashomon_test()
    elif config.experiment_id.startswith('X'):
        runner = SheafExactnessRunner(config)
        metrics = runner.run_axiom_tests()
    else:
        runner = ExperimentRunner(config)
        metrics = runner.run(resume_if_exists=args.resume_if_exists)

    # Print final metrics
    print("\n" + "=" * 60)
    print(f"Experiment {config.experiment_id} complete!")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
