#!/usr/bin/env python3
"""
HOLOGRAPH Experiment Config Generator

Generates all YAML config files for the full experiment suite as defined in
the Experiment Foundation Document (docs/experiment_foundation.md).

Experiments:
- E1: Main Benchmark (CausalBench - 4 datasets)
- E3: Hidden Confounders (Latent variables)
- E5: Rashomon Stress Test (Contradiction detection/resolution)
- A1-A6: Ablation Studies
- X1-X4: Sheaf Exactness Validation

Usage:
    python experiments/generate_configs.py
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List
from copy import deepcopy


# =============================================================================
# Base Configuration (verified in wet run)
# =============================================================================
BASE_CONFIG = {
    # Method
    "method": "holograph",

    # Optimization
    "learning_rate": 0.01,
    "lambda_descent": 1.0,
    "lambda_spec": 0.1,
    "lambda_reg": 0.0001,
    "max_steps": 1000,
    "use_natural_gradient": True,

    # LLM Configuration
    "llm_provider": "sglang",
    "llm_model_role": "primary",  # deepseek-ai/DeepSeek-V3.2-Exp-thinking-on
    "use_llm": True,
    "use_embeddings": False,  # Disabled for now (requires separate embedding server)
    "use_active_queries": True,

    # Budget limits (verified in wet run)
    "max_queries_per_step": 3,
    "query_interval": 50,
    "max_total_queries": 100,
    "max_total_tokens": 500000,

    # Output
    "output_dir": "experiments/outputs",
    "save_checkpoints": True,
    "checkpoint_interval": 100,
}


# =============================================================================
# E1: Main Benchmark Configurations
# =============================================================================
E1_DATASETS = [
    {
        "name": "sachs",
        "dataset": "sachs",
        "n_vars": 11,  # Fixed size for SACHS
        "description": "SACHS protein signaling network (11 proteins, 17 edges)",
    },
    {
        "name": "er_small",
        "dataset": "er",
        "n_vars": 20,
        "dataset_kwargs": {"edge_prob": 0.2},
        "description": "Erdos-Renyi random graph (n=20, p=0.2)",
    },
    {
        "name": "er_medium",
        "dataset": "er",
        "n_vars": 50,
        "dataset_kwargs": {"edge_prob": 0.15},
        "description": "Erdos-Renyi random graph (n=50, p=0.15)",
    },
    {
        "name": "sf_medium",
        "dataset": "sf",
        "n_vars": 50,
        "dataset_kwargs": {"avg_degree": 2.0},
        "description": "Scale-free graph (n=50, avg_degree=2.0)",
    },
]


def generate_e1_configs() -> List[Dict[str, Any]]:
    """Generate E1: Main Benchmark configs."""
    configs = []

    for ds in E1_DATASETS:
        config = deepcopy(BASE_CONFIG)
        config.update({
            "experiment_id": f"E1_{ds['name']}",
            "description": f"Main Benchmark: {ds['description']}",
            "hypothesis": "HOLOGRAPH achieves lower SHD than baselines",
            "claims_supported": ["C1", "C2"],

            "dataset": ds["dataset"],
            "n_vars": ds["n_vars"],
            "seed": 42,
        })

        if "dataset_kwargs" in ds:
            config["dataset_kwargs"] = ds["dataset_kwargs"]

        # Adjust steps based on problem size
        if ds["n_vars"] >= 50:
            config["max_steps"] = 1500
            config["query_interval"] = 75

        configs.append(config)

    return configs


# =============================================================================
# E3: Hidden Confounder Configurations
# =============================================================================
E3_SETTINGS = [
    {"n_observed": 20, "n_latent": 3, "edge_prob": 0.2},
    {"n_observed": 30, "n_latent": 5, "edge_prob": 0.2},
    {"n_observed": 50, "n_latent": 8, "edge_prob": 0.15},
]


def generate_e3_configs() -> List[Dict[str, Any]]:
    """Generate E3: Hidden Confounder configs."""
    configs = []

    for i, setting in enumerate(E3_SETTINGS, 1):
        config = deepcopy(BASE_CONFIG)
        config.update({
            "experiment_id": f"E3_latent_{setting['n_observed']}o_{setting['n_latent']}l",
            "description": f"Hidden Confounders: {setting['n_observed']} observed, {setting['n_latent']} latent",
            "hypothesis": "Algebraic projection correctly handles hidden confounders",
            "claims_supported": ["C3"],

            "dataset": "latent",
            "n_vars": setting["n_observed"],  # n_vars refers to observed variables
            "dataset_kwargs": {
                "n_observed": setting["n_observed"],
                "n_latent": setting["n_latent"],
                "edge_prob": setting["edge_prob"],
            },
            "seed": 42,

            # Increase steps for harder problems
            "max_steps": 1500,
            "query_interval": 50,
        })
        configs.append(config)

    return configs


# =============================================================================
# E5: Rashomon Stress Test Configuration
# =============================================================================
def generate_e5_configs() -> List[Dict[str, Any]]:
    """Generate E5: Rashomon Stress Test config."""
    config = deepcopy(BASE_CONFIG)
    config.update({
        "experiment_id": "E5_rashomon",
        "description": "Rashomon Stress Test: Contradiction detection and resolution",
        "hypothesis": "HOLOGRAPH detects contradictions >=95%, resolves via latent variables >=70%",
        "claims_supported": ["C1", "C5"],

        "dataset": "latent",
        "n_vars": 30,
        "dataset_kwargs": {
            "n_observed": 30,
            "n_latent": 5,
            "edge_prob": 0.2,
        },
        "seed": 42,

        # Rashomon specific settings
        "max_steps": 1500,
        "query_interval": 30,  # More frequent queries for contradiction detection
        "max_queries_per_step": 5,

        # Detection threshold for descent loss spike
        "detection_threshold": 2.0,
    })

    return [config]


# =============================================================================
# A1-A6: Ablation Studies
# =============================================================================
ABLATION_DEFINITIONS = {
    "A1": {
        "name": "no_natural_gradient",
        "description": "Ablation: Standard SGD vs Natural Gradient",
        "hypothesis": "Natural gradient improves convergence speed and accuracy",
        "claims_supported": ["C2"],
        "changes": {"use_natural_gradient": False},
    },
    "A2": {
        "name": "no_sheaf_consistency",
        "description": "Ablation: Without Frobenius descent loss (lambda_descent=0)",
        "hypothesis": "Sheaf consistency is essential for multi-source integration",
        "claims_supported": ["C1"],
        "changes": {"lambda_descent": 0.0},
    },
    "A3": {
        "name": "no_spectral_reg",
        "description": "Ablation: Without spectral regularization (lambda_spec=0)",
        "hypothesis": "Spectral regularization prevents numerical instability",
        "claims_supported": ["C4"],
        "changes": {"lambda_spec": 0.0},
    },
    "A4": {
        "name": "no_active_queries",
        "description": "Ablation: Random queries vs EFE-based selection",
        "hypothesis": "EFE-based selection improves query efficiency",
        "claims_supported": ["C7"],
        "changes": {"use_active_queries": False},
    },
    "A5": {
        "name": "fast_model",
        "description": "Ablation: Fast model (thinking-off) vs Primary (thinking-on)",
        "hypothesis": "Extended thinking improves complex causal reasoning",
        "claims_supported": ["C1"],
        "changes": {"llm_model_role": "fast"},
    },
    "A6": {
        "name": "no_llm",
        "description": "Ablation: Pure optimization without LLM guidance",
        "hypothesis": "LLM guidance is essential for semantic grounding",
        "claims_supported": ["C1", "C2"],
        "changes": {
            "use_llm": False,
            "use_embeddings": False,
            "use_active_queries": False,
        },
    },
}

# Ablations run on these datasets
ABLATION_DATASETS = ["sachs", "er_medium"]


def generate_ablation_configs() -> List[Dict[str, Any]]:
    """Generate A1-A6: Ablation configs."""
    configs = []

    for abl_id, abl_def in ABLATION_DEFINITIONS.items():
        for ds_key in ABLATION_DATASETS:
            # Find dataset config from E1
            ds = next(d for d in E1_DATASETS if d["name"] == ds_key)

            config = deepcopy(BASE_CONFIG)
            config.update({
                "experiment_id": f"{abl_id}_{ds_key}",
                "description": f"{abl_def['description']} on {ds_key}",
                "hypothesis": abl_def["hypothesis"],
                "claims_supported": abl_def["claims_supported"],

                "dataset": ds["dataset"],
                "n_vars": ds["n_vars"],
                "seed": 42,
            })

            if "dataset_kwargs" in ds:
                config["dataset_kwargs"] = ds["dataset_kwargs"]

            # Apply ablation changes
            config.update(abl_def["changes"])

            configs.append(config)

    return configs


# =============================================================================
# X1-X4: Sheaf Exactness Validation
# =============================================================================
SHEAF_AXIOMS = {
    "X1": {
        "name": "identity",
        "description": "Sheaf Axiom: Identity (rho_UU = id)",
        "hypothesis": "Identity axiom holds to numerical precision (<1e-6)",
        "claims_supported": ["T1"],
    },
    "X2": {
        "name": "transitivity",
        "description": "Sheaf Axiom: Transitivity (rho_ZU = rho_ZV o rho_VU)",
        "hypothesis": "Transitivity error < epsilon for all context triples",
        "claims_supported": ["T2"],
    },
    "X3": {
        "name": "locality",
        "description": "Sheaf Axiom: Locality (restrictions agree on overlaps)",
        "hypothesis": "Locality error < epsilon for all overlapping contexts",
        "claims_supported": ["T3"],
    },
    "X4": {
        "name": "gluing",
        "description": "Sheaf Axiom: Gluing (reconstruct global from local)",
        "hypothesis": "Gluing produces SHD=0 reconstruction",
        "claims_supported": ["T4"],
    },
}

# Sheaf validation on scale-free graphs with different context depths
SHEAF_SETTINGS = [
    {"n_vars": 30, "context_depth": 3},
    {"n_vars": 50, "context_depth": 5},
    {"n_vars": 100, "context_depth": 10},
]


def generate_sheaf_configs() -> List[Dict[str, Any]]:
    """Generate X1-X4: Sheaf Exactness configs."""
    configs = []

    for axiom_id, axiom_def in SHEAF_AXIOMS.items():
        for setting in SHEAF_SETTINGS:
            config = deepcopy(BASE_CONFIG)
            config.update({
                "experiment_id": f"{axiom_id}_n{setting['n_vars']}_d{setting['context_depth']}",
                "description": f"{axiom_def['description']} (n={setting['n_vars']}, depth={setting['context_depth']})",
                "hypothesis": axiom_def["hypothesis"],
                "claims_supported": axiom_def["claims_supported"],

                "dataset": "sf",  # Scale-free for structured hierarchies
                "n_vars": setting["n_vars"],
                "dataset_kwargs": {"avg_degree": 2.0},
                "seed": 42,

                # Sheaf validation specific
                "context_depth": setting["context_depth"],
                "max_steps": 500,  # Shorter runs for axiom checks
                "query_interval": 25,
            })

            configs.append(config)

    return configs


# =============================================================================
# Main Generator
# =============================================================================
def save_config(config: Dict[str, Any], output_dir: Path):
    """Save config as YAML file."""
    exp_id = config["experiment_id"]
    filepath = output_dir / f"{exp_id}.yaml"

    # Add header comment
    header = f"# {config['description']}\n# Hypothesis: {config['hypothesis']}\n\n"

    with open(filepath, 'w') as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return filepath


def main():
    # Setup output directory
    output_dir = Path(__file__).parent / "configs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track generated configs
    generated = []

    print("=" * 60)
    print("HOLOGRAPH Experiment Config Generator")
    print("=" * 60)

    # Generate E1: Main Benchmark
    print("\n[E1] Generating Main Benchmark configs...")
    for config in generate_e1_configs():
        path = save_config(config, output_dir)
        generated.append(config["experiment_id"])
        print(f"  - {config['experiment_id']}")

    # Generate E3: Hidden Confounders
    print("\n[E3] Generating Hidden Confounder configs...")
    for config in generate_e3_configs():
        path = save_config(config, output_dir)
        generated.append(config["experiment_id"])
        print(f"  - {config['experiment_id']}")

    # Generate E5: Rashomon
    print("\n[E5] Generating Rashomon Stress Test config...")
    for config in generate_e5_configs():
        path = save_config(config, output_dir)
        generated.append(config["experiment_id"])
        print(f"  - {config['experiment_id']}")

    # Generate A1-A6: Ablations
    print("\n[A1-A6] Generating Ablation configs...")
    for config in generate_ablation_configs():
        path = save_config(config, output_dir)
        generated.append(config["experiment_id"])
        print(f"  - {config['experiment_id']}")

    # Generate X1-X4: Sheaf Exactness
    print("\n[X1-X4] Generating Sheaf Exactness configs...")
    for config in generate_sheaf_configs():
        path = save_config(config, output_dir)
        generated.append(config["experiment_id"])
        print(f"  - {config['experiment_id']}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Generated {len(generated)} config files in {output_dir}")
    print("=" * 60)

    # Group by experiment type
    e1 = [x for x in generated if x.startswith("E1_")]
    e3 = [x for x in generated if x.startswith("E3_")]
    e5 = [x for x in generated if x.startswith("E5_")]
    ablations = [x for x in generated if x.startswith("A")]
    sheaf = [x for x in generated if x.startswith("X")]

    print(f"\nBreakdown:")
    print(f"  E1 (Benchmark):     {len(e1)} configs")
    print(f"  E3 (Confounders):   {len(e3)} configs")
    print(f"  E5 (Rashomon):      {len(e5)} configs")
    print(f"  A1-A6 (Ablations):  {len(ablations)} configs")
    print(f"  X1-X4 (Sheaf):      {len(sheaf)} configs")

    return generated


if __name__ == "__main__":
    main()
