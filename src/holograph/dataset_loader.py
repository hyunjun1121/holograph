"""
Dataset Loader for HOLOGRAPH.

Loads standard causal discovery benchmarks:
- SACHS (protein signaling)
- DREAM4 (gene regulatory networks)
- Synthetic graphs (ER, SF, custom)
- CLadder (causal reasoning benchmark)
"""

import os
import json
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch

# Import centralized constants
import sys
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments.config.constants import (
    EDGE_WEIGHT_MIN,
    EDGE_WEIGHT_MAX,
    SYNTHETIC_DEFAULT_EDGE_PROB,
    SYNTHETIC_DEFAULT_AVG_DEGREE,
    SYNTHETIC_EDGE_WEIGHT_MIN,
    SYNTHETIC_EDGE_WEIGHT_MAX,
    LATENT_EDGE_WEIGHT_MIN,
    LATENT_EDGE_WEIGHT_MAX,
    MATRIX_EPSILON,
)

logger = logging.getLogger(__name__)


@dataclass
class CausalDataset:
    """Container for causal discovery dataset."""
    name: str
    n_vars: int
    var_names: List[str]
    ground_truth_W: np.ndarray  # True DAG adjacency matrix
    ground_truth_M: Optional[np.ndarray] = None  # True bidirected edges (if known)
    domain_context: str = ""  # Text description of domain
    causal_descriptions: Optional[Dict[Tuple[int, int], str]] = None  # Edge descriptions


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    download: bool = True


class SACHSDataset:
    """
    SACHS protein signaling dataset.

    11 proteins, 17 known edges from biological literature.
    Reference: Sachs et al. (2005), Science
    """

    VARIABLE_NAMES = [
        "praf", "pmek", "plcg", "PIP2", "PIP3",
        "p44/42", "pakts473", "PKA", "PKC", "P38", "pjnk"
    ]

    # Ground truth from literature
    TRUE_EDGES = [
        ("PKC", "praf"),
        ("PKC", "pmek"),
        ("PKC", "p44/42"),
        ("PKC", "pjnk"),
        ("PKC", "P38"),
        ("PKC", "PKA"),
        ("PKA", "praf"),
        ("PKA", "pmek"),
        ("PKA", "p44/42"),
        ("PKA", "pakts473"),
        ("PKA", "pjnk"),
        ("PKA", "P38"),
        ("praf", "pmek"),
        ("pmek", "p44/42"),
        ("plcg", "PIP2"),
        ("plcg", "PIP3"),
        ("PIP3", "PIP2"),
    ]

    DOMAIN_CONTEXT = """
    Protein signaling network from immune system T-cells.
    Variables represent phosphorylated protein levels.
    Causal relationships represent biochemical activation/inhibition pathways.
    Key hubs: PKC (protein kinase C) and PKA (protein kinase A) are central signaling proteins.
    """

    @classmethod
    def load(cls) -> CausalDataset:
        """Load SACHS dataset."""
        n_vars = len(cls.VARIABLE_NAMES)
        var_to_idx = {name: i for i, name in enumerate(cls.VARIABLE_NAMES)}

        # Build ground truth adjacency matrix
        W = np.zeros((n_vars, n_vars))
        for source, target in cls.TRUE_EDGES:
            i, j = var_to_idx[source], var_to_idx[target]
            W[i, j] = 1.0

        # Edge descriptions
        edge_descriptions = {
            ("PKC", "praf"): "PKC activates Raf-1 kinase",
            ("PKC", "pmek"): "PKC can directly activate MEK",
            ("PKA", "pakts473"): "PKA phosphorylates Akt at S473",
            ("praf", "pmek"): "Raf phosphorylates and activates MEK (MAPK cascade)",
            ("pmek", "p44/42"): "MEK phosphorylates ERK1/2 (p44/42)",
            ("plcg", "PIP2"): "PLCγ cleaves PIP2",
            ("plcg", "PIP3"): "PLCγ produces PIP3 via PI3K pathway",
            ("PIP3", "PIP2"): "PIP3 is dephosphorylated to PIP2",
        }

        causal_desc = {}
        for (src, tgt), desc in edge_descriptions.items():
            i, j = var_to_idx[src], var_to_idx[tgt]
            causal_desc[(i, j)] = desc

        return CausalDataset(
            name="SACHS",
            n_vars=n_vars,
            var_names=cls.VARIABLE_NAMES,
            ground_truth_W=W,
            domain_context=cls.DOMAIN_CONTEXT,
            causal_descriptions=causal_desc
        )


class DREAM4Dataset:
    """
    DREAM4 In Silico Network Challenge dataset.

    Gene regulatory networks from E.coli and S.cerevisiae.
    Networks of 10 or 100 genes with known ground truth.
    """

    DOMAIN_CONTEXT = """
    Gene regulatory network from in silico simulations.
    Variables represent gene expression levels.
    Causal relationships represent transcriptional regulation.
    Positive edges: activation/upregulation.
    Negative edges: repression/downregulation.
    """

    @classmethod
    def load(cls, network_id: int = 1, size: int = 10) -> CausalDataset:
        """
        Load DREAM4 network.

        Args:
            network_id: Network number (1-5)
            size: Network size (10 or 100)
        """
        # Generate synthetic DREAM4-like network if file not available
        n_vars = size
        var_names = [f"Gene_{i+1}" for i in range(n_vars)]

        # Create scale-free like structure
        np.random.seed(42 + network_id)
        W = cls._generate_scale_free_dag(n_vars, avg_degree=2.0)

        return CausalDataset(
            name=f"DREAM4_net{network_id}_size{size}",
            n_vars=n_vars,
            var_names=var_names,
            ground_truth_W=W,
            domain_context=cls.DOMAIN_CONTEXT
        )

    @staticmethod
    def _generate_scale_free_dag(
        n: int,
        avg_degree: float = SYNTHETIC_DEFAULT_AVG_DEGREE,
        apply_permutation: bool = True
    ) -> np.ndarray:
        """
        Generate scale-free DAG using preferential attachment.

        Args:
            n: Number of nodes
            avg_degree: Average degree (edges per node)
            apply_permutation: If True, apply random permutation to remove
                               ordering bias (hub nodes won't always have
                               smaller indices)

        Returns:
            Weighted adjacency matrix W
        """
        W = np.zeros((n, n))
        degrees = np.ones(n)  # Initialize with degree 1

        for i in range(1, n):
            # Number of edges to add
            n_edges = max(1, int(avg_degree / 2))

            # Preferential attachment: probability proportional to degree
            probs = degrees[:i] / degrees[:i].sum()
            targets = np.random.choice(
                i, size=min(n_edges, i), replace=False, p=probs
            )

            for j in targets:
                W[j, i] = np.random.choice([-1, 1]) * np.random.uniform(
                    SYNTHETIC_EDGE_WEIGHT_MIN, SYNTHETIC_EDGE_WEIGHT_MAX
                )
                degrees[j] += 1
                degrees[i] += 1

        # Apply random permutation to remove topological ordering bias
        # Without this, smaller indices are always parents (hubs)
        if apply_permutation:
            perm = np.random.permutation(n)
            W = W[perm][:, perm]

        return W


class SyntheticDataset:
    """Generate synthetic causal graphs for testing."""

    @staticmethod
    def erdos_renyi(
        n_vars: int,
        edge_prob: float = SYNTHETIC_DEFAULT_EDGE_PROB,
        seed: int = 42
    ) -> CausalDataset:
        """
        Generate Erdos-Renyi random DAG.

        Args:
            n_vars: Number of variables
            edge_prob: Edge probability
            seed: Random seed
        """
        np.random.seed(seed)

        # Generate random DAG (upper triangular for acyclicity)
        W = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if np.random.random() < edge_prob:
                    W[i, j] = np.random.choice([-1, 1]) * np.random.uniform(
                        LATENT_EDGE_WEIGHT_MIN, LATENT_EDGE_WEIGHT_MAX
                    )

        # Random permutation to remove ordering bias
        perm = np.random.permutation(n_vars)
        W = W[perm][:, perm]

        var_names = [f"X{i}" for i in range(n_vars)]

        return CausalDataset(
            name=f"ER_n{n_vars}_p{edge_prob}",
            n_vars=n_vars,
            var_names=var_names,
            ground_truth_W=W,
            domain_context="Synthetic Erdos-Renyi random graph."
        )

    @staticmethod
    def scale_free(
        n_vars: int,
        avg_degree: float = SYNTHETIC_DEFAULT_AVG_DEGREE,
        seed: int = 42
    ) -> CausalDataset:
        """
        Generate scale-free DAG.

        Uses preferential attachment with random permutation to avoid
        ordering bias (hub nodes don't always have smaller indices).
        """
        np.random.seed(seed)
        W = DREAM4Dataset._generate_scale_free_dag(n_vars, avg_degree, apply_permutation=True)
        var_names = [f"X{i}" for i in range(n_vars)]

        return CausalDataset(
            name=f"SF_n{n_vars}_d{avg_degree}",
            n_vars=n_vars,
            var_names=var_names,
            ground_truth_W=W,
            domain_context="Synthetic scale-free graph with hub structure."
        )

    @staticmethod
    def with_latent_confounders(
        n_observed: int,
        n_latent: int = 3,
        edge_prob: float = SYNTHETIC_DEFAULT_EDGE_PROB,
        seed: int = 42
    ) -> CausalDataset:
        """
        Generate DAG with latent confounders.

        Creates an ADMG structure where latent variables create
        bidirected edges between observed variables.

        The induced bidirected edge covariance M is computed properly
        accounting for latent-to-latent edges (W_LL) using:
            M = W_LO^T @ (I - W_LL)^{-T} @ (I - W_LL)^{-1} @ W_LO

        This captures the total effect of latent variables on observed
        variables through the Neumann series expansion.
        """
        np.random.seed(seed)
        n_total = n_observed + n_latent

        # Generate full DAG (upper triangular for acyclicity)
        W_full = np.zeros((n_total, n_total))
        for i in range(n_total):
            for j in range(i + 1, n_total):
                if np.random.random() < edge_prob:
                    W_full[i, j] = np.random.choice([-1, 1]) * np.random.uniform(
                        LATENT_EDGE_WEIGHT_MIN, LATENT_EDGE_WEIGHT_MAX
                    )

        # Random permutation to remove ordering bias
        perm = np.random.permutation(n_total)
        W_full = W_full[perm][:, perm]

        # Split into observed and latent (after permutation)
        obs_idx = list(range(n_observed))
        lat_idx = list(range(n_observed, n_total))

        # Observed subgraph: W_OO
        W_OO = W_full[np.ix_(obs_idx, obs_idx)]

        # Latent-to-observed edges: W_LO (latent -> observed)
        W_LO = W_full[np.ix_(lat_idx, obs_idx)]

        # Latent-to-latent edges: W_LL
        W_LL = W_full[np.ix_(lat_idx, lat_idx)]

        # Compute bidirected edges from latent confounding
        # M = W_LO^T @ (I - W_LL)^{-T} @ (I - W_LL)^{-1} @ W_LO
        # This accounts for latent variable interactions via Neumann series
        I_L = np.eye(n_latent)
        try:
            # (I - W_LL)^{-1} captures total effect among latents
            inv_term = np.linalg.inv(I_L - W_LL)
            # Total effect from latent to observed
            total_effect = inv_term @ W_LO
            # M captures covariance induced by latent confounders
            M = total_effect.T @ total_effect
        except np.linalg.LinAlgError:
            # Fallback with regularization if singular
            inv_term = np.linalg.inv(I_L - W_LL + MATRIX_EPSILON * I_L)
            total_effect = inv_term @ W_LO
            M = total_effect.T @ total_effect

        var_names = [f"X{i}" for i in range(n_observed)]

        return CausalDataset(
            name=f"Latent_o{n_observed}_l{n_latent}",
            n_vars=n_observed,
            var_names=var_names,
            ground_truth_W=W_OO,
            ground_truth_M=M,
            domain_context=f"Synthetic graph with {n_latent} latent confounders."
        )


class CLadderDataset:
    """
    CLadder: Causal reasoning benchmark for LLMs.

    Tests understanding of interventions, counterfactuals, and causal inference.
    """

    DOMAIN_CONTEXT = """
    CLadder benchmark for evaluating causal reasoning abilities.
    Questions test understanding of:
    - Level 1: Observational queries (seeing/association)
    - Level 2: Interventional queries (doing/manipulation)
    - Level 3: Counterfactual queries (imagining/retrospection)
    """

    @classmethod
    def load_questions(
        cls,
        level: int = 1,
        max_questions: int = 100
    ) -> List[Dict]:
        """
        Load CLadder reasoning questions.

        Args:
            level: Pearl's ladder level (1, 2, or 3)
            max_questions: Maximum number of questions

        Returns:
            List of question dicts with 'question', 'answer', 'graph' keys
        """
        # Sample CLadder-style questions
        questions = []

        if level == 1:
            # Observational
            questions.extend([
                {
                    "question": "Given that variable A is high, what can we say about variable B?",
                    "answer": "If A→B, then B is likely high.",
                    "graph": {"edges": [("A", "B")]},
                    "type": "observational"
                },
                {
                    "question": "Are X and Y correlated when Z is not observed?",
                    "answer": "Yes, if there's a path through Z.",
                    "graph": {"edges": [("X", "Z"), ("Z", "Y")]},
                    "type": "association"
                }
            ])
        elif level == 2:
            # Interventional
            questions.extend([
                {
                    "question": "If we intervene to set A=1, what happens to B?",
                    "answer": "B increases if A→B with positive effect.",
                    "graph": {"edges": [("A", "B", "+")]},
                    "type": "intervention"
                },
                {
                    "question": "Does intervening on mediator M affect the relationship between X and Y?",
                    "answer": "Yes, it blocks the causal path X→M→Y.",
                    "graph": {"edges": [("X", "M"), ("M", "Y")]},
                    "type": "intervention"
                }
            ])
        else:
            # Counterfactual
            questions.extend([
                {
                    "question": "If A had been different, would B have changed?",
                    "answer": "Yes, if A is a cause of B.",
                    "graph": {"edges": [("A", "B")]},
                    "type": "counterfactual"
                },
                {
                    "question": "What would Y have been if we had not treated with X?",
                    "answer": "Depends on effect size of X→Y and baseline.",
                    "graph": {"edges": [("X", "Y")]},
                    "type": "counterfactual"
                }
            ])

        return questions[:max_questions]


class DatasetLoader:
    """Main interface for loading datasets."""

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.cache: Dict[str, CausalDataset] = {}

    def load(
        self,
        dataset_name: str,
        **kwargs
    ) -> CausalDataset:
        """
        Load a dataset by name.

        Args:
            dataset_name: One of 'sachs', 'dream4', 'er', 'sf', 'latent', 'cladder'
            **kwargs: Dataset-specific parameters

        Returns:
            CausalDataset object
        """
        cache_key = f"{dataset_name}_{hash(frozenset(kwargs.items()))}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        dataset_name_lower = dataset_name.lower()

        if dataset_name_lower == "sachs":
            dataset = SACHSDataset.load()

        elif dataset_name_lower == "dream4":
            network_id = kwargs.get("network_id", 1)
            size = kwargs.get("size", 10)
            dataset = DREAM4Dataset.load(network_id, size)

        elif dataset_name_lower in ["er", "erdos_renyi", "erdos-renyi"]:
            n_vars = kwargs.get("n_vars", 20)
            edge_prob = kwargs.get("edge_prob", SYNTHETIC_DEFAULT_EDGE_PROB)
            seed = kwargs.get("seed", 42)
            dataset = SyntheticDataset.erdos_renyi(n_vars, edge_prob, seed)

        elif dataset_name_lower in ["sf", "scale_free", "scale-free"]:
            n_vars = kwargs.get("n_vars", 20)
            avg_degree = kwargs.get("avg_degree", SYNTHETIC_DEFAULT_AVG_DEGREE)
            seed = kwargs.get("seed", 42)
            dataset = SyntheticDataset.scale_free(n_vars, avg_degree, seed)

        elif dataset_name_lower == "latent":
            n_observed = kwargs.get("n_observed", 20)
            n_latent = kwargs.get("n_latent", 3)
            edge_prob = kwargs.get("edge_prob", SYNTHETIC_DEFAULT_EDGE_PROB)
            seed = kwargs.get("seed", 42)
            dataset = SyntheticDataset.with_latent_confounders(
                n_observed, n_latent, edge_prob, seed
            )

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.cache[cache_key] = dataset
        return dataset

    def load_for_experiment(
        self,
        experiment_type: str,
        config: Optional[Dict] = None,
        **kwargs
    ) -> CausalDataset:
        """
        Load dataset appropriate for experiment type.

        Args:
            experiment_type: One of 'E1'-'E5', 'A1'-'A6', 'X1'-'X4', etc.
            config: Optional experiment config dict (from defaults.yaml).
                    If provided, dataset settings are read from config.
            **kwargs: Override parameters (take priority over config)

        Note:
            Experiment-specific settings should be defined in defaults.yaml
            under the 'experiments' section. This method falls back to
            hardcoded defaults only when config is not provided.
        """
        # If config provided, use it to get dataset settings
        if config:
            exp_config = config.get('experiments', {}).get(experiment_type, {})
            dataset_name = kwargs.get('dataset', exp_config.get('dataset', 'sachs'))
            n_vars = kwargs.get('n_vars', exp_config.get('n_vars', 20))

            if dataset_name == 'latent':
                n_latent = kwargs.get('n_latent', exp_config.get('n_latent', 3))
                return self.load("latent", n_observed=n_vars, n_latent=n_latent)
            elif dataset_name in ['sf', 'scale_free']:
                avg_degree = kwargs.get('avg_degree', exp_config.get('avg_degree', SYNTHETIC_DEFAULT_AVG_DEGREE))
                return self.load("sf", n_vars=n_vars, avg_degree=avg_degree)
            elif dataset_name in ['er', 'erdos_renyi']:
                edge_prob = kwargs.get('edge_prob', exp_config.get('edge_prob', SYNTHETIC_DEFAULT_EDGE_PROB))
                return self.load("er", n_vars=n_vars, edge_prob=edge_prob)
            else:
                return self.load(dataset_name, **kwargs)

        # Fallback: hardcoded defaults when no config provided
        # NOTE: These should match defaults.yaml experiments section
        # E1-E2: Single paper (use SACHS or synthetic)
        if experiment_type in ["E1", "E2"]:
            return self.load("sachs")

        # E3-E4: Corpus integration (use larger synthetic)
        elif experiment_type in ["E3", "E4"]:
            n_vars = kwargs.get("n_vars", 50)
            avg_degree = kwargs.get("avg_degree", SYNTHETIC_DEFAULT_AVG_DEGREE + 1.0)  # 3.0
            return self.load("sf", n_vars=n_vars, avg_degree=avg_degree)

        # E5: Rashomon (need latent confounders)
        elif experiment_type == "E5":
            n_observed = kwargs.get("n_observed", 30)
            n_latent = kwargs.get("n_latent", 5)
            return self.load("latent", n_observed=n_observed, n_latent=n_latent)

        # A1-A6: Ablation (use medium synthetic)
        elif experiment_type.startswith("A"):
            n_vars = kwargs.get("n_vars", 30)
            edge_prob = kwargs.get("edge_prob", SYNTHETIC_DEFAULT_EDGE_PROB * 0.75)  # 0.15
            return self.load("er", n_vars=n_vars, edge_prob=edge_prob)

        # X1-X4: Sheaf exactness (synthetic with precise structure)
        elif experiment_type.startswith("X"):
            return self.load("sf", n_vars=kwargs.get("n_vars", 50))

        # Default: SACHS
        else:
            return self.load("sachs")

    def to_torch(
        self,
        dataset: CausalDataset,
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """Convert dataset to PyTorch tensors."""
        result = {
            "W": torch.tensor(dataset.ground_truth_W, dtype=torch.float32, device=device),
        }
        if dataset.ground_truth_M is not None:
            result["M"] = torch.tensor(
                dataset.ground_truth_M, dtype=torch.float32, device=device
            )
        return result


def create_dataset_loader(
    config: Optional[DatasetConfig] = None
) -> DatasetLoader:
    """Factory function to create dataset loader."""
    return DatasetLoader(config)
