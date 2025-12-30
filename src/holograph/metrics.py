"""
Evaluation metrics for causal discovery.

Implements standard metrics:
- SHD (Structural Hamming Distance)
- F1 Score (Edge prediction)
- SID (Structural Intervention Distance)
"""

import numpy as np
from typing import Tuple, Dict, Optional
from collections import defaultdict

# Import centralized constants
import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments.config.constants import (
    RASHOMON_DETECTION_STD_THRESHOLD,
    IDENTITY_AXIOM_THRESHOLD,
    TRANSITIVITY_AXIOM_THRESHOLD,
    LOCALITY_AXIOM_THRESHOLD,
    GLUING_AXIOM_THRESHOLD,
)


def compute_shd(
    pred: np.ndarray,
    true: np.ndarray,
    ignore_direction: bool = False
) -> int:
    """
    Compute Structural Hamming Distance.

    SHD = #(missing edges) + #(extra edges) + #(reversed edges)

    Args:
        pred: Predicted adjacency matrix (binary)
        true: Ground truth adjacency matrix (binary)
        ignore_direction: If True, only count edge presence, not direction

    Returns:
        SHD score (lower is better)
    """
    assert pred.shape == true.shape, "Matrices must have same shape"

    pred = (pred != 0).astype(int)
    true = (true != 0).astype(int)

    if ignore_direction:
        # Symmetrize: edge exists if (i,j) or (j,i)
        pred = np.maximum(pred, pred.T)
        true = np.maximum(true, true.T)
        # Only count upper triangle
        pred = np.triu(pred, k=1)
        true = np.triu(true, k=1)
        diff = np.abs(pred - true)
        return int(np.sum(diff))

    # Count edge differences
    diff = pred - true

    # Missing edges: true=1, pred=0
    missing = np.sum((diff == -1))

    # Extra edges: true=0, pred=1
    extra = np.sum((diff == 1))

    # Reversed edges: both have edge but different direction
    # Edge (i,j) exists in both but direction reversed
    both_exist_pred = (pred + pred.T) > 0
    both_exist_true = (true + true.T) > 0
    overlap = both_exist_pred & both_exist_true

    # Check for reversals in overlap
    reversed_count = 0
    n = pred.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if overlap[i, j] or overlap[j, i]:
                pred_dir = pred[i, j] - pred[j, i]  # 1 if i->j, -1 if j->i
                true_dir = true[i, j] - true[j, i]
                if pred_dir != 0 and true_dir != 0 and pred_dir != true_dir:
                    reversed_count += 1

    return int(missing + extra + reversed_count)


def compute_edge_metrics(
    pred: np.ndarray,
    true: np.ndarray
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 for edge prediction.

    Args:
        pred: Predicted adjacency matrix (binary)
        true: Ground truth adjacency matrix (binary)

    Returns:
        Dictionary with precision, recall, f1
    """
    pred = (pred != 0).astype(int)
    true = (true != 0).astype(int)

    # Flatten for binary classification metrics
    pred_flat = pred.flatten()
    true_flat = true.flatten()

    tp = np.sum((pred_flat == 1) & (true_flat == 1))
    fp = np.sum((pred_flat == 1) & (true_flat == 0))
    fn = np.sum((pred_flat == 0) & (true_flat == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def compute_f1(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Compute F1 score for edge prediction.

    Args:
        pred: Predicted adjacency matrix
        true: Ground truth adjacency matrix

    Returns:
        F1 score (higher is better, range [0, 1])
    """
    return compute_edge_metrics(pred, true)['f1']


def compute_sid(
    pred: np.ndarray,
    true: np.ndarray,
    n_samples: int = 1000
) -> float:
    """
    Compute Structural Intervention Distance (approximate).

    SID measures the number of incorrectly inferred causal effects
    under all single-node interventions.

    This is an approximation based on reachability analysis.

    Args:
        pred: Predicted adjacency matrix
        true: Ground truth adjacency matrix
        n_samples: Number of intervention samples

    Returns:
        SID score (lower is better)
    """
    n = pred.shape[0]
    pred = (pred != 0).astype(int)
    true = (true != 0).astype(int)

    # Compute transitive closures (reachability)
    def transitive_closure(adj: np.ndarray) -> np.ndarray:
        """Compute reachability matrix via matrix powers."""
        n = adj.shape[0]
        reach = adj.copy().astype(float)
        power = adj.copy().astype(float)
        for _ in range(n - 1):
            power = power @ adj
            reach = reach + power
        return (reach > 0).astype(int)

    pred_reach = transitive_closure(pred)
    true_reach = transitive_closure(true)

    # SID counts disagreements in causal effects
    # A causal effect P(Y|do(X)) differs if reachability differs
    sid = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                # Effect of do(X_i) on X_j
                pred_effect = pred_reach[i, j]
                true_effect = true_reach[i, j]
                if pred_effect != true_effect:
                    sid += 1

    return float(sid)


def compute_all_metrics(
    pred: np.ndarray,
    true: np.ndarray
) -> Dict[str, float]:
    """
    Compute all standard causal discovery metrics.

    Args:
        pred: Predicted adjacency matrix
        true: Ground truth adjacency matrix

    Returns:
        Dictionary with all metrics
    """
    edge_metrics = compute_edge_metrics(pred, true)

    return {
        'shd': compute_shd(pred, true),
        'shd_undirected': compute_shd(pred, true, ignore_direction=True),
        'sid': compute_sid(pred, true),
        **edge_metrics
    }


class RashomonMetrics:
    """
    Metrics for Rashomon Stress Test (E5).

    Measures contradiction detection and resolution capabilities.
    """

    def __init__(self):
        self.detection_results = []
        self.resolution_results = []

    def record_detection(
        self,
        contradiction_injected: bool,
        descent_loss_delta: float,
        threshold: float = RASHOMON_DETECTION_STD_THRESHOLD
    ):
        """
        Record whether a contradiction was correctly detected.

        Args:
            contradiction_injected: Whether contradiction was actually present
            descent_loss_delta: Change in descent loss (normalized by std)
            threshold: Detection threshold in std units
        """
        detected = descent_loss_delta > threshold
        self.detection_results.append({
            'injected': contradiction_injected,
            'detected': detected,
            'loss_delta': descent_loss_delta
        })

    def record_resolution(
        self,
        contradiction_resolved: bool,
        latent_proposed: bool,
        shd_improvement: float
    ):
        """
        Record whether a contradiction was correctly resolved.

        Args:
            contradiction_resolved: Whether the contradiction was resolved
            latent_proposed: Whether a latent variable was proposed
            shd_improvement: SHD improvement after resolution (positive = better)
        """
        self.resolution_results.append({
            'resolved': contradiction_resolved,
            'latent_proposed': latent_proposed,
            'shd_improvement': shd_improvement
        })

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute Rashomon stress test metrics.

        Returns:
            Dictionary with detection_rate, resolution_rate, false_positive_rate
        """
        # Detection metrics
        injected = [r for r in self.detection_results if r['injected']]
        not_injected = [r for r in self.detection_results if not r['injected']]

        if injected:
            detection_rate = sum(r['detected'] for r in injected) / len(injected)
        else:
            detection_rate = 0.0

        if not_injected:
            false_positive_rate = sum(r['detected'] for r in not_injected) / len(not_injected)
        else:
            false_positive_rate = 0.0

        # Resolution metrics
        if self.resolution_results:
            resolution_rate = sum(r['resolved'] for r in self.resolution_results) / len(self.resolution_results)
            latent_proposal_rate = sum(r['latent_proposed'] for r in self.resolution_results) / len(self.resolution_results)
            avg_shd_improvement = np.mean([r['shd_improvement'] for r in self.resolution_results])
        else:
            resolution_rate = 0.0
            latent_proposal_rate = 0.0
            avg_shd_improvement = 0.0

        return {
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'resolution_rate': resolution_rate,
            'latent_proposal_rate': latent_proposal_rate,
            'avg_shd_improvement': avg_shd_improvement
        }


class SheafExactnessMetrics:
    """
    Metrics for Sheaf Exactness Validation (X1-X4).

    Measures deviation from presheaf axioms.
    """

    def __init__(self):
        self.identity_errors = []
        self.transitivity_errors = []
        self.locality_errors = []
        self.gluing_residuals = []

    def record_identity(self, error: float):
        """Record identity axiom error (X1)."""
        self.identity_errors.append(error)

    def record_transitivity(self, error: float):
        """Record transitivity axiom error (X2)."""
        self.transitivity_errors.append(error)

    def record_locality(self, error: float):
        """Record locality error (X3)."""
        self.locality_errors.append(error)

    def record_gluing(self, residual: float):
        """Record gluing residual (X4)."""
        self.gluing_residuals.append(residual)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute sheaf exactness metrics.

        Returns:
            Dictionary with max and mean errors for each axiom
        """
        metrics = {}

        if self.identity_errors:
            metrics['identity_max'] = max(self.identity_errors)
            metrics['identity_mean'] = np.mean(self.identity_errors)

        if self.transitivity_errors:
            metrics['transitivity_max'] = max(self.transitivity_errors)
            metrics['transitivity_mean'] = np.mean(self.transitivity_errors)

        if self.locality_errors:
            metrics['locality_max'] = max(self.locality_errors)
            metrics['locality_mean'] = np.mean(self.locality_errors)

        if self.gluing_residuals:
            metrics['gluing_max'] = max(self.gluing_residuals)
            metrics['gluing_mean'] = np.mean(self.gluing_residuals)

        return metrics

    def passes_thresholds(
        self,
        identity_thresh: float = IDENTITY_AXIOM_THRESHOLD,
        transitivity_thresh: float = TRANSITIVITY_AXIOM_THRESHOLD,
        locality_thresh: float = LOCALITY_AXIOM_THRESHOLD,
        gluing_thresh: float = GLUING_AXIOM_THRESHOLD
    ) -> Dict[str, bool]:
        """
        Check if metrics pass success criteria.

        Returns:
            Dictionary with pass/fail for each axiom
        """
        metrics = self.compute_metrics()

        return {
            'identity_passes': metrics.get('identity_max', float('inf')) < identity_thresh,
            'transitivity_passes': metrics.get('transitivity_max', float('inf')) < transitivity_thresh,
            'locality_passes': metrics.get('locality_max', float('inf')) < locality_thresh,
            'gluing_passes': metrics.get('gluing_max', float('inf')) < gluing_thresh
        }
