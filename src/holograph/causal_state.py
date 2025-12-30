"""
Causal State representation for HOLOGRAPH.

Represents the continuous belief state θ = (W, M) where:
- W: Directed edge weights (continuous relaxation of DAG)
- L: Cholesky factor of error covariance M = LL^T (bi-directed edges)
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

# Import centralized constants
import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments.config.constants import (
    RANDOM_INIT_EDGE_DENSITY,
    PARAM_INIT_SCALE,
    DISCRETIZATION_THRESHOLD,
)


@dataclass
class CausalState:
    """
    Continuous representation of an ADMG (Acyclic Directed Mixed Graph).

    Attributes:
        W: Weighted adjacency matrix for directed edges. Shape (N, N).
           W[i,j] represents the causal effect of variable i on variable j.
        L: Lower-triangular Cholesky factor. Shape (N, N).
           The error covariance M = L @ L.T represents bi-directed edges
           (latent confounding).
        variable_names: Optional list of variable names for interpretability.
    """
    W: torch.Tensor
    L: torch.Tensor
    variable_names: Optional[list] = None

    def __post_init__(self):
        """Validate tensor shapes and properties."""
        assert self.W.dim() == 2, f"W must be 2D, got {self.W.dim()}D"
        assert self.L.dim() == 2, f"L must be 2D, got {self.L.dim()}D"
        assert self.W.shape[0] == self.W.shape[1], "W must be square"
        assert self.L.shape[0] == self.L.shape[1], "L must be square"
        assert self.W.shape == self.L.shape, "W and L must have same shape"

        if self.variable_names is not None:
            assert len(self.variable_names) == self.W.shape[0], \
                "variable_names must match matrix dimension"

    @property
    def M(self) -> torch.Tensor:
        """Compute error covariance matrix M = LL^T."""
        return self.L @ self.L.t()

    @property
    def n_vars(self) -> int:
        """Number of variables in the causal model."""
        return self.W.shape[0]

    @property
    def device(self) -> torch.device:
        """Device where tensors are stored."""
        return self.W.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of tensors."""
        return self.W.dtype

    def to(self, device: torch.device) -> 'CausalState':
        """Move state to specified device."""
        return CausalState(
            W=self.W.to(device),
            L=self.L.to(device),
            variable_names=self.variable_names
        )

    def clone(self) -> 'CausalState':
        """Create a deep copy of the state."""
        return CausalState(
            W=self.W.clone(),
            L=self.L.clone(),
            variable_names=self.variable_names.copy() if self.variable_names else None
        )

    def detach(self) -> 'CausalState':
        """Detach from computation graph."""
        return CausalState(
            W=self.W.detach(),
            L=self.L.detach(),
            variable_names=self.variable_names
        )

    def requires_grad_(self, requires_grad: bool = True) -> 'CausalState':
        """Set requires_grad for optimization."""
        self.W.requires_grad_(requires_grad)
        self.L.requires_grad_(requires_grad)
        return self

    @classmethod
    def random_init(
        cls,
        n_vars: int,
        edge_density: float = RANDOM_INIT_EDGE_DENSITY,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
        variable_names: Optional[list] = None
    ) -> 'CausalState':
        """
        Initialize random causal state.

        Args:
            n_vars: Number of variables
            edge_density: Expected density of non-zero edges in W
            device: Target device
            dtype: Data type
            seed: Random seed for reproducibility
            variable_names: Optional list of variable names
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Initialize W with sparse structure
        W = torch.randn(n_vars, n_vars, device=device, dtype=dtype) * PARAM_INIT_SCALE
        mask = torch.rand(n_vars, n_vars, device=device) < edge_density
        W = W * mask.float()
        W.fill_diagonal_(0)  # No self-loops

        # Initialize L as identity + small noise (ensures PSD M)
        L = torch.eye(n_vars, device=device, dtype=dtype)
        L = L + torch.tril(torch.randn(n_vars, n_vars, device=device, dtype=dtype) * PARAM_INIT_SCALE)

        return cls(W=W, L=L, variable_names=variable_names)

    @classmethod
    def from_numpy(
        cls,
        W: np.ndarray,
        L: Optional[np.ndarray] = None,
        variable_names: Optional[list] = None,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ) -> 'CausalState':
        """Create CausalState from numpy arrays."""
        W_tensor = torch.from_numpy(W).to(device=device, dtype=dtype)

        if L is None:
            L = np.eye(W.shape[0])
        L_tensor = torch.from_numpy(L).to(device=device, dtype=dtype)

        return cls(W=W_tensor, L=L_tensor, variable_names=variable_names)

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Export to numpy arrays."""
        return (
            self.W.detach().cpu().numpy(),
            self.L.detach().cpu().numpy()
        )

    def discretize(self, threshold: float = DISCRETIZATION_THRESHOLD) -> np.ndarray:
        """
        Discretize continuous W to binary adjacency matrix.

        Args:
            threshold: Absolute value threshold for edge existence

        Returns:
            Binary adjacency matrix as numpy array
        """
        W_np = self.W.detach().cpu().numpy()
        return (np.abs(W_np) > threshold).astype(int)

    def spectral_radius(self) -> float:
        """
        Compute spectral radius ρ(W) for stability check.

        The spectral radius must be < 1 for stable Neumann series
        in the algebraic latent projection.
        """
        eigenvalues = torch.linalg.eigvals(self.W)
        return float(torch.max(torch.abs(eigenvalues)).item())

    def acyclicity_penalty(self) -> torch.Tensor:
        """
        Compute NOTEARS-style acyclicity constraint h(W).

        h(W) = Tr(e^{W ∘ W}) - N = 0 iff W is acyclic.
        """
        W_sq = self.W * self.W  # Element-wise square
        expm = torch.matrix_exp(W_sq)
        return torch.trace(expm) - self.n_vars

    def save(self, path: str) -> None:
        """Save state to file."""
        torch.save({
            'W': self.W,
            'L': self.L,
            'variable_names': self.variable_names
        }, path)

    @classmethod
    def load(cls, path: str, device: torch.device = torch.device('cpu')) -> 'CausalState':
        """Load state from file."""
        data = torch.load(path, map_location=device)
        return cls(
            W=data['W'],
            L=data['L'],
            variable_names=data.get('variable_names')
        )

    def __repr__(self) -> str:
        return (
            f"CausalState(n_vars={self.n_vars}, "
            f"spectral_radius={self.spectral_radius():.4f}, "
            f"acyclicity={self.acyclicity_penalty().item():.4f})"
        )
