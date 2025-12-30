"""
Sheaf Engine for HOLOGRAPH.

Implements the Algebraic Latent Projection and Frobenius Descent Conditions
for sheaf-theoretic consistency in causal discovery.

Mathematical Foundation:
- Objects: Linear SEMs (ADMGs) represented as (W, M)
- Restriction: Algebraic Latent Projection ρ_UV
- Gluing: Frobenius Descent Loss minimization
"""

import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Import centralized constants
import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments.config.constants import (
    MATRIX_EPSILON,
    SPECTRAL_MARGIN,
    CONTRADICTION_DETECTION_THRESHOLD,
    get_scale_dependent_threshold,
)

from .causal_state import CausalState


@dataclass
class ContextOverlap:
    """
    Represents an overlap between two contexts for sheaf descent.

    Attributes:
        context_i: Name/ID of first context
        context_j: Name/ID of second context
        indices_i: Variable indices in context i
        indices_j: Variable indices in context j
        intersection: Variable indices in the overlap V_i ∩ V_j
    """
    context_i: str
    context_j: str
    indices_i: List[int]
    indices_j: List[int]
    intersection: List[int]


class SheafEngine:
    """
    Implements the Algebraic Latent Projection and Descent Conditions.

    The sheaf engine ensures global consistency of local causal models
    by projecting them onto their intersections and minimizing the
    Frobenius norm of the discrepancies.

    Reference: Mathematical Specification Section 1.1
    """

    def __init__(
        self,
        epsilon: float = MATRIX_EPSILON,
        use_pseudoinverse: bool = True,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize SheafEngine.

        Args:
            epsilon: Regularization for matrix inversion
            use_pseudoinverse: Use SVD-based pseudoinverse for stability
            device: Computation device
        """
        self.epsilon = epsilon
        self.use_pseudoinverse = use_pseudoinverse
        self.device = device

    def partition_indices(
        self,
        n_vars: int,
        observed_indices: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Partition variables into Observed (O) and Hidden (H) sets.

        Args:
            n_vars: Total number of variables
            observed_indices: Indices of observed variables

        Returns:
            Tuple of (observed_indices, hidden_indices)
        """
        all_indices = set(range(n_vars))
        observed_set = set(observed_indices)
        hidden_indices = sorted(list(all_indices - observed_set))
        return list(observed_indices), hidden_indices

    def partition_matrix(
        self,
        M: torch.Tensor,
        observed_indices: List[int],
        hidden_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Partition a matrix into O-O, O-H, H-O, H-H blocks.

        Args:
            M: Matrix to partition (N x N)
            observed_indices: Indices of observed variables
            hidden_indices: Indices of hidden variables

        Returns:
            Tuple of (M_OO, M_OH, M_HO, M_HH)
        """
        O = observed_indices
        H = hidden_indices

        M_OO = M[torch.tensor(O)][:, torch.tensor(O)]
        M_OH = M[torch.tensor(O)][:, torch.tensor(H)] if H else torch.empty(len(O), 0, device=M.device)
        M_HO = M[torch.tensor(H)][:, torch.tensor(O)] if H else torch.empty(0, len(O), device=M.device)
        M_HH = M[torch.tensor(H)][:, torch.tensor(H)] if H else torch.empty(0, 0, device=M.device)

        return M_OO, M_OH, M_HO, M_HH

    def compute_absorption_matrix(
        self,
        W_OH: torch.Tensor,
        W_HH: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the absorption matrix A = W_OH(I - W_HH)^{-1}.

        The absorption matrix captures how hidden variable effects
        propagate to observed variables.

        Args:
            W_OH: Edges from observed to hidden (O x H)
            W_HH: Edges among hidden variables (H x H)

        Returns:
            Absorption matrix A (O x H)
        """
        if W_HH.shape[0] == 0:
            # No hidden variables
            return torch.empty(W_OH.shape[0], 0, device=W_OH.device)

        n_hidden = W_HH.shape[0]
        I = torch.eye(n_hidden, device=W_HH.device, dtype=W_HH.dtype)

        # Compute (I - W_HH)^{-1}
        diff = I - W_HH

        if self.use_pseudoinverse:
            # SVD-based pseudoinverse for numerical stability
            inv_term = torch.linalg.pinv(diff, rcond=self.epsilon)
        else:
            # Standard inverse with regularization
            diff_reg = diff + self.epsilon * I
            inv_term = torch.linalg.inv(diff_reg)

        return W_OH @ inv_term

    def algebraic_latent_projection(
        self,
        state: CausalState,
        observed_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Algebraic Latent Projection ρ_UV.

        Projects a global causal state onto a subset of observed variables,
        accounting for the effects of hidden confounders.

        Formula:
            W_tilde = W_OO + A @ W_HO
            M_tilde = M_OO + A @ M_HH @ A^T + M_OH @ A^T + A @ M_HO

        where A = W_OH @ (I - W_HH)^{-1}

        Args:
            state: Global causal state (W, L)
            observed_indices: Indices of variables to project onto

        Returns:
            Tuple of (W_tilde, M_tilde) - projected matrices
        """
        O, H = self.partition_indices(state.n_vars, observed_indices)

        # Partition W
        W_OO, W_OH, W_HO, W_HH = self.partition_matrix(state.W, O, H)

        # Partition M = L @ L^T
        M = state.M
        M_OO, M_OH, M_HO, M_HH = self.partition_matrix(M, O, H)

        if len(H) == 0:
            # No hidden variables - identity projection
            return W_OO, M_OO

        # Compute absorption matrix
        A = self.compute_absorption_matrix(W_OH, W_HH)

        # Project W
        W_tilde = W_OO + A @ W_HO

        # Project M (including cross-terms for transitivity)
        M_tilde = (
            M_OO +
            A @ M_HH @ A.t() +
            M_OH @ A.t() +
            A @ M_HO
        )

        return W_tilde, M_tilde

    def compute_descent_loss(
        self,
        state: CausalState,
        overlaps: List[ContextOverlap]
    ) -> torch.Tensor:
        """
        Compute Frobenius Descent Loss over context overlaps.

        L_descent = Σ_{i,j} (||W_i|_V_{ij} - W_j|_V_{ij}||_F^2 +
                            ||M_i|_V_{ij} - M_j|_V_{ij}||_F^2)

        This enforces the sheaf descent condition: local sections
        must agree on their overlaps.

        Args:
            state: Global causal state
            overlaps: List of context overlaps

        Returns:
            Total descent loss (scalar tensor)
        """
        if not overlaps:
            return torch.tensor(0.0, device=state.device)

        loss = torch.tensor(0.0, device=state.device)

        for overlap in overlaps:
            # Project global state onto each context
            W_i, M_i = self.algebraic_latent_projection(state, overlap.indices_i)
            W_j, M_j = self.algebraic_latent_projection(state, overlap.indices_j)

            # Map intersection indices to local indices
            # This is a simplified version - in practice, need index mapping
            intersection_size = len(overlap.intersection)

            if intersection_size > 0:
                # For now, assume intersection aligns with beginning of indices
                # In full implementation, need proper index alignment
                W_i_inter = W_i[:intersection_size, :intersection_size]
                W_j_inter = W_j[:intersection_size, :intersection_size]
                M_i_inter = M_i[:intersection_size, :intersection_size]
                M_j_inter = M_j[:intersection_size, :intersection_size]

                # Frobenius norm squared
                loss = loss + torch.norm(W_i_inter - W_j_inter, p='fro') ** 2
                loss = loss + torch.norm(M_i_inter - M_j_inter, p='fro') ** 2

        return loss

    def compute_spectral_penalty(
        self,
        state: CausalState,
        margin: float = SPECTRAL_MARGIN
    ) -> torch.Tensor:
        """
        Compute spectral stability regularization.

        L_spec = max(0, ρ(W) - 1 + δ)^2

        Ensures spectral radius < 1 for stable Neumann series.

        Args:
            state: Causal state
            margin: Safety margin δ

        Returns:
            Spectral penalty (scalar tensor)
        """
        # Use Frobenius norm as differentiable upper bound on spectral radius
        # ||W||_F >= ||W||_2 >= ρ(W), so this is a conservative penalty
        # This avoids eigvals backward which can fail for ill-conditioned matrices
        W_sq = state.W * state.W
        frobenius_norm = torch.sqrt(torch.sum(W_sq) + self.epsilon)

        # Alternative: power iteration (more accurate but slower)
        # spectral_radius = self._power_iteration_spectral_radius(state.W)

        penalty = torch.clamp(frobenius_norm - (1 - margin), min=0) ** 2
        return penalty

    def _power_iteration_spectral_radius(
        self,
        W: torch.Tensor,
        n_iter: int = 10
    ) -> torch.Tensor:
        """
        Estimate spectral radius using power iteration.

        More numerically stable than eigenvalue decomposition.
        """
        n = W.shape[0]
        v = torch.ones(n, device=W.device, dtype=W.dtype) / np.sqrt(n)

        for _ in range(n_iter):
            Wv = W @ v
            norm = torch.norm(Wv) + self.epsilon
            v = Wv / norm

        # Rayleigh quotient
        return torch.abs(v @ W @ v)

    def verify_identity_axiom(
        self,
        state: CausalState,
        indices: List[int]
    ) -> float:
        """
        Verify presheaf identity axiom: ρ_UU = id.

        For Sheaf Exactness Validation (X1).

        Args:
            state: Causal state
            indices: Full variable indices

        Returns:
            Frobenius norm of deviation from identity
        """
        W_proj, M_proj = self.algebraic_latent_projection(state, indices)

        # Compare with original (restricted to these indices)
        W_orig = state.W[torch.tensor(indices)][:, torch.tensor(indices)]
        M_orig = state.M[torch.tensor(indices)][:, torch.tensor(indices)]

        error = (
            torch.norm(W_proj - W_orig, p='fro').item() +
            torch.norm(M_proj - M_orig, p='fro').item()
        )
        return error

    def verify_transitivity_axiom(
        self,
        state: CausalState,
        indices_U: List[int],
        indices_V: List[int],
        indices_Z: List[int]
    ) -> float:
        """
        Verify presheaf transitivity axiom: ρ_ZU = ρ_ZV ∘ ρ_VU.

        For Sheaf Exactness Validation (X2).

        Args:
            state: Causal state
            indices_U, indices_V, indices_Z: Nested context indices (U ⊃ V ⊃ Z)

        Returns:
            Frobenius norm of composition error
        """
        # Direct projection U → Z
        W_direct, M_direct = self.algebraic_latent_projection(state, indices_Z)

        # Composed projection: U → V → Z
        # First, create intermediate state from projection to V
        W_V, M_V = self.algebraic_latent_projection(state, indices_V)

        # Create intermediate CausalState for V
        # Note: This is a simplified version; need Cholesky factor for M_V
        L_V = torch.linalg.cholesky(M_V + self.epsilon * torch.eye(M_V.shape[0], device=M_V.device))
        state_V = CausalState(W=W_V, L=L_V)

        # Project V → Z
        # Map indices_Z to local indices in V
        V_to_global = {v: i for i, v in enumerate(indices_V)}
        local_Z = [V_to_global[z] for z in indices_Z if z in V_to_global]

        if local_Z:
            W_composed, M_composed = self.algebraic_latent_projection(state_V, local_Z)

            error = (
                torch.norm(W_direct - W_composed, p='fro').item() +
                torch.norm(M_direct - M_composed, p='fro').item()
            )
        else:
            error = float('inf')  # Invalid nesting

        return error

    def find_max_residual_pair(
        self,
        state: CausalState,
        overlaps: List[ContextOverlap]
    ) -> Tuple[Tuple[str, str], float]:
        """
        Find the context pair with maximum descent residual.

        Used for topological obstruction detection.

        Args:
            state: Causal state
            overlaps: List of context overlaps

        Returns:
            Tuple of ((context_i, context_j), residual)
        """
        max_residual = 0.0
        max_pair = (None, None)

        for overlap in overlaps:
            W_i, M_i = self.algebraic_latent_projection(state, overlap.indices_i)
            W_j, M_j = self.algebraic_latent_projection(state, overlap.indices_j)

            intersection_size = len(overlap.intersection)
            if intersection_size > 0:
                W_i_inter = W_i[:intersection_size, :intersection_size]
                W_j_inter = W_j[:intersection_size, :intersection_size]

                residual = torch.norm(W_i_inter - W_j_inter, p='fro').item()

                if residual > max_residual:
                    max_residual = residual
                    max_pair = (overlap.context_i, overlap.context_j)

        return max_pair, max_residual

    def detect_contradiction(
        self,
        W_i: torch.Tensor,
        W_j: torch.Tensor,
        threshold: float = CONTRADICTION_DETECTION_THRESHOLD,
        use_scale_dependent: bool = False,
        base_threshold: float = 0.1
    ) -> Optional[Tuple[int, int]]:
        """
        Detect contradictory edges between two context projections.

        Used for Rashomon Stress Test (E5).

        Detects two types of contradictions:
        1. Same position, opposite sign (X→Y with positive vs negative effect)
        2. Reversed direction (W_i[a,b] significant AND W_j[b,a] significant)

        Args:
            W_i, W_j: Projected W matrices for two contexts
            threshold: Minimum magnitude for edge existence (ignored if use_scale_dependent)
            use_scale_dependent: If True, compute threshold based on matrix size
            base_threshold: Base threshold for scale-dependent mode (default: 0.1)

        Returns:
            Tuple (row, col) of contradicting edge, or None

        Note:
            Frobenius norm scales with sqrt(n_vars), so a fixed threshold may be
            too loose for large graphs or too strict for small graphs.
            Set use_scale_dependent=True for automatic scaling.
        """
        n = W_i.shape[0]

        # Compute scale-dependent threshold if requested
        if use_scale_dependent:
            threshold = get_scale_dependent_threshold(base_threshold, n, method='sqrt')

        # Check for reversed edge contradictions (X→Y in one, Y→X in other)
        for a in range(n):
            for b in range(n):
                if a != b:
                    # Check if W_i has edge a→b and W_j has edge b→a
                    edge_i_ab = torch.abs(W_i[a, b]) > threshold
                    edge_j_ba = torch.abs(W_j[b, a]) > threshold

                    if edge_i_ab and edge_j_ba:
                        return (a, b)

                    # Check if W_j has edge a→b and W_i has edge b→a
                    edge_j_ab = torch.abs(W_j[a, b]) > threshold
                    edge_i_ba = torch.abs(W_i[b, a]) > threshold

                    if edge_j_ab and edge_i_ba:
                        return (a, b)

        # Check for sign disagreements at same position
        significant_i = torch.abs(W_i) > threshold
        significant_j = torch.abs(W_j) > threshold
        both_significant = significant_i & significant_j
        sign_disagree = torch.sign(W_i) != torch.sign(W_j)
        contradictions = both_significant & sign_disagree

        if contradictions.any():
            idx = torch.nonzero(contradictions, as_tuple=False)[0]
            return (idx[0].item(), idx[1].item())

        return None
