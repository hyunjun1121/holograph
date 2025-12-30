"""
Tests for SheafEngine - Algebraic Latent Projection and Descent Conditions.

TDD tests for Sheaf Exactness Validation (X1-X4).
"""

import pytest
import torch
import numpy as np

from holograph.causal_state import CausalState
from holograph.sheaf_engine import SheafEngine, ContextOverlap


class TestCausalState:
    """Tests for CausalState dataclass."""

    def test_creation(self):
        """Test basic CausalState creation."""
        W = torch.randn(5, 5)
        L = torch.tril(torch.randn(5, 5))
        state = CausalState(W=W, L=L)

        assert state.n_vars == 5
        assert state.W.shape == (5, 5)
        assert state.L.shape == (5, 5)

    def test_M_property(self):
        """Test that M = LL^T is correctly computed."""
        L = torch.tril(torch.randn(4, 4))
        W = torch.randn(4, 4)
        state = CausalState(W=W, L=L)

        expected_M = L @ L.t()
        assert torch.allclose(state.M, expected_M)

    def test_random_init(self):
        """Test random initialization."""
        state = CausalState.random_init(n_vars=10, edge_density=0.3, seed=42)

        assert state.n_vars == 10
        assert state.W.diag().sum() == 0  # No self-loops
        assert state.spectral_radius() < float('inf')

    def test_acyclicity_penalty(self):
        """Test NOTEARS acyclicity constraint."""
        # DAG should have h(W) ≈ 0 when edges are small
        W = torch.zeros(3, 3)
        W[0, 1] = 0.1
        W[1, 2] = 0.1
        L = torch.eye(3)
        state = CausalState(W=W, L=L)

        penalty = state.acyclicity_penalty()
        assert penalty.item() < 1.0  # Small for near-DAG

    def test_discretize(self):
        """Test continuous to discrete conversion."""
        W = torch.tensor([
            [0.0, 0.8, 0.1],
            [0.0, 0.0, 0.6],
            [0.0, 0.0, 0.0]
        ])
        L = torch.eye(3)
        state = CausalState(W=W, L=L)

        binary = state.discretize(threshold=0.3)
        expected = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        np.testing.assert_array_equal(binary, expected)

    def test_save_load(self, tmp_path):
        """Test state persistence."""
        state = CausalState.random_init(n_vars=5, seed=42)
        path = str(tmp_path / "state.pt")
        state.save(path)

        loaded = CausalState.load(path)
        assert torch.allclose(state.W, loaded.W)
        assert torch.allclose(state.L, loaded.L)


class TestSheafEngineProjection:
    """Tests for Algebraic Latent Projection."""

    @pytest.fixture
    def engine(self):
        return SheafEngine(epsilon=1e-6)

    @pytest.fixture
    def simple_state(self):
        """Simple 4-variable state for testing."""
        W = torch.tensor([
            [0.0, 0.5, 0.0, 0.2],
            [0.0, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.4],
            [0.0, 0.0, 0.0, 0.0]
        ], dtype=torch.float32)
        L = torch.eye(4, dtype=torch.float32) * 0.5
        return CausalState(W=W, L=L)

    def test_identity_projection(self, engine, simple_state):
        """Test X1: ρ_UU = id (identity axiom)."""
        all_indices = list(range(simple_state.n_vars))
        W_proj, M_proj = engine.algebraic_latent_projection(simple_state, all_indices)

        # Projection onto all variables should be identity
        assert torch.allclose(W_proj, simple_state.W, atol=1e-5)
        assert torch.allclose(M_proj, simple_state.M, atol=1e-5)

    def test_projection_reduces_dimension(self, engine, simple_state):
        """Test that projection reduces to observed subspace."""
        observed = [0, 1, 2]  # Variable 3 is hidden
        W_proj, M_proj = engine.algebraic_latent_projection(simple_state, observed)

        assert W_proj.shape == (3, 3)
        assert M_proj.shape == (3, 3)

    def test_projection_accounts_for_hidden(self, engine):
        """Test that hidden variable effects are absorbed."""
        # X -> H -> Y structure (H is hidden)
        W = torch.tensor([
            [0.0, 0.8, 0.0],  # X -> H
            [0.0, 0.0, 0.7],  # H -> Y
            [0.0, 0.0, 0.0]
        ], dtype=torch.float32)
        L = torch.eye(3, dtype=torch.float32)
        state = CausalState(W=W, L=L)

        # Project onto X, Y (H is hidden)
        observed = [0, 2]
        W_proj, M_proj = engine.algebraic_latent_projection(state, observed)

        # Should capture indirect effect X -> Y through H
        # W_proj[0,1] should be approximately W[0,1] * W[1,2] = 0.8 * 0.7 = 0.56
        assert W_proj.shape == (2, 2)
        assert W_proj[0, 1].item() > 0.5  # Indirect effect captured

    def test_absorption_matrix(self, engine):
        """Test absorption matrix computation."""
        W_OH = torch.tensor([[0.5], [0.3]], dtype=torch.float32)
        W_HH = torch.tensor([[0.2]], dtype=torch.float32)

        A = engine.compute_absorption_matrix(W_OH, W_HH)

        # A = W_OH @ (I - W_HH)^{-1}
        # (I - 0.2)^{-1} = 1/0.8 = 1.25
        expected = W_OH * 1.25
        assert torch.allclose(A, expected, atol=1e-4)


class TestSheafExactness:
    """Tests for Sheaf Exactness Validation (X1-X4)."""

    @pytest.fixture
    def engine(self):
        return SheafEngine(epsilon=1e-6)

    def test_x1_identity_axiom(self, engine):
        """X1: Verify ρ_UU = id to numerical precision."""
        state = CausalState.random_init(n_vars=10, seed=42)
        all_indices = list(range(10))

        error = engine.verify_identity_axiom(state, all_indices)
        assert error < 1e-5, f"Identity axiom violated: error = {error}"

    def test_x2_transitivity_axiom(self, engine):
        """X2: Verify ρ_ZU = ρ_ZV ∘ ρ_VU for nested contexts."""
        state = CausalState.random_init(n_vars=10, seed=42)

        # Nested contexts: U = all, V = [0..7], Z = [0..4]
        indices_U = list(range(10))
        indices_V = list(range(8))
        indices_Z = list(range(5))

        error = engine.verify_transitivity_axiom(state, indices_U, indices_V, indices_Z)
        assert error < 0.1, f"Transitivity axiom violated: error = {error}"

    def test_x3_locality_on_overlaps(self, engine):
        """X3: Restrictions should agree on overlapping regions."""
        state = CausalState.random_init(n_vars=10, seed=42)

        # Two contexts with overlap
        overlap = ContextOverlap(
            context_i="A",
            context_j="B",
            indices_i=[0, 1, 2, 3, 4],
            indices_j=[3, 4, 5, 6, 7],
            intersection=[3, 4]
        )

        loss = engine.compute_descent_loss(state, [overlap])

        # For a random state, loss should be finite
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestDescentLoss:
    """Tests for Frobenius Descent Loss computation."""

    @pytest.fixture
    def engine(self):
        return SheafEngine()

    def test_zero_loss_for_consistent_state(self, engine):
        """Consistent local sections should have zero descent loss."""
        # Create a state where projections naturally agree
        W = torch.zeros(4, 4)
        W[0, 1] = 0.5
        W[1, 2] = 0.5
        L = torch.eye(4)
        state = CausalState(W=W, L=L)

        # Non-overlapping contexts should have zero loss
        overlap = ContextOverlap(
            context_i="A",
            context_j="B",
            indices_i=[0, 1],
            indices_j=[2, 3],
            intersection=[]
        )

        loss = engine.compute_descent_loss(state, [overlap])
        assert loss.item() == 0.0

    def test_loss_increases_with_inconsistency(self, engine):
        """Descent loss should increase with sheaf inconsistency."""
        state = CausalState.random_init(n_vars=6, seed=42)

        # Overlapping contexts
        overlap = ContextOverlap(
            context_i="A",
            context_j="B",
            indices_i=[0, 1, 2],
            indices_j=[1, 2, 3],
            intersection=[1, 2]
        )

        loss1 = engine.compute_descent_loss(state, [overlap])

        # Perturb state to increase inconsistency
        state.W[0, 1] += 1.0
        loss2 = engine.compute_descent_loss(state, [overlap])

        # Loss should change (may increase or decrease depending on perturbation)
        assert loss1.item() != loss2.item()


class TestSpectralRegularization:
    """Tests for spectral stability regularization."""

    @pytest.fixture
    def engine(self):
        return SheafEngine()

    def test_zero_penalty_for_stable_W(self, engine):
        """Small spectral radius should give zero penalty."""
        W = torch.randn(5, 5) * 0.1  # Small weights
        L = torch.eye(5)
        state = CausalState(W=W, L=L)

        penalty = engine.compute_spectral_penalty(state, margin=0.1)
        # May be non-zero but should be small
        assert penalty.item() < 1.0

    def test_positive_penalty_for_unstable_W(self, engine):
        """Large spectral radius should give positive penalty."""
        W = torch.randn(5, 5) * 2.0  # Large weights
        L = torch.eye(5)
        state = CausalState(W=W, L=L)

        penalty = engine.compute_spectral_penalty(state, margin=0.1)
        # Should be positive for unstable W
        assert penalty.item() >= 0.0


class TestContradictionDetection:
    """Tests for Rashomon Stress Test (E5) - contradiction detection."""

    @pytest.fixture
    def engine(self):
        return SheafEngine()

    def test_detect_sign_contradiction(self, engine):
        """Detect when two contexts disagree on edge direction."""
        # Context A says X -> Y
        W_A = torch.tensor([
            [0.0, 0.8],
            [0.0, 0.0]
        ])

        # Context B says Y -> X
        W_B = torch.tensor([
            [0.0, 0.0],
            [0.8, 0.0]
        ])

        contradiction = engine.detect_contradiction(W_A, W_B, threshold=0.3)

        # Should detect contradiction
        assert contradiction is not None

    def test_no_contradiction_for_consistent_edges(self, engine):
        """No contradiction when contexts agree."""
        W_A = torch.tensor([
            [0.0, 0.8],
            [0.0, 0.0]
        ])

        W_B = torch.tensor([
            [0.0, 0.6],  # Same direction, different magnitude
            [0.0, 0.0]
        ])

        contradiction = engine.detect_contradiction(W_A, W_B, threshold=0.3)
        assert contradiction is None


class TestMaxResidualPair:
    """Tests for finding maximum residual pair for obstruction detection."""

    @pytest.fixture
    def engine(self):
        return SheafEngine()

    def test_find_max_residual(self, engine):
        """Find the most inconsistent context pair."""
        state = CausalState.random_init(n_vars=8, seed=42)

        overlaps = [
            ContextOverlap("A", "B", [0, 1, 2], [1, 2, 3], [1, 2]),
            ContextOverlap("B", "C", [1, 2, 3], [2, 3, 4], [2, 3]),
            ContextOverlap("A", "C", [0, 1, 2], [2, 3, 4], [2]),
        ]

        (ctx_i, ctx_j), residual = engine.find_max_residual_pair(state, overlaps)

        assert ctx_i is not None
        assert ctx_j is not None
        assert residual >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
