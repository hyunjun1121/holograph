# Control Flow Specification: HOLOGRAPH System Architecture

## 1. System Architecture Flowchart

The following diagram illustrates the closed-loop control system. The **HOLOGRAPH Agent** operates as a filter, refining a continuous belief state $\theta = (W, M)$ by actively foraging for information from the **LLM Environment**. The **Sheaf Engine** acts as a topological constraint, ensuring that the global belief state remains consistent across local context projections.

```mermaid
graph TD
    subgraph "Environment (Latent SCM)"
        LLM[Large Language Model]
        Contexts[Context Topology X]
    end

    subgraph "HOLOGRAPH Agent (Active Inference)"
        Prior[Prior Belief P(θ)]
        Planner[EFE Minimizer / Planner]
        Sensor[Semantic Embedding φ]
        Optimizer[Natural Gradient Descent]
    end

    subgraph "Sheaf Consistency Engine"
        Projector[Algebraic Latent Projection L]
        Descent[Frobenius Descent Loss]
        Spectral[Spectral Stability Check]
    end

    %% Data Flow
    Prior --> Planner
    Planner -- "Action a* (Query)" --> LLM
    LLM -- "Observation y (Text)" --> Sensor
    Sensor -- "Embedding φ(y)" --> Optimizer
    
    %% Optimization Loop
    Optimizer -- "Candidate θ" --> Projector
    Contexts -- "Overlap V_ij" --> Projector
    Projector -- "Projected Matrices" --> Descent
    Descent -- "Gradient ∇L_descent" --> Optimizer
    Spectral -- "Gradient ∇L_spec" --> Optimizer
    
    %% Belief Update
    Optimizer -- "Updated Posterior Q(θ|y)" --> Prior
    
    %% Error Handling
    Descent -- "High Residual (Obstruction)" --> Planner
    Planner -- "Discriminator Query" --> LLM
```

---

## 2. Rigorous Pseudocode: The Active Causal Loop

This specification implements the **Continuous Belief-Update Loop** defined in the Mathematical Specification. It bridges the discrete nature of text (queries/responses) with the continuous optimization of the causal manifold (ADMGs).

```python
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

# Type Aliases for Mathematical Objects
AdjacencyMatrix = torch.Tensor  # W in R^{NxN}
CovarianceFactor = torch.Tensor # L in R^{NxN} (Lower Triangular)
ContextID = str
Query = str
TextEmbedding = torch.Tensor

@dataclass
class CausalState:
    """
    Represents the continuous belief state θ = (W, M).
    W: Directed edge weights (continuous relaxation of DAG).
    L: Cholesky factor of error covariance M = LL^T (bi-directed edges).
    """
    W: AdjacencyMatrix
    L: CovarianceFactor
    
    @property
    def M(self) -> torch.Tensor:
        return self.L @ self.L.t()

class SheafEngine:
    """
    Implements the Algebraic Latent Projection and Descent Conditions.
    Ref: Math Spec Section 1.1
    """
    def algebraic_latent_projection(self, state: CausalState, 
                                    observed_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes ρ_UV via Neumann series and Schur complements.
        Formula: W_tilde = W_OO + W_OH(I - W_HH)^(-1)W_HO
        """
        # Partition matrices into Observed (O) and Hidden (H)
        W_OO, W_OH, W_HO, W_HH = self._partition(state.W, observed_indices)
        M_OO, M_OH, M_HO, M_HH = self._partition(state.M, observed_indices)
        
        # Compute Absorption Matrix A = W_OH(I - W_HH)^(-1)
        I = torch.eye(W_HH.shape[0])
        # Note: In practice, use torch.linalg.solve for stability
        inv_term = torch.linalg.inv(I - W_HH) 
        A = W_OH @ inv_term
        
        # Project W and M
        W_tilde = W_OO + A @ W_HO
        M_tilde = M_OO + A @ M_HH @ A.t() + M_OH @ A.t() + A @ M_HO
        
        return W_tilde, M_tilde

    def compute_descent_loss(self, state: CausalState, 
                             overlaps: List[Dict]) -> torch.Tensor:
        """
        Computes Frobenius Descent Loss over context overlaps.
        L_descent = Σ || L(θ_i) - L(θ_j) ||_F^2
        """
        loss = torch.tensor(0.0)
        for overlap in overlaps:
            # Project global state onto Context i and Context j
            W_i, M_i = self.algebraic_latent_projection(state, overlap['indices_i'])
            W_j, M_j = self.algebraic_latent_projection(state, overlap['indices_j'])
            
            # Restrict both to the intersection variables
            # (Assuming indices are aligned via Semantic Normalization Functor)
            loss += torch.norm(W_i - W_j, p='fro')**2
            loss += torch.norm(M_i - M_j, p='fro')**2
            
        return loss

class HolographAgent:
    """
    Implements Active Inference via Natural Gradient Descent.
    Ref: Math Spec Section 3 & 4
    """
    def __init__(self, num_vars: int, learning_rate: float = 0.01):
        self.theta = self._initialize_priors(num_vars)
        self.sheaf = SheafEngine()
        self.lr = learning_rate

    def compute_expected_free_energy(self, candidate_query: Query, 
                                     current_state: CausalState) -> float:
        """
        Approximates G(π).
        G(a) = Ambiguity + Risk (Divergence from Priors/Constraints)
        """
        # 1. Epistemic Value: Expected Information Gain
        # Approximated by the variance of the posterior predictive distribution
        # projected onto the query's semantic focus.
        epistemic_val = self._estimate_information_gain(candidate_query, current_state)
        
        # 2. Pragmatic Value: Constraint Satisfaction (Acyclicity + Descent)
        # We want queries that help resolve high descent loss regions.
        pragmatic_val = self._estimate_constraint_resolution(candidate_query)
        
        return -epistemic_val - pragmatic_val # Minimize G -> Maximize Value

    def natural_gradient_step(self, observation_embedding: TextEmbedding, 
                              overlaps: List[Dict]):
        """
        Updates belief state θ using Regularized Natural Gradient.
        θ_new = θ - η * G(θ)^(-1) * ∇L
        """
        # 1. Compute Standard Gradients
        # Loss = Semantic Energy + Descent Loss + Acyclicity + Spectral Penalty
        L_sem = self._semantic_energy(observation_embedding, self.theta)
        L_descent = self.sheaf.compute_descent_loss(self.theta, overlaps)
        L_acyclic = self._acyclicity_constraint(self.theta.W)
        L_spec = self._spectral_penalty(self.theta.W)
        
        total_loss = L_sem + L_descent + L_acyclic + L_spec
        total_loss.backward()
        
        # 2. Compute Interventional Metric Tensor G(θ)
        # Sum over all atomic interventions (continuous definition)
        G_metric = self._compute_interventional_metric(self.theta)
        
        # 3. Tikhonov Regularization (Damping)
        G_reg = G_metric + 1e-4 * torch.eye(G_metric.shape[0])
        
        # 4. Natural Gradient Update
        with torch.no_grad():
            nat_grad = torch.linalg.solve(G_reg, self.theta.grad)
            self.theta -= self.lr * nat_grad
            self.theta.grad.zero_()

    def active_causal_loop(self, max_steps: int, epsilon: float):
        """
        Main Control Loop (MOD-01, MOD-03)
        """
        for t in range(max_steps):
            # 1. Generate Candidate Queries (Action Space)
            candidates = self._generate_candidate_prompts(self.theta)
            
            # 2. Select Action minimizing EFE
            best_query = min(candidates, key=lambda q: self.compute_expected_free_energy(q, self.theta))
            
            # 3. Execute Action (LLM Call)
            observation_text = self.llm_interface.query(best_query)
            obs_embedding = self.semantic_encoder(observation_text)
            
            # 4. Update Beliefs (Natural Gradient)
            self.natural_gradient_step(obs_embedding, self.context_overlaps)
            
            # 5. Check Convergence
            if self.sheaf.compute_descent_loss(self.theta, self.context_overlaps) < epsilon:
                break
                
            # 6. Error Handling (Topological Obstruction)
            if self._detect_obstruction():
                self._resolve_topological_obstruction()

```

---

## 3. Recursive Error Handling: Topological Obstruction Resolution

When the Sheaf Engine detects a non-trivial obstruction (i.e., the Frobenius Descent Loss plateaus above $\epsilon$), it implies a logical contradiction in the source material that cannot be resolved by simple averaging. The system must enter a recursive refinement mode.

### Logic Specification

1.  **Detection:**
    *   Monitor $\mathcal{L}_{\text{descent}}$ moving average.
    *   Trigger if $\Delta \mathcal{L}_{\text{descent}} \approx 0$ AND $\mathcal{L}_{\text{descent}} > \epsilon$.
    *   Identify the specific overlap pair $(U_i, U_j)$ contributing the maximum residual.

2.  **Diagnosis (The "Cohomological Alert"):**
    *   Extract the conflicting sub-structures: $\mathcal{G}_i = \mathfrak{L}(\theta, V_i)|_{V_{ij}}$ and $\mathcal{G}_j = \mathfrak{L}(\theta, V_j)|_{V_{ij}}$.
    *   Identify the specific edge $(X, Y)$ where $\text{sign}(w_{XY}^{(i)}) \neq \text{sign}(w_{XY}^{(j)})$ (e.g., Context A says $X \to Y$, Context B says $Y \to X$).

3.  **Resolution Strategy (Discriminator Query):**
    *   Construct a **Discriminator Prompt**:
        > "In the context of [Context A], X is claimed to cause Y. However, in [Context B], Y is claimed to cause X. Identify the latent mediating variable or condition that resolves this apparent contradiction."
    *   **Action:** Inject this query into the `active_causal_loop` with high priority (override EFE planner).

4.  **Belief Expansion (Manifold Surgery):**
    *   If the LLM identifies a latent variable $Z$, **expand the state space**:
        *   $N \leftarrow N + 1$.
        *   Augment $W$ and $L$ with a new row/column for $Z$.
    *   This effectively lifts the obstruction by embedding the cycle into a higher-dimensional DAG (resolving the projection artifact).

### Pseudocode Implementation

```python
    def _resolve_topological_obstruction(self):
        """
        Handles cases where local contexts cannot be glued (High Descent Loss).
        Corresponds to resolving a non-trivial 1-cocycle.
        """
        # 1. Identify worst offender
        worst_pair, residual = self.sheaf.find_max_residual_pair(self.theta)
        
        # 2. Construct Discriminator Query
        context_a, context_b = worst_pair
        conflict_edge = self._find_conflicting_edge(context_a, context_b)
        
        prompt = (f"Conflict detected between {context_a} and {context_b} "
                  f"regarding relationship {conflict_edge}. "
                  "Specify the latent mechanism or conditional independence "
                  "that resolves this.")
        
        # 3. Force Execution
        resolution_text = self.llm_interface.query(prompt)
        
        # 4. Manifold Surgery (Add Latent Variable)
        if "latent variable" in resolution_text:
            new_var_name = self._extract_variable_name(resolution_text)
            self._expand_state_space(new_var_name)
            # Re-initialize priors for the new variable rows/cols
            self._reinitialize_local_priors(new_var_name)
```