**Title:** HOLOGRAPH: Active Causal Discovery via Continuous Sheaf Alignment and Natural Gradient Descent

### 1. Abstract
We propose **HOLOGRAPH**, a rigorous mathematical framework for extracting and unifying causal structures from Large Language Models (LLMs). Existing paradigms (e.g., DEMOCRITUS) rely on heuristic graph expansion and conflate linguistic probability with causal mechanisms. HOLOGRAPH replaces these ad-hoc methods with a first-principles approach combining **Continuous Active Inference** and **Sheaf Theory**. We treat the LLM as a "noisy sensor" observing a latent Structural Causal Model (SCM). Unlike previous discrete search methods, we relax the causal structure into a continuous manifold of **Acyclic Directed Mixed Graphs (ADMGs)**. We model local causal claims as sections of a sheaf, where consistency between contexts is enforced via **Algebraic Latent Projections**—differentiable matrix operations that account for hidden confounders. The system evolves via **Natural Gradient Descent** on a Free Energy landscape, optimizing for both information gain and global topological consistency. This framework guarantees numerical stability through spectral regularization and provides formal bounds on internal causal consistency, moving the field from "textual visualizations" to rigorous, identifiable **Causal Priors**.

### 2. Problem Identification
The predecessor system, DEMOCRITUS, demonstrated that LLMs contain latent causal logic. However, its implementation suffered from three critical mathematical failures:

1.  **The Map-Territory Fallacy:** It equated the probability of a token sequence with the existence of a physical mechanism. It lacked a measurement model to bridge $P(\text{Text})$ and $P(\text{Causality})$.
2.  **Discrete Heuristics:** The use of Breadth-First Search (BFS) and discrete graph operations prevented the use of powerful gradient-based optimization techniques. It failed to account for the continuous nature of uncertainty in causal strength.
3.  **Topological Naivety:** By ignoring latent confounders when merging domains, previous methods violated the axioms of causal marginalization. Merging a model of "Economics" with "Politics" without accounting for unobserved common causes leads to spurious cycles, not valid causal graphs.

### 3. Theoretical Foundation
HOLOGRAPH replaces heuristics with a tripartite formal framework:

**A. The Measurement Model (Continuous Bayesian Mechanics)**
We define the LLM as an instrument yielding noisy measurements $y$ of a latent causal structure $\theta = (W, M)$, where $W$ represents directed edges and $M$ represents error covariances (bi-directed edges). We define a **Semantic Energy Function** based on the distance in a Reproducing Kernel Hilbert Space (RKHS) between the text embedding and the graph embedding, allowing us to treat causal discovery as energy minimization.

**B. Active Inference via Natural Gradient**
Instead of discrete search (MCTS), we employ **Natural Gradient Descent (NGD)**. The agent optimizes a continuous query policy to minimize **Expected Free Energy (EFE)**. To ensure stability, we define a **Continuous Interventional Metric Tensor** that remains non-singular even near the boundaries of identifiability (via Tikhonov regularization). This allows the agent to smoothly traverse the manifold of causal hypotheses, seeking regions of high epistemic value.

**C. Sheaf-Theoretic Consistency (Algebraic Latent Projections)**
We formalize "Contexts" as a topological space.
*   **Objects:** The objects in our category are **Linear SEMs** (ADMGs), not simple DAGs.
*   **Restriction:** Moving from a super-context to a sub-context is modeled by the **Algebraic Latent Projection** operator $\mathfrak{L}$. This operator uses Neumann series and Schur complements to analytically project the effects of hidden variables onto the observed subspace.
*   **Gluing:** Global consistency is enforced not by logical matching, but by minimizing the **Frobenius Descent Loss** between the projected matrices of overlapping contexts.

### 4. Methodology

#### Step 1: Differentiable Extraction
We initialize the continuous adjacency matrices $(W, M)$ using a **Semantic Initialization Functor**. Textual assertions are mapped to initial weights in the continuous ADMG space. Unlike regex extraction, this process initializes a dense, differentiable prior that allows for "soft" causal claims (e.g., $w_{ij} = 0.7$) rather than binary edges.

#### Step 2: Algebraic Sheafification
We construct the Global Causal Model by optimizing for the **Descent Condition**.
*   **Latent Projection:** For every pair of overlapping contexts $U$ and $V$, we compute the algebraic projection of their respective models onto the intersection $U \cap V$. This explicitly accounts for cross-correlations ($M_{OH}$) between observed and hidden variables.
*   **Spectral Regularization:** To ensure the numerical stability of the projection (specifically the inversion of $(I - W_{HH})$), we apply a **Spectral Radius penalty** $\mathcal{L}_{\text{spec}}$, guaranteeing that the feedback loops through hidden variables do not cause gradient explosion.

#### Step 3: Active Causal Exploration
The system iterates through a **Continuous Belief-Update Loop**:
1.  **Planner:** The agent computes the gradient of the Expected Free Energy with respect to the query parameters. It selects prompts that target edges with high variance in the posterior distribution $Q(\theta|y)$.
2.  **Update:** Upon receiving LLM feedback, the system updates the global parameters $\theta$ using **Natural Gradient Descent**. The update is preconditioned by the regularized Interventional Metric Tensor, ensuring the step size reflects the causal geometry (interventional distinctness) rather than Euclidean distance.
3.  **Constraint Satisfaction:** Throughout optimization, the **Acyclicity Constraint** $h(W) = 0$ is enforced via a Lagrangian multiplier, ensuring the final structure is a valid ADMG.

#### Step 4: Post-Hoc Identifiability Verification
Once the continuous optimization converges to a stable structure $\theta^*$, we threshold the matrices to obtain a discrete graph $\mathcal{G}^*$. We then apply the discrete **ID Algorithm** (Shpitser & Pearl) as a validation step. The system outputs the **Identification Frontier**: the specific subset of causal queries that are mathematically solvable given the learned structure.

### 5. Expected Mathematical Guarantees

1.  **Frobenius Consistency:** We guarantee that the inferred global model minimizes the discrepancy between local contexts on their overlaps, accounting for latent confounders. The global model is the "center of mass" of the local projections in the causal manifold.
2.  **Numerical Stability:** Through Spectral Stability Regularization and Tikhonov damping of the Fisher Information Matrix, we guarantee that the optimization trajectory remains bounded and non-singular, avoiding the divergences common in recursive causal models.
3.  **Internal Consistency:** We provide a formal guarantee of **Internal Consistency**. While we cannot guarantee the LLM's assertions are empirically true, we guarantee that the output set of causal effects $\mathfrak{F}$ contains *only* those estimands that are structurally identifiable within the inferred model $\mathcal{G}^*$. Unidentifiable queries are rigorously filtered out.