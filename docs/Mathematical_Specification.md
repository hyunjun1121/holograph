Here is the corrected **Mathematical Specification**.

This version resolves the discontinuity in the optimization landscape by defining the **Interventional Metric Tensor** over the set of all atomic interventions, rendering the manifold smooth. It also reframes the final corollary to claim **Internal Consistency** rather than absolute truth-conditional soundness, acknowledging the dependence on the learned structure.

***

\section{Mathematical Specification}

\subsection{The Sheaf of Causal Contexts via General Algebraic Projections}

We model the global causal structure as a sheaf of Linear Structural Equation Models (LSEMs). To satisfy the axioms of a presheaf under the category of ADMGs, the restriction map must account for existing correlations between observed and hidden variables when projecting further.

\begin{definition}[The Category of Linear SEMs]
Let $\mathbf{LSEM}$ be the category where objects are tuples $\theta = (W, M)$, with $W \in \mathbb{R}^{N \times N}$ representing directed coefficients and $M \in \mathbb{R}^{N \times N}$ (symmetric, positive semi-definite) representing the error covariance.
\end{definition}

\begin{definition}[The General Algebraic Presheaf]
We define the presheaf $\mathcal{F}: \mathfrak{C}^{\text{op}} \to \mathbf{LSEM}$. For a refinement $V \xrightarrow{\iota} U$, partition the variables into Observed ($O$) and Hidden ($H$). The restriction map $\rho_{UV}$ is the **General Algebraic Latent Projection**:
$$ \rho_{UV}(W_U, M_U) = (\tilde{W}, \tilde{M}) $$
Let $A = W_{OH}(I - W_{HH})^{-1}$ be the \textit{absorption matrix}. The projected structures are:
\begin{align*}
\tilde{W} &= W_{OO} + A W_{HO} \\
\tilde{M} &= M_{OO} + A M_{HH} A^T + M_{OH} A^T + A M_{HO}
\end{align*}
The inclusion of the cross-terms $M_{OH} A^T + A M_{HO}$ ensures that the projection is closed under composition ($\rho_{ZU} = \rho_{ZV} \circ \rho_{VU}$), satisfying the transitivity axiom required for $\mathcal{F}$ to be a valid presheaf.
\end{definition}

\begin{axiom}[Frobenius Descent Condition]
We enforce consistency on the intersection of contexts $V_{ij}$ using the Frobenius norm on the projected matrices:
$$ \mathcal{L}_{\text{descent}} = \sum_{i,j} \left( \| \tilde{W}_i|_{V_{ij}} - \tilde{W}_j|_{V_{ij}} \|_F^2 + \| \tilde{M}_i|_{V_{ij}} - \tilde{M}_j|_{V_{ij}} \|_F^2 \right) < \epsilon $$
\end{axiom}

\subsection{Bayesian Mechanics via Continuous Relaxation}

We employ a continuous relaxation of the causal structure. To ensure the invertibility of $(I - W_{HH})$ required for the projection, we enforce spectral stability.

\begin{definition}[Continuous Causal State]
Let $W \in \mathbb{R}^{N \times N}$ be the weighted adjacency matrix. Let $L \in \mathbb{R}^{N \times N}$ be lower-triangular such that $M = L L^T$.
The structural constraint $h(W)$ ensures global acyclicity:
$$ h(W) = \text{Tr}(e^{W \circ W}) - N = 0 $$
\end{definition}

\begin{definition}[Spectral Stability Regularization]
To prevent gradient explosion during the computation of the Neumann series $(I - W_{HH})^{-1}$, we impose a penalty on the spectral radius $\rho(W)$:
$$ \mathcal{L}_{\text{spec}}(W) = \max(0, \rho(W) - 1 + \delta)^2 $$
where $\delta > 0$ is a safety margin. This ensures that the spectral radius remains strictly below 1 throughout the optimization.
\end{definition}

\begin{definition}[Semantic Energy Function]
The likelihood of observing text $y$ given structure $\theta = (W, L)$ is:
$$ P(y | \theta) = \frac{1}{Z} \exp\left( -\beta \| \phi(y) - \Psi(W, L L^T) \|^2_{\mathcal{H}} \right) $$
\end{definition}

\subsection{Active Inference and Epistemic Foraging}

The agent optimizes a query policy $\pi$ to minimize the Expected Free Energy.

\begin{theorem}[Optimal Query Policy]
The optimal policy $\pi^*$ minimizes the functional $G(\pi)$ using a Lagrangian formulation:
$$ G(\pi) = \mathbb{E}_{Q(y|\pi)} \left[ D_{KL}(Q(\theta|y) || Q(\theta)) + \lambda_1 \mathcal{L}_{\text{descent}} + \lambda_2 \mathcal{L}_{\text{spec}} \right] + \nu h(\mathbb{E}[W]) $$
This objective balances epistemic value (KL divergence) against global consistency ($\mathcal{L}_{\text{descent}}$), numerical stability ($\mathcal{L}_{\text{spec}}$), and topological validity ($h(W)$).
\end{theorem}

\subsection{Continuous Interventional Information Geometry}

To ensure a smooth optimization landscape, we define the metric tensor over the full space of atomic interventions, independent of the current discrete identifiability status.

\begin{definition}[Continuous Interventional Metric]
Let $\mathcal{V}$ be the set of all variables in the system. The metric tensor $G(\theta)$ is defined as the sum of Fisher Information Matrices over all possible atomic interventions $do(X=x)$ for all $X \in \mathcal{V}$:
$$ G(\theta) = \sum_{X \in \mathcal{V}} \mathbb{E}_{y} \left[ (\nabla_\theta \ln P(y|do(X);\theta))(\nabla_\theta \ln P(y|do(X);\theta))^T \right] + \lambda_{\text{reg}} I $$
This definition is strictly continuous with respect to $\theta$. The Tikhonov damping term $\lambda_{\text{reg}} I$ ensures that the matrix remains invertible even in regions of the parameter space where certain effects are unidentifiable (i.e., where the Fisher Information is singular), allowing Natural Gradient Descent to traverse these regions stably.
\end{definition}

\subsection{Post-Hoc Identifiability Verification}

The discrete topological analysis is applied strictly as a post-processing step after the continuous optimization has converged.

\begin{definition}[The Identification Frontier]
Let $\mathcal{G}^*$ be the discretized ADMG obtained by thresholding the converged parameters $\theta^*$. The Identification Frontier $\mathfrak{F}$ is the set of queries $Q$ for which the discrete **ID Algorithm** (Shpitser & Pearl) returns a valid estimand given $\mathcal{G}^*$.
\end{definition}

\begin{corollary}[Internal Consistency]
The system guarantees that the output set $\mathfrak{F}$ is internally consistent with the learned structure $\mathcal{G}^*$. Specifically, if a query is in $\mathfrak{F}$, there exists a valid adjustment formula derived from $\mathcal{G}^*$. We do not guarantee that $\mathcal{G}^*$ corresponds to the ground truth causal mechanism, only that the reported estimands are mathematically solvable within the inferred model.
\end{corollary}