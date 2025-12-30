# HOLOGRAPH: Experiment Foundation Document

**Generated**: 2025-12-30
**Target Venue**: ICML 2026
**Status**: Revised - Expanded Scope (Component Interactions + Sheaf Exactness)

---

## Executive Summary

HOLOGRAPH proposes a novel framework for causal discovery from LLMs combining:
1. **Continuous relaxation** of causal graphs (vs. discrete BFS in DEMOCRITUS)
2. **Sheaf-theoretic consistency** via Algebraic Latent Projections
3. **Natural Gradient Descent** for geometry-aware optimization
4. **Active query selection** via Expected Free Energy minimization

This document establishes the experimental foundation for validating these claims against state-of-the-art methods.

---

## 1. Environment Specification

| Resource | Value |
|----------|-------|
| Cluster | IZAR (EPFL) |
| GPU Type | NVIDIA V100 (Volta) |
| GPUs per Node | 2 |
| CPUs per Node | 40 |
| Memory per Node | ~192GB |
| Total GPUs Available | 146 |
| Time Limit | Unlimited |

---

## 2. Claims to Validate

### 2.1 Primary Claims (P0 - Required for Acceptance)

| ID | Claim | Type | Validation |
|----|-------|------|------------|
| **C1** | HOLOGRAPH extracts more consistent causal structures than DEMOCRITUS | Performance | SHD, F1 vs baseline on standard benchmarks |
| **C2** | Continuous relaxation + NGD converges faster than discrete BFS/MCTS | Efficiency | Wall-clock time, #LLM queries to convergence |
| **C3** | Algebraic Latent Projection correctly handles hidden confounders | Correctness | Synthetic experiments with known ground truth ADMGs |
| **C4** | Spectral regularization prevents numerical instability | Stability | Track spectral radius, gradient norms during training |

### 2.2 Secondary Claims (P1 - Strengthen Paper)

| ID | Claim | Type | Validation |
|----|-------|------|------------|
| **C5** | Frobenius Descent Loss correlates with semantic coherence | Interpretability | Correlation analysis on multi-context scenarios |
| **C6** | System identifies more causal queries via ID algorithm | Generalization | Size of Identification Frontier vs baselines |
| **C7** | Active query selection (EFE) reduces queries vs random | Efficiency | Ablation: EFE planner vs uniform sampling |

### 2.3 Implicit Claims (Reviewers Will Ask)

| ID | Claim | Validation Strategy |
|----|-------|---------------------|
| **C8** | LLM responses contain meaningful causal signal | Validate on benchmarks where ground truth exists |
| **C9** | Semantic embedding captures causal structure | Ablation: compare different embedding methods |
| **C10** | Continuous→discrete thresholding preserves identifiability | Compare ID frontier before/after thresholding |

### 2.4 Mathematical Guarantees to Validate

| ID | Theorem | Experimental Check |
|----|---------|-------------------|
| **T1** | Frobenius Descent: L_descent < ε | Show convergence curves |
| **T2** | Spectral Stability: ρ(W) < 1 | Log spectral radius throughout training |
| **T3** | Acyclicity: h(W) = Tr(e^{W∘W}) - N = 0 | Verify constraint satisfaction at convergence |
| **T4** | Internal Consistency | Post-hoc ID algorithm validation |

---

## 3. Literature Research Summary

### 3.1 Baseline Methods (Must Compare)

| Method | Type | Key Paper | Why Compare |
|--------|------|-----------|-------------|
| **DEMOCRITUS** | LLM→Causal Graph | [Mahadevan 2024](https://arxiv.org/abs/2512.07796) | Direct predecessor, discrete BFS approach |
| **NOTEARS** | Continuous DAG | [Zheng 2018](https://arxiv.org/abs/1803.01422) | Foundational continuous constraint |
| **DAGMA** | Continuous DAG | [Bello 2022](https://arxiv.org/abs/2209.08037) | SOTA continuous optimization |
| **SDCD** | Stable Diff. CD | [2024](https://arxiv.org/abs/2311.10263) | Addresses NOTEARS instability |
| **LLM-CD** | LLM for CD | [Jiralerspong 2024](https://arxiv.org/abs/2402.01207) | Efficient LLM-based CD |

### 3.2 Benchmarks

| Benchmark | Description | Use Case |
|-----------|-------------|----------|
| **CausalBench** | Comprehensive LLM causal reasoning | Main evaluation |
| **CLadder** | Causal ladder reasoning (NeurIPS 2023) | Multi-level reasoning |
| **CausalProbe 2024** | Fresh causal QA (post-Jan 2024) | Avoid data contamination |
| **Alarm/Insurance** | Classic Bayesian networks | Structure learning |
| **SACHS** | Biological network (11 vars) | Real-world validation |
| **Synthetic ADMGs** | Generated with known ground truth | Controlled experiments |

### 3.3 Evaluation Metrics

| Metric | Formula/Description | Primary Use |
|--------|---------------------|-------------|
| **SHD** | Structural Hamming Distance (insertions + deletions + reversals) | Structure accuracy |
| **F1** | Harmonic mean of edge precision/recall | Edge prediction |
| **SID** | Structural Intervention Distance | Causal effect accuracy |
| **NLL** | Negative log-likelihood on held-out data | Probabilistic fit |
| **#Queries** | Number of LLM calls to convergence | Efficiency |
| **Wall Time** | Total runtime | Practical efficiency |
| **ID Frontier Size** | Number of identifiable queries in output | Utility |

### 3.4 Related Work on Sheaf Neural Networks

- [Neural Sheaf Diffusion](https://arxiv.org/abs/2202.04579) - Bodnar et al. 2022
- [Sheaf Diffusion Goes Nonlinear](https://proceedings.mlr.press/v251/zaghen24a.html) - GRaM Workshop 2024
- Key insight: Sheaf structures improve heterophily handling in GNNs

---

## 4. Experiment Taxonomy

### 4.1 Core Comparison Experiments (P0)

| Exp ID | Name | Claim | Setup |
|--------|------|-------|-------|
| **E1** | Main Benchmark | C1 | HOLOGRAPH vs {DEMOCRITUS, NOTEARS, DAGMA, SDCD} on CausalBench |
| **E2** | Convergence Speed | C2 | #Queries and wall-time to reach fixed SHD threshold |
| **E3** | Hidden Confounder Handling | C3 | Synthetic ADMGs with latent variables |
| **E4** | Numerical Stability | C4 | Track ρ(W), gradient norms, loss NaNs |
| **E5** | **Rashomon Stress Test** | C1, C5 | 의도적 모순 Context 3개 주입 (A: X→Y, B: Y→X, C: Overlap) → Descent Loss 급증 감지 + Latent Variable 생성 성공률 측정 |

#### E5: Rashomon Stress Test - Detailed Protocol

**Purpose:** HOLOGRAPH가 모순을 "평균화"하는 것이 아니라, **감지(Detection)** 후 **해결(Resolution)**한다는 것을 입증

**Setup:**
```python
# Contradictory Context Injection
context_A = {"claim": "X causes Y", "edge": (X, Y, +1.0)}
context_B = {"claim": "Y causes X", "edge": (Y, X, +1.0)}
context_C = {"overlap": [X, Y], "no_direct_claim": True}
```

**Metrics:**
| Metric | Definition | Success Criterion |
|--------|------------|-------------------|
| **Detection Rate** | L_descent 급증 감지 (Δ > 2σ) | ≥ 95% |
| **Resolution Rate** | Latent Variable Z 제안 성공 | ≥ 70% |
| **Post-Resolution SHD** | Z 추가 후 ground truth와 비교 | Baseline 대비 ≥20% 개선 |
| **False Positive Rate** | 모순 없는 상황에서 잘못된 Z 생성 | ≤ 5% |

**Baseline Comparison:**
- DEMOCRITUS: 모순 시 어떻게 처리하는가? (예상: 임의 선택 또는 무시)
- NOTEARS/DAGMA: 구조적 모순 handling 없음 (예상: 잘못된 평균화)

### 4.2 Ablation Studies (P0)

| Exp ID | Component | Ablation | Expected Impact |
|--------|-----------|----------|-----------------|
| **A1** | Algebraic Latent Projection | → Naive marginalization | -15-30% F1 (critical) |
| **A2** | Natural Gradient Descent | → Vanilla SGD | Slower convergence, instability |
| **A3** | Spectral Regularization | Remove L_spec | NaN losses, divergence |
| **A4** | Tikhonov Damping | λ_reg = 0 | Singular FIM, crashes |
| **A5** | EFE Planner | → Random queries | 2-3x more queries needed |
| **A6** | Acyclicity Constraint | Remove h(W) | Invalid graphs (cycles) |

### 4.3 Component Interaction Ablations (P0 - Extended)

Study pairwise synergies between components. Each row removes two components simultaneously.

| Exp ID | Components Removed | Hypothesis | Expected Outcome |
|--------|-------------------|------------|------------------|
| **I1** | ALP + NGD | Core synergy | Worse than sum of individual ablations (super-additive) |
| **I2** | ALP + Spectral Reg | Stability coupling | ALP needs spectral reg for stable projections |
| **I3** | NGD + Tikhonov | Geometry dependencies | NGD requires damping; failure without it |
| **I4** | EFE + ALP | Query-structure synergy | Active queries target sheaf inconsistencies |
| **I5** | Spectral Reg + Acyclicity | Constraint interaction | Both needed for valid optimization landscape |
| **I6** | NGD + EFE | Optimization-planning | Independent contributions (additive) |

**Analysis**: Compute interaction effect = I_xy - (A_x + A_y - baseline)
- Positive interaction → components are synergistic
- Negative interaction → components are redundant
- Near-zero → independent contributions

### 4.4 Sheaf Exactness Validation (P0 - Theoretical)

Validate that the Algebraic Latent Projection satisfies presheaf axioms empirically.

| Exp ID | Axiom | Validation Method | Success Criterion |
|--------|-------|-------------------|-------------------|
| **X1** | Identity: ρ_UU = id | Measure ‖ρ_UU(θ) - θ‖_F | < 1e-6 (numerical precision) |
| **X2** | Transitivity: ρ_ZU = ρ_ZV ∘ ρ_VU | Measure ‖ρ_ZU(θ) - ρ_ZV(ρ_VU(θ))‖_F | < ε (descent threshold) |
| **X3** | Locality: restriction respects overlaps | ρ_VU(θ)\|_{V∩W} = ρ_WU(θ)\|_{V∩W} | Frobenius norm < ε |
| **X4** | Gluing: global section from local | Reconstruct global θ from local restrictions | SHD(θ, θ_reconstructed) = 0 |

**Experimental Setup for X1-X4:**
```python
# For each context hierarchy U ⊃ V ⊃ Z:
contexts = generate_nested_contexts(N_vars=50, depths=[3, 5, 10])
for U, V, Z in context_triples:
    # X1: Identity
    assert frobenius_norm(rho(theta, U, U) - theta) < 1e-6

    # X2: Transitivity
    direct = rho(theta, U, Z)
    composed = rho(rho(theta, U, V), V, Z)
    transitivity_error = frobenius_norm(direct - composed)

    # X3: Locality (on overlaps)
    W = another_context_overlapping_V
    locality_error = frobenius_norm(
        restrict(rho(theta, U, V), V_cap_W) -
        restrict(rho(theta, U, W), V_cap_W)
    )

    # X4: Gluing
    local_sections = {ctx: rho(theta, global_ctx, ctx) for ctx in contexts}
    theta_glued = glue(local_sections)  # Minimize Frobenius descent
    gluing_error = SHD(discretize(theta), discretize(theta_glued))
```

**Metrics for Sheaf Exactness:**
| Metric | Definition | Target |
|--------|------------|--------|
| **Transitivity Error** | max_triples ‖ρ_ZU - ρ_ZV∘ρ_VU‖_F | < 0.01 |
| **Locality Error** | max_overlaps ‖ρ_i\|_O - ρ_j\|_O‖_F | < ε |
| **Gluing Residual** | L_descent at convergence | < ε |
| **Cocycle Obstruction** | H¹(X, F) approximation via residuals | Report magnitude |

### 4.5 Scaling Experiments (P1)

| Exp ID | Dimension | Range | Metric |
|--------|-----------|-------|--------|
| **S1** | Graph Size | N ∈ {10, 25, 50, 100, 200} | SHD, Runtime |
| **S2** | Context Count | K ∈ {2, 4, 8, 16} | Descent Loss, Coherence |
| **S3** | LLM Size | {GPT-3.5, GPT-4, Claude-3} | Causal signal quality |

### 4.6 Robustness Experiments (P1)

| Exp ID | Factor | Setup |
|--------|--------|-------|
| **R1** | Random Seeds | 5 seeds per configuration |
| **R2** | Hyperparameter Sensitivity | Grid over {lr, λ_descent, λ_spec, λ_reg} |
| **R3** | Noise in LLM Responses | Inject random contradictions |

### 4.7 Analysis Experiments (P2)

| Exp ID | Analysis | Visualization |
|--------|----------|---------------|
| **V1** | Loss Landscape | 2D slices of L_total around optima |
| **V2** | Descent Loss vs Coherence | Scatter plot with correlation |
| **V3** | Identification Frontier | Bar chart comparing methods |
| **V4** | Convergence Trajectories | SHD vs #queries curves |
| **V5** | Obstruction Resolution | Case study of contradiction handling |

---

## 5. Resource Planning

### 5.1 Compute Budget Estimate (Expanded)

| Experiment Group | GPU Type | Hours/Run | Runs | Seeds | Total GPU-Hours |
|------------------|----------|-----------|------|-------|-----------------|
| E1: Main Benchmark | V100 | 8 | 5 methods × 4 datasets | 5 | 800 |
| E2: Convergence | V100 | 4 | 5 methods × 4 datasets | 3 | 240 |
| E3: Hidden Confounders | V100 | 6 | 5 methods × 3 settings | 5 | 450 |
| E4: Stability | V100 | 2 | 1 method × 10 configs | 3 | 60 |
| **E5: Rashomon Stress** | V100 | 4 | 5 methods × 5 contradiction scenarios | 5 | **500** |
| A1-A6: Single Ablations | V100 | 4 | 6 ablations × 2 datasets | 5 | 240 |
| **I1-I6: Interaction Ablations** | V100 | 6 | 6 pairs × 2 datasets | 5 | **360** |
| **X1-X4: Sheaf Exactness** | V100 | 4 | 4 axioms × 3 context depths | 5 | **240** |
| S1-S3: Scaling | V100 | 12 | 15 configurations | 3 | 540 |
| R1-R3: Robustness | V100 | 4 | 20 configurations | 5 | 400 |
| V1-V5: Analysis | V100 | 2 | 10 analyses | 1 | 20 |
| **Subtotal** | | | | | **3,850** |
| **Buffer (20%)** | | | | | **770** |
| **Total** | | | | | **4,620** |

### 5.2 LLM API Budget

| LLM | Est. Queries | Cost/1K queries | Total |
|-----|--------------|-----------------|-------|
| GPT-4 | 50,000 | ~$30 | ~$1,500 |
| GPT-3.5-turbo | 200,000 | ~$2 | ~$400 |
| Claude-3 (scaling) | 20,000 | ~$15 | ~$300 |
| **Total** | | | **~$2,200** |

### 5.3 Execution Priority (Final)

```
Priority 0 (Week 1-2): E1, E3, E5, A1, A3, X1-X4 (core claims + Rashomon + sheaf)
Priority 1 (Week 2-3): E2, E4, A2, A4-A6, I1-I3 (supporting + key interactions)
Priority 2 (Week 3-4): I4-I6, S1-S3, R1-R3 (remaining interactions + scaling)
Priority 3 (Week 4-5): V1-V5 (analysis & visualization)
```

---

## 6. SLURM Configuration Template

```bash
#!/bin/bash
#SBATCH --job-name=holograph_${EXP_ID}
#SBATCH --output=experiments/slurm/logs/%x_%A_%a.out
#SBATCH --error=experiments/slurm/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --gres=gpu:volta:1
#SBATCH --time=24:00:00
#SBATCH --array=0-4  # 5 seeds

# Environment
source ~/.bashrc
conda activate holograph

# Configuration
SEED=$((42 + SLURM_ARRAY_TASK_ID))
CONFIG_FILE="experiments/configs/${EXP_ID}.yaml"

# Run with checkpoint support
python -m holograph.run \
    --config ${CONFIG_FILE} \
    --seed ${SEED} \
    --output_dir experiments/outputs/${EXP_ID}/seed_${SEED} \
    --resume_if_exists
```

---

## 7. Output Format Specification

### 7.1 Self-Documenting Results JSON

```json
{
  "_metadata": {
    "experiment_id": "E1_benchmark_causalbench",
    "description": "Main benchmark comparison on CausalBench",
    "hypothesis": "HOLOGRAPH achieves lower SHD than DEMOCRITUS",
    "claims_supported": ["C1"],
    "config": {
      "method": "holograph",
      "dataset": "causalbench",
      "lr": 0.01,
      "lambda_descent": 1.0,
      "lambda_spec": 0.1,
      "lambda_reg": 1e-4
    },
    "git_commit": "abc123",
    "slurm_job_id": "12345678",
    "seed": 42,
    "timestamp": "2025-12-30T10:00:00Z"
  },
  "results": {
    "shd": 12.4,
    "f1": 0.847,
    "sid": 8.2,
    "num_queries": 1250,
    "wall_time_seconds": 3600,
    "id_frontier_size": 45,
    "final_loss": {
      "total": 0.032,
      "semantic": 0.015,
      "descent": 0.008,
      "acyclic": 0.001,
      "spectral": 0.008
    },
    "convergence": {
      "epochs": 150,
      "final_spectral_radius": 0.87
    }
  },
  "artifacts": {
    "learned_graph": "outputs/E1_benchmark_causalbench/seed_42/graph.npz",
    "training_log": "outputs/E1_benchmark_causalbench/seed_42/training.csv",
    "checkpoint": "outputs/E1_benchmark_causalbench/seed_42/checkpoint.pt"
  }
}
```

### 7.2 Directory Structure

```
experiments/
├── configs/
│   ├── E1_benchmark_causalbench.yaml
│   ├── E2_convergence.yaml
│   └── ...
├── outputs/
│   └── E1_benchmark_causalbench/
│       ├── seed_42/
│       │   ├── results.json
│       │   ├── graph.npz
│       │   ├── training.csv
│       │   └── checkpoint.pt
│       └── ...
├── slurm/
│   ├── scripts/
│   └── logs/
└── analysis/
    ├── tables/
    └── figures/
```

---

## 8. Key Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| LLM API rate limits | High | Medium | Implement exponential backoff, cache responses |
| NOTEARS instability at scale | Medium | High | Use SDCD constraint as backup |
| Sheaf projection numerical issues | Medium | High | SVD-based pseudoinverse fallback |
| Ground truth unavailable for real data | High | Medium | Rely on synthetic + proxy metrics |
| Context contradictions too frequent | Medium | Medium | Robust obstruction resolution module |

---

## 9. Success Criteria

### For ICML 2026 Acceptance:

**Core Claims:**
- [ ] **E1**: HOLOGRAPH achieves ≥10% lower SHD than DEMOCRITUS on 3+ benchmarks
- [ ] **E2**: HOLOGRAPH converges in ≤50% the queries of discrete methods
- [ ] **E3**: Algebraic projection shows clear advantage on ADMG benchmarks
- [ ] **E5**: Rashomon Stress Test - Detection ≥95%, Resolution ≥70%, FPR ≤5%
- [ ] **A1-A6**: All ablations confirm component necessity

**Component Interactions (New):**
- [ ] **I1-I6**: At least 3 pairs show super-additive effects (synergy confirmed)
- [ ] **I1**: ALP + NGD synergy is the strongest (core design validated)

**Sheaf Theory Validation (New):**
- [ ] **X1**: Identity axiom holds to numerical precision (< 1e-6)
- [ ] **X2**: Transitivity error < ε for all context triples
- [ ] **X3**: Locality error < ε for all overlapping contexts
- [ ] **X4**: Gluing produces SHD = 0 reconstruction

**Scaling & Robustness:**
- [ ] **S1**: Method scales to N≥100 variables
- [ ] **R1**: Results consistent across 5 seeds (std < 10% of mean)

---

## 10. References

1. [Large Causal Models from LLMs (DEMOCRITUS)](https://arxiv.org/abs/2512.07796) - Mahadevan 2024
2. [NOTEARS: Continuous DAG Learning](https://arxiv.org/abs/1803.01422) - Zheng 2018
3. [DAGMA: DAGs via Algebraic Manifolds](https://arxiv.org/abs/2209.08037) - Bello 2022
4. [Stable Differentiable Causal Discovery](https://arxiv.org/abs/2311.10263) - 2024
5. [CausalBench for LLMs](https://arxiv.org/abs/2404.06349) - 2024
6. [CLadder Benchmark](https://arxiv.org/abs/2312.04350) - NeurIPS 2023
7. [Neural Sheaf Diffusion](https://arxiv.org/abs/2202.04579) - Bodnar 2022
8. [Unveiling Causal Reasoning in LLMs](https://neurips.cc/virtual/2024/poster/96872) - NeurIPS 2024
9. [Natural Gradient Methods](https://jmlr.org/papers/v21/17-678.html) - JMLR 2020

---

## APPROVAL CHECKPOINT (Final)

**Final Scope Summary:**
- **E5: Rashomon Stress Test** - 모순 감지 및 해결 능력 검증 (핵심 차별점)
- **I1-I6**: Component Interaction Ablations (pairwise synergy analysis)
- **X1-X4**: Sheaf Exactness Validation (presheaf axiom verification)
- **Compute budget**: **~4,620 GPU-hours**
- **Target venue**: **ICML 2026**

---

**STATUS: APPROVED** - Proceeding to Phase 5 (Implementation)
