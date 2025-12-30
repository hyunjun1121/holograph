# HOLOGRAPH: Supplemental Materials

## Sheaf-Theoretic Framework for LLM-Guided Causal Discovery

**ICML 2026 Submission**

---

## Contents

```
holograph_supplemental/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── src/
│   └── holograph/               # Core implementation
│       ├── __init__.py
│       ├── sheaf_engine.py      # Sheaf-theoretic projection (Sec. 2.1-2.4)
│       ├── causal_state.py      # Causal state representation (Sec. 2.5-2.6)
│       ├── agent.py             # Active learning agent (Sec. 2.8)
│       ├── llm_interface.py     # LLM query interface
│       ├── query_generator.py   # EFE-based query selection (Eq. 11)
│       ├── experiment.py        # Experiment runner
│       ├── dataset_loader.py    # Dataset generation
│       ├── metrics.py           # Evaluation metrics (SHD, F1, SID)
│       └── ...
├── experiments/
│   ├── configs/                 # YAML experiment configurations
│   ├── config/
│   │   └── constants.py         # Hyperparameter definitions (Table 5)
│   ├── generate_configs.py      # Config generator script
│   └── outputs/                 # Raw experiment results (162 runs)
│       ├── E1_*/                # Main benchmarks (Table 1)
│       ├── E3_*/                # Hidden confounders (Table 4)
│       ├── E5_*/                # Rashomon stress test (Sec. 3.6)
│       ├── A1-A6_*/             # Ablation studies (Table 3)
│       └── X1-X4_*/             # Sheaf axiom validation (Table 2)
├── paper/
│   ├── data/
│   │   └── experiment_stats.json  # Aggregated statistics
│   └── scripts/
│       └── aggregate_results.py   # Result aggregation script
├── docs/
│   ├── Mathematical_Specification.md
│   ├── Control_Flow.md
│   └── experiment_foundation.md
└── tests/
    └── test_sheaf_engine.py     # Unit tests
```

---

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy, SciPy
- NetworkX
- (Optional) Access to LLM API for full experiments

---

## Reproducing Experiments

### 1. Generate Experiment Configurations

```bash
python experiments/generate_configs.py
```

This creates YAML configs in `experiments/configs/` for all experiments:
- **E1**: Main benchmarks (ER, SF, Sachs)
- **E3**: Hidden confounder experiments
- **E5**: Rashomon stress test
- **A1-A6**: Ablation studies
- **X1-X4**: Sheaf axiom validation

### 2. Run a Single Experiment

```bash
python -m holograph.run --config experiments/configs/E1_sachs.yaml --seed 42
```

### 3. Run All Experiments (SLURM)

```bash
# Generate SLURM scripts
python experiments/submit_all.sh

# Submit to cluster
sbatch experiments/slurm/submit_all.sbatch
```

### 4. Aggregate Results

```bash
python paper/scripts/aggregate_results.py
```

This generates:
- `paper/data/experiment_stats.json`: Aggregated statistics
- `paper/data/results_table.tex`: LaTeX tables

---

## Code-to-Paper Mapping

### Core Algorithms (Section 2)

| Paper Section | Equation | Implementation |
|--------------|----------|----------------|
| Sec. 2.2 Absorption Matrix | Eq. 2 | `sheaf_engine.py:135` |
| Sec. 2.2 W Projection | Eq. 3 | `sheaf_engine.py:208` |
| Sec. 2.2 M Projection | Eq. 4 | `sheaf_engine.py:211-215` |
| Sec. 2.4 Spectral Penalty | Eq. 6 | `sheaf_engine.py:297` |
| Sec. 2.5 Acyclicity | Eq. 7 | `causal_state.py:198-199` |
| Sec. 2.6 Natural Gradient | Eq. 8-9 | `causal_state.py:220-250` |
| Sec. 2.7 Total Loss | Eq. 10 | `experiment.py:180-195` |
| Sec. 2.8 EFE Query Selection | Eq. 11 | `query_generator.py:85-120` |

### Experiments (Section 3)

| Paper Table/Section | Experiment ID | Output Directory |
|--------------------|---------------|------------------|
| Table 1 (Benchmarks) | E1_* | `outputs/E1_*` |
| Table 2 (Sheaf Axioms) | X1-X4 | `outputs/X*` |
| Table 3 (Ablations) | A1-A6 | `outputs/A*` |
| Table 4 (Latent) | E3_* | `outputs/E3_*` |
| Sec. 3.6 (Rashomon) | E5 | `outputs/E5_rashomon` |

---

## Experiment Output Format

Each experiment produces the following files per seed:

```
outputs/{experiment_id}/seed_{N}/
├── results.json      # Final metrics (SHD, F1, SID, loss)
├── graph.npz         # Learned adjacency matrix
├── training.csv      # Training curve (loss per step)
└── training.log      # Detailed training log
```

### results.json Schema

```json
{
  "_metadata": {
    "experiment_id": "E1_sachs",
    "seed": 42,
    "dataset": "SACHS",
    "wall_time_seconds": 0.36
  },
  "results": {
    "shd": 16,
    "f1": 0.085,
    "sid": 18,
    "final_loss_total": 7.5e-05,
    "training_steps": 150,
    "num_queries": 11
  }
}
```

---

## Hyperparameters (Table 5)

All hyperparameters are defined in `experiments/config/constants.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MATRIX_EPSILON` | 10^-6 | Numerical stability threshold |
| `SPECTRAL_MARGIN` | 0.1 | Spectral radius constraint margin |
| `FISHER_MIN_VALUE` | 0.01 | Fisher information floor |
| `QUERY_UNCERTAINTY_THRESHOLD` | 0.3 | EFE query threshold |
| `EDGE_THRESHOLD` | 0.01 | Edge discretization threshold |
| `DISCRETIZATION_THRESHOLD` | 0.3 | Graph discretization threshold |
| `LLM_DEFAULT_TEMPERATURE` | 0.1 | LLM sampling temperature |
| `LLM_DEFAULT_MAX_TOKENS` | 4096 | Max tokens per query |

---

## Dataset Information

### Synthetic Datasets

| Dataset | Nodes | Edges | Parameters |
|---------|-------|-------|------------|
| ER (n=20) | 20 | ~40 | p=0.2 |
| ER (n=50) | 50 | ~185 | p=0.15 |
| SF (n=50) | 50 | ~100 | avg_degree=2.0 |
| Latent | 20-50 | varies | n_latent=3-8 |

### Real Dataset

**Sachs et al. (2005)**: Protein signaling network
- 11 proteins (nodes)
- 17 causal edges (ground truth)
- Single-cell flow cytometry data

---

## Verification

To verify the reported results match the raw data:

```bash
# Check aggregated statistics
python -c "
import json
with open('paper/data/experiment_stats.json') as f:
    stats = json.load(f)

# E1_er_medium results (Table 1)
e1 = stats['E1_er_medium']['metrics']
print(f'E1 ER (n=50) SHD: {e1[\"shd\"][\"mean\"]:.1f} ± {e1[\"shd\"][\"std\"]:.1f}')

# E5 Rashomon results (Section 3.6)
e5 = stats['E5_rashomon']['metrics']
print(f'E5 SHD: {e5[\"shd\"][\"mean\"]:.1f} ± {e5[\"shd\"][\"std\"]:.1f}')
"
```

Expected output:
```
E1 ER (n=50) SHD: 193.2 ± 13.6
E5 SHD: 89.8 ± 5.7
```

---

## License

This code is released under the MIT License for academic use.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{holograph2026,
  title={HOLOGRAPH: Sheaf-Theoretic Framework for LLM-Guided Causal Discovery},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
