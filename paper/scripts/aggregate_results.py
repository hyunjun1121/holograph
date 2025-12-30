#!/usr/bin/env python3
"""
Aggregate experiment results across seeds for HOLOGRAPH ICML 2026 paper.

Generates:
- paper/data/experiment_stats.json: Full statistics
- paper/data/results_table.tex: LaTeX table rows
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "experiments" / "outputs"
DATA_DIR = PROJECT_ROOT / "paper" / "data"


def load_all_results():
    """Load all results.json files from experiments/outputs/."""
    results = defaultdict(list)

    for exp_dir in OUTPUTS_DIR.iterdir():
        if not exp_dir.is_dir():
            continue
        if exp_dir.name in ("dry_run", "wet_run"):
            continue

        exp_id = exp_dir.name

        for seed_dir in exp_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue

            results_file = seed_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        data = json.load(f)
                    data["_seed_dir"] = str(seed_dir)
                    results[exp_id].append(data)
                except Exception as e:
                    print(f"Error loading {results_file}: {e}")

    return results


def compute_stats(values):
    """Compute mean and std for a list of values."""
    if not values:
        return {"mean": None, "std": None, "n": 0}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(arr)
    }


def aggregate_performance_metrics(exp_results):
    """Aggregate performance metrics (SHD, F1, SID) for E/A experiments."""
    shd_vals = []
    f1_vals = []
    sid_vals = []
    precision_vals = []
    recall_vals = []
    wall_time_vals = []
    loss_total_vals = []
    spectral_vals = []
    acyclicity_vals = []
    num_queries_vals = []
    training_steps_vals = []

    for r in exp_results:
        res = r.get("results", {})
        if "shd" in res:
            shd_vals.append(res["shd"])
        if "f1" in res:
            f1_vals.append(res["f1"])
        if "sid" in res:
            sid_vals.append(res["sid"])
        if "precision" in res:
            precision_vals.append(res["precision"])
        if "recall" in res:
            recall_vals.append(res["recall"])
        if "wall_time_seconds" in res:
            wall_time_vals.append(res["wall_time_seconds"])
        if "num_queries" in res:
            num_queries_vals.append(res["num_queries"])
        if "training_steps" in res:
            training_steps_vals.append(res["training_steps"])

        # Loss components
        loss = res.get("final_loss", {})
        if "total" in loss:
            loss_total_vals.append(loss["total"])
        if "spectral_radius" in loss:
            spectral_vals.append(loss["spectral_radius"])
        if "acyclicity" in loss:
            acyclicity_vals.append(loss["acyclicity"])

    return {
        "shd": compute_stats(shd_vals),
        "f1": compute_stats(f1_vals),
        "sid": compute_stats(sid_vals),
        "precision": compute_stats(precision_vals),
        "recall": compute_stats(recall_vals),
        "wall_time_seconds": compute_stats(wall_time_vals),
        "num_queries": compute_stats(num_queries_vals),
        "training_steps": compute_stats(training_steps_vals),
        "final_loss_total": compute_stats(loss_total_vals),
        "final_spectral_radius": compute_stats(spectral_vals),
        "final_acyclicity": compute_stats(acyclicity_vals),
    }


def aggregate_sheaf_metrics(exp_results):
    """Aggregate sheaf axiom metrics for X experiments."""
    identity_max = []
    identity_mean = []
    transitivity_max = []
    transitivity_mean = []
    locality_max = []
    locality_mean = []
    gluing_max = []
    gluing_mean = []

    identity_passes = []
    transitivity_passes = []
    locality_passes = []
    gluing_passes = []

    wall_time_vals = []
    loss_total_vals = []

    for r in exp_results:
        res = r.get("results", {})

        if "identity_max" in res:
            identity_max.append(res["identity_max"])
        if "identity_mean" in res:
            identity_mean.append(res["identity_mean"])
        if "transitivity_max" in res:
            transitivity_max.append(res["transitivity_max"])
        if "transitivity_mean" in res:
            transitivity_mean.append(res["transitivity_mean"])
        if "locality_max" in res:
            locality_max.append(res["locality_max"])
        if "locality_mean" in res:
            locality_mean.append(res["locality_mean"])
        if "gluing_max" in res:
            gluing_max.append(res["gluing_max"])
        if "gluing_mean" in res:
            gluing_mean.append(res["gluing_mean"])

        if "passes_identity_passes" in res:
            identity_passes.append(res["passes_identity_passes"])
        if "passes_transitivity_passes" in res:
            transitivity_passes.append(res["passes_transitivity_passes"])
        if "passes_locality_passes" in res:
            locality_passes.append(res["passes_locality_passes"])
        if "passes_gluing_passes" in res:
            gluing_passes.append(res["passes_gluing_passes"])

        if "wall_time_seconds" in res:
            wall_time_vals.append(res["wall_time_seconds"])

        loss = res.get("final_loss", {})
        if "total" in loss:
            loss_total_vals.append(loss["total"])

    return {
        "identity_max": compute_stats(identity_max),
        "identity_mean": compute_stats(identity_mean),
        "transitivity_max": compute_stats(transitivity_max),
        "transitivity_mean": compute_stats(transitivity_mean),
        "locality_max": compute_stats(locality_max),
        "locality_mean": compute_stats(locality_mean),
        "gluing_max": compute_stats(gluing_max),
        "gluing_mean": compute_stats(gluing_mean),
        "identity_pass_rate": sum(identity_passes) / len(identity_passes) if identity_passes else None,
        "transitivity_pass_rate": sum(transitivity_passes) / len(transitivity_passes) if transitivity_passes else None,
        "locality_pass_rate": sum(locality_passes) / len(locality_passes) if locality_passes else None,
        "gluing_pass_rate": sum(gluing_passes) / len(gluing_passes) if gluing_passes else None,
        "wall_time_seconds": compute_stats(wall_time_vals),
        "final_loss_total": compute_stats(loss_total_vals),
    }


def get_experiment_type(exp_id):
    """Determine experiment type from ID."""
    if exp_id.startswith("X"):
        return "sheaf"
    elif exp_id.startswith("A"):
        return "ablation"
    elif exp_id.startswith("E"):
        return "main"
    return "unknown"


def format_stat(stat, fmt=".2f"):
    """Format a stat as mean ± std."""
    if stat is None or stat.get("mean") is None:
        return "—"
    mean = stat["mean"]
    std = stat["std"]
    if fmt == ".2f":
        return f"{mean:.2f} ± {std:.2f}"
    elif fmt == ".3f":
        return f"{mean:.3f} ± {std:.3f}"
    elif fmt == ".1e":
        return f"{mean:.1e} ± {std:.1e}"
    elif fmt == ".0f":
        return f"{mean:.0f} ± {std:.0f}"
    return f"{mean} ± {std}"


def generate_latex_tables(aggregated):
    """Generate LaTeX table content from aggregated results."""
    lines = []

    # Table 1: Main Benchmark Results (E1 experiments)
    lines.append("% Table 1: Main Benchmark Results")
    lines.append("% Experiment & Dataset & SHD ↓ & F1 ↑ & SID ↓ & Time (s) \\\\")
    lines.append("")

    for exp_id in sorted(aggregated.keys()):
        if not exp_id.startswith("E1"):
            continue
        stats = aggregated[exp_id]
        if "metrics" not in stats:
            continue
        m = stats["metrics"]
        dataset = stats.get("dataset", exp_id)

        shd = format_stat(m.get("shd"), ".1f") if m.get("shd") else "—"
        f1 = format_stat(m.get("f1"), ".3f") if m.get("f1") else "—"
        sid = format_stat(m.get("sid"), ".1f") if m.get("sid") else "—"
        time_s = format_stat(m.get("wall_time_seconds"), ".1f") if m.get("wall_time_seconds") else "—"

        # Clean up the shd/f1/sid format
        shd_mean = m.get("shd", {}).get("mean", "—")
        shd_std = m.get("shd", {}).get("std", 0)
        f1_mean = m.get("f1", {}).get("mean", 0)
        f1_std = m.get("f1", {}).get("std", 0)
        sid_mean = m.get("sid", {}).get("mean", "—")
        sid_std = m.get("sid", {}).get("std", 0)
        time_mean = m.get("wall_time_seconds", {}).get("mean", "—")
        time_std = m.get("wall_time_seconds", {}).get("std", 0)

        if isinstance(shd_mean, (int, float)):
            shd_str = f"${shd_mean:.1f} \\pm {shd_std:.1f}$"
        else:
            shd_str = "—"

        if isinstance(f1_mean, (int, float)):
            f1_str = f"${f1_mean:.3f} \\pm {f1_std:.3f}$"
        else:
            f1_str = "—"

        if isinstance(sid_mean, (int, float)):
            sid_str = f"${sid_mean:.1f} \\pm {sid_std:.1f}$"
        else:
            sid_str = "—"

        if isinstance(time_mean, (int, float)):
            time_str = f"${time_mean:.1f} \\pm {time_std:.1f}$"
        else:
            time_str = "—"

        lines.append(f"HOLOGRAPH & {dataset} & {shd_str} & {f1_str} & {sid_str} & {time_str} \\\\")

    lines.append("")
    lines.append("% Table 2: Ablation Results")
    lines.append("% Variant & SHD ↓ & Loss ↓ & Queries \\\\")
    lines.append("")

    for exp_id in sorted(aggregated.keys()):
        if not exp_id.startswith("A"):
            continue
        stats = aggregated[exp_id]
        if "metrics" not in stats:
            continue
        m = stats["metrics"]
        desc = stats.get("description", exp_id)

        shd_mean = m.get("shd", {}).get("mean", "—")
        shd_std = m.get("shd", {}).get("std", 0)
        loss_mean = m.get("final_loss_total", {}).get("mean", "—")
        loss_std = m.get("final_loss_total", {}).get("std", 0)
        queries_mean = m.get("num_queries", {}).get("mean", 0)

        if isinstance(shd_mean, (int, float)):
            shd_str = f"${shd_mean:.1f} \\pm {shd_std:.1f}$"
        else:
            shd_str = "—"

        if isinstance(loss_mean, (int, float)):
            loss_str = f"${loss_mean:.2e}$"
        else:
            loss_str = "—"

        if isinstance(queries_mean, (int, float)):
            queries_str = f"${queries_mean:.0f}$"
        else:
            queries_str = "—"

        # Shorten description
        short_desc = desc.replace("Ablation: ", "").replace(" on er_medium", "").replace(" on sachs", "")

        lines.append(f"{exp_id} ({short_desc}) & {shd_str} & {loss_str} & {queries_str} \\\\")

    lines.append("")
    lines.append("% Table 3: Sheaf Axiom Verification")
    lines.append("% Experiment & n & Identity & Transitivity & Locality & Gluing \\\\")
    lines.append("")

    for exp_id in sorted(aggregated.keys()):
        if not exp_id.startswith("X"):
            continue
        stats = aggregated[exp_id]
        if "metrics" not in stats:
            continue
        m = stats["metrics"]

        # Extract n from experiment ID (e.g., X1_n50_d5 -> n=50)
        n_vars = "—"
        if "_n" in exp_id:
            try:
                n_vars = exp_id.split("_n")[1].split("_")[0]
            except:
                pass

        id_rate = m.get("identity_pass_rate")
        tr_rate = m.get("transitivity_pass_rate")
        loc_rate = m.get("locality_pass_rate")
        gl_rate = m.get("gluing_pass_rate")

        id_str = f"\\checkmark ({id_rate*100:.0f}\\%)" if id_rate == 1.0 else f"$\\times$ ({id_rate*100:.0f}\\%)" if id_rate is not None else "—"
        tr_str = f"\\checkmark ({tr_rate*100:.0f}\\%)" if tr_rate == 1.0 else f"$\\times$ ({tr_rate*100:.0f}\\%)" if tr_rate is not None else "—"
        loc_str = f"\\checkmark ({loc_rate*100:.0f}\\%)" if loc_rate == 1.0 else f"$\\times$ ({loc_rate*100:.0f}\\%)" if loc_rate is not None else "—"
        gl_str = f"\\checkmark ({gl_rate*100:.0f}\\%)" if gl_rate == 1.0 else f"$\\times$ ({gl_rate*100:.0f}\\%)" if gl_rate is not None else "—"

        # Add locality error for context
        loc_max = m.get("locality_max", {})
        if loc_max.get("mean") is not None:
            loc_str += f" (err: {loc_max['mean']:.2f})"

        lines.append(f"{exp_id} & n={n_vars} & {id_str} & {tr_str} & {loc_str} & {gl_str} \\\\")

    lines.append("")
    lines.append("% Table 4: Hidden Confounder Experiments (E3)")
    lines.append("% Config & Observed & Latent & SHD ↓ & SID ↓ & Time (s) \\\\")
    lines.append("")

    for exp_id in sorted(aggregated.keys()):
        if not exp_id.startswith("E3"):
            continue
        stats = aggregated[exp_id]
        if "metrics" not in stats:
            continue
        m = stats["metrics"]

        # Parse config from exp_id (e.g., E3_latent_20o_3l)
        parts = exp_id.split("_")
        n_obs = parts[2].replace("o", "") if len(parts) > 2 else "—"
        n_lat = parts[3].replace("l", "") if len(parts) > 3 else "—"

        shd_mean = m.get("shd", {}).get("mean", "—")
        shd_std = m.get("shd", {}).get("std", 0)
        sid_mean = m.get("sid", {}).get("mean", "—")
        sid_std = m.get("sid", {}).get("std", 0)
        time_mean = m.get("wall_time_seconds", {}).get("mean", "—")
        time_std = m.get("wall_time_seconds", {}).get("std", 0)

        if isinstance(shd_mean, (int, float)):
            shd_str = f"${shd_mean:.1f} \\pm {shd_std:.1f}$"
        else:
            shd_str = "—"

        if isinstance(sid_mean, (int, float)):
            sid_str = f"${sid_mean:.1f} \\pm {sid_std:.1f}$"
        else:
            sid_str = "—"

        if isinstance(time_mean, (int, float)):
            time_str = f"${time_mean:.1f} \\pm {time_std:.1f}$"
        else:
            time_str = "—"

        lines.append(f"{exp_id} & {n_obs} & {n_lat} & {shd_str} & {sid_str} & {time_str} \\\\")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("HOLOGRAPH Results Aggregation")
    print("=" * 60)

    # Load all results
    print("\nLoading results from experiments/outputs/...")
    results = load_all_results()
    print(f"Found {len(results)} experiment types")

    # Aggregate by experiment
    aggregated = {}

    for exp_id, exp_results in results.items():
        exp_type = get_experiment_type(exp_id)

        # Get metadata from first result
        first = exp_results[0] if exp_results else {}
        metadata = first.get("_metadata", {})

        entry = {
            "experiment_id": exp_id,
            "experiment_type": exp_type,
            "description": metadata.get("description", ""),
            "hypothesis": metadata.get("hypothesis", ""),
            "claims_supported": metadata.get("claims_supported", []),
            "dataset": metadata.get("dataset", ""),
            "n_seeds": len(exp_results),
            "seeds": [r.get("_metadata", {}).get("seed") for r in exp_results],
        }

        # Aggregate metrics based on type
        if exp_type == "sheaf":
            entry["metrics"] = aggregate_sheaf_metrics(exp_results)
        else:
            entry["metrics"] = aggregate_performance_metrics(exp_results)

        aggregated[exp_id] = entry

        print(f"  {exp_id}: {len(exp_results)} seeds, type={exp_type}")

    # Save JSON
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "experiment_stats.json"
    with open(json_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Generate and save LaTeX
    latex_content = generate_latex_tables(aggregated)
    tex_path = DATA_DIR / "results_table.tex"
    with open(tex_path, "w") as f:
        f.write(latex_content)
    print(f"Saved: {tex_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Count by type
    type_counts = defaultdict(int)
    for exp_id in aggregated:
        type_counts[get_experiment_type(exp_id)] += 1

    print(f"\nExperiment counts:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c} experiments")

    # Key findings
    print("\n--- Key Findings ---")

    # Sheaf axiom summary
    print("\nSheaf Axiom Pass Rates (X experiments):")
    for exp_id in sorted(aggregated.keys()):
        if not exp_id.startswith("X"):
            continue
        m = aggregated[exp_id].get("metrics", {})
        id_rate = m.get("identity_pass_rate", 0) * 100 if m.get("identity_pass_rate") else 0
        tr_rate = m.get("transitivity_pass_rate", 0) * 100 if m.get("transitivity_pass_rate") else 0
        loc_rate = m.get("locality_pass_rate", 0) * 100 if m.get("locality_pass_rate") else 0
        gl_rate = m.get("gluing_pass_rate", 0) * 100 if m.get("gluing_pass_rate") else 0
        loc_err = m.get("locality_max", {}).get("mean", 0)
        print(f"  {exp_id}: Id={id_rate:.0f}% Tr={tr_rate:.0f}% Loc={loc_rate:.0f}% (err={loc_err:.2f}) Gl={gl_rate:.0f}%")

    # Best performing E1
    print("\nMain Benchmark Results (E1):")
    for exp_id in sorted(aggregated.keys()):
        if not exp_id.startswith("E1"):
            continue
        m = aggregated[exp_id].get("metrics", {})
        shd = m.get("shd", {}).get("mean", "N/A")
        f1 = m.get("f1", {}).get("mean", 0)
        print(f"  {exp_id}: SHD={shd:.1f}, F1={f1:.3f}" if isinstance(shd, float) else f"  {exp_id}: SHD={shd}, F1={f1}")

    # Ablation comparison
    print("\nAblation Results (A experiments) - Loss comparison:")
    for exp_id in sorted(aggregated.keys()):
        if not exp_id.startswith("A"):
            continue
        m = aggregated[exp_id].get("metrics", {})
        loss = m.get("final_loss_total", {}).get("mean", "N/A")
        desc = aggregated[exp_id].get("description", "")
        print(f"  {exp_id}: Loss={loss:.2e} - {desc}" if isinstance(loss, float) else f"  {exp_id}: Loss={loss}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
