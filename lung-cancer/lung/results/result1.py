"""Results management for proteomics pipeline."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def setup_results(output_dir):
    """Create results folders."""
    figs = os.path.join(output_dir, "figures")
    models = os.path.join(output_dir, "models")
    metrics = os.path.join(output_dir, "metrics")
    
    for d in [figs, models, metrics]:
        os.makedirs(d, exist_ok=True)
    
    return {"figures": figs, "models": models, "metrics": metrics}


def save_figure(fig, name, results_dir):
    """Save figure."""
    path = os.path.join(results_dir["figures"], name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Figure saved: {path}")
    return path


def save_metrics(data, name, results_dir):
    """Save metrics as JSON."""
    path = os.path.join(results_dir["metrics"], f"{name}.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Metrics saved: {path}")
    return path


def save_model(model, name, results_dir):
    """Save model using joblib."""
    import joblib
    path = os.path.join(results_dir["models"], f"{name}.pkl")
    joblib.dump(model, path)
    print(f"✓ Model saved: {path}")
    return path


def load_model(name, results_dir):
    """Load model."""
    import joblib
    path = os.path.join(results_dir["models"], f"{name}.pkl")
    return joblib.load(path)


def save_summary(results, results_dir):
    """Save model evaluation summary."""
    summary = {
        "best_model": max(results.items(), key=lambda x: x[1]["mean"])[0],
        "scores": {k: {"mean": float(v["mean"]), "std": float(v["std"])} 
                   for k, v in results.items()}
    }
    return save_metrics(summary, "summary", results_dir)