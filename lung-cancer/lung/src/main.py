"""Main execution pipeline for proteomics ML project"""

import os
import numpy as np
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, MZML_DIR, RANDOM_STATE, MZ_MIN, DELTA_M
from data_loader import find_mzml_files, load_processed_data
from preprocessing import (
    process_all_mzml_files, combine_vectors, create_multi_resolution_features
)
from feature_selection import stability_selection, apply_feature_selection
from models import get_models, evaluate_models, permutation_test
from visualization import (
    plot_roc_curves, plot_heatmap_fingerprint, 
    plot_importance_scatter, mz_range_from_feature
)


def run_full_pipeline():
    """Run the complete analysis pipeline"""
    
    print("=" * 60)
    print("PROTEOMICS ML PIPELINE")
    print("=" * 60)
    
    # Step 1: Find mzML files
    print("\n[1/7] Finding mzML files...")
    files = find_mzml_files(MZML_DIR)
    print(f"Found {len(files)} mzML files")
    
    # Step 2: Process mzML to binned vectors
    print("\n[2/7] Processing mzML files...")
    process_all_mzml_files(MZML_DIR, OUTPUT_DIR, files)
    
    # Step 3: Combine into X, y
    print("\n[3/7] Combining vectors...")
    X, y = combine_vectors(OUTPUT_DIR)
    
    # Step 4: Create multi-resolution features
    print("\n[4/7] Creating multi-resolution features...")
    bins5, bins10 = create_multi_resolution_features(X)
    np.save(os.path.join(OUTPUT_DIR, "X_bins5.npy"), bins5)
    np.save(os.path.join(OUTPUT_DIR, "X_bins10.npy"), bins10)
    print(f"Δm=5 shape: {bins5.shape}")
    print(f"Δm=10 shape: {bins10.shape}")
    
    # Step 5: Stability selection on multi-resolution features
    print("\n[5/7] Performing stability selection...")
    X_multi = np.hstack([X, bins5, bins10])
    stable_feats, stability_scores = stability_selection(X_multi, y)
    np.save(os.path.join(OUTPUT_DIR, "stable_feats_multi.npy"), stable_feats)
    np.save(os.path.join(OUTPUT_DIR, "stability_scores_multi.npy"), stability_scores)
    
    X_multi_stable = X_multi[:, stable_feats]
    np.save(os.path.join(OUTPUT_DIR, "X_multi_stable.npy"), X_multi_stable)
    print(f"X_multi_stable shape: {X_multi_stable.shape}")
    
    # Step 6: Evaluate models
    print("\n[6/7] Evaluating models...")
    models = get_models()
    results = evaluate_models(X_multi_stable, y, models)
    
    # Step 7: Visualizations
    print("\n[7/7] Generating visualizations...")
    
    # ROC curves
    fig_roc = plot_roc_curves(models, X_multi_stable, y)
    fig_roc.savefig(os.path.join(OUTPUT_DIR, "roc_comparison.png"), dpi=300)
    print(f"ROC plot saved to {OUTPUT_DIR}/roc_comparison.png")
    
    # Heatmap fingerprint
    stable_bins = np.load(os.path.join(OUTPUT_DIR, "stable_bins.npy"))
    mz_centers = MZ_MIN + stable_bins * DELTA_M + (DELTA_M / 2)
    fig_heatmap = plot_heatmap_fingerprint(
        X, y, stable_bins, mz_centers,
        output_path=os.path.join(OUTPUT_DIR, "figure_heatmap_fingerprint.png")
    )
    
    # Permutation test for significance
    print("\nPerforming permutation test for SVM...")
    svm_model = models["SVM (RBF)"]
    permutation_test(svm_model, X_multi_stable, y, n_permutations=200)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = run_full_pipeline()