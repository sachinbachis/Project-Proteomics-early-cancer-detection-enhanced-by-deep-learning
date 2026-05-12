"""Visualization utilities for the proteomics ML project"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold  # <-- Add this line

from config import MZ_MIN, DELTA_M


def mz_range_from_feature(feature_idx, stable_bins, X_stable_shape, X_multi_shape=None):
    """
    Map feature index to m/z range.
    
    Args:
        feature_idx: index in X_multi_stable (0-197)
        stable_bins: original stable bin indices for fine features
        X_stable_shape: shape of X_stable (n_samples, 153)
        X_multi_shape: shape of X_multi (n_samples, 603) - optional
    """
    # Block 1: Fine stable features (153 features)
    if 0 <= feature_idx < 153:
        fine_bin = int(stable_bins[feature_idx])
        mz_low = MZ_MIN + fine_bin * DELTA_M
        mz_high = mz_low + DELTA_M
        return ("fine (Δm=1, stable)", mz_low, mz_high)
    
    # Block 2: Medium bins (Δm=5, 300 features)
    if 153 <= feature_idx < 453:
        k = feature_idx - 153
        mz_low = MZ_MIN + k * 5.0
        mz_high = mz_low + 5.0
        return ("medium (Δm=5)", mz_low, mz_high)
    
    # Block 3: Coarse bins (Δm=10, 150 features)
    k = feature_idx - 453
    mz_low = MZ_MIN + k * 10.0
    mz_high = mz_low + 10.0
    return ("coarse (Δm=10)", mz_low, mz_high)


def plot_roc_curves(models, X, y, title="ROC Curve Comparison"):
    """Plot ROC curves for multiple models"""
    plt.figure(figsize=(8, 6))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        for train, test in cv.split(X, y):
            model.fit(X[train], y[train])
            probs = model.predict_proba(X[test])[:, 1]
            fpr, tpr, _ = roc_curve(y[test], probs)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, label=f"{name} (AUC = {mean_auc:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()


def plot_heatmap_fingerprint(X, y, stable_bins, mz_centers, output_path=None):
    """
    Plot heatmap, class means, and difference spectrum for stable fine bins.
    """
    # Z-score per feature
    col_mean = X.mean(axis=0, keepdims=True)
    col_std = X.std(axis=0, keepdims=True) + 1e-12
    Xz = (X - col_mean) / col_std
    
    # Sort by label
    order = np.argsort(y)
    Xz_sorted = Xz[order]
    y_sorted = y[order]
    
    # Class means
    tum = (y == 1)
    nor = (y == 0)
    mean_tum = X[tum].mean(axis=0)
    mean_nor = X[nor].mean(axis=0)
    diff = mean_tum - mean_nor
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Heatmap
    ax1 = plt.subplot(3, 1, 1)
    im = ax1.imshow(Xz_sorted, aspect='auto', interpolation='nearest')
    ax1.set_title("Spectral Fingerprint Heatmap (Z-scored)")
    ax1.set_ylabel("Samples (Normal → Tumor)")
    boundary = np.sum(y_sorted == 0)
    ax1.axhline(boundary - 0.5)
    plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.02)
    
    tick_count = 10
    tick_pos = np.linspace(0, X.shape[1]-1, tick_count).astype(int)
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels([f"{mz_centers[i]:.0f}" for i in tick_pos])
    
    # Class means
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(mz_centers, mean_nor, label="Normal mean")
    ax2.plot(mz_centers, mean_tum, label="Tumor mean")
    ax2.set_title("Class Mean Spectra")
    ax2.set_ylabel("Mean intensity")
    ax2.legend()
    
    # Difference
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(mz_centers, diff)
    ax3.axhline(0)
    ax3.set_title("Difference Spectrum (Tumor − Normal)")
    ax3.set_xlabel("m/z")
    ax3.set_ylabel("Mean difference")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved figure: {output_path}")
    
    return plt.gcf()


def plot_importance_scatter(mz_positions, importance_values, title="Important Spectral Regions"):
    """Create scatter plot of feature importances vs m/z"""
    plt.figure(figsize=(10, 4))
    plt.scatter(mz_positions, importance_values)
    plt.xlabel("m/z")
    plt.ylabel("Feature Importance")
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()