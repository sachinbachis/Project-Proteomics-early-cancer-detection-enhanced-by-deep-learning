"""Feature selection using stability selection"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

from config import REPEATS, SUBSAMPLE_RATIO, K_FEATURES, STABILITY_THRESHOLD, RANDOM_STATE


def stability_selection(X, y, repeats=REPEATS, subsample_ratio=SUBSAMPLE_RATIO,
                        k_features=K_FEATURES, threshold=STABILITY_THRESHOLD):
    """
    Perform stability selection using ANOVA F-test.
    
    Returns:
        stable_features: indices of stable features
        stability_scores: selection frequency for each feature
    """
    n_samples, n_features = X.shape
    rng = np.random.default_rng(RANDOM_STATE)
    
    counts = np.zeros(n_features, dtype=np.int32)
    
    for r in range(repeats):
        idx = rng.choice(n_samples, int(subsample_ratio * n_samples), replace=False)
        X_sub, y_sub = X[idx], y[idx]
        
        selector = SelectKBest(score_func=f_classif, k=min(k_features, n_features))
        selector.fit(X_sub, y_sub)
        selected = selector.get_support(indices=True)
        
        counts[selected] += 1
    
    stability_scores = counts / repeats
    stable_features = np.where(stability_scores >= threshold)[0]
    
    print(f"Stable features found: {len(stable_features)}")
    
    return stable_features, stability_scores


def apply_feature_selection(X, stable_features):
    """Filter X to only stable features"""
    return X[:, stable_features]