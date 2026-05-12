"""Machine learning models and evaluation"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def get_models():
    """Return dictionary of models to evaluate"""
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=42))
        ]),
        "SVM (RBF)": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            random_state=42
        )
    }


def evaluate_model(model, X, y, cv_folds=5, scoring="roc_auc"):
    """Evaluate model using cross-validation"""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return scores


def evaluate_models(X, y, models=None, cv_folds=5, scoring="roc_auc"):
    """Evaluate multiple models and return results"""
    if models is None:
        models = get_models()
    
    results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        results[name] = {
            "scores": scores,
            "mean": scores.mean(),
            "std": scores.std()
        }
        print(f"{name}: Mean {scoring} = {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return results


def permutation_test(model, X, y, n_permutations=200, scoring="roc_auc"):
    """Perform permutation test to validate model significance"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    score, perm_scores, pvalue = permutation_test_score(
        model, X, y,
        scoring=scoring,
        cv=cv,
        n_permutations=n_permutations,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"Observed {scoring}: {score:.4f}")
    print(f"Permutation mean: {perm_scores.mean():.4f}")
    print(f"p-value: {pvalue:.6f}")
    
    return score, perm_scores, pvalue