"""Data loading and file management utilities"""

import os
import glob
import re
import numpy as np


def find_mzml_files(mzml_dir):
    """Find all mzML files in directory"""
    files = glob.glob(os.path.join(mzml_dir, "*.mzML")) + \
            glob.glob(os.path.join(mzml_dir, "*.mzml"))
    
    # Deduplicate case-insensitively
    unique_files = sorted(set(os.path.normcase(f) for f in files))
    return unique_files


def label_from_filename(filename):
    """Extract label (Tumor=1, Normal=0) from filename"""
    name = os.path.basename(filename)
    match = re.search(r'(\d+)([TN])', name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot find T/N in filename: {name}")
    return 1 if match.group(2).upper() == "T" else 0


def load_processed_data(output_dir):
    """Load preprocessed X and y arrays"""
    X = np.load(os.path.join(output_dir, "X.npy"))
    y = np.load(os.path.join(output_dir, "y.npy"))
    return X, y


def load_feature_selection_results(output_dir):
    """Load stability selection results"""
    stable_bins = np.load(os.path.join(output_dir, "stable_bins.npy"))
    stability_scores = np.load(os.path.join(output_dir, "stability_scores.npy"))
    return stable_bins, stability_scores