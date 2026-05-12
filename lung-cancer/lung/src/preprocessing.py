"""Preprocessing for proteomics data."""

import os
import glob
import re
import numpy as np
from pyteomics import mzml

MZ_MIN, MZ_MAX, DELTA = 300.0, 1800.0, 1.0
N_BINS = int((MZ_MAX - MZ_MIN) / DELTA)


def find_mzml_files(mzml_dir):
    """Find all mzML files."""
    files = glob.glob(os.path.join(mzml_dir, "*.mzML")) + glob.glob(os.path.join(mzml_dir, "*.mzml"))
    return sorted(set(os.path.normcase(f) for f in files))


def label_from_filename(filename):
    """Extract label from filename (T=1, N=0)."""
    match = re.search(r'(\d+)([TN])', os.path.basename(filename), flags=re.IGNORECASE)
    return 1 if match and match.group(2).upper() == "T" else 0


def bin_mzml_file(filepath):
    """Convert mzML to binned vector."""
    vec = np.zeros(N_BINS)
    with mzml.MzML(filepath) as reader:
        for spec in reader:
            if spec.get("ms level") != 1:
                continue
            mzs = spec.get("m/z array")
            ints = spec.get("intensity array")
            if mzs is None:
                continue
            mask = (mzs >= MZ_MIN) & (mzs < MZ_MAX)
            idx = ((mzs[mask] - MZ_MIN) / DELTA).astype(int)
            np.add.at(vec, idx, ints[mask])
    return np.log1p(vec / vec.sum()) if vec.sum() > 0 else vec


def process_all_mzml_files(mzml_dir, output_dir, files=None):
    """Process all mzML files."""
    os.makedirs(output_dir, exist_ok=True)
    files = files or find_mzml_files(mzml_dir)
    saved, skipped = 0, []
    
    for i, fp in enumerate(files, 1):
        base = os.path.basename(fp)
        if os.path.exists(os.path.join(output_dir, base + ".npy")):
            continue
        try:
            vec = bin_mzml_file(fp)
            np.save(os.path.join(output_dir, base + ".npy"), vec.astype(np.float32))
            with open(os.path.join(output_dir, base + ".label.txt"), "w") as f:
                f.write(str(label_from_filename(base)))
            saved += 1
            print(f"[{i}/{len(files)}] processed: {base}")
        except:
            skipped.append(base)
            print(f"[{i}/{len(files)}] skipped: {base}")
    
    print(f"Processed: {saved}, Skipped: {len(skipped)}")
    return saved, skipped


def combine_vectors(output_dir):
    """Combine vectors into X and y."""
    vec_files = sorted(glob.glob(os.path.join(output_dir, "*.mzML.npy")))
    if not vec_files:
        return None, None
    
    X = np.vstack([np.load(v) for v in vec_files])
    y = np.array([int(open(os.path.join(output_dir, os.path.basename(v).replace(".npy", ".label.txt"))).read()) for v in vec_files])
    
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    print(f"X: {X.shape}, y: {np.bincount(y)}")
    return X, y


def create_multi_resolution_features(X):
    """Create multi-resolution features."""
    n = X.shape[0]
    return X.reshape(n, 300, 5).sum(axis=2), X.reshape(n, 150, 10).sum(axis=2)