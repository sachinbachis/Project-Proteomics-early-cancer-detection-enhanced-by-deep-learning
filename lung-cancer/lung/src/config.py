"""Configuration constants for the proteomics ML pipeline"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MZML_DIR = r"C:\Users\Sachin Sharma\Music\raw_80datasets"
OUTPUT_DIR = os.path.join(MZML_DIR, "processed_bins")

# Binning parameters
MZ_MIN = 300.0
MZ_MAX = 1800.0
DELTA_M = 1.0
N_BINS = int((MZ_MAX - MZ_MIN) / DELTA_M)

# Feature selection
REPEATS = 50
SUBSAMPLE_RATIO = 0.8
K_FEATURES = 200
STABILITY_THRESHOLD = 0.6

# Model parameters
RANDOM_STATE = 42
CV_FOLDS = 5
CV_REPEATS = 20