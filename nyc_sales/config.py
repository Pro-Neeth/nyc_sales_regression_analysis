"""Paths and settings shared across the pipeline."""

from pathlib import Path

import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'nyc-rolling-sales.csv'
REPORTS_DIR = PROJECT_ROOT / 'reports'
KAGGLE_DATASET_URL = 'https://www.kaggle.com/datasets/new-york-city/nyc-property-sales'

RANDOM_STATE = 42

# EDA
PAIRPLOT_SAMPLE_SIZE = 10000

# Modeling
ALPHAS = np.logspace(-3, 3, 50)  # 50 values from 0.001 to 1000
OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 4
RIDGE_MAX_ITER = 10000
LASSO_MAX_ITER = 5000
