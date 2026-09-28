"""Paths and settings shared across the pipeline."""

from pathlib import Path

import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'nyc-rolling-sales.csv'
REPORTS_DIR = PROJECT_ROOT / 'reports'
KAGGLE_DATASET_URL = 'https://www.kaggle.com/datasets/new-york-city/nyc-property-sales'

RANDOM_STATE = 42

# Cleaning
MIN_SALE_PRICE = 100_000  # cheaper sales are non-market transfers recorded at token prices, not market sales

# EDA
PAIRPLOT_SAMPLE_SIZE = 10000

# Modeling
RIDGE_ALPHAS = np.logspace(-3, 3, 50)  # 50 values from 0.001 to 1000
LASSO_ALPHAS = np.logspace(-5, 0, 50)  # 50 values from 0.00001 to 1; at 1 Lasso zeroes every coefficient
GRADIENT_BOOSTING_GRID = {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [15, 31, 63]}
OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 4
RIDGE_MAX_ITER = 10000
LASSO_MAX_ITER = 5000
