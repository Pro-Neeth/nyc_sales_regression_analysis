"""Run the whole analysis end to end: load -> clean -> EDA -> features -> modeling."""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nyc_sales.cleaning import clean_data, load_raw_data
from nyc_sales.config import RAW_DATA_PATH, REPORTS_DIR
from nyc_sales.eda import run_eda
from nyc_sales.features import engineer_features
from nyc_sales.modeling import run_modeling

logger = logging.getLogger(__name__)


def run_pipeline(data_path=RAW_DATA_PATH, reports_dir=REPORTS_DIR, skip_eda=False, show=False):
    """Run every stage, save the figures and results under reports_dir, and return (results, comparison table)."""
    figures_dir = Path(reports_dir) / 'figures'
    results_dir = Path(reports_dir) / 'results'

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.precision', 3)

    # 1-2. Initial cleaning and investigation of anomalies
    nyc_df = load_raw_data(data_path)
    logger.info("Loaded %s: %d rows x %d columns", data_path, *nyc_df.shape)
    nyc_df = clean_data(nyc_df, verbose=not skip_eda)
    logger.info("Cleaned data: %d rows x %d columns", *nyc_df.shape)

    # 3-4. Distribution & correlation analysis
    if not skip_eda:
        save_figures(run_eda(nyc_df), figures_dir, show)

    # 5. Feature engineering & final preprocessing
    nyc_df_ols = engineer_features(nyc_df, verbose=not skip_eda)
    logger.info("Model data: %d rows x %d columns", *nyc_df_ols.shape)

    # 6. Regression methods & comparative analysis
    results, comparison_df, fig = run_modeling(nyc_df_ols)
    save_figures({'07_model_comparison.png': fig}, figures_dir, show)

    results_dir.mkdir(parents=True, exist_ok=True)
    for name, results_df in results.items():
        results_df.to_csv(results_dir / f"{name.lower().replace(' ', '_')}_nested_cv.csv", index=False)
    comparison_df.to_csv(results_dir / 'model_comparison.csv', index=False)
    logger.info("Saved figures to %s and results to %s", figures_dir, results_dir)

    return results, comparison_df


def save_figures(figures, figures_dir, show=False):
    """Save {filename: figure} into figures_dir, optionally display them, then close them."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    for filename, fig in figures.items():
        fig.savefig(figures_dir / filename, dpi=100, bbox_inches='tight')
    if show:
        plt.show()
    for fig in figures.values():
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python -m nyc_sales',
        description='Clean the NYC rolling sales data, run the EDA and compare a mean baseline, Ridge, Lasso '
                    'and gradient boosting with nested cross-validation.')
    parser.add_argument('--data', type=Path, default=RAW_DATA_PATH,
                        help='path to nyc-rolling-sales.csv (default: data/raw/nyc-rolling-sales.csv)')
    parser.add_argument('--reports-dir', type=Path, default=REPORTS_DIR,
                        help='where figures/ and results/ are written (default: reports/)')
    parser.add_argument('--skip-eda', action='store_true',
                        help='skip the analysis printouts and EDA plots; only clean, engineer features and model')
    parser.add_argument('--show', action='store_true', help='also display the figures in windows')
    args = parser.parse_args(argv)

    # Log to stdout so log lines stay in order with the printed results. Only this package logs at
    # INFO; libraries such as matplotlib stay at WARNING.
    logging.basicConfig(format='%(asctime)s | %(message)s', datefmt='%H:%M:%S', stream=sys.stdout)
    logging.getLogger('nyc_sales').setLevel(logging.INFO)
    try:
        run_pipeline(args.data, args.reports_dir, skip_eda=args.skip_eda, show=args.show)
    except FileNotFoundError as exc:
        parser.exit(1, f"error: {exc}\n")
