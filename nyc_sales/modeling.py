"""Ridge vs. Lasso regression with nested cross-validation"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from nyc_sales.config import ALPHAS, INNER_CV_SPLITS, LASSO_MAX_ITER, OUTER_CV_SPLITS, RANDOM_STATE, RIDGE_MAX_ITER

# Models compared, in the order they are run
MODELS = {
    'Ridge': lambda: Ridge(max_iter=RIDGE_MAX_ITER),
    'Lasso': lambda: Lasso(max_iter=LASSO_MAX_ITER),
}


def run_modeling(nyc_df_ols):
    """Evaluate Ridge and Lasso with nested CV and compare them.

    Returns ({model name: per-fold results}, comparison table, comparison figure).
    """
    # Separate features and target
    X = nyc_df_ols.drop('SALE PRICE', axis=1)
    y = nyc_df_ols['SALE PRICE']

    # Define columns
    continuous_cols = ['RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'LAND SQUARE FEET',
                       'GROSS SQUARE FEET', 'AGE OF PROPERTY']
    target_encode_cols = ['NEIGHBORHOOD']
    ohe_cols = [col for col in X.columns if col not in continuous_cols and col not in target_encode_cols]

    # Create preprocessing pipeline. Target encoding replaces each neighborhood with its mean sale price;
    # being part of the pipeline, it is fit inside each fold and never sees the held-out data.
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), continuous_cols),
            ('target_encoder', TargetEncoder(target_type='continuous', smooth='auto', cv=5),
             target_encode_cols),
            ('passthrough', 'passthrough', ohe_cols)
        ],
        remainder='drop'
    )

    # Initialize cross-validation
    outer_cv = KFold(n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = KFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = {name: nested_cv(X, y, preprocessor, make_regressor, name, outer_cv, inner_cv)
               for name, make_regressor in MODELS.items()}

    # Fit on full data for visualization purposes only
    viz_results = {}
    for name, make_regressor in MODELS.items():
        grid_full = GridSearchCV(
            Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', make_regressor())
            ]),
            {'regressor__alpha': ALPHAS},
            cv=inner_cv,
            scoring='r2',
            n_jobs=-1,
            return_train_score=True
        )
        grid_full.fit(X, y)
        viz_results[name] = pd.DataFrame(grid_full.cv_results_)

    ridge_results_df, lasso_results_df = results['Ridge'], results['Lasso']
    fig = plot_model_results(viz_results['Ridge'], viz_results['Lasso'], ridge_results_df, lasso_results_df, y)

    # Final comparison table
    ridge_gap = ridge_results_df['train_r2'] - ridge_results_df['test_r2']
    lasso_gap = lasso_results_df['train_r2'] - lasso_results_df['test_r2']

    print("\n" + "=" * 70)
    print("NESTED CV MODEL COMPARISON")
    print("=" * 70)
    comparison_df = pd.DataFrame({
        'Model': ['Ridge', 'Lasso'],
        'Mean Test R²': [ridge_results_df['test_r2'].mean(), lasso_results_df['test_r2'].mean()],
        'Std Test R²': [ridge_results_df['test_r2'].std(), lasso_results_df['test_r2'].std()],
        'Mean Train R²': [ridge_results_df['train_r2'].mean(), lasso_results_df['train_r2'].mean()],
        'Overfit Gap': [ridge_gap.mean(), lasso_gap.mean()],
        'Mean RMSE': [ridge_results_df['test_rmse'].mean(), lasso_results_df['test_rmse'].mean()],
        'Mean MAE': [ridge_results_df['test_mae'].mean(), lasso_results_df['test_mae'].mean()],
        'Best Alpha': [ridge_results_df['best_alpha'].mode()[0], lasso_results_df['best_alpha'].mode()[0]]
    })
    print(comparison_df.to_string(index=False))
    print("=" * 70)

    return results, comparison_df, fig


def nested_cv(X, y, preprocessor, make_regressor, name, outer_cv, inner_cv):
    """Tune alpha with an inner grid search and score the best model on each held-out outer fold.

    Prints the per-fold and summary metrics and returns one row of results per outer fold.
    """
    print("\n" + "=" * 70)
    print(f"{name.upper()} REGRESSION WITH NESTED CV")
    print("=" * 70)

    nested_results = []

    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\nOuter Fold {fold_id + 1}/{outer_cv.get_n_splits()}")

        # Split data for this outer fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Create fresh pipeline for this fold
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', make_regressor())
        ])

        # GridSearchCV for hyperparameter tuning
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid={'regressor__alpha': ALPHAS},
            cv=inner_cv,
            scoring='r2',
            n_jobs=-1,
            return_train_score=False
        )

        # Fit grid search on training data
        grid.fit(X_train, y_train)

        # Get best model and predict on held-out test fold
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        # Also get training predictions to monitor overfitting
        y_train_pred = best_model.predict(X_train)

        # Calculate metrics
        test_r2 = r2_score(y_test, y_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_pred)

        nested_results.append({
            'outer_fold': fold_id,
            'best_alpha': grid.best_params_['regressor__alpha'],
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        })

        print(f"  Best alpha: {grid.best_params_['regressor__alpha']:.4f}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test RMSE: ${test_rmse:,.2f}")

    results_df = pd.DataFrame(nested_results)

    print("\n" + "=" * 70)
    print(f"{name.upper()} REGRESSION SUMMARY")
    print("=" * 70)
    print(f"Mean Test R²: {results_df['test_r2'].mean():.4f} (+/- {results_df['test_r2'].std():.4f})")
    print(f"Mean Train R²: {results_df['train_r2'].mean():.4f} (+/- {results_df['train_r2'].std():.4f})")
    print(f"Mean Test RMSE: ${results_df['test_rmse'].mean():,.2f} (+/- ${results_df['test_rmse'].std():,.2f})")
    print(f"Mean Test MAE: ${results_df['test_mae'].mean():,.2f}")
    print(f"Most Common Best Alpha: {results_df['best_alpha'].mode()[0]:.4f}")

    return results_df


def plot_model_results(ridge_viz_results, lasso_viz_results, ridge_results_df, lasso_results_df, y):
    """Ridge vs. Lasso: CV score against alpha, RMSE curves, per-fold R² and the overfitting gap.

    The *_viz_results are the cv_results_ of the full-data grid searches; the *_results_df are
    the nested CV results.
    """
    alphas = ALPHAS

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Finding Best Alpha for Ridge
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogx(alphas, ridge_viz_results['mean_test_score'], 'b-', linewidth=2.5, label='CV Score')
    ax1.fill_between(alphas,
                     ridge_viz_results['mean_test_score'] - ridge_viz_results['std_test_score'],
                     ridge_viz_results['mean_test_score'] + ridge_viz_results['std_test_score'],
                     alpha=0.2, color='b')
    ax1.semilogx(alphas, ridge_viz_results['mean_train_score'], 'r--', linewidth=2, label='Train Score', alpha=0.6)
    ax1.axvline(ridge_results_df['best_alpha'].mode()[0], color='g', linestyle='--', linewidth=2,
                label=f'Most Common α = {ridge_results_df["best_alpha"].mode()[0]:.3f}')
    ax1.set_xlabel('Regularization Strength', fontsize=11, fontweight='bold')
    ax1.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax1.set_title('Ridge: Cross-Validation Performance vs Alpha', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Finding Best Alpha for Lasso
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogx(alphas, lasso_viz_results['mean_test_score'], 'b-', linewidth=2.5, label='CV Score')
    ax2.fill_between(alphas,
                     lasso_viz_results['mean_test_score'] - lasso_viz_results['std_test_score'],
                     lasso_viz_results['mean_test_score'] + lasso_viz_results['std_test_score'],
                     alpha=0.2, color='b')
    ax2.semilogx(alphas, lasso_viz_results['mean_train_score'], 'r--', linewidth=2, label='Train Score', alpha=0.6)
    ax2.axvline(lasso_results_df['best_alpha'].mode()[0], color='g', linestyle='--', linewidth=2,
                label=f'Most Common alpha = {lasso_results_df["best_alpha"].mode()[0]:.3f}')
    ax2.set_xlabel('Regularization Strength', fontsize=11, fontweight='bold')
    ax2.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax2.set_title('Lasso: Cross-Validation Performance vs Alpha', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # RMSE curves for Ridge vs Lasso
    ax3 = fig.add_subplot(gs[0, 2])
    ridge_rmse_cv = np.sqrt(-ridge_viz_results['mean_test_score'] * y.var() + y.var())
    lasso_rmse_cv = np.sqrt(-lasso_viz_results['mean_test_score'] * y.var() + y.var())
    ax3.semilogx(alphas, ridge_rmse_cv, 'b-', linewidth=2.5, label='Ridge CV RMSE', alpha=0.8)
    ax3.semilogx(alphas, lasso_rmse_cv, 'r-', linewidth=2.5, label='Lasso CV RMSE', alpha=0.8)
    ax3.set_xlabel('Regularization Strength', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Estimated RMSE', fontsize=11, fontweight='bold')
    ax3.set_title('Ridge vs Lasso: RMSE Comparison', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Ridge Performance Across Folds
    ax4 = fig.add_subplot(gs[1, 0])
    folds = ridge_results_df['outer_fold'] + 1
    ax4.plot(folds, ridge_results_df['test_r2'], 'o-', linewidth=2.5, markersize=8,
             color='#2E86AB', label='Test R²')
    ax4.plot(folds, ridge_results_df['train_r2'], 's--', linewidth=2, markersize=7,
             color='#A23B72', alpha=0.7, label='Train R²')
    ax4.axhline(ridge_results_df['test_r2'].mean(), color='#2E86AB', linestyle=':',
                linewidth=2, label=f'Mean Test R² = {ridge_results_df["test_r2"].mean():.3f}')
    ax4.set_xlabel('Outer CV Fold', fontsize=11, fontweight='bold')
    ax4.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax4.set_title('Ridge: Performance Across Folds', fontsize=13, fontweight='bold')
    ax4.set_xticks(folds)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Lasso Performance Across Folds
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(folds, lasso_results_df['test_r2'], 'o-', linewidth=2.5, markersize=8,
             color='#2E86AB', label='Test R²')
    ax5.plot(folds, lasso_results_df['train_r2'], 's--', linewidth=2, markersize=7,
             color='#A23B72', alpha=0.7, label='Train R²')
    ax5.axhline(lasso_results_df['test_r2'].mean(), color='#2E86AB', linestyle=':',
                linewidth=2, label=f'Mean Test R² = {lasso_results_df["test_r2"].mean():.3f}')
    ax5.set_xlabel('Outer CV Fold', fontsize=11, fontweight='bold')
    ax5.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax5.set_title('Lasso: Performance Across Folds', fontsize=13, fontweight='bold')
    ax5.set_xticks(folds)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Overfitting Analysis
    ax6 = fig.add_subplot(gs[2, 1])
    ridge_gap = ridge_results_df['train_r2'] - ridge_results_df['test_r2']
    lasso_gap = lasso_results_df['train_r2'] - lasso_results_df['test_r2']
    ax6.boxplot([ridge_gap, lasso_gap], tick_labels=['Ridge', 'Lasso'],
                patch_artist=True,
                boxprops=dict(facecolor='#A8DADC', alpha=0.7),
                medianprops=dict(color='#E63946', linewidth=2))
    ax6.axhline(0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='No Overfitting')
    ax6.set_ylabel('Train R² - Test R² (Overfitting Gap)', fontsize=11, fontweight='bold')
    ax6.set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')

    return fig
