"""Mean baseline, Ridge, Lasso and gradient boosting compared with nested cross-validation"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from nyc_sales.config import (GRADIENT_BOOSTING_GRID, INNER_CV_SPLITS, LASSO_ALPHAS, LASSO_MAX_ITER, OUTER_CV_SPLITS,
                              RANDOM_STATE, RIDGE_ALPHAS, RIDGE_MAX_ITER)

# Models compared, in the order they are run, with the hyperparameter grid tuned in the inner CV
MODELS = {
    'Baseline': (lambda: DummyRegressor(strategy='mean'), {}),
    'Ridge': (lambda: Ridge(max_iter=RIDGE_MAX_ITER), {'alpha': RIDGE_ALPHAS}),
    # precompute=True: with ~140 features and ~28k rows, the Gram matrix makes small-alpha fits far faster
    'Lasso': (lambda: Lasso(max_iter=LASSO_MAX_ITER, precompute=True), {'alpha': LASSO_ALPHAS}),
    'Gradient Boosting': (lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE), GRADIENT_BOOSTING_GRID),
}

# Plot colors: blue for test/validation scores and orange for train scores, a one-hue blue ramp for point
# density, and gray inks for text and chrome
TEST_COLOR = '#2a78d6'
TRAIN_COLOR = '#eb6834'
FOLD_COLOR = '#104281'
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    'density', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
SECONDARY_INK = '#52514e'
MUTED_INK = '#898781'
AXIS_COLOR = '#c3c2b7'
PLOT_STYLE = {
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'axes.edgecolor': AXIS_COLOR, 'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.axisbelow': True, 'grid.color': '#e1e0d9', 'grid.linewidth': 0.7,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'axes.titlelocation': 'left',
    'axes.titlepad': 10, 'axes.labelsize': 10, 'axes.labelcolor': SECONDARY_INK,
    'xtick.color': AXIS_COLOR, 'ytick.color': AXIS_COLOR,
    'xtick.labelcolor': SECONDARY_INK, 'ytick.labelcolor': SECONDARY_INK,
    'legend.frameon': False, 'legend.fontsize': 9, 'lines.linewidth': 1.5,
}


def run_modeling(nyc_df_ols):
    """Evaluate every model in MODELS with nested CV and compare them.

    Returns ({model name: per-fold results}, comparison table, comparison figure).
    """
    # Separate features and target
    X = nyc_df_ols.drop('SALE PRICE', axis=1)
    y = nyc_df_ols['SALE PRICE']

    preprocessor = make_preprocessor(X)

    # Initialize cross-validation
    outer_cv = KFold(n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = KFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results, predictions = {}, {}
    for name, (make_regressor, param_grid) in MODELS.items():
        results[name], predictions[name] = nested_cv(X, y, preprocessor, make_regressor, param_grid, name,
                                                     outer_cv, inner_cv)

    # Validation curves over alpha on the full data, for visualization purposes only
    validation_curves = {}
    for name in ['Ridge', 'Lasso']:
        make_regressor, param_grid = MODELS[name]
        validation_curves[name] = validation_curve(
            Pipeline([('preprocessor', preprocessor), ('regressor', make_regressor())]), X, y,
            param_name='regressor__alpha', param_range=param_grid['alpha'], cv=inner_cv, scoring='r2', n_jobs=-1)

    # Final comparison table
    print("\n" + "=" * 70)
    print("NESTED CV MODEL COMPARISON")
    print("=" * 70)
    comparison_df = pd.DataFrame([{
        'Model': name,
        'Mean Test R²': results_df['test_r2'].mean(),
        'Std Test R²': results_df['test_r2'].std(),
        'Mean Train R²': results_df['train_r2'].mean(),
        'Overfit Gap': (results_df['train_r2'] - results_df['test_r2']).mean(),
        'Mean RMSE (log)': results_df['test_rmse_log'].mean(),
        'Median % Error': results_df['test_median_pct_error'].mean(),
        'Most Common Params': results_df['best_params'].mode()[0],
    } for name, results_df in results.items()])
    print(comparison_df.to_string(index=False, float_format='{:.4f}'.format))
    print("=" * 70)

    best_name = comparison_df.loc[comparison_df['Mean Test R²'].idxmax(), 'Model']
    fig = plot_model_results(results, validation_curves, y, predictions[best_name], best_name)

    return results, comparison_df, fig


def make_preprocessor(X):
    """Scale the continuous features, target-encode NEIGHBORHOOD and pass the one-hot columns of X through."""
    # Define columns
    continuous_cols = ['RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'LAND SQUARE FEET',
                       'GROSS SQUARE FEET', 'AGE OF PROPERTY']
    target_encode_cols = ['NEIGHBORHOOD']
    ohe_cols = [col for col in X.columns if col not in continuous_cols and col not in target_encode_cols]

    # Target encoding replaces each neighborhood with its mean sale price; being part of the pipeline, it is fit
    # inside each fold and never sees the held-out data.
    return ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), continuous_cols),
            ('target_encoder', TargetEncoder(target_type='continuous', smooth='auto',
                                             cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)),
             target_encode_cols),
            ('passthrough', 'passthrough', ohe_cols)
        ],
        remainder='drop'
    )


def median_pct_error(y_true, y_pred):
    """Median absolute error of the predicted price as a % of the actual price, given log1p(SALE PRICE) values."""
    actual_price, predicted_price = np.expm1(np.asarray(y_true)), np.expm1(np.asarray(y_pred))
    return np.median(np.abs(predicted_price - actual_price) / actual_price) * 100


def nested_cv(X, y, preprocessor, make_regressor, param_grid, name, outer_cv, inner_cv):
    """Tune the model with an inner grid search and score the best model on each held-out outer fold.

    Prints the per-fold and summary metrics and returns (one row of results per outer fold,
    the out-of-fold prediction for every sample).
    """
    print("\n" + "=" * 70)
    print(f"{name.upper()} WITH NESTED CV")
    print("=" * 70)

    nested_results = []
    oof_predictions = pd.Series(np.nan, index=y.index)

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
            param_grid={f'regressor__{param}': values for param, values in param_grid.items()},
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
        oof_predictions.iloc[test_idx] = y_pred

        # Also get training predictions to monitor overfitting
        y_train_pred = best_model.predict(X_train)

        # Calculate metrics. R² and RMSE are on the log1p(SALE PRICE) scale the models are fit on, while the
        # percentage error compares predicted and actual prices in dollars
        test_r2 = r2_score(y_test, y_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
        test_median_pct_error = median_pct_error(y_test, y_pred)
        best_params = ', '.join(f"{param.removeprefix('regressor__')}={value:.3g}"
                                for param, value in grid.best_params_.items()) or '-'

        nested_results.append({
            'outer_fold': fold_id,
            'best_params': best_params,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse_log': test_rmse_log,
            'test_median_pct_error': test_median_pct_error
        })

        print(f"  Best params: {best_params}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test median error: {test_median_pct_error:.1f}%")

    results_df = pd.DataFrame(nested_results)

    print("\n" + "=" * 70)
    print(f"{name.upper()} SUMMARY")
    print("=" * 70)
    print(f"Mean Test R²: {results_df['test_r2'].mean():.4f} (+/- {results_df['test_r2'].std():.4f})")
    print(f"Mean Train R²: {results_df['train_r2'].mean():.4f} (+/- {results_df['train_r2'].std():.4f})")
    print(f"Mean Test RMSE (log scale): {results_df['test_rmse_log'].mean():.4f} "
          f"(+/- {results_df['test_rmse_log'].std():.4f})")
    print(f"Mean Test median error: {results_df['test_median_pct_error'].mean():.1f}%")
    print(f"Most Common Best Params: {results_df['best_params'].mode()[0]}")

    return results_df, oof_predictions


def plot_model_results(results, validation_curves, y, best_predictions, best_name):
    """Test R² and median error per model, predicted vs. actual prices for the best model, train vs. test R²,
    and the Ridge and Lasso validation curves over alpha.

    results are the nested CV results per model, validation_curves the (train, validation) scores of the
    full-data alpha sweeps, and best_predictions the out-of-fold predictions of the best_name model.
    """
    names = list(results)
    positions = np.arange(len(names))[::-1]  # first model at the top

    def dollars(value, _):
        if value >= 1e9:
            return f'${value / 1e9:g}B'
        return f'${value / 1e6:g}M' if value >= 1e6 else f'${value / 1e3:g}k'

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), layout='constrained')
        fig.suptitle(f'Model comparison with nested cross-validation '
                     f'({OUTER_CV_SPLITS} outer × {INNER_CV_SPLITS} inner folds)',
                     x=0.01, ha='left', fontsize=15, fontweight='bold', color=INK)

        # Test R² and median error per model: the bar is the mean over the outer folds, the dots are the folds
        max_pct_error = max(results[name]['test_median_pct_error'].max() for name in names)
        for ax, column, title, x_label, label_format, x_max in [
                (axes[0, 0], 'test_r2', 'Test R² (higher is better)',
                 'R² on held-out folds', '{:.3f}', 1),
                (axes[0, 1], 'test_median_pct_error', 'Median % error (lower is better)',
                 'Median absolute % error of the predicted price', '{:.1f}%', max_pct_error * 1.25)]:
            fold_values = [results[name][column] for name in names]
            means = [values.mean() for values in fold_values]
            ax.barh(positions, means, height=0.28, color=TEST_COLOR, zorder=2)
            for position, values, mean in zip(positions, fold_values, means):
                ax.scatter(values, [position] * len(values), s=34, color=FOLD_COLOR, edgecolors=SURFACE,
                           linewidths=1.5, zorder=3, clip_on=False)
                # round() + 0 prints a tiny negative mean as 0.000 rather than -0.000
                ax.annotate(label_format.format(round(mean, 3) + 0), xy=(max(values.max(), mean), position),
                            xytext=(8, 0), textcoords='offset points', va='center', color=SECONDARY_INK)
            ax.set_xlim(0, x_max)
            ax.set_yticks(positions, names)
            ax.tick_params(axis='y', length=0)
            ax.grid(axis='y', visible=False)
            ax.set_title(title)
            ax.set_xlabel(f'{x_label} (bar = mean, dots = folds)')
        axes[0, 1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}%'))

        # Predicted vs. actual price for the best model, from its out-of-fold predictions
        ax = axes[0, 2]
        actual_price, predicted_price = np.expm1(y), np.expm1(best_predictions)
        cells = ax.hexbin(actual_price, predicted_price, gridsize=45, xscale='log', yscale='log', bins='log',
                          cmap=DENSITY_CMAP, mincnt=1, linewidths=0, zorder=2)
        limits = [min(actual_price.min(), predicted_price.min()), max(actual_price.max(), predicted_price.max())]
        ax.plot(limits, limits, color=MUTED_INK, linewidth=1, zorder=3)
        ax.annotate('perfect prediction', xy=(limits[1], limits[1]), xytext=(-4, -14), textcoords='offset points',
                    ha='right', color=MUTED_INK, fontsize=9)
        colorbar = fig.colorbar(cells, ax=ax)
        colorbar.set_label('Sales per cell', color=SECONDARY_INK)
        colorbar.outline.set_visible(False)
        ax.xaxis.set_major_formatter(FuncFormatter(dollars))
        ax.yaxis.set_major_formatter(FuncFormatter(dollars))
        ax.set_title(f'{best_name}: predicted vs. actual sale price')
        ax.set_xlabel('Actual sale price')
        ax.set_ylabel('Predicted sale price (out of fold)')

        # Train vs. test R² per model, the gap between them is the overfitting. The baseline only predicts the
        # mean, so it has nothing to overfit and is left out to zoom in on the gaps
        ax = axes[1, 0]
        fitted_names = [name for name in names if name != 'Baseline']
        fitted_positions = np.arange(len(fitted_names))[::-1]
        train_means = [results[name]['train_r2'].mean() for name in fitted_names]
        test_means = [results[name]['test_r2'].mean() for name in fitted_names]
        for position, train_r2, test_r2 in zip(fitted_positions, train_means, test_means):
            ax.plot([test_r2, train_r2], [position, position], color=AXIS_COLOR, zorder=2)
            ax.annotate(f'gap {train_r2 - test_r2:.3f}', xy=(max(train_r2, test_r2), position), xytext=(10, 0),
                        textcoords='offset points', va='center', color=SECONDARY_INK, fontsize=9)
        ax.scatter(train_means, fitted_positions, s=50, color=TRAIN_COLOR, edgecolors=SURFACE, linewidths=1.5,
                   zorder=3, label='Train')
        ax.scatter(test_means, fitted_positions, s=50, color=TEST_COLOR, edgecolors=SURFACE, linewidths=1.5,
                   zorder=3, label='Test')
        x_range = max(train_means) - min(test_means)
        ax.set_xlim(min(test_means) - 0.3 * x_range, max(train_means) + 0.6 * x_range)
        ax.set_ylim(-0.5, len(fitted_names) - 0.5)
        ax.set_yticks(fitted_positions, fitted_names)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='y', visible=False)
        ax.legend(loc='upper right')
        ax.set_title('Train vs. test R² (the gap shows overfitting)')
        ax.set_xlabel('Mean R² over the outer folds')

        # Validation curves: R² against alpha from the full-data sweeps
        for ax, name in [(axes[1, 1], 'Ridge'), (axes[1, 2], 'Lasso')]:
            alphas = MODELS[name][1]['alpha']
            train_scores, test_scores = validation_curves[name]
            test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)
            best = test_mean.argmax()
            ax.fill_between(alphas, test_mean - test_std, test_mean + test_std, color=TEST_COLOR, alpha=0.1,
                            linewidth=0)
            ax.plot(alphas, train_scores.mean(axis=1), color=TRAIN_COLOR, label='Train')
            ax.plot(alphas, test_mean, color=TEST_COLOR, label='Validation (±1 std)')
            ax.scatter(alphas[best], test_mean[best], s=50, color=TEST_COLOR, edgecolors=SURFACE, linewidths=1.5,
                       zorder=3)
            ax.annotate(f'best α = {alphas[best]:.3g}', xy=(alphas[best], test_mean[best]), xytext=(0, 10),
                        textcoords='offset points', ha='center', color=SECONDARY_INK, fontsize=9)
            ax.set_xscale('log')
            ax.legend(loc='lower left')
            ax.set_title(f'{name}: validation curve over α')
            ax.set_xlabel('Regularization strength α')
            ax.set_ylabel('R²')

    return fig
