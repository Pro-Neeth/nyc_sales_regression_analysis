"""The analysis behind the cleaning decisions in the README: what the cleaned data covers, why sales under
MIN_SALE_PRICE are excluded, and how excluding them changed Ridge's R²

Run from the repository root: python -m nyc_sales.cleaning_analysis [--data PATH]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from nyc_sales.cleaning import clean_data, load_raw_data
from nyc_sales.config import MIN_SALE_PRICE, OUTER_CV_SPLITS, RANDOM_STATE, RAW_DATA_PATH, RIDGE_ALPHAS
from nyc_sales.features import engineer_features
from nyc_sales.modeling import make_preprocessor, median_pct_error

# Price floors compared at the end; 0 keeps every sale over $10, as the original version of the pipeline did
FLOORS = [0, 1_000, 10_000, 50_000, MIN_SALE_PRICE, 200_000, 300_000]
BOROUGH_CODES = {'Manhattan': 1, 'Brooklyn': 2, 'Queens': 3, 'Bronx': 4, 'Staten Island': 5}


def run_cleaning_analysis(raw_df):
    """Print what the cleaned data covers, the evidence that sales under MIN_SALE_PRICE aren't market sales, and how
    excluding them changed Ridge's R².

    The price analysis starts from the data as the original pipeline cleaned it, keeping every sale over $10, and
    scores every Ridge model on the same outer folds. Returns the printed tables.
    """
    # 1. Scope. Condo and co-op units almost never record square footage, so the square footage filters drop them
    category = raw_df['BUILDING CLASS CATEGORY'].str.strip()
    condo_or_coop = category.str.contains('CONDO|COOP')
    gross_sqft = pd.to_numeric(raw_df['GROSS SQUARE FEET'], errors='coerce')
    land_sqft = pd.to_numeric(raw_df['LAND SQUARE FEET'], errors='coerce')
    no_sqft = gross_sqft.isna() | (gross_sqft == 0) | land_sqft.isna() | (land_sqft == 0)
    cleaned_category = clean_data(raw_df.copy())['BUILDING CLASS CATEGORY'].astype(str).str.strip()

    print("=" * 70)
    print("WHAT THE CLEANED DATA COVERS")
    print("=" * 70)
    print(f"Condo and co-op units: {condo_or_coop.mean() * 100:.1f}% of raw sales, "
          f"{no_sqft[condo_or_coop].mean() * 100:.1f}% of them without square footage")
    print(f"One- to three-family homes: {cleaned_category.str[:2].isin(['01', '02', '03']).mean() * 100:.1f}% "
          f"of cleaned sales")

    # 2. The data as the original pipeline cleaned it, keeping every sale over $10
    nyc_df = clean_data(raw_df.copy(), min_sale_price=0)
    nyc_df = nyc_df[nyc_df['SALE PRICE'] > 10].reset_index(drop=True)
    price = nyc_df['SALE PRICE']
    market = (price >= MIN_SALE_PRICE).to_numpy()
    model_df = engineer_features(nyc_df)
    X = model_df.drop('SALE PRICE', axis=1)
    y = model_df['SALE PRICE']

    # Out-of-fold predictions for every sale from Ridge trained only on the sales at or above each floor, on the same
    # outer folds. The model trained on MIN_SALE_PRICE+ sales estimates each property's market value, including for
    # the cheaper sales it never sees
    fold = np.empty(len(y), dtype=int)
    outer_cv = KFold(n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold_id, (_, test_idx) in enumerate(outer_cv.split(X)):
        fold[test_idx] = fold_id
    predictions = {floor: out_of_fold_predictions(X, y, fold, (price >= floor).to_numpy()) for floor in FLOORS}
    original_pred, market_pred = predictions[0], predictions[MIN_SALE_PRICE]

    # Compare the cheaper sales with market sales: price per square foot, how the price relates to the estimated
    # market value, and each group's share of the original model's squared error
    band = pd.cut(price, [0, 10_000, MIN_SALE_PRICE, np.inf], right=False,
                  labels=['$11 - $9,999', f'$10,000 - ${MIN_SALE_PRICE - 1:,}', f'${MIN_SALE_PRICE:,}+'])
    in_band = {label: (band == label).to_numpy() for label in band.cat.categories}
    price_per_sqft = price / nyc_df['GROSS SQUARE FEET']
    price_to_estimate = price / np.expm1(market_pred)
    squared_error = (y - original_pred) ** 2

    # How often the same lot (borough, block and lot) also sold at a market price within the 12 months of data
    raw_sales = raw_df.drop(columns=['Unnamed: 0']).drop_duplicates()
    raw_market = pd.to_numeric(raw_sales['SALE PRICE'], errors='coerce') >= MIN_SALE_PRICE
    market_sales_per_lot = raw_market.groupby([raw_sales['BOROUGH'], raw_sales['BLOCK'], raw_sales['LOT']]).sum()
    lots = pd.MultiIndex.from_arrays([nyc_df['BOROUGH'].astype(str).map(BOROUGH_CODES), nyc_df['BLOCK'],
                                      nyc_df['LOT']])
    other_market_sales = market_sales_per_lot.reindex(lots).to_numpy() - market  # a market sale counts itself

    band_table = pd.DataFrame({
        'sales': [rows.sum() for rows in in_band.values()],
        'median price per sq ft': [price_per_sqft[rows].median() for rows in in_band.values()],
        'corr. with estimated value': [np.corrcoef(y[rows], market_pred[rows])[0, 1] for rows in in_band.values()],
        'median price / estimated value': [price_to_estimate[rows].median() for rows in in_band.values()],
        'lot also sold at market %': [(other_market_sales[rows] >= 1).mean() * 100 for rows in in_band.values()],
        'share of squared error %': [squared_error[rows].sum() / squared_error.sum() * 100
                                     for rows in in_band.values()],
    }, index=list(in_band))

    print("\n" + "=" * 70)
    print(f"SALES UNDER ${MIN_SALE_PRICE:,}")
    print("=" * 70)
    print(f"Sales kept by the original cleaning (over $10): {len(y)}, of which {(~market).sum()} "
          f"({(~market).mean() * 100:.1f}%) under ${MIN_SALE_PRICE:,}")
    print("Estimated value: Ridge trained only on sales of at least "
          f"${MIN_SALE_PRICE:,}; correlation on the log scale\n")
    print(band_table.round({'median price per sq ft': 2, 'corr. with estimated value': 3,
                            'median price / estimated value': 3, 'lot also sold at market %': 1,
                            'share of squared error %': 1}).to_string())

    print(f"\nMost common prices under ${MIN_SALE_PRICE:,}:")
    print(price[~market].value_counts().head(6).to_string())

    mid_band = in_band[f'$10,000 - ${MIN_SALE_PRICE - 1:,}']
    print(f"\nSales of $10,000 - ${MIN_SALE_PRICE - 1:,} priced at $100+ per sq ft: "
          f"{(mid_band & (price_per_sqft >= 100).to_numpy()).sum()} of {mid_band.sum()}")
    print(f"Sales of ${MIN_SALE_PRICE:,}+ priced under $20 per sq ft: "
          f"{(market & (price_per_sqft < 20).to_numpy()).sum()} of {market.sum()}")

    at_50k = (price == 50_000).to_numpy()
    examples = nyc_df.loc[at_50k, ['NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'GROSS SQUARE FEET']].assign(
        **{'estimated value': np.expm1(market_pred[at_50k]).round(-3)})
    print(f"\nThe {at_50k.sum()} sales recorded at exactly $50,000 have estimated values from "
          f"${examples['estimated value'].min():,.0f} to ${examples['estimated value'].max():,.0f}. The highest:")
    print(examples.sort_values('estimated value', ascending=False).head(5).to_string(index=False))

    # 3. How much of the R² gain is a change in what gets scored and how much is a better model, on the same
    # held-out sales
    market_y = y[market]
    decomposition = pd.DataFrame({
        'R²': [r2_score(y, original_pred), r2_score(market_y, original_pred[market]),
               r2_score(market_y, market_pred[market])],
        'median % error': [median_pct_error(y, original_pred), median_pct_error(market_y, original_pred[market]),
                           median_pct_error(market_y, market_pred[market])],
        f'underpricing of ${MIN_SALE_PRICE:,}+ sales %': [
            np.nan, np.expm1((market_y - original_pred[market]).mean()) * 100,
            np.expm1((market_y - market_pred[market]).mean()) * 100],
    }, index=['trained on all sales over $10, scored on all', f'same model, scored on ${MIN_SALE_PRICE:,}+ only',
              f'trained and scored on ${MIN_SALE_PRICE:,}+ only'])

    print("\n" + "=" * 70)
    print("WHERE THE R² GAIN COMES FROM")
    print("=" * 70)
    print(decomposition.round(3).to_string())

    # 4. Moving the floor: R² keeps rising as it goes up, so the floor can't be chosen by R²
    floor_rows = []
    for floor in FLOORS:
        kept = (price >= floor).to_numpy()
        floor_rows.append({
            'floor': 'none (over $10)' if floor == 0 else f'${floor:,}',
            'sales kept': kept.sum(),
            'R² on kept sales': r2_score(y[kept], predictions[floor][kept]),
            'median % error on kept sales': median_pct_error(y[kept], predictions[floor][kept]),
            f'R² on ${MIN_SALE_PRICE:,}+ sales': (r2_score(market_y, predictions[floor][market])
                                                  if floor <= MIN_SALE_PRICE else np.nan),
        })
    floor_table = pd.DataFrame(floor_rows)

    print("\n" + "=" * 70)
    print("MOVING THE FLOOR")
    print("=" * 70)
    print(floor_table.round(3).to_string(index=False))

    return {'bands': band_table, 'decomposition': decomposition, 'floors': floor_table}


def out_of_fold_predictions(X, y, fold, train_rows):
    """Predict every sale with Ridge (alpha chosen by leave-one-out CV) fit on the train_rows of the other folds."""
    predictions = pd.Series(np.nan, index=y.index)
    for fold_id in range(fold.max() + 1):
        train = train_rows & (fold != fold_id)
        test = fold == fold_id
        model = Pipeline([('preprocessor', make_preprocessor(X)), ('regressor', RidgeCV(alphas=RIDGE_ALPHAS))])
        predictions[test] = model.fit(X[train], y[train]).predict(X[test])
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python -m nyc_sales.cleaning_analysis',
        description='Print the analysis behind the cleaning decisions described in the README.')
    parser.add_argument('--data', type=Path, default=RAW_DATA_PATH,
                        help='path to nyc-rolling-sales.csv (default: data/raw/nyc-rolling-sales.csv)')
    args = parser.parse_args()

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    try:
        run_cleaning_analysis(load_raw_data(args.data))
    except FileNotFoundError as exc:
        parser.exit(1, f"error: {exc}\n")
