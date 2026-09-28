"""Loading, initial cleaning and removal of anomalies (notebook sections 1-2)."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from nyc_sales.config import KAGGLE_DATASET_URL, RAW_DATA_PATH


def load_raw_data(path=RAW_DATA_PATH):
    """Load the nyc rolling sales data set."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download nyc-rolling-sales.csv from "
            f"{KAGGLE_DATASET_URL} and place it there, or pass --data PATH."
        )
    return pd.read_csv(path)


def clean_data(nyc_df, verbose=False):
    """Fix the dtypes and drop the anomalous samples.

    With verbose=True, the analysis behind each cleaning decision is printed along the way.
    """
    if verbose:
        # Check first few samples of data set to ensure proper loading
        print(nyc_df.head())
        nyc_df.info()

    # Indexing column, no use of it
    nyc_df = nyc_df.drop(columns=['Unnamed: 0'])

    # Remove duplicate values
    nyc_df = nyc_df.drop_duplicates().reset_index(drop=True)

    # Extract year and month then drop SALE DATE
    nyc_df['SALE DATE'] = pd.to_datetime(nyc_df['SALE DATE'])
    nyc_df['SALE YEAR'] = nyc_df['SALE DATE'].dt.year
    nyc_df['SALE MONTH'] = nyc_df['SALE DATE'].dt.month

    nyc_df = nyc_df.drop(columns=['SALE DATE'])

    # Convert columns to numeric format
    num_col = ['APARTMENT NUMBER', 'ZIP CODE', 'LAND SQUARE FEET', 'GROSS SQUARE FEET', 'SALE PRICE']
    nyc_df[num_col] = nyc_df[num_col].apply(pd.to_numeric, errors='coerce')

    # Convert to categorical format
    col_list = ['BOROUGH', 'NEIGHBORHOOD', 'ADDRESS', 'BUILDING CLASS CATEGORY', 'TAX CLASS AT PRESENT',
                'BUILDING CLASS AT PRESENT', 'TAX CLASS AT TIME OF SALE', 'BUILDING CLASS AT TIME OF SALE']

    nyc_df[col_list] = nyc_df[col_list].astype('category')

    # For clarity purposes
    nyc_df['BOROUGH'] = nyc_df['BOROUGH'].cat.rename_categories({
        1: 'Manhattan',
        2: 'Brooklyn',
        3: 'Queens',
        4: 'Bronx',
        5: 'Staten Island'
    })

    if verbose:
        # Print number of unique values per column
        unique_counts = nyc_df.nunique()

        print("Number of unique values in each column:\n")
        for col, count in unique_counts.items():
            print(f"{col}: {count}")

    # Drop EASE-MENT because only one unique value
    nyc_df = nyc_df.drop(columns=['EASE-MENT'])

    if verbose:
        # Check the metrics for the numerical features
        print(nyc_df.describe().apply(lambda x: x.apply('{0:.2f}'.format)).transpose())

    # Replace missing values with NaN
    with warnings.catch_warnings():
        # pandas 2.x warns that replace() on categorical columns changes in pandas 3, hence pandas<3 in requirements.txt
        warnings.simplefilter('ignore', FutureWarning)
        nyc_df = nyc_df.replace(['', ' ', '  ', '-'], np.nan)

    if verbose:
        # Print percentage of $0 sales
        freq = ((nyc_df['SALE PRICE'] == 0).sum())/(len(nyc_df))*100
        print(f'Percent of $0 sales: {freq}%\n')

        # Print percentage of NaN values
        print('Sum of NaNs by column and percentage: \n')
        print(nyc_df.isna().sum()[nyc_df.isna().any()])
        print("----------------------------------")
        print(nyc_df.isna().mean() * 100)

    # Dropping $0 sales, they are transfers of ownership rather than real sales
    nyc_df = nyc_df[nyc_df['SALE PRICE'] != 0].reset_index(drop=True)

    # High amount of NA values and also specific apartment number won't be a strong feature, so we drop
    nyc_df = nyc_df.drop(columns=['APARTMENT NUMBER'])

    if verbose:
        # Check if the two columns have missing values in the same samples
        missing_col1 = nyc_df[nyc_df['TAX CLASS AT PRESENT'].isna()].index
        missing_col2 = nyc_df[nyc_df['BUILDING CLASS AT PRESENT'].isna()].index

        print(f"Are these the same samples? {missing_col1.tolist() == missing_col2.tolist()}")

        # Taking a look to see other correlations
        missing_df = nyc_df[nyc_df['TAX CLASS AT PRESENT'].isna()]
        print(missing_df.head(20))

        # Checking if it may also be resulted from anomalies in other columns
        value = 0
        column = 'TOTAL UNITS'
        all_equal = (missing_df[column] == value).all()
        print(f"All values in '{column}' equal '{value}': {'Yes' if all_equal else 'No'}")

        print(missing_df['LAND SQUARE FEET'].unique())
        print(missing_df['GROSS SQUARE FEET'].unique())

    # Dropping NaN values in Tax class at present
    nyc_df = nyc_df.dropna(subset=['TAX CLASS AT PRESENT'])

    if verbose:
        # Checking if they have the same unique values or not
        print(nyc_df['TAX CLASS AT PRESENT'].unique())
        print(nyc_df['TAX CLASS AT TIME OF SALE'].unique())

    # Re-Mapping the values, the tax subclasses (e.g. 2A, 2B) collapse into their main class
    nyc_df['TAX CLASS AT PRESENT'] = nyc_df['TAX CLASS AT PRESENT'].astype(str).str[0].astype(int)
    nyc_df['TAX CLASS AT PRESENT'] = nyc_df['TAX CLASS AT PRESENT'].astype('category')

    if verbose:
        # Abnormal is defined as when some features = 0 when they shouldnt or if houses are sold for $1

        # Inconsistent is defined as when residential + commercial > total
        numeric_cols = ['SALE PRICE', 'LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT', 'ZIP CODE', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS']

        for col in numeric_cols:
            if col in ['RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS']:
                continue

            abnormal = ((nyc_df[col] <= 10).sum() / len(nyc_df)) * 100 if col == 'SALE PRICE' else ((nyc_df[col] == 0).sum() / len(nyc_df)) * 100
            nan_val = (nyc_df[col].isna().sum() / len(nyc_df)) * 100
            total = abnormal + nan_val

            print(f"{col}: Abnormal={abnormal:.2f}% | NaN={nan_val:.2f}% | Total={total:.2f}%")

        print(f"\nRESIDENTIAL/COMMERCIAL/TOTAL UNITS:")
        nan_pct = ((nyc_df[['RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS']].isna().any(axis=1)).sum() / len(nyc_df)) * 100

        # Only check inconsistency where all three have non-null values
        non_na = nyc_df[['RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS']].notna().all(axis=1)
        inconsistent = ((non_na) & (nyc_df['TOTAL UNITS'] < (nyc_df['RESIDENTIAL UNITS'] + nyc_df['COMMERCIAL UNITS']))).sum() / len(nyc_df) * 100

        print(f"  NaN={nan_pct:.2f}% | Inconsistent={inconsistent:.2f}% | Total={nan_pct + inconsistent:.2f}%")

        correlation_matrix = nyc_df.corr(numeric_only=True)
        print(correlation_matrix)

    # Dropping zip codes and inconsistent unit size
    nyc_df = nyc_df[nyc_df['ZIP CODE'] != 0].reset_index(drop=True)
    nyc_df = nyc_df[nyc_df['TOTAL UNITS'] >= nyc_df['RESIDENTIAL UNITS'] + nyc_df['COMMERCIAL UNITS']].reset_index(drop=True)

    if verbose:
        # Extracting samples year built = 0
        extracted_df = nyc_df.loc[nyc_df['YEAR BUILT'] == 0]

        print(extracted_df.shape)
        unique_counts = extracted_df.nunique()

        print("Number of unique values in each column:\n")
        for col, count in unique_counts.items():
            print(f"{col}: {count}")

    # Dropping samples where YEAR BUILT was never recorded
    nyc_df = nyc_df[nyc_df['YEAR BUILT'] != 0].reset_index(drop=True)

    if verbose:
        print(nyc_df.skew(numeric_only=True))

    # Dropping anomalies among LSF, GSF, and Sale price
    nyc_df = nyc_df[nyc_df['LAND SQUARE FEET'] != 0].reset_index(drop=True)
    nyc_df = nyc_df[nyc_df['GROSS SQUARE FEET'] != 0].reset_index(drop=True)
    nyc_df = nyc_df[nyc_df['SALE PRICE'] > 10].reset_index(drop=True)
    nyc_df = nyc_df.dropna(subset=['GROSS SQUARE FEET', 'LAND SQUARE FEET', 'SALE PRICE']).reset_index(drop=True)
    return nyc_df
