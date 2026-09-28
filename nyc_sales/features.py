"""Feature engineering and final preprocessing"""

import numpy as np
import pandas as pd


# Function to convert months to season

def month_to_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


def engineer_features(nyc_df, verbose=False):
    """Turn the cleaned data into the log-scaled, one-hot encoded data used for modeling.

    With verbose=True, the analysis behind the feature decisions is printed along the way.
    """
    # Dropping ADDRESS, it is more of an identifier
    nyc_df = nyc_df.drop(['ADDRESS'], axis=1)

    # Creating new feature AGE OF PROPERTY and dropping SALE YEAR and YEAR BUILT
    nyc_df['AGE OF PROPERTY'] = nyc_df['SALE YEAR'] - nyc_df['YEAR BUILT']
    nyc_df = nyc_df.drop(['SALE YEAR', 'YEAR BUILT'], axis=1)

    # Dropping Total Units, it is a linear combination of RESIDENTIAL UNITS and COMMERCIAL UNITS
    nyc_df = nyc_df.drop(['TOTAL UNITS'], axis=1)

    # Dropping BLOCK and LOT, they act more like IDs
    nyc_df = nyc_df.drop(['BLOCK', 'LOT'], axis=1)

    if verbose:
        # Create dataframes for mismatches of each type of class
        tax_class_mismatch = nyc_df[nyc_df['TAX CLASS AT PRESENT'] != nyc_df['TAX CLASS AT TIME OF SALE']]
        print(f"TAX CLASS mismatches: {len(tax_class_mismatch)/len(nyc_df)*100}% of samples")

        building_class_mismatch = nyc_df[nyc_df['BUILDING CLASS AT PRESENT'] != nyc_df['BUILDING CLASS AT TIME OF SALE']]
        print(f"BUILDING CLASS mismatches: {len(building_class_mismatch)/len(nyc_df)*100}% samples")

        either_mismatch = nyc_df[
            (nyc_df['TAX CLASS AT PRESENT'] != nyc_df['TAX CLASS AT TIME OF SALE']) |
            (nyc_df['BUILDING CLASS AT PRESENT'] != nyc_df['BUILDING CLASS AT TIME OF SALE'])
        ]
        print(f"Either mismatch: {len(either_mismatch)/len(nyc_df)*100}% of samples")

        both_mismatch = nyc_df[
            (nyc_df['TAX CLASS AT PRESENT'] != nyc_df['TAX CLASS AT TIME OF SALE']) &
            (nyc_df['BUILDING CLASS AT PRESENT'] != nyc_df['BUILDING CLASS AT TIME OF SALE'])
        ]
        print(f"Both mismatch: {len(both_mismatch)/len(nyc_df)*100}% of samples")

    # The AT PRESENT classes were recorded when the dataset was published, after the sale, so they aren't known
    # at the time of sale. Few samples changed class anyway, so keep only the TIME OF SALE columns
    nyc_df = nyc_df.drop(['TAX CLASS AT PRESENT', 'BUILDING CLASS AT PRESENT'], axis=1)

    # BUILDING CLASS AT TIME OF SALE is a more detailed version of BUILDING CLASS CATEGORY
    nyc_df = nyc_df.drop(['BUILDING CLASS CATEGORY'], axis=1)

    # NEIGHBORHOOD captures location better than the postal ZIP CODE
    nyc_df = nyc_df.drop(['ZIP CODE'], axis=1)

    # Converting neighborhood to object type because it needs special encoding scheme later on
    nyc_df['NEIGHBORHOOD'] = nyc_df['NEIGHBORHOOD'].astype(object)

    if verbose:
        print(nyc_df.describe().apply(lambda x: x.apply('{0:.2f}'.format)).transpose())

    # Create SALE SEASON column
    nyc_df['SALE SEASON'] = nyc_df['SALE MONTH'].apply(month_to_season)

    # Convert to category dtype
    nyc_df['SALE SEASON'] = nyc_df['SALE SEASON'].astype('category')

    # Drop SALE MONTH now that SALE SEASON replaces it
    nyc_df = nyc_df.drop(['SALE MONTH'], axis=1)

    # One Hot Encoding
    columns_to_encode = ['BOROUGH', 'TAX CLASS AT TIME OF SALE', 'BUILDING CLASS AT TIME OF SALE', 'SALE SEASON']

    for col in nyc_df.select_dtypes(include='category').columns:
        nyc_df[col] = nyc_df[col].astype(str)

    nyc_df_encoded = pd.get_dummies(nyc_df, columns=columns_to_encode, dtype=int)

    if verbose:
        print(nyc_df_encoded.skew(numeric_only=True))

    # Create dataframe for model specifically
    nyc_df_ols = nyc_df_encoded.copy()

    # Perform log+1 transformation to account for 0 values
    nyc_df_ols['SALE PRICE'] = np.log1p(nyc_df_encoded['SALE PRICE'])
    nyc_df_ols['LAND SQUARE FEET'] = np.log1p(nyc_df_encoded['LAND SQUARE FEET'])
    nyc_df_ols['GROSS SQUARE FEET'] = np.log1p(nyc_df_encoded['GROSS SQUARE FEET'])
    nyc_df_ols['RESIDENTIAL UNITS'] = np.log1p(nyc_df_encoded['RESIDENTIAL UNITS'])
    nyc_df_ols['COMMERCIAL UNITS'] = np.log1p(nyc_df_encoded['COMMERCIAL UNITS'])

    return nyc_df_ols
