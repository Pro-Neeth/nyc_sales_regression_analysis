"""Distribution and correlation analysis of the cleaned data (notebook sections 3-4)."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from nyc_sales.config import PAIRPLOT_SAMPLE_SIZE, RANDOM_STATE


def run_eda(nyc_df):
    """Print the numeric and categorical summaries and build the EDA figures.

    Returns {filename: figure} for the caller to save or show.
    """
    figures = {}

    # Checking skew
    print(nyc_df.skew(numeric_only=True))

    # Plotting the histogram distributions with thresholds to reduce extreme outliers
    columns_to_exclude = ['SALE YEAR', 'SALE MONTH', 'BLOCK', 'LOT', 'APARTMENT NUMBER']
    numeric_cols = nyc_df.select_dtypes(include=['number']).columns.difference(columns_to_exclude)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        if col in ['YEAR BUILT', 'ZIP CODE']:
            axes[i].hist(nyc_df[col], bins=100)
            axes[i].set_title(f'Count of {col}')
            axes[i].set_ylabel('Count')
            axes[i].set_xlabel(col)

        elif col in ['COMMERCIAL UNITS']:
            threshold_upper = nyc_df[col].quantile(.99)

            nyc_df_plot = nyc_df[(nyc_df[col] < threshold_upper)]
            axes[i].hist(nyc_df_plot[col], bins=100)
            axes[i].set_title(f'Count of {col}')
            axes[i].set_ylabel('Count')
            axes[i].set_xlabel(col)

        else:
            threshold_upper = nyc_df[col].quantile(.95)
            nyc_df_plot = nyc_df[(nyc_df[col] < threshold_upper)]
            axes[i].hist(nyc_df_plot[col], bins=100)
            axes[i].set_title(f'Count of {col}')
            axes[i].set_ylabel('Count')
            axes[i].set_xlabel(col)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    figures['01_numeric_distributions.png'] = fig

    # Create correlation matrix for numerical features
    corr_matrix = nyc_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    figures['02_correlation_heatmap.png'] = fig

    # Random sample to optimize plotting for Sale price, LSF, GSF and also optimized for outliers
    sample_df = nyc_df.sample(n=PAIRPLOT_SAMPLE_SIZE, random_state=RANDOM_STATE)

    cols_plot = ['SALE PRICE', 'GROSS SQUARE FEET', 'LAND SQUARE FEET', 'YEAR BUILT',
                 'ZIP CODE', 'TOTAL UNITS', 'COMMERCIAL UNITS', 'RESIDENTIAL UNITS']
    nyc_df_plot = sample_df[cols_plot].copy()

    mask = pd.Series([True] * len(nyc_df_plot), index=nyc_df_plot.index)

    for col in cols_plot:
        if col in ['SALE PRICE', 'GROSS SQUARE FEET', 'LAND SQUARE FEET', 'ZIP CODE', 'YEAR BUILT']:
            threshold_upper = nyc_df_plot[col].quantile(.95)
            mask &= (nyc_df_plot[col] < threshold_upper) & (nyc_df_plot[col] > 1)

        elif col in ['TOTAL UNITS', 'COMMERCIAL UNITS', 'RESIDENTIAL UNITS']:
            threshold_upper = nyc_df_plot[col].quantile(1)
            mask &= (nyc_df_plot[col] < threshold_upper)

    nyc_df_plot = nyc_df_plot[mask]

    figures['03_pairplot.png'] = sns.pairplot(nyc_df_plot, diag_kind='kde').figure

    # Printing Unique Values for each categorical data type
    cat_cols = nyc_df.select_dtypes(include=['category']).columns.tolist()
    print(cat_cols)
    for col in cat_cols:
        unique_vals = nyc_df[col].unique()
        print(f"\n{col} ({len(unique_vals)} unique):")
        print(f"  {unique_vals}")

    cat_cols_plot = ['BOROUGH', 'TAX CLASS AT PRESENT', 'TAX CLASS AT TIME OF SALE',
                     'BUILDING CLASS AT PRESENT', 'BUILDING CLASS AT TIME OF SALE', 'BUILDING CLASS CATEGORY']

    fig, axes = plt.subplots(2, 3, figsize=(20, 6))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols_plot):

        # Building class features have too many unique values to plot, so only the top 5 are shown
        if col in ['BUILDING CLASS AT PRESENT', 'BUILDING CLASS AT TIME OF SALE', 'BUILDING CLASS CATEGORY']:
            top_5 = nyc_df[col].value_counts().head(5).index
            sns.countplot(data=nyc_df[nyc_df[col].isin(top_5)], y=col, ax=axes[i], order=top_5)
            axes[i].set_title(f'Top 5 {col}')
        else:
            sns.countplot(data=nyc_df, y=col, ax=axes[i], order=nyc_df[col].value_counts().index)
            axes[i].set_title(f'Distribution of {col}')

        axes[i].set_xlabel('Count')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    figures['04_categorical_distributions.png'] = fig

    # Filter sale price to 95th percentile
    price_95 = nyc_df['SALE PRICE'].quantile(0.95)
    nyc_df_filtered = nyc_df[nyc_df['SALE PRICE'] <= price_95].copy()

    for col, filename in [('BOROUGH', '05_sale_price_by_borough.png'),
                          ('TAX CLASS AT TIME OF SALE', '06_sale_price_by_tax_class.png')]:
        fig = plt.figure(figsize=(10, 8))

        order = nyc_df_filtered[col].value_counts().index.tolist()

        sns.boxplot(x=col, y='SALE PRICE', data=nyc_df_filtered,
                    hue=col, order=order, palette='Set2',
                    legend=False)

        plt.title(f'Box Plot for SALE PRICE and {col}')
        plt.xlabel(col)
        plt.ylabel('SALE PRICE')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        figures[filename] = fig

    return figures
