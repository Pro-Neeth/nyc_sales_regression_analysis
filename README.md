# NYC Property Sale Price Prediction

Predicting New York City property sale prices from the NYC Department of Finance rolling sales data
(84,548 sales from September 2016 to August 2017). The pipeline cleans the raw records, explores them,
engineers features and compares a mean baseline, **Ridge**, **Lasso** and **gradient boosting** using nested
cross-validation, with neighborhoods target-encoded inside each fold.

## Results

Nested cross-validation (5 outer folds × 4 inner folds), predicting `log1p(SALE PRICE)` for 27,856 market sales:

| Model | Mean test R² | Std test R² | Mean train R² | Overfit gap | RMSE (log) | Median error | Most common hyperparameters |
|-------|-------------:|------------:|--------------:|------------:|-----------:|-------------:|-----------------------------|
| Baseline (predicts the mean) | 0.000 | 0.000 | 0.000 | 0.000 | 0.860 | 44.6% | – |
| Ridge | 0.734 | 0.013 | 0.749 | 0.015 | 0.443 | 19.3% | α = 2.02 |
| Lasso | 0.733 | 0.012 | 0.747 | 0.014 | 0.444 | 19.3% | α = 0.000105 |
| Gradient boosting | **0.744** | 0.013 | 0.790 | 0.047 | **0.435** | **18.2%** | learning rate 0.05, 63 leaves |

Gradient boosting is the most accurate model: its median prediction is off by 18% of the sale price, against 45%
for always predicting the mean. It beats Ridge on all five outer folds, but by a modest margin (0.003 to 0.022 R²).
Ridge and Lasso barely overfit; boosting fits the training data more tightly but still scores best on held-out
data. Both linear models pick an α inside their search range, so the grids cover the optimum.

Most of the improvement over the original version of this pipeline came from excluding non-market sales, not from the
model choice. The next section shows how much of that R² gain is a better model and how much is a change in what gets
scored.

![Model comparison](reports/figures/07_model_comparison.png)

## Excluding non-market sales

The dataset's description warns that "many sales occur with a nonsensically small dollar amount: $0 most commonly",
and that these are transfers of deeds between parties, such as parents passing a home to a child. The original
version of this pipeline dropped the $0 sales and those of $10 or less, but non-market transfers also appear at token
prices above that. The data doesn't record what each one is (a gift, an estate or trust transfer, a deal between
related parties), only a price unrelated to the property. The pipeline therefore keeps only sales of $100,000 or more.

Of the 28,516 sales the original cleaning kept, 660 were under $100,000. To check whether their prices reflect the
property, each sale is compared with the estimate of a Ridge model trained only on sales of $100,000 or more:

| Sale price | Sales | Median price per sq ft | Correlation with estimated value | Share of Ridge's squared error |
|------------|------:|-----------------------:|---------------------------------:|-------------------------------:|
| $11 – $9,999 | 227 | $0.35 | 0.06 | 51.7% |
| $10,000 – $99,999 | 433 | $16.83 | −0.03 | 17.8% |
| $100,000 and up | 27,856 | $353.69 | 0.86 | 30.6% |

- For market sales, the price tracks the property (a correlation of 0.86 on the log scale). For the cheaper sales it
  doesn't at all: the 22 sales recorded at exactly $50,000 are of properties estimated at anywhere from $424,000 to
  $4.8M. The median sale between $10,000 and $99,999 went for 5% of its estimated value.
- Their prices cluster on round numbers: $100 (60 sales), $500 (53), $25,000 (51), $1,000 (41) and $10,000 (41).
- The same lot is about three times as likely to also sell for $100,000 or more within the same year (22–24%,
  against 7% for market sales), consistent with a paper transfer before or after a real sale.

These 660 sales were 2.3% of the rows but produced 69% of Ridge's out-of-fold squared error. The cutoff costs few
genuine sales: only 15 of the 433 between $10,000 and $99,999 are priced at $100 per square foot or more. A few
non-market sales probably remain above it (59 sales of $100,000 or more are priced under $20 per square foot).

### Where the R² gain comes from

Excluding these sales raised Ridge's R² from 0.447 to 0.734, but most of that is a change in what gets scored rather
than a better model. On the same held-out sales:

| Ridge | R² | Median error | Underpricing of $100,000+ sales |
|-------|---:|-------------:|--------------------------------:|
| Trained on all sales over $10, scored on all of them | 0.447 | 23.0% | – |
| Same model, scored on $100,000+ sales only | 0.703 | 22.5% | 9.9% |
| Trained and scored on $100,000+ sales only | 0.734 | 19.4% | −0.1% |

- 0.256 of the 0.287 gain comes from no longer scoring the model on transfers whose prices no property-based model
  can predict.
- The remaining 0.031 is a real improvement. The transfers pulled the fit down, so the model underpriced genuine
  sales by about 10%; without them the bias disappears and the median error on genuine sales falls from 22.5% to 19.4%.
- The floor wasn't chosen by R², which keeps rising as the floor goes up (0.771 at $200,000 and 0.797 at $300,000)
  because each step drops more real but hard-to-price sales. The real improvement, measured on the same $100,000+
  sales, levels off as the floor approaches $100,000: 0.703 with the original cutoff, 0.717 at $1,000, 0.723 at
  $10,000, 0.733 at $50,000 and 0.734 at $100,000.

`python -m nyc_sales.cleaning_analysis` reproduces every number in this section, along with the scope figures under
Notes.

## Pipeline

| Stage | Module | What it does |
|-------|--------|--------------|
| Cleaning | `nyc_sales/cleaning.py` | Fixes dtypes, splits the sale date, and drops duplicates, sales under $100,000, missing tax classes, and zero or missing ZIP code, year built and square footage (84,548 → 27,856 rows) |
| EDA | `nyc_sales/eda.py` | Skew and category summaries; histograms, correlation heatmap, pairplot, category counts and sale price box plots |
| Features | `nyc_sales/features.py` | Adds age of property and sale season; keeps the tax and building class at the time of sale (the "at present" columns are recorded after the sale, so using them would leak future information); drops identifier-like columns; one-hot encodes; log-scales price, area and unit counts |
| Modeling | `nyc_sales/modeling.py` | Scales continuous features and target-encodes `NEIGHBORHOOD` inside each fold, then tunes a mean baseline, Ridge, Lasso and gradient boosting in nested CV and compares them |
| Orchestration | `nyc_sales/pipeline.py` | Runs the stages in order and saves the figures and results |

## Getting started

Requires Python 3.11 or newer (tested with 3.12).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `nyc-rolling-sales.csv` from the
[NYC Property Sales dataset on Kaggle](https://www.kaggle.com/datasets/new-york-city/nyc-property-sales)
and save it as `data/raw/nyc-rolling-sales.csv`. From a terminal:

```bash
curl -L -o nyc-property-sales.zip https://www.kaggle.com/api/v1/datasets/download/new-york-city/nyc-property-sales
unzip nyc-property-sales.zip -d data/raw
```

The results in this README come from the file with SHA-256 checksum
`97228c77c1558c70cac0d8949556fd6dc43d70d8deebc66d6333d76e5ee817be` (check it with
`sha256sum data/raw/nyc-rolling-sales.csv`, or `certutil -hashfile data\raw\nyc-rolling-sales.csv SHA256` on Windows).

## Usage

Run from the repository root:

```bash
python -m nyc_sales              # full run: cleaning, EDA printouts and figures, modeling
python -m nyc_sales --skip-eda   # cleaning, features and modeling only
python -m nyc_sales --show       # also open the figures in windows
python -m nyc_sales --data path/to/nyc-rolling-sales.csv --reports-dir path/to/output
python -m nyc_sales.cleaning_analysis   # the analysis behind the price cutoff (prints only, about a minute)
```

A full run takes about 3 minutes on a 12-core machine. Each run writes:

- `reports/figures/`: the six EDA figures (`01`–`06`) and the model comparison (`07_model_comparison.png`)
- `reports/results/`: per-fold nested CV results for each model (`baseline_nested_cv.csv`, `ridge_nested_cv.csv`,
  `lasso_nested_cv.csv`, `gradient_boosting_nested_cv.csv`) and `model_comparison.csv`

The stages can also be used from Python:

```python
from nyc_sales.cleaning import load_raw_data, clean_data
from nyc_sales.features import engineer_features
from nyc_sales.modeling import run_modeling

nyc_df = clean_data(load_raw_data())
results, comparison_df, fig = run_modeling(engineer_features(nyc_df))
```

## Project structure

```
├── data/raw/                       # nyc-rolling-sales.csv goes here (not committed)
├── notebooks/
│   └── nyc_sales_analysis.ipynb    # original exploratory analysis, with commentary
├── nyc_sales/
│   ├── config.py                   # paths, price cutoff and hyperparameter grids
│   ├── cleaning.py                 # loading and cleaning
│   ├── cleaning_analysis.py        # the analysis behind the price cutoff and the R² breakdown
│   ├── eda.py                      # distribution and correlation analysis
│   ├── features.py                 # feature engineering
│   ├── modeling.py                 # baseline, Ridge, Lasso and gradient boosting with nested CV
│   └── pipeline.py                 # command-line entry point
├── reports/
│   ├── figures/
│   └── results/
└── requirements.txt
```

## Notes

- R² and RMSE are on the log scale the models are trained on (`log1p(SALE PRICE)`). The median error is in
  dollars: the median of |predicted − actual| / actual on each outer fold, averaged over the folds.
- Every random step (the CV splits, the target encoder's internal folds and gradient boosting) is seeded with
  `RANDOM_STATE` in `config.py`, so reruns produce byte-identical figures and CSVs, including in a fresh environment
  installed from `requirements.txt`.
- The model covers houses and whole buildings: 87% of the cleaned sales are one- to three-family homes. Condo and
  co-op units are 43% of the raw sales, but 99.8% of them have no recorded square footage, so almost none survive
  cleaning.
- pandas is pinned below 3.0 because the cleaning step relies on how pandas 2.x replaces values in
  categorical columns.
- The notebook was written in Google Colab (its first cell mounts Google Drive). The `nyc_sales` package started
  as the runnable version of the same analysis and has since moved on: it drops sales under $100,000 instead of
  $10, uses the time-of-sale tax and building classes, and adds the baseline and gradient boosting models.
