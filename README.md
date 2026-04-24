# Sydney Property Price & Type Prediction

A  machine learning pipeline built on a Sydney real estate dataset to **predict property prices (regression)** 
and **predict property types (classification)**. Developed as part of COMP9321 – Data Services Engineering at UNSW Sydney.

---

## Project Overview

This project solves two real-world ML problems on 6,000+ Sydney property sale records spanning 2016–2021:

- **Part I — Regression:** Predict the sale price of a property (float output)
- **Part II — Classification:** Predict the property type (House, Apartment, Townhouse, etc. — 14 classes)

The solution is modular, production-ready, and accepts `train.csv` + `test.csv` as CLI arguments, 
outputting two separate prediction CSVs.

---

## Model Performance

| Task | Model | Metric | Train | Test |
|---|---|---|---|---|
| Regression | HistGradientBoostingRegressor | MAE | $196,327 | $305,069 |
| Classification | Random Forest Classifier | Weighted F1 | 0.990 | 0.897 |

---

### Preprocessing (Shared)
- Dropped 30+ irrelevant/leaky features (demographic stats, lifestyle ratings, redundant geo fields)
- Resolved multicollinearity: merged `time_to_cbd_public_transport` + `time_to_cbd_driving` → `commute_time` (mean of both)
- Zero-value imputation: treated `0` values as missing; filled with column means where appropriate

### Feature Engineering

| Feature | Description | Used In |
|---|---|---|
| `num_of_rooms` | `num_bed + num_bath` | Both |
| `inverse_cbd_distance` | `1 / (km_from_cbd + 1)` | Both |
| `commute_time` | Average of public + driving CBD time | Both |
| `is_sold_in_2021` | Binary flag — pandemic boom year | Regression |
| `years_diff` | `2022 - year_sold` — property age proxy | Regression |
| `is_cbd` | Binary: `km_from_cbd < 10` | Regression |
| `suburb_median_price` | Type-aware median: house/apartment/average | Regression |
| `region_group` | Ordinal encoding of 15 Sydney regions by mean price | Regression |
| `low_suburb_price` / `high_suburb_price` | Binary flags for South West / Eastern Suburbs extremes | Regression |

### Regression — HistGradientBoostingRegressor
```python
HistGradientBoostingRegressor(
    learning_rate=0.15,
    max_iter=500,
    max_depth=7,
    loss='absolute_error',
    random_state=42
)
```
- **Why HGB over others?** Handles missing values natively, no feature scaling required, 
  captures non-linear price patterns (e.g. 2021 COVID boom). MAE loss function chosen 
  for robustness against high-value outliers (multi-million dollar estates).
- **Benchmarked against:** SVM Regressor (MAE ~$500k), Ridge Regression (MAE ~$460k)

### Classification — Random Forest Classifier
```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=16,
    random_state=42,
    class_weight='balanced'
)
```
- **Why RF over others?** Handles severely imbalanced classes (House: 4,110 records vs. 
  Studio: <5 records) via `class_weight='balanced'`. No scaling needed, robust to noise.
- **Top features by importance:** `property_size` (18.1%), `inverse_cbd_distance` (10.9%), 
  `num_of_rooms` (9.4%), `commute_time` (8.2%)
- **Benchmarked against:** KNN Classifier (poor on rare classes, computationally expensive)

---

##  Key Insights from EDA (Visualisation Notebook)

- **2021 COVID Boom:** Average property price jumped from ~$1.42M (2019) to ~$2.1M (2021) — 
  a 48% spike captured via `is_sold_in_2021` binary flag
- **Regional pricing:** Eastern Suburbs mean price ($3.4M) is ~3.8x South West ($0.9M)
- **Room count effect:** Properties with 14–15 rooms average ~$4.5M vs ~$1M for 2-room properties
- **Commute sweet spot:** Properties 10–30 minutes from CBD command the highest average prices (~$2.25M), 
  likely reflecting Inner West/North Shore demand

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/sydney-property-ml

# Install dependencies
pip install -r requirements.txt

# Run the pipeline (outputs two CSVs)
python3 z5521431.py train.csv test.csv
```

**Outputs:**
- `z5521431.regression.csv` — Predicted prices (id, price)
- `z5521431.classification.csv` — Predicted property types (id, type)

---
