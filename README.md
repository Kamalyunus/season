# Category-Level Sales Forecasting Pipeline

Minimal, production-ready forecasting system using MSTL decomposition, temperature-modulated seasonality, and LightGBM with automatic hyperparameter optimization.

## What's New (Nov 2025)

Recent improvements for better accuracy and production readiness:

- **Damped Trend Forecasting**: Prevents unrealistic trend extrapolation (configurable `damping_trend: 0.95`)
- **lag_1 Feature Added**: Yesterday's sales now included (`lag_days: [1, 7, 14, 28]`)
- **Future Inputs from CSV**: Optional `future_prices.csv`, `future_promos.csv`, `future_temperature.csv` for planned business data
- **Sales-Weighted Metrics**: Validation and tuning now weight by sales volume (high-volume categories have more influence)
- **Improved Hyperparameter Tuning**: Uses full forecasting pipeline per trial (more realistic, consistent with production)
- **Fixed MSTL Logic**: Correctly handles 1D seasonality output (weekly-only when yearly extraction fails)
- **Enhanced Synthetic Data**: Stronger yearly seasonality (60% amplitude) and lower noise (8%) for realistic testing
- **Organized Visualizations**: All plots saved to `forecast_plots/` with date-based naming and metrics in titles
- **Training Progress**: LightGBM now shows training progress (verbose mode enabled)

See [CLAUDE.md](CLAUDE.md#recent-improvements-2025-11) for detailed technical notes.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate demo data
python demo_synthetic_data.py

# Run forecasting pipeline
python run.py

# Validate performance
python validate.py

# (Optional) Tune hyperparameters once
python tune_hyperparameters.py

# (Optional) Visualize results
python visualize_forecast.py
```

## Execution Order

### 1️⃣ First-Time Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Generate demo data (or prepare your own SKU-level data)
python demo_synthetic_data.py
```

**Output**: `demo_sku_data.csv` (or use your own data file)

---

### 2️⃣ Initial Pipeline Run

```bash
# Run the full forecasting pipeline
python run.py
```

**What it does**:
1. Aggregates SKU → Category level
2. MSTL decomposition
3. Trend forecasting
4. Feature engineering
5. Train seasonal model
6. Train LightGBM (uses defaults from config.yaml)
7. Generate forecasts

**Output**:
- `category_day_aggregated.csv`
- `pooled_training_data.csv`
- `category_forecasts_30day.csv`

---

### 3️⃣ Validate Performance

```bash
# Run time series cross-validation
python validate.py
```

**What it does**:
- 3-4 fold expanding window validation
- Tests forecasting on historical data
- Calculates sales-weighted metrics (WAPE, MAPE, etc.)

**Output**:
- `validation_metrics.csv`
- `validation_predictions.csv`
- Console output with WAPE (e.g., ~11-14% before tuning)

---

### 4️⃣ (Optional) Visualize Results

```bash
# Generate all plots
python visualize_forecast.py
```

**Output** (all saved to `forecast_plots/` folder):
- `forecast_analysis_{category}_{test_date}.png` (date-based naming, includes MAPE/Bias in title)
- `weekly_pattern.png` (average weekly seasonal pattern)
- `feature_importance.png` (LightGBM feature ranking)

---

### 5️⃣ (Optional) Hyperparameter Tuning

**⚠️ Do this ONCE after initial validation**

```bash
# Tune LightGBM hyperparameters (takes 15-30 minutes)
python tune_hyperparameters.py
```

**What it does**:
- Runs 50 Optuna trials
- Each trial runs full pipeline with different hyperparameters
- Optimizes sales-weighted MAPE
- Saves best parameters

**Output**:
- `best_hyperparameters.yaml` (auto-loaded by future runs)

**After tuning, re-run validation to see improvement**:
```bash
python validate.py  # Should see ~2pp WAPE improvement
```

---

### 6️⃣ Production Workflow

Once tuned, your regular workflow is:

```bash
# Optional: Provide future business inputs
# Create these CSV files if you have planned data:
# - future_prices.csv
# - future_promos.csv
# - future_temperature.csv

# Generate forecasts
python run.py  # Auto-loads best_hyperparameters.yaml

# Check results
cat category_forecasts_30day.csv
```

---

### Execution Flow Diagram

```
┌─────────────────────────────────────────────┐
│ SETUP (Once)                                │
├─────────────────────────────────────────────┤
│ 1. pip install -r requirements.txt          │
│ 2. python demo_synthetic_data.py            │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│ INITIAL RUN                                 │
├─────────────────────────────────────────────┤
│ 3. python run.py                            │
│ 4. python validate.py                       │
│    → Check WAPE (e.g., ~13%)                │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│ OPTIMIZATION (One-time, Optional)           │
├─────────────────────────────────────────────┤
│ 5. python tune_hyperparameters.py          │
│    → Saves best_hyperparameters.yaml        │
│ 6. python validate.py                       │
│    → Check improved WAPE (e.g., ~11%)       │
│ 7. python visualize_forecast.py (optional)  │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│ PRODUCTION (Recurring)                      │
├─────────────────────────────────────────────┤
│ • Update data files                         │
│ • Optional: Add future_*.csv files          │
│ • python run.py (auto-uses tuned params)    │
│ • Use category_forecasts_30day.csv          │
└─────────────────────────────────────────────┘
```

---

### When to Re-Run What

| Scenario | Command | Why |
|----------|---------|-----|
| **New data available** | `python run.py` | Generate fresh forecasts |
| **Changed config** | `python validate.py` | Check impact on performance |
| **Added/removed categories** | `python tune_hyperparameters.py` → `python validate.py` | Re-optimize for new data structure |
| **WAPE degraded by >2pp** | `python tune_hyperparameters.py` | Re-tune hyperparameters |
| **Need forecast visuals** | `python visualize_forecast.py` | Generate plots |
| **Data volume 2× changed** | `python tune_hyperparameters.py` | Re-optimize for new scale |

---

### Key Points

1. **run.py is your main script** - Use this in production to generate forecasts
2. **validate.py checks quality** - Run after major changes to verify WAPE
3. **tune_hyperparameters.py is one-time** - Only re-run when data changes significantly
4. **Order matters for tuning** - Must run `run.py` first to generate `category_day_aggregated.csv`
5. **Tuned params auto-load** - After tuning, `run.py` and `validate.py` automatically use `best_hyperparameters.yaml`

## File Structure

```
season/
├── config.yaml                      # All pipeline parameters
├── best_hyperparameters.yaml        # Tuned params (auto-generated)
├── forecaster.py                    # Core forecasting class (441 lines)
├── config_loader.py                 # Configuration utility (82 lines)
├── run.py                           # Main pipeline execution (262 lines)
├── validate.py                      # Time series cross-validation (273 lines)
├── tune_hyperparameters.py          # Hyperparameter optimization (233 lines)
├── visualize_forecast.py            # Forecast visualization (361 lines)
├── demo_synthetic_data.py           # Generate synthetic data (184 lines)
├── forecast_plots/                  # Generated visualizations (organized)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Pipeline Architecture

```
SKU-Day Data
    ↓
Aggregate to Category (filter noisy categories, interpolate low instock)
    ↓
MSTL Decomposition (weekly + yearly seasonality)
    ↓
Trend Forecasting (exponential smoothing)
    ↓
Feature Engineering (lags, rolling stats, cyclical encoding)
    ↓
LightGBM Training (auto-loads tuned params if available)
    ↓
Forecast Generation (temperature-modulated seasonality)
```

## Key Features

### 1. Category Filtering
Automatically removes categories with < N SKUs (too noisy for reliable forecasting).

**Configuration**:
```yaml
preprocessing:
  min_skus_per_category: 10  # Filter categories with < 10 SKUs
```

**Example output**:
```
SKU count per category:
  Fresh_Category_1: 15 SKUs ✓
  Fresh_Category_2: 15 SKUs ✓
  Small_Category: 8 SKUs ✗ (filtered)
Retained 2/3 categories with >= 10 SKUs
```

**When to adjust**:
- Small retailers: 5-8
- Medium retailers: 10-15
- Large retailers: 15-20

### 2. Sales Interpolation
Corrects sales during stockouts (low instock periods) using linear interpolation.

**Configuration**:
```yaml
preprocessing:
  instock_threshold: 0.85  # Interpolate when instock < 85%
```

**How it works**:
```
Day    Instock    Sales_Raw    Sales_Interpolated
1      95%        100          100 (no change)
2      95%        105          105 (no change)
3      70%        60           102.5 (interpolated)
4      75%        65           107.5 (interpolated)
5      95%        110          110 (no change)
```

**When to adjust**:
- Fresh products: 0.80-0.90 (higher sensitivity)
- Non-perishables: 0.70-0.80
- Fast movers: 0.85-0.95

**Important**: `instock_rate` is NOT used as a model feature (prevents data leakage).

### 3. Temperature Modulation
Simple mathematical adjustment of yearly seasonality based on temperature.

**Formula**:
```python
seasonal_yearly = base_pattern × (1 + temp_sensitivity × temp_deviation)
# Bounded between 0.7 and 1.3 (±30% adjustment)
```

**Configuration**:
```yaml
decomposition:
  temp_sensitivity: 0.02  # Default: moderate temperature effect
```

**Tuning guide**:
```yaml
# Low sensitivity (0.01): Canned goods, shelf-stable items
temp_sensitivity: 0.01

# Medium sensitivity (0.02): Most fresh items [DEFAULT]
temp_sensitivity: 0.02

# High sensitivity (0.05): Ice cream, cold beverages, seasonal produce
temp_sensitivity: 0.05
```

**Example**:
```
Temperature: 30°C (hot day, +2 std deviations)
Sensitivity: 0.02
Adjustment: 1 + 0.02 × 2 = 1.04 (4% increase)

For ice cream:
  Base seasonal: +50 units
  Modulated: +50 × 1.04 = +52 units
```

### 4. Automatic Hyperparameter Loading
**Once you run tuning, optimal parameters are used automatically - no manual copying!**

**How it works**:
1. `forecaster.py` checks if `best_hyperparameters.yaml` exists
2. If found → loads tuned LightGBM parameters
3. If not found → falls back to `config.yaml` defaults

**Usage**:
```bash
# One-time tuning
python tune_hyperparameters.py

# All future runs automatically use tuned params
python run.py       # ✓ Auto-loads best_hyperparameters.yaml
python validate.py  # ✓ Uses tuned params
```

## Configuration

### Main Parameters (config.yaml)

```yaml
# Data Quality & Preprocessing
preprocessing:
  min_skus_per_category: 10      # Filter categories with < N SKUs
  instock_threshold: 0.85        # Interpolate sales when instock < 85%

# Decomposition
decomposition:
  weekly_period: 7
  yearly_period: 365
  seasonal_smoothing: 13
  temp_sensitivity: 0.02         # Temperature effect strength (0.0-0.1)

# Trend Forecasting (Exponential Smoothing)
trend:
  smoothing_level: 0.8
  smoothing_trend: 0.2
  damping_trend: 0.95            # NEW: Damping prevents over-extrapolation (0.8-0.98)

# Features
features:
  lag_days: [1, 7, 14, 28]       # NEW: lag_1 added (yesterday's sales)
  rolling_windows: [7, 28]       # Rolling window sizes
  min_non_nan_pct: 0.1          # Min data coverage to include feature

# Input Files (NEW: Optional future inputs)
input:
  sku_data: "demo_sku_data.csv"
  future_prices: "future_prices.csv"         # Optional: planned pricing
  future_promos: "future_promos.csv"         # Optional: promo calendar
  future_temperature: "future_temperature.csv"  # Optional: weather forecasts

# LightGBM (auto-loaded from tuned params if available)
lightgbm:
  n_estimators: 600
  learning_rate: 0.1
  max_depth: 5
  num_leaves: 15
  min_child_samples: 46
  subsample: 0.76
  colsample_bytree: 0.8
  reg_alpha: 0.4
  reg_lambda: 0.33
  random_state: 42

# Validation
validation:
  n_splits: 3                    # Number of CV folds
  train_window_days: 730         # Fixed training window size (must be >= 730 for yearly seasonality)

# Hyperparameter Tuning
optuna:
  n_trials: 50                   # Number of trials (20-200)
  timeout_seconds: 3600          # Max time (optional)

  lgbm_search_space:             # Parameter ranges to search
    learning_rate: [0.01, 0.1]
    n_estimators: [300, 1000]
    max_depth: [4, 12]
    num_leaves: [15, 63]
    min_child_samples: [10, 50]
    subsample: [0.6, 1.0]
    colsample_bytree: [0.6, 1.0]
    reg_alpha: [0.0, 1.0]
    reg_lambda: [0.0, 1.0]
```

## Hyperparameter Tuning

### Running Tuning

```bash
python tune_hyperparameters.py
```

**What happens**:
- Runs 50 Optuna trials (configurable)
- Uses 3-fold time series cross-validation
- **NEW**: Each trial runs the full forecasting pipeline (MSTL, trend, feature engineering, etc.)
- **NEW**: Optimizes sales-weighted MAPE (high-volume periods have more influence)
- Searches for optimal LightGBM parameters
- Saves results to `best_hyperparameters.yaml`
- Takes ~15-30 minutes (slower but more accurate than before)

**Example output**:
```
Best trial: 22
Best CV MAPE: 4.12%

Best Parameters:
  n_estimators: 369
  learning_rate: 0.093
  max_depth: 7
  num_leaves: 59
  ...

✓ Saved best parameters to: best_hyperparameters.yaml
```

### Validation Strategy

Hyperparameter tuning uses **the same validation strategy as validate.py** to ensure consistency:

**Fixed-Size Moving Window Cross-Validation**:
- Each fold trains on exactly **730 days** (2 years) for yearly seasonality
- **Step size = forecast horizon** (e.g., 7 days) for contiguous test periods
- **Works backward from most recent data** (last fold ends at max_date)
- **Sales-weighted MAPE** across folds (high-volume periods weighted more)

**Example with 3 folds, 7-day horizon**:
```
Fold 1: Train [Day 1, Day 730]   → Test [Day 731, Day 737]
Fold 2: Train [Day 8, Day 737]   → Test [Day 738, Day 744]
Fold 3: Train [Day 15, Day 744]  → Test [Day 745, Day 751]
```

**Why this matters**:
- Hyperparameters optimized for **same conditions** as production forecasting
- Fixed training window ensures stable yearly seasonality extraction
- Most recent data validation reflects realistic performance

### When to Re-Tune

- Data significantly changed (2× more/less data)
- New categories added
- Changed `temp_sensitivity` or feature configuration
- Validation WAPE increased by >2pp

### Performance Improvement

**Before tuning** (defaults from config.yaml):
```
Validation WAPE: ~13-14%
```

**After tuning** (optimized):
```
Validation WAPE: ~11-12%
Improvement: ~2 percentage points
```

## Validation

### Time Series Cross-Validation

**Method**: Fixed-size moving window with 3 folds (configurable)

**Strategy**:
- Each fold trains on exactly **730 days** (2 years) for yearly seasonality
- Test periods are **contiguous** (step = forecast horizon)
- Works **backward from most recent data** (last fold ends at max_date)
- Ensures validation on latest data patterns

**Example with 3 folds, 7-day horizon, max_date = 2025-12-31**:
```
Fold 1: Train [2024-06-15 to 2025-12-10] → Test [2025-12-11 to 2025-12-17]
Fold 2: Train [2024-06-22 to 2025-12-17] → Test [2025-12-18 to 2025-12-24]
Fold 3: Train [2024-06-29 to 2025-12-24] → Test [2025-12-25 to 2025-12-31]
```

**Maximum possible splits**: Displayed when running `validate.py` based on your data:
```
Data Summary:
  Total days available: 1460 (from 2022-01-01 to 2025-12-31)
  Training window size: 730 days
  Forecast horizon: 7 days
  Requested splits: 3
  Maximum possible splits: 104

✓ You could increase to 104 splits for more robust validation.
```

### Metrics

- **WAPE** (Weighted Absolute Percentage Error): Primary metric
- **MAPE** (Mean Absolute Percentage Error)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **Bias** (Over/under prediction tendency)

**NEW**: All metrics are now **sales-weighted** - high-volume categories have more influence than low-volume categories.

**Example output**:
```
Overall Metrics (weighted by sales volume):
  MAE:   144.8
  RMSE:  179.5
  MAPE:  11.3%
  WAPE:  11.7%  ← Primary metric
  Bias:  -4.9%
  Total Sales: 125,432  ← Total volume across all folds

Metrics by Category:
                     MAE    RMSE   MAPE   WAPE  Bias_%  Total_Sales
Fresh_Category_1  114.10  144.32  11.53  11.78   -2.48    45,230
Fresh_Category_2  153.04  182.64  12.43  12.63   -5.88    52,105
Fresh_Category_3  167.30  211.48   9.98  10.58   -6.32    28,097
```

## Data Requirements

### Input Format (SKU-Day Level)

Required columns:
```
sku_id          : Unique SKU identifier
category        : Category name
date            : Date (YYYY-MM-DD)
sales           : Sales quantity
instock_rate    : Instock % (0-100)
price           : Average price
temperature     : Temperature in °C
holiday_gap     : Days to nearest holiday
main_promo      : Main promo flag (0/1)
other_promo     : Other promo count
```

**Minimum requirements**:
- At least 2 years of daily data (730 days)
- At least 10 SKUs per category
- Complete temperature data
- No missing dates

### Optional Future Inputs (NEW)

You can optionally provide planned business data as CSV files for more accurate forecasts:

**future_prices.csv** (Planned pricing schedule):
```csv
date,category,price
2026-01-01,Category_A,5.99
2026-01-02,Category_A,4.99  # Promotional price
```

**future_promos.csv** (Promotional calendar):
```csv
date,category,main_promo,other_promo
2026-01-01,Category_A,0,0
2026-01-02,Category_A,1,2  # Main promo + 2 other promos
```

**future_temperature.csv** (Weather forecasts):
```csv
date,category,temperature
2026-01-01,Category_A,15.5
2026-01-02,Category_A,16.2
```

**Templates**: See `future_*_template.csv` files for examples.

**Fallback behavior** (if files not provided):
- Prices: Last known price
- Promos: No promotions (conservative)
- Temperature: Historical day-of-year average

### Output Files

| File | Description |
|------|-------------|
| `category_day_aggregated.csv` | Aggregated category-level data |
| `pooled_training_data.csv` | Engineered features for training |
| `category_forecasts_30day.csv` | Forecasts by category |
| `validation_metrics.csv` | Validation metrics by fold/category |
| `validation_predictions.csv` | Actuals vs forecasts |
| `best_hyperparameters.yaml` | Tuned LightGBM parameters |

## API Reference

### CategoryForecaster Class

```python
from forecaster import CategoryForecaster

# Initialize
forecaster = CategoryForecaster('config.yaml')

# Methods
forecaster.aggregate_to_category(sku_df)  → DataFrame
    # Aggregate SKU-day to category-day with preprocessing
    # - Filters categories by min SKUs
    # - Interpolates sales during stockouts

forecaster.decompose(category_df)
    # MSTL decomposition for each category
    # - Extracts weekly and yearly seasonality

forecaster.forecast_trend()
    # Exponential smoothing for trend forecasting

forecaster.prepare_features() → DataFrame
    # Engineer features from decomposed data

forecaster.train_seasonal_model(pooled_df)
    # Calculate temperature sensitivity (mathematical formula)

forecaster.train_lgbm(pooled_df)
    # Train LightGBM (auto-loads tuned params if available)

forecaster.generate_forecast(
    category_df,
    future_temps,
    future_promos,
    future_prices=None
) → DataFrame
    # Generate forecasts for all categories
```

## Usage Example

```python
from forecaster import CategoryForecaster
import pandas as pd

# Initialize
forecaster = CategoryForecaster('config.yaml')

# Load data
sku_data = pd.read_csv('demo_sku_data.csv')

# Aggregate (with preprocessing)
category_data = forecaster.aggregate_to_category(sku_data)

# Decompose
forecaster.decompose(category_data)

# Forecast trend
forecaster.forecast_trend()

# Prepare features
pooled_data = forecaster.prepare_features()

# Train models
forecaster.train_seasonal_model(pooled_data)
forecaster.train_lgbm(pooled_data)  # Auto-loads tuned params

# Generate forecast
future_temps = {
    'Fresh_Category_1': [25, 26, 24, 25, 27, 28, 26],
    'Fresh_Category_2': [25, 26, 24, 25, 27, 28, 26],
    'Fresh_Category_3': [25, 26, 24, 25, 27, 28, 26]
}

future_promos = {
    'Fresh_Category_1': pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=7),
        'main_promo': [0, 0, 5, 5, 0, 0, 0],
        'other_promo': [0, 0, 0, 0, 0, 0, 0]
    })
}

forecasts = forecaster.generate_forecast(
    category_data,
    future_temps,
    future_promos
)

print(forecasts)
```

## Troubleshooting

### Common Issues

**1. "No categories have >= N SKUs"**
```
Solution: Lower min_skus_per_category in config.yaml
preprocessing:
  min_skus_per_category: 5  # Reduced from 10
```

**2. "MSTL decomposition failed"**
```
Cause: Insufficient data (<2 years)
Solution: Increase data collection period
```

**3. "Tuned parameters worse than defaults"**
```
Cause: Overfitting to validation set
Solution:
1. Increase validation.n_splits to 4-5
2. Run more trials (n_trials: 100)
3. Delete best_hyperparameters.yaml and re-tune
```

**5. "Yearly seasonality shows as 0 in validation charts"**
```
Cause: Validation train_window_days too short (730 days = 2 years borderline)
Solution: Increase training window in config.yaml:
validation:
  train_window_days: 1095  # 3 years for robust yearly seasonality
Note: This reduces max possible validation folds but improves reliability
```

**4. "High WAPE (>20%)"**
```
Possible causes:
1. Noisy data → Increase min_skus_per_category
2. Poor seasonality → Adjust temp_sensitivity
3. Needs tuning → Run tune_hyperparameters.py
```

## Performance Tips

### Improve Accuracy
1. Run hyperparameter tuning
2. Adjust temp_sensitivity for product type
3. Increase min_skus_per_category (filter noise)
4. Use 3-4 years of data instead of 2

### Improve Speed
1. Reduce n_trials to 20-30 for quick tuning
2. Reduce n_splits to 2-3 validation folds
3. Reduce n_estimators to 300-400

### Reduce Overfitting
1. Increase regularization (reg_alpha, reg_lambda)
2. Reduce model complexity (max_depth, num_leaves)
3. More validation folds (n_splits: 4-5)

## Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
statsmodels>=0.14.0
lightgbm>=4.0.0
pyyaml>=6.0
optuna>=3.0.0
matplotlib>=3.5.0
scikit-learn>=1.2.0
```

Install all:
```bash
pip install -r requirements.txt
```

## Best Practices

1. **Always validate**: Run `validate.py` after configuration changes
2. **Version control configs**: Commit `config.yaml` and `best_hyperparameters.yaml`
3. **Monitor in production**: Track WAPE over time
4. **Re-tune periodically**: When data changes significantly
5. **Test before deploying**: Validate on holdout period