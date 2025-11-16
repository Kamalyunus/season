"""
Simplified Model Validation
Time series cross-validation with performance metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from forecaster import CategoryForecaster
from config_loader import get_config
import warnings
warnings.filterwarnings('ignore')


def create_cv_splits(category_df, config):
    """Create time series CV splits with FIXED sliding window

    Returns splits with fixed training window size (not expanding).
    Each fold has the same training window size, sliding forward in time.

    Works BACKWARD from most recent data:
    - Last fold always tests on most recent period (ends at max_date)
    - Folds slide backward in time (older data discarded if insufficient)
    - Ensures validation on latest data (most realistic for production)
    """

    max_date = category_df['date'].max()
    min_date = category_df['date'].min()
    total_days = (max_date - min_date).days

    n_splits = config.get('validation.n_splits')
    horizon = config.get('forecast.horizon_days')
    train_window_days = config.get('validation.train_window_days')

    # Validate minimum training window for MSTL
    if train_window_days < 730:
        raise ValueError(f"train_window_days must be >= 730 for yearly seasonality. Got {train_window_days}")

    # Calculate maximum possible splits based on available data
    # Formula: max_splits = (total_days - train_window_days) / horizon
    max_possible_splits = (total_days - train_window_days) // horizon

    # Inform user about data capacity
    print(f"\nData Summary:")
    print(f"  Total days available: {total_days} (from {min_date.date()} to {max_date.date()})")
    print(f"  Training window size: {train_window_days} days")
    print(f"  Forecast horizon: {horizon} days")
    print(f"  Requested splits: {n_splits}")
    print(f"  Maximum possible splits: {max_possible_splits}")

    if n_splits > max_possible_splits:
        print(f"\n⚠ WARNING: Requested {n_splits} splits but only {max_possible_splits} possible with current data.")
        print(f"  Recommendation: Set validation.n_splits to {max_possible_splits} or less in config.yaml")
    elif n_splits < max_possible_splits:
        print(f"\n✓ You could increase to {max_possible_splits} splits for more robust validation.")

    # Ensure enough data for all folds
    # Need: train_window + (n_splits * horizon) for contiguous test periods
    required_days = train_window_days + (n_splits * horizon)
    if total_days < required_days:
        raise ValueError(
            f"Insufficient data: need {required_days} days for {n_splits} folds, "
            f"have {total_days} days. Reduce n_splits or increase data."
        )

    # Step size = forecast horizon (contiguous test periods)
    step = horizon

    # Create splits working BACKWARD from most recent data
    # Last fold always ends at max_date (most recent period)
    splits = []
    for i in range(n_splits):
        # Work backward: last fold (i=n_splits-1) ends at max_date
        fold_offset = (n_splits - 1 - i) * step

        # Test period
        test_end = max_date - pd.Timedelta(days=fold_offset)
        test_start = test_end - pd.Timedelta(days=horizon - 1)

        # Training period (fixed window ending just before test)
        train_end = test_start - pd.Timedelta(days=1)
        train_start = train_end - pd.Timedelta(days=train_window_days - 1)

        # Only include if we have enough historical data
        if train_start >= min_date:
            splits.append((train_start, train_end, test_start, test_end))
        else:
            print(f"Warning: Fold {i+1} skipped - train_start before min_date (need more data)")

    return splits


def calculate_metrics(y_true, y_pred):
    """Calculate forecast accuracy metrics"""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
    bias = np.mean(y_pred - y_true)
    bias_pct = bias / np.mean(y_true) * 100
    total_sales = np.sum(y_true)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'WAPE': wape,
        'Bias': bias,
        'Bias_%': bias_pct,
        'Total_Sales': total_sales
    }


def validate(category_df, config):
    """Run time series cross-validation"""

    print("="*80)
    print("MODEL VALIDATION")
    print("="*80)

    splits = create_cv_splits(category_df, config)
    print(f"\nFolds: {len(splits)}")
    print(f"Forecast horizon: {config.get('forecast.horizon_days')} days\n")

    all_results = []
    all_predictions = []

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(splits, 1):
        print("="*80)
        print(f"Fold {fold_idx}/{len(splits)}")
        print("="*80)
        train_days = (train_end - train_start).days + 1
        print(f"Train: {train_start} to {train_end} ({train_days} days)")
        print(f"Test:  {test_start} to {test_end}\n")

        # Split data (fixed window)
        train_df = category_df[
            (category_df['date'] >= train_start) &
            (category_df['date'] <= train_end)
        ].copy()
        test_df = category_df[
            (category_df['date'] >= test_start) &
            (category_df['date'] <= test_end)
        ].copy()

        # Train forecaster
        forecaster = CategoryForecaster()
        forecaster.decompose(train_df)
        forecaster.forecast_trend()

        pooled = forecaster.prepare_features()
        forecaster.train_seasonal_model(pooled)
        forecaster.train_lgbm(pooled)

        # Prepare future inputs from test data
        categories = train_df['category'].unique()
        future_temps = {}
        future_promos = {}
        future_prices = {}

        for cat in categories:
            cat_test = test_df[test_df['category'] == cat].sort_values('date')
            if len(cat_test) > 0:
                future_temps[cat] = cat_test['temperature'].values.tolist()
                future_promos[cat] = cat_test[['date', 'main_promo', 'other_promo']].copy()
                future_prices[cat] = cat_test['price'].values.tolist()

        # Generate forecasts
        try:
            forecasts = forecaster.generate_forecast(train_df, future_temps, future_promos, future_prices)

            # Merge with actuals
            comparison = forecasts.merge(
                test_df[['date', 'category', 'sales']],
                on=['date', 'category'],
                how='inner'
            )

            # Calculate metrics by category
            for cat in comparison['category'].unique():
                cat_comp = comparison[comparison['category'] == cat]

                if len(cat_comp) > 0:
                    metrics = calculate_metrics(
                        cat_comp['sales'].values,
                        cat_comp['forecast'].values
                    )
                    metrics['category'] = cat
                    metrics['fold'] = fold_idx
                    all_results.append(metrics)

                    cat_comp['fold'] = fold_idx
                    all_predictions.append(cat_comp)

                    print(f"{cat}:")
                    print(f"  MAE: {metrics['MAE']:.1f} | RMSE: {metrics['RMSE']:.1f}")
                    print(f"  MAPE: {metrics['MAPE']:.1f}% | WAPE: {metrics['WAPE']:.1f}%")
                    print(f"  Bias: {metrics['Bias']:.1f} ({metrics['Bias_%']:.1f}%)")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Save results
    results_df = pd.DataFrame(all_results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    results_df.to_csv(config.get('output.validation_metrics'), index=False)
    predictions_df.to_csv(config.get('output.validation_predictions'), index=False)

    print(f"\n✓ Saved: {config.get('output.validation_metrics')}")
    print(f"✓ Saved: {config.get('output.validation_predictions')}")

    # Summary report
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    print("\nOverall Metrics (weighted by sales volume):")
    print("-"*80)

    # Calculate weighted averages using Total_Sales as weights
    total_weight = results_df['Total_Sales'].sum()
    overall_weighted = {}
    for metric in ['MAE', 'RMSE', 'MAPE', 'WAPE', 'Bias_%']:
        overall_weighted[metric] = (results_df[metric] * results_df['Total_Sales']).sum() / total_weight

    print(f"  MAE:   {overall_weighted['MAE']:.1f}")
    print(f"  RMSE:  {overall_weighted['RMSE']:.1f}")
    print(f"  MAPE:  {overall_weighted['MAPE']:.1f}%")
    print(f"  WAPE:  {overall_weighted['WAPE']:.1f}%")
    print(f"  Bias:  {overall_weighted['Bias_%']:.1f}%")
    print(f"  Total Sales: {total_weight:,.0f}")

    print("\nMetrics by Category:")
    print("-"*80)
    by_category = results_df.groupby('category').agg({
        'MAE': 'mean',
        'RMSE': 'mean',
        'MAPE': 'mean',
        'WAPE': 'mean',
        'Bias_%': 'mean',
        'Total_Sales': 'sum'  # Total sales across all folds
    }).round(2)
    print(by_category.to_string())

    print("\n" + "="*80)

    return results_df, predictions_df


def main():
    """Run validation"""

    config = get_config()

    # Load data
    try:
        category_df = pd.read_csv(config.get('output.category_data'))
        category_df['date'] = pd.to_datetime(category_df['date'])
        print(f"\n✓ Loaded: {len(category_df)} records\n")
    except FileNotFoundError:
        print("Error: category_day_aggregated.csv not found")
        print("Run: python run.py first")
        return

    # Run validation
    validate(category_df, config)


if __name__ == "__main__":
    main()
