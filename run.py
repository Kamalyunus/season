"""
Main Pipeline Script
Runs the complete forecasting workflow
"""

import pandas as pd
import numpy as np
from forecaster import CategoryForecaster
from config_loader import get_config


def prepare_future_inputs(category_df, categories, horizon):
    """Prepare temperature and promo inputs for forecasting"""

    last_date = category_df['date'].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)

    print("\nPreparing future inputs...")

    future_temps = {}
    future_promos = {}

    for category in categories:
        cat_data = category_df[category_df['category'] == category]

        # Temperature: use historical average for same day of year
        temps = []
        for future_date in future_dates:
            doy = future_date.dayofyear
            hist_temp = cat_data[
                (cat_data['date'].dt.dayofyear >= doy - 3) &
                (cat_data['date'].dt.dayofyear <= doy + 3)
            ]['temperature'].mean()

            temps.append(hist_temp if not pd.isna(hist_temp) else cat_data['temperature'].mean())

        future_temps[category] = temps

        # Promos: assume no promos (conservative forecast)
        future_promos[category] = pd.DataFrame({
            'date': future_dates,
            'main_promo': 0,
            'other_promo': 0
        })

        print(f"  ✓ {category}")

    return future_temps, future_promos


def main():
    """Run complete forecasting pipeline"""

    print("="*80)
    print("CATEGORY SALES FORECASTING PIPELINE")
    print("="*80)

    config = get_config()

    # Load data
    print("\nLoading data...")
    try:
        sku_df = pd.read_csv('demo_sku_data.csv')
        sku_df['date'] = pd.to_datetime(sku_df['date'])
        print(f"✓ Loaded: {len(sku_df)} SKU-day records")
    except FileNotFoundError:
        print("Error: demo_sku_data.csv not found")
        print("Run: python demo_synthetic_data.py")
        return

    # Initialize forecaster
    forecaster = CategoryForecaster()

    # Pipeline steps
    print("\n" + "="*80)
    print("PIPELINE EXECUTION")
    print("="*80)

    # 1. Aggregate
    category_df = forecaster.aggregate_to_category(sku_df)
    category_df.to_csv(config.get('output.category_data'), index=False)
    print(f"✓ Saved: {config.get('output.category_data')}")

    # 2. Decompose
    forecaster.decompose(category_df)

    # 3. Forecast trend
    forecaster.forecast_trend()

    # 4. Engineer features
    pooled_df = forecaster.prepare_features()
    pooled_df.to_csv(config.get('output.pooled_data'), index=False)
    print(f"✓ Saved: {config.get('output.pooled_data')}")

    # 5. Train models
    forecaster.train_seasonal_model(pooled_df)
    forecaster.train_lgbm(pooled_df)

    # 6. Prepare future inputs
    categories = category_df['category'].unique()
    future_temps, future_promos = prepare_future_inputs(
        category_df, categories, forecaster.horizon
    )

    # 7. Generate forecasts
    forecasts = forecaster.generate_forecast(category_df, future_temps, future_promos)
    forecasts.to_csv(config.get('output.forecasts'), index=False)
    print(f"\n✓ Saved: {config.get('output.forecasts')}")

    # 8. Summary
    print("\n" + "="*80)
    print("FORECAST SUMMARY")
    print("="*80)

    for category in forecasts['category'].unique():
        cat_fc = forecasts[forecasts['category'] == category]
        print(f"\n{category}:")
        print(f"  Period: {cat_fc['date'].min()} to {cat_fc['date'].max()}")
        print(f"  Daily avg: {cat_fc['forecast'].mean():.1f}")
        print(f"  Total: {cat_fc['forecast'].sum():.1f}")
        print(f"  Range: {cat_fc['forecast'].min():.1f} - {cat_fc['forecast'].max():.1f}")

    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - {config.get('output.category_data')}")
    print(f"  - {config.get('output.pooled_data')}")
    print(f"  - {config.get('output.forecasts')}")
    print("\nNext steps:")
    print("  - Run validation: python validate.py")
    print("  - Tune hyperparameters: python tune_hyperparameters.py")
    print("  - Visualize: python visualize_forecast.py")


if __name__ == "__main__":
    main()
