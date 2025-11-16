"""
Main Pipeline Script
Runs the complete forecasting workflow
"""

import pandas as pd
import numpy as np
from forecaster import CategoryForecaster
from config_loader import get_config


def load_future_prices(config, category_df, categories, horizon):
    """Load future prices from CSV if available"""

    future_prices_file = config.get('input.future_prices')

    try:
        price_df = pd.read_csv(future_prices_file)
        price_df['date'] = pd.to_datetime(price_df['date'])

        last_date = category_df['date'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)

        future_prices = {}
        for category in categories:
            cat_prices = price_df[
                (price_df['category'] == category) &
                (price_df['date'].isin(future_dates))
            ].sort_values('date')

            if len(cat_prices) == horizon:
                future_prices[category] = cat_prices['price'].tolist()
            else:
                # Partial data: use last known price for missing dates
                last_price = category_df[category_df['category'] == category]['price'].iloc[-1]
                future_prices[category] = [last_price] * horizon

        print(f"✓ Loaded future prices from {future_prices_file}")
        return future_prices

    except FileNotFoundError:
        print(f"ℹ Future prices file not found ({future_prices_file}), using last known prices")
        return None


def load_future_temperature(config, category_df, categories, horizon):
    """Load future temperature from CSV if available"""

    future_temp_file = config.get('input.future_temperature')

    try:
        temp_df = pd.read_csv(future_temp_file)
        temp_df['date'] = pd.to_datetime(temp_df['date'])

        last_date = category_df['date'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)

        future_temps = {}
        for category in categories:
            cat_temps = temp_df[
                (temp_df['category'] == category) &
                (temp_df['date'].isin(future_dates))
            ].sort_values('date')

            if len(cat_temps) == horizon:
                future_temps[category] = cat_temps['temperature'].tolist()
            else:
                # Partial/missing data: return None to trigger fallback
                return None

        print(f"✓ Loaded future temperatures from {future_temp_file}")
        return future_temps

    except FileNotFoundError:
        print(f"ℹ Temperature forecast file not found ({future_temp_file}), using historical averages")
        return None


def load_future_promos(config, category_df, categories, horizon):
    """Load future promotions from CSV if available"""

    future_promos_file = config.get('input.future_promos')

    try:
        promo_df = pd.read_csv(future_promos_file)
        promo_df['date'] = pd.to_datetime(promo_df['date'])

        last_date = category_df['date'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)

        future_promos = {}
        for category in categories:
            cat_promos = promo_df[
                (promo_df['category'] == category) &
                (promo_df['date'].isin(future_dates))
            ].sort_values('date')

            if len(cat_promos) == horizon:
                future_promos[category] = cat_promos[['date', 'main_promo', 'other_promo']].copy()
            else:
                # Partial/missing data: return None to trigger fallback
                return None

        print(f"✓ Loaded future promotions from {future_promos_file}")
        return future_promos

    except FileNotFoundError:
        print(f"ℹ Promotions file not found ({future_promos_file}), assuming no promos")
        return None


def prepare_future_inputs_fallback(category_df, categories, horizon):
    """Fallback: generate future inputs from historical data"""

    last_date = category_df['date'].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)

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

    return future_temps, future_promos


def prepare_future_inputs(config, category_df, categories, horizon):
    """Prepare temperature and promo inputs for forecasting (from files or fallback)"""

    print("\nPreparing future inputs...")

    # Try loading from files first
    future_temps = load_future_temperature(config, category_df, categories, horizon)
    future_promos = load_future_promos(config, category_df, categories, horizon)

    # Fallback to historical estimates if files not available
    if future_temps is None or future_promos is None:
        fallback_temps, fallback_promos = prepare_future_inputs_fallback(
            category_df, categories, horizon
        )
        if future_temps is None:
            future_temps = fallback_temps
        if future_promos is None:
            future_promos = fallback_promos

        for category in categories:
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
    sku_data_file = config.get('input.sku_data')
    try:
        sku_df = pd.read_csv(sku_data_file)
        sku_df['date'] = pd.to_datetime(sku_df['date'])
        print(f"✓ Loaded: {len(sku_df)} SKU-day records from {sku_data_file}")
    except FileNotFoundError:
        print(f"Error: {sku_data_file} not found")
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
        config, category_df, categories, forecaster.horizon
    )
    future_prices = load_future_prices(
        config, category_df, categories, forecaster.horizon
    )

    # 7. Generate forecasts
    forecasts = forecaster.generate_forecast(category_df, future_temps, future_promos, future_prices)
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
