"""
Demo: Test the forecasting pipeline with synthetic data
Run this to see the complete workflow before using your actual data
"""

import pandas as pd
import numpy as np
from category_forecast import CategoryForecaster
from validate_model import ForecastValidator


def generate_synthetic_data(n_categories=3, n_skus_per_category=20, n_days=900):
    """
    Generate realistic synthetic SKU-day sales data

    Simulates:
    - Weekly seasonality (weekend spike)
    - Yearly seasonality (summer peak for fresh)
    - Trend (slight growth)
    - Temperature effects
    - Promotional lift
    - Out-of-stock impact
    """
    print(f"Generating synthetic data...")
    print(f"  Categories: {n_categories}")
    print(f"  SKUs per category: {n_skus_per_category}")
    print(f"  Days: {n_days} (~{n_days/365:.1f} years)")

    np.random.seed(42)

    # Generate dates
    start_date = pd.Timestamp('2022-01-01')
    dates = pd.date_range(start_date, periods=n_days, freq='D')

    # Generate categories and SKUs
    categories = [f'Fresh_Category_{i+1}' for i in range(n_categories)]

    all_data = []

    for cat_idx, category in enumerate(categories):
        # Category-specific parameters
        base_sales_per_sku = 50 + cat_idx * 20  # Different volumes
        trend_slope = 0.05 + cat_idx * 0.02     # Different growth rates

        for sku_id in range(n_skus_per_category):
            sku_name = f'{category}_SKU_{sku_id+1}'

            # SKU-specific base sales
            sku_base = base_sales_per_sku * (0.5 + np.random.random())

            for day_idx, date in enumerate(dates):
                # Temporal features
                day_of_week = date.dayofweek
                day_of_year = date.dayofyear

                # 1. TREND (linear growth)
                trend = sku_base * (1 + trend_slope * day_idx / 365)

                # 2. WEEKLY SEASONALITY (weekend spike)
                weekly_seasonal = 1.0
                if day_of_week == 5:  # Saturday
                    weekly_seasonal = 1.3
                elif day_of_week == 6:  # Sunday
                    weekly_seasonal = 1.2
                elif day_of_week in [0, 1]:  # Monday, Tuesday
                    weekly_seasonal = 0.85

                # 3. YEARLY SEASONALITY (summer peak for fresh)
                yearly_seasonal = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2)

                # 4. TEMPERATURE (correlated with yearly seasonality + noise)
                base_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2)
                temperature = base_temp + np.random.normal(0, 3)

                # Temperature effect (fresh products sell more when warm)
                temp_effect = 1 + 0.01 * (temperature - 15)

                # 5. PROMOTIONS (random)
                main_promo = 1 if np.random.random() < 0.05 else 0  # 5% of days
                other_promo = 1 if np.random.random() < 0.10 else 0  # 10% of days

                promo_lift = 1.0
                if main_promo:
                    promo_lift *= 1.5  # 50% lift
                if other_promo:
                    promo_lift *= 1.2  # 20% lift

                # 6. PRICE (mostly stable, occasional changes)
                base_price = 10 + cat_idx * 2
                price = base_price * (1 + np.random.normal(0, 0.1))

                # Price elasticity
                price_effect = (base_price / price) ** 1.5

                # 7. INSTOCK RATE (usually high, sometimes low)
                instock_rate = min(100, max(0, np.random.normal(95, 10)))

                # If low instock, sales are capped
                instock_effect = instock_rate / 100

                # 8. HOLIDAY EFFECT
                # Simulate major holidays
                is_holiday_week = (
                    (day_of_year >= 355) or  # New Year
                    (day_of_year >= 165 and day_of_year <= 175) or  # Summer holiday
                    (day_of_year >= 350)  # Christmas
                )

                if is_holiday_week:
                    holiday_gap = min(5, max(-14, int(np.random.normal(0, 7))))
                    holiday_effect = 1.2 if holiday_gap < 0 else 0.9  # Pre-holiday boost
                else:
                    holiday_gap = 0
                    holiday_effect = 1.0

                # FINAL SALES (combine all effects + noise)
                expected_sales = (
                    trend *
                    weekly_seasonal *
                    yearly_seasonal *
                    temp_effect *
                    promo_lift *
                    price_effect *
                    holiday_effect *
                    instock_effect
                )

                # Add noise
                noise = np.random.normal(1, 0.15)
                sales = max(0, expected_sales * noise)

                # Store record
                all_data.append({
                    'date': date,
                    'category': category,
                    'sku_id': sku_name,
                    'sales': round(sales, 2),
                    'price': round(price, 2),
                    'instock_rate': round(instock_rate, 1),
                    'temperature': round(temperature, 1),
                    'holiday_gap': holiday_gap,
                    'main_promo': main_promo,
                    'other_promo': other_promo
                })

    df = pd.DataFrame(all_data)
    print(f"✓ Generated {len(df):,} records")

    return df


def demo_forecasting():
    """
    Complete demo of the forecasting pipeline
    """
    print("="*80)
    print("DEMO: Category Forecasting with Synthetic Data")
    print("="*80)
    print()

    # Step 1: Generate synthetic data
    sku_df = generate_synthetic_data(
        n_categories=3,
        n_skus_per_category=15,
        n_days=1460  # 4 years
    )

    # Save for inspection
    sku_df.to_csv('demo_sku_data.csv', index=False)
    print(f"✓ Saved demo_sku_data.csv")
    print()

    # Step 2: Initialize forecaster
    print("="*80)
    print("FORECASTING PIPELINE")
    print("="*80)

    forecaster = CategoryForecaster(forecast_horizon=30)

    # Step 3: Aggregate to category-day
    category_df = forecaster.aggregate_sku_to_category(sku_df)
    category_df.to_csv('demo_category_data.csv', index=False)
    print(f"✓ Saved demo_category_data.csv")

    # Step 4: Decompose
    forecaster.decompose_time_series(category_df)

    # Step 5: Forecast trend
    forecaster.forecast_trend()

    # Step 6: Prepare pooled data
    pooled_df = forecaster.prepare_pooled_data()
    pooled_df.to_csv('demo_pooled_data.csv', index=False)
    print(f"✓ Saved demo_pooled_data.csv")

    # Step 7: Train models
    forecaster.train_seasonal_adjuster(pooled_df)
    forecaster.train_lgbm(pooled_df)

    # Step 8: Prepare future inputs
    last_date = category_df['date'].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)

    categories = category_df['category'].unique()

    # Generate future temperatures (use seasonal pattern)
    future_temps = {}
    future_promos = {}

    for category in categories:
        # Simulate temperature forecast (realistic pattern)
        future_doy = future_dates.dayofyear
        temps = [15 + 10 * np.sin(2 * np.pi * doy / 365 - np.pi/2) + np.random.normal(0, 2)
                for doy in future_doy]
        future_temps[category] = temps

        # Simulate planned promos (sparse)
        promo_dates = np.random.choice(future_dates, size=2, replace=False)
        future_promos[category] = pd.DataFrame({
            'date': future_dates,
            'main_promo': [1 if d in promo_dates else 0 for d in future_dates],
            'other_promo': [1 if np.random.random() < 0.1 else 0 for _ in future_dates]
        })

    # Step 9: Generate forecast
    forecasts = forecaster.generate_forecast(
        category_df,
        future_temps,
        future_promos
    )

    forecasts.to_csv('demo_forecasts.csv', index=False)
    print(f"\n✓ Saved demo_forecasts.csv")

    # Display sample forecasts
    print("\n" + "="*80)
    print("SAMPLE FORECASTS (First 7 Days)")
    print("="*80)
    print(forecasts.head(21).to_string(index=False))

    # Step 10: Validation
    print("\n" + "="*80)
    print("MODEL VALIDATION")
    print("="*80)

    validator = ForecastValidator(n_splits=3, forecast_horizon=30, gap=7)

    try:
        results_df, predictions_df = validator.validate(category_df)

        results_df.to_csv('demo_validation_metrics.csv', index=False)
        predictions_df.to_csv('demo_validation_predictions.csv', index=False)

        print(f"\n✓ Saved demo_validation_metrics.csv")
        print(f"✓ Saved demo_validation_predictions.csv")

        # Summary
        validator.summary_report(results_df)

        # Try to plot
        try:
            validator.plot_results(results_df, predictions_df, 'demo_validation_plots.png')
        except Exception as e:
            print(f"\nNote: Could not generate plots (matplotlib issue): {e}")

    except Exception as e:
        print(f"Validation error: {e}")
        print("Continuing with forecast output...")

    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - demo_sku_data.csv: Synthetic SKU-day data")
    print("  - demo_category_data.csv: Aggregated category-day data")
    print("  - demo_pooled_data.csv: Training data with all features")
    print("  - demo_forecasts.csv: 30-day forecasts by category")
    print("  - demo_validation_metrics.csv: Validation metrics")
    print("  - demo_validation_predictions.csv: Validation predictions")
    print("  - demo_validation_plots.png: Validation visualizations")
    print("\nNext steps:")
    print("  1. Inspect the CSV files to understand the data structure")
    print("  2. Review validation metrics to see model performance")
    print("  3. Modify example_usage.py to use your actual data")
    print("="*80)


if __name__ == "__main__":
    demo_forecasting()
