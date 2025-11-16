"""
Visualization: Historical Sales + Forecast vs Actuals
Shows if the model captures seasonal patterns
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from forecaster import CategoryForecaster
import warnings
warnings.filterwarnings('ignore')


def plot_forecast_vs_actual(category_df, predictions_df, output_path='forecast_analysis.png'):
    """
    Create comprehensive visualization showing:
    1. Historical sales with decomposed components
    2. Test period: Actuals vs Forecast
    3. Component breakdown to verify seasonal pattern capture

    Parameters:
    -----------
    category_df : pd.DataFrame
        Historical category-day data
    predictions_df : pd.DataFrame
        Validation predictions with actuals
    output_path : str
        Where to save the plot
    """

    # Get unique categories and folds
    categories = predictions_df['category'].unique()
    folds = sorted(predictions_df['fold'].unique())

    # Create plots for each category and fold
    for category in categories:
        print(f"\nCreating plots for {category}...")

        # Get historical data for this category
        hist_data = category_df[category_df['category'] == category].copy()
        hist_data = hist_data.sort_values('date')

        for fold in folds:
            # Get predictions for this category and fold
            fold_preds = predictions_df[
                (predictions_df['category'] == category) &
                (predictions_df['fold'] == fold)
            ].copy()

            if len(fold_preds) == 0:
                continue

            fold_preds = fold_preds.sort_values('date')

            # Calculate metrics for title
            fold_preds['error'] = fold_preds['forecast'] - fold_preds['sales']
            fold_preds['error_pct'] = (fold_preds['error'] / fold_preds['sales']) * 100

            mape = abs(fold_preds['error_pct']).mean()
            bias = fold_preds['error'].mean()
            bias_pct = (bias / fold_preds['sales'].mean()) * 100

            # Create figure with subplots
            fig, axes = plt.subplots(4, 1, figsize=(16, 12))
            test_start_str = fold_preds['date'].min().strftime('%Y-%m-%d')
            test_end_str = fold_preds['date'].max().strftime('%Y-%m-%d')
            fig.suptitle(f'{category} - Test Period: {test_start_str} to {test_end_str} | MAPE: {mape:.1f}% | Bias: {bias_pct:+.1f}%',
                        fontsize=16, fontweight='bold')

            # Get test period dates
            test_start = fold_preds['date'].min()
            test_end = fold_preds['date'].max()

            # Get historical data up to test start
            hist_before_test = hist_data[hist_data['date'] < test_start]

            # Limit to last 180 days before test for better visibility
            lookback_days = 180
            if len(hist_before_test) > lookback_days:
                hist_plot = hist_before_test.tail(lookback_days)
            else:
                hist_plot = hist_before_test

            # === Plot 1: Full Historical Sales + Forecast vs Actual ===
            ax1 = axes[0]

            # Historical sales (before test)
            ax1.plot(hist_plot['date'], hist_plot['sales'],
                    color='gray', alpha=0.6, linewidth=1.5, label='Historical Sales')

            # Test period actuals
            ax1.plot(fold_preds['date'], fold_preds['sales'],
                    color='black', linewidth=2, marker='o', markersize=4,
                    label='Actual Sales (Test)')

            # Test period forecasts
            ax1.plot(fold_preds['date'], fold_preds['forecast'],
                    color='red', linewidth=2, marker='s', markersize=4,
                    linestyle='--', label='Forecast')

            # Shade test period
            ax1.axvspan(test_start, test_end, alpha=0.1, color='yellow',
                       label='Test Period')

            ax1.set_ylabel('Sales', fontsize=11, fontweight='bold')
            ax1.set_title('Sales History + Forecast vs Actual', fontsize=12)
            ax1.legend(loc='upper left', fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

            # === Plot 2: Trend Component ===
            ax2 = axes[1]

            # Can't plot historical trend from hist_data as it's not decomposed
            # Only show forecast trend
            ax2.plot(fold_preds['date'], fold_preds['forecast_trend'],
                    color='blue', linewidth=2, marker='o', markersize=3,
                    label='Forecast Trend')

            ax2.axvspan(test_start, test_end, alpha=0.1, color='yellow')
            ax2.set_ylabel('Trend', fontsize=11, fontweight='bold')
            ax2.set_title('Trend Component', fontsize=12)
            ax2.legend(loc='upper left', fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

            # === Plot 3: Seasonal Components ===
            ax3 = axes[2]

            ax3.plot(fold_preds['date'], fold_preds['forecast_seasonal_yearly'],
                    color='orange', linewidth=2, marker='o', markersize=3,
                    label='Yearly Seasonal')

            ax3.plot(fold_preds['date'], fold_preds['forecast_seasonal_weekly'],
                    color='green', linewidth=2, marker='s', markersize=3,
                    label='Weekly Seasonal')

            ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax3.axvspan(test_start, test_end, alpha=0.1, color='yellow')
            ax3.set_ylabel('Seasonal Effect', fontsize=11, fontweight='bold')
            ax3.set_title('Seasonal Components (Weekly + Yearly)', fontsize=12)
            ax3.legend(loc='upper left', fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

            # === Plot 4: Forecast Error ===
            ax4 = axes[3]

            # Error bars (metrics already calculated above for title)
            colors = ['red' if x > 0 else 'green' for x in fold_preds['error']]
            ax4.bar(fold_preds['date'], fold_preds['error'],
                   color=colors, alpha=0.6, width=0.8)

            ax4.axhline(0, color='black', linestyle='-', linewidth=1)
            ax4.set_ylabel('Forecast Error', fontsize=11, fontweight='bold')
            ax4.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax4.set_title('Forecast Error (Forecast - Actual)', fontsize=12)
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

            # Add metrics text
            mae = abs(fold_preds['error']).mean()
            mape = abs(fold_preds['error_pct']).mean()
            bias = fold_preds['error'].mean()

            metrics_text = f'MAE: {mae:.1f} | MAPE: {mape:.1f}% | Bias: {bias:.1f}'
            ax4.text(0.02, 0.95, metrics_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Rotate x-axis labels for all subplots
            for ax in axes:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            plt.tight_layout()

            # Save figure with test start date in filename
            test_start_str = test_start.strftime('%Y-%m-%d')
            filename = output_path.replace('.png', f'_{category}_{test_start_str}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {filename}")
            plt.close()


def plot_weekly_pattern_verification(predictions_df, output_path='weekly_pattern.png'):
    """
    Verify weekly seasonal pattern is correctly captured
    """

    categories = predictions_df['category'].unique()

    fig, axes = plt.subplots(len(categories), 1, figsize=(12, 4*len(categories)))
    if len(categories) == 1:
        axes = [axes]

    fig.suptitle('Weekly Seasonal Pattern Verification', fontsize=16, fontweight='bold')

    for idx, category in enumerate(categories):
        ax = axes[idx]

        cat_data = predictions_df[predictions_df['category'] == category].copy()
        cat_data['day_name'] = pd.to_datetime(cat_data['date']).dt.day_name()
        cat_data['day_of_week'] = pd.to_datetime(cat_data['date']).dt.dayofweek

        # Group by day of week
        weekly_pattern = cat_data.groupby('day_of_week').agg({
            'forecast_seasonal_weekly': 'mean',
            'day_name': 'first'
        }).reset_index()

        weekly_pattern = weekly_pattern.sort_values('day_of_week')

        # Bar plot
        colors = ['red' if x < 0 else 'green' for x in weekly_pattern['forecast_seasonal_weekly']]
        bars = ax.bar(weekly_pattern['day_name'], weekly_pattern['forecast_seasonal_weekly'],
                     color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=10, fontweight='bold')

        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel('Weekly Seasonal Effect', fontsize=11, fontweight='bold')
        ax.set_title(f'{category} - Average Weekly Pattern', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlabel('Day of Week', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved weekly pattern plot: {output_path}")
    plt.close()


def plot_feature_importance(category_df, output_path='feature_importance.png'):
    """
    Plot feature importance from trained LightGBM model
    """
    print("\nTraining model to extract feature importance...")

    # Train forecaster
    forecaster = CategoryForecaster()
    forecaster.decompose(category_df)
    forecaster.forecast_trend()

    pooled = forecaster.prepare_features()
    forecaster.train_seasonal_model(pooled)
    forecaster.train_lgbm(pooled)

    # Get feature importance
    importance = forecaster.lgbm_model.feature_importances_
    features = forecaster.lgbm_features

    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=True)

    print(f"✓ Extracted importance for {len(features)} features")

    # Plot
    _, ax = plt.subplots(figsize=(10, max(8, len(features) * 0.3)))

    colors = plt.cm.viridis(importance_df['importance'] / importance_df['importance'].max())

    bars = ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
    ax.set_xlabel('Importance (Gain)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title('LightGBM Feature Importance\n(Higher = More Important for Predictions)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{width:.0f}',
               ha='left', va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature importance plot: {output_path}")

    # Print top features
    print("\n" + "="*60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*60)
    top_features = importance_df.tail(10).iloc[::-1]
    for _, row in top_features.iterrows():
        print(f"  {row['feature']:30s} {row['importance']:10.1f}")
    print("="*60)

    plt.close()

    return importance_df


def main():
    """
    Generate comprehensive forecast visualizations
    """
    import os

    # Create plots directory
    plots_dir = 'forecast_plots'
    os.makedirs(plots_dir, exist_ok=True)

    print("="*80)
    print("FORECAST VISUALIZATION")
    print("="*80)
    print(f"Plots will be saved to: {plots_dir}/")
    print()

    # Load data
    try:
        category_df = pd.read_csv('category_day_aggregated.csv')
        category_df['date'] = pd.to_datetime(category_df['date'])
        print(f"✓ Loaded category data: {len(category_df)} records")
    except FileNotFoundError:
        print("Error: category_day_aggregated.csv not found")
        print("Please run example_usage.py first")
        return

    try:
        predictions_df = pd.read_csv('validation_predictions.csv')
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        print(f"✓ Loaded validation predictions: {len(predictions_df)} records")
    except FileNotFoundError:
        print("Error: validation_predictions.csv not found")
        print("Please run validate_model.py first")
        return

    print("\nGenerating plots...")

    # Generate detailed forecast vs actual plots
    plot_forecast_vs_actual(category_df, predictions_df, f'{plots_dir}/forecast_analysis.png')

    # Generate weekly pattern verification
    plot_weekly_pattern_verification(predictions_df, f'{plots_dir}/weekly_pattern.png')

    # Generate feature importance plot
    plot_feature_importance(category_df, f'{plots_dir}/feature_importance.png')

    print("\n" + "="*80)
    print("✓ Visualization Complete!")
    print("="*80)
    print("\nGenerated plots show:")
    print("  1. Historical sales context + forecast vs actual")
    print("  2. Trend component over time")
    print("  3. Seasonal components (weekly + yearly)")
    print("  4. Forecast errors with metrics")
    print("  5. Weekly pattern verification")
    print("  6. Feature importance ranking")


if __name__ == "__main__":
    main()
