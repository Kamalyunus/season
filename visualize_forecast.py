"""
Visualization: Historical Sales + Forecast vs Actuals
Shows if the model captures seasonal patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
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

            # Create figure with subplots
            fig, axes = plt.subplots(4, 1, figsize=(16, 12))
            fig.suptitle(f'{category} - Fold {fold}: Historical Context + Forecast Validation',
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

            plot_start = hist_plot['date'].min()

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

            fold_preds['error'] = fold_preds['forecast'] - fold_preds['sales']
            fold_preds['error_pct'] = (fold_preds['error'] / fold_preds['sales']) * 100

            # Error bars
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

            # Save figure
            filename = output_path.replace('.png', f'_{category}_fold{fold}.png')
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


def main():
    """
    Generate comprehensive forecast visualizations
    """
    print("="*80)
    print("FORECAST VISUALIZATION")
    print("="*80)

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
    plot_forecast_vs_actual(category_df, predictions_df, 'forecast_analysis.png')

    # Generate weekly pattern verification
    plot_weekly_pattern_verification(predictions_df, 'weekly_pattern.png')

    print("\n" + "="*80)
    print("✓ Visualization Complete!")
    print("="*80)
    print("\nGenerated plots show:")
    print("  1. Historical sales context + forecast vs actual")
    print("  2. Trend component over time")
    print("  3. Seasonal components (weekly + yearly)")
    print("  4. Forecast errors with metrics")
    print("  5. Weekly pattern verification")


if __name__ == "__main__":
    main()
