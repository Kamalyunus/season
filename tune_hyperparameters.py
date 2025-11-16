"""
Hyperparameter Tuning with Optuna
Optimizes LightGBM parameters using full forecasting pipeline
"""

import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_percentage_error
import warnings
from config_loader import load_config
from forecaster import CategoryForecaster

warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """Tune LightGBM hyperparameters using Optuna"""

    def __init__(self, config_path='config.yaml'):
        self.config = load_config(config_path)
        self.best_params = {}

    def create_cv_splits(self, category_df, n_splits=3):
        """Create time series cross-validation splits"""
        max_date = category_df['date'].max()
        min_date = category_df['date'].min()
        total_days = (max_date - min_date).days

        min_train = self.config.get('validation.min_train_days')
        horizon = self.config.get('forecast.horizon_days')

        available_days = total_days - min_train - horizon
        step_size = max(30, available_days // (n_splits + 1))

        splits = []
        for i in range(n_splits):
            train_end = max_date - pd.Timedelta(days=(n_splits - i) * step_size + horizon)
            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.Timedelta(days=horizon - 1)

            if train_end >= min_date + pd.Timedelta(days=min_train):
                splits.append((train_end, test_start, test_end))

        return splits

    def objective_lgbm(self, trial, category_df, cv_splits):
        """Objective function for LightGBM tuning using full forecasting pipeline"""

        # Sample hyperparameters
        search_space = self.config['optuna']['lgbm_search_space']

        trial_params = {
            'n_estimators': trial.suggest_int('n_estimators', *search_space['n_estimators']),
            'learning_rate': trial.suggest_float('learning_rate', *search_space['learning_rate'], log=True),
            'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
            'num_leaves': trial.suggest_int('num_leaves', *search_space['num_leaves']),
            'min_child_samples': trial.suggest_int('min_child_samples', *search_space['min_child_samples']),
            'subsample': trial.suggest_float('subsample', *search_space['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *search_space['colsample_bytree']),
            'reg_alpha': trial.suggest_float('reg_alpha', *search_space['reg_alpha']),
            'reg_lambda': trial.suggest_float('reg_lambda', *search_space['reg_lambda']),
        }

        # Cross-validation using full pipeline
        fold_mapes = []
        fold_sales = []

        for train_end, test_start, test_end in cv_splits:
            try:
                # Split data
                train_df = category_df[category_df['date'] <= train_end].copy()
                test_df = category_df[(category_df['date'] >= test_start) &
                                     (category_df['date'] <= test_end)].copy()

                if len(train_df) < 730 or len(test_df) == 0:
                    continue

                # Initialize forecaster with trial hyperparameters
                forecaster = CategoryForecaster(lgbm_params_override=trial_params)

                # Run full pipeline on training data
                forecaster.decompose(train_df)
                forecaster.forecast_trend()
                pooled = forecaster.prepare_features()
                forecaster.train_seasonal_model(pooled)
                forecaster.train_lgbm(pooled)

                # Prepare future inputs from test data (like validate.py)
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

                # Generate forecasts using full pipeline
                forecasts = forecaster.generate_forecast(train_df, future_temps, future_promos, future_prices)

                # Merge with actuals
                comparison = forecasts.merge(
                    test_df[['date', 'category', 'sales']],
                    on=['date', 'category'],
                    how='inner'
                )

                if len(comparison) > 0:
                    mape = mean_absolute_percentage_error(
                        comparison['sales'].values,
                        comparison['forecast'].values
                    ) * 100
                    total_sales = comparison['sales'].sum()

                    fold_mapes.append(mape)
                    fold_sales.append(total_sales)

            except Exception:
                continue

        # Calculate sales-weighted MAPE
        if fold_mapes and fold_sales:
            weighted_mape = sum(m * s for m, s in zip(fold_mapes, fold_sales)) / sum(fold_sales)
            return weighted_mape
        else:
            return 100.0


    def tune(self, category_df):
        """Run hyperparameter tuning for LightGBM using full forecasting pipeline"""
        print("="*80)
        print("HYPERPARAMETER TUNING WITH OPTUNA")
        print("Using full forecasting pipeline for realistic evaluation")
        print("="*80)

        # Create CV splits
        cv_splits = self.create_cv_splits(category_df)
        print(f"\nCross-validation folds: {len(cv_splits)}")
        for i, (train_end, test_start, test_end) in enumerate(cv_splits):
            print(f"  Fold {i+1}: Train until {train_end.date()}, Test: {test_start.date()} to {test_end.date()}")

        n_trials = self.config.get('optuna.n_trials')
        timeout = self.config.get('optuna.timeout_seconds')

        # Tune LightGBM
        print("\n" + "="*80)
        print("Tuning LightGBM Parameters")
        print("="*80)
        print("Note: This may take longer as each trial runs the full pipeline")

        study_lgbm = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study_lgbm.optimize(
            lambda trial: self.objective_lgbm(trial, category_df, cv_splits),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        print(f"\n✓ Best MAPE: {study_lgbm.best_value:.2f}%")
        print(f"✓ Best params: {study_lgbm.best_params}")
        self.best_params['lightgbm'] = study_lgbm.best_params

        # Save best parameters to config
        print("\n" + "="*80)
        print("Saving Best Parameters")
        print("="*80)

        # Update config with best params
        for key, value in self.best_params['lightgbm'].items():
            self.config.update(f'lightgbm.{key}', value)

        # Save to file
        output_path = self.config.get('output.best_params')
        self.config.save(output_path)

        print(f"✓ Saved best parameters to: {output_path}")
        print("\nThese parameters are automatically used by run.py and validate.py")

        return self.best_params


def main():
    """Run hyperparameter tuning"""

    # Load data
    try:
        category_df = pd.read_csv('category_day_aggregated.csv')
        category_df['date'] = pd.to_datetime(category_df['date'])
        print(f"\n✓ Loaded category data: {len(category_df)} records")
        print(f"  Categories: {category_df['category'].nunique()}")
        print(f"  Date range: {category_df['date'].min().date()} to {category_df['date'].max().date()}\n")
    except FileNotFoundError:
        print("Error: category_day_aggregated.csv not found")
        print("Please run the forecasting pipeline first: python run.py")
        return

    # Run tuning
    tuner = HyperparameterTuner()
    best_params = tuner.tune(category_df)

    print("\n" + "="*80)
    print("✓ TUNING COMPLETE")
    print("="*80)
    print("\nBest LightGBM Parameters:")
    for key, value in best_params['lightgbm'].items():
        print(f"  {key}: {value}")
    print("\nThese parameters will be automatically used by:")
    print("  - python run.py")
    print("  - python validate.py")


if __name__ == "__main__":
    main()
