"""
Hyperparameter Tuning with Optuna
Optimizes LightGBM parameters
"""

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error
import warnings
from config_loader import load_config

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
        gap = self.config.get('validation.gap_days')

        available_days = total_days - min_train - horizon - gap
        step_size = max(30, available_days // (n_splits + 1))

        splits = []
        for i in range(n_splits):
            train_end = max_date - pd.Timedelta(days=(n_splits - i) * step_size + horizon + gap)
            test_start = train_end + pd.Timedelta(days=gap + 1)
            test_end = test_start + pd.Timedelta(days=horizon - 1)

            if train_end >= min_date + pd.Timedelta(days=min_train):
                splits.append((train_end, test_start, test_end))

        return splits

    def objective_lgbm(self, trial, pooled_df, cv_splits, feature_cols):
        """Objective function for LightGBM tuning"""

        # Sample hyperparameters
        search_space = self.config['optuna']['lgbm_search_space']

        params = {
            'objective': 'regression',
            'n_estimators': trial.suggest_int('n_estimators', *search_space['n_estimators']),
            'learning_rate': trial.suggest_float('learning_rate', *search_space['learning_rate'], log=True),
            'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
            'num_leaves': trial.suggest_int('num_leaves', *search_space['num_leaves']),
            'min_child_samples': trial.suggest_int('min_child_samples', *search_space['min_child_samples']),
            'subsample': trial.suggest_float('subsample', *search_space['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *search_space['colsample_bytree']),
            'reg_alpha': trial.suggest_float('reg_alpha', *search_space['reg_alpha']),
            'reg_lambda': trial.suggest_float('reg_lambda', *search_space['reg_lambda']),
            'n_estimators': 300,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        # Cross-validation
        mapes = []

        for train_end, test_start, test_end in cv_splits:
            train_df = pooled_df[pooled_df['date'] <= train_end]
            test_df = pooled_df[(pooled_df['date'] >= test_start) & (pooled_df['date'] <= test_end)]

            train_clean = train_df.dropna(subset=feature_cols + ['sales'])
            test_clean = test_df.dropna(subset=feature_cols + ['sales'])

            if len(train_clean) == 0 or len(test_clean) == 0:
                continue

            X_train, y_train = train_clean[feature_cols], train_clean['sales']
            X_test, y_test = test_clean[feature_cols], test_clean['sales']

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train)],
                     callbacks=[lgb.early_stopping(30, verbose=False)])

            y_pred = model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            mapes.append(mape)

        return np.mean(mapes) if mapes else 100.0


    def tune(self, pooled_df):
        """Run hyperparameter tuning for LightGBM"""
        print("="*80)
        print("HYPERPARAMETER TUNING WITH OPTUNA")
        print("="*80)

        # Create CV splits
        cv_splits = self.create_cv_splits(pooled_df)
        print(f"\nCross-validation folds: {len(cv_splits)}")

        n_trials = self.config.get('optuna.n_trials')
        timeout = self.config.get('optuna.timeout_seconds')

        # Tune LightGBM
        print("\n" + "="*80)
        print("Tuning LightGBM Parameters")
        print("="*80)

        lgbm_features = [col for col in pooled_df.columns if col not in
                        ['date', 'category', 'sales', 'sku_id', 'remainder', 'observed']]

        # Filter to available features
        available_features = [f for f in lgbm_features if f in pooled_df.columns]

        study_lgbm = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study_lgbm.optimize(
            lambda trial: self.objective_lgbm(trial, pooled_df, cv_splits, available_features),
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
        print("\nTo use these parameters, copy them to config.yaml")

        return self.best_params


def main():
    """Run hyperparameter tuning"""

    # Load data
    try:
        pooled_df = pd.read_csv('pooled_training_data.csv')
        pooled_df['date'] = pd.to_datetime(pooled_df['date'])
        print(f"\n✓ Loaded pooled data: {len(pooled_df)} records\n")
    except FileNotFoundError:
        print("Error: pooled_training_data.csv not found")
        print("Please run the forecasting pipeline first")
        return

    # Run tuning
    tuner = HyperparameterTuner()
    best_params = tuner.tune(pooled_df)

    print("\n" + "="*80)
    print("✓ TUNING COMPLETE")
    print("="*80)
    print("\nBest LightGBM Parameters:")
    for key, value in best_params['lightgbm'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
