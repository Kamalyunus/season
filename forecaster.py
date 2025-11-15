"""
Simplified Category-Level Sales Forecasting Pipeline
Uses MSTL decomposition + Temperature Modulation + LightGBM
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


class CategoryForecaster:
    """Simplified forecasting pipeline with config-driven parameters"""

    def __init__(self, config_path='config.yaml'):
        from config_loader import load_config
        self.config = load_config(config_path)
        self.horizon = self.config.get('forecast.horizon_days')

        # Model storage
        self.categories = None
        self.decompositions = {}
        self.trend_forecasts = {}
        self.temp_stats = {}
        self.temp_sensitivity = None
        self.lgbm_model = None
        self.lgbm_features = None

    def aggregate_to_category(self, sku_df):
        """Aggregate SKU-day to category-day level with quality filters"""
        print("\nAggregating SKU → Category level...")

        # Get config parameters
        min_skus = self.config.get('preprocessing.min_skus_per_category', 10)
        instock_threshold = self.config.get('preprocessing.instock_threshold', 0.85)

        # Count SKUs per category to filter noisy categories
        sku_count = sku_df.groupby('category')['sku_id'].nunique()
        valid_categories = sku_count[sku_count >= min_skus].index.tolist()

        print(f"  SKU count per category:")
        for cat, count in sku_count.items():
            status = "✓" if cat in valid_categories else "✗ (filtered)"
            print(f"    {cat}: {count} SKUs {status}")

        # Filter to only valid categories
        sku_df = sku_df[sku_df['category'].isin(valid_categories)].copy()

        if len(valid_categories) == 0:
            raise ValueError(f"No categories have >= {min_skus} SKUs. Lower min_skus_per_category in config.")

        print(f"  Retained {len(valid_categories)}/{len(sku_count)} categories with >= {min_skus} SKUs")

        def weighted_avg(group, col, weight='sales'):
            return np.average(group[col], weights=group[weight]) if group[weight].sum() > 0 else group[col].mean()

        # Aggregate to category-day level
        agg = sku_df.groupby(['category', 'date']).apply(
            lambda x: pd.Series({
                'sales': x['sales'].sum(),
                'instock_rate': weighted_avg(x, 'instock_rate'),
                'price': weighted_avg(x, 'price'),
                'temperature': x['temperature'].iloc[0],
                'holiday_gap': x['holiday_gap'].iloc[0],
                'main_promo': x['main_promo'].sum(),
                'other_promo': x['other_promo'].sum(),
                'assortment_count': x['sku_id'].nunique()
            })
        ).reset_index()

        agg['date'] = pd.to_datetime(agg['date'])
        agg = agg.sort_values(['category', 'date']).reset_index(drop=True)

        # Apply linear interpolation to sales when instock rate is low
        print(f"\nApplying sales interpolation for low instock (threshold: {instock_threshold})...")

        interpolated_count = 0
        for cat in valid_categories:
            cat_mask = agg['category'] == cat
            low_instock_mask = cat_mask & (agg['instock_rate'] < instock_threshold)

            if low_instock_mask.sum() > 0:
                cat_data = agg[cat_mask].copy()

                # Mark low instock points as NaN for interpolation
                cat_data.loc[cat_data['instock_rate'] < instock_threshold, 'sales'] = np.nan

                # Apply linear interpolation
                cat_data['sales'] = cat_data['sales'].interpolate(method='linear', limit_direction='both')

                # Update main dataframe
                agg.loc[cat_mask, 'sales'] = cat_data['sales'].values

                interpolated_count += low_instock_mask.sum()
                print(f"  {cat}: {low_instock_mask.sum()} days interpolated")

        print(f"✓ Total {interpolated_count} days with sales interpolated due to low instock")
        print(f"✓ {len(agg)} records, {agg['category'].nunique()} categories")
        print(f"  Date range: {agg['date'].min()} to {agg['date'].max()}")

        return agg

    def decompose(self, category_df):
        """MSTL decomposition for each category"""
        print("\nDecomposing time series (MSTL)...")

        self.categories = category_df['category'].unique()
        weekly_p = self.config.get('decomposition.weekly_period')
        yearly_p = self.config.get('decomposition.yearly_period')
        seasonal_s = self.config.get('decomposition.seasonal_smoothing')

        for cat in self.categories:
            data = category_df[category_df['category'] == cat].sort_values('date')

            try:
                mstl = MSTL(data['sales'].values, periods=(weekly_p, yearly_p),
                           stl_kwargs={'seasonal': seasonal_s})
                result = mstl.fit()

                # Handle 1D vs 2D seasonal output
                if result.seasonal.ndim == 2:
                    seasonal_weekly = result.seasonal[:, 0]
                    seasonal_yearly = result.seasonal[:, 1]
                else:
                    # Less than 2 years: only weekly extracted
                    if len(data) < 730:
                        seasonal_weekly = result.seasonal
                        seasonal_yearly = np.zeros_like(seasonal_weekly)
                    else:
                        seasonal_yearly = result.seasonal
                        seasonal_weekly = np.zeros_like(seasonal_yearly)

                self.decompositions[cat] = {
                    'data': data,
                    'trend': result.trend,
                    'seasonal_weekly': seasonal_weekly,
                    'seasonal_yearly': seasonal_yearly,
                    'remainder': result.resid
                }
                print(f"  ✓ {cat}")

            except Exception as e:
                print(f"  ✗ {cat}: {e}")

    def forecast_trend(self):
        """Exponential smoothing for trend forecasting"""
        print("\nForecasting trends...")

        level = self.config.get('trend.smoothing_level')
        slope = self.config.get('trend.smoothing_trend')

        for cat in self.decompositions:
            trend = self.decompositions[cat]['trend']

            try:
                model = ExponentialSmoothing(trend, trend='add', seasonal=None,
                                            initialization_method='estimated')
                fitted = model.fit(smoothing_level=level, smoothing_trend=slope)
                forecast = fitted.forecast(self.horizon)

            except:
                # Fallback: linear extrapolation
                last_n = min(14, len(trend))
                slope_val = (trend.iloc[-1] - trend.iloc[-last_n]) / last_n
                forecast = trend.iloc[-1] + slope_val * np.arange(1, self.horizon + 1)

            self.trend_forecasts[cat] = forecast
            print(f"  ✓ {cat}: {forecast[0]:.1f} → {forecast[-1]:.1f}")

    def prepare_features(self):
        """Engineer features from decomposed data"""
        print("\nEngineering features...")

        all_data = []
        lag_days = self.config.get('features.lag_days')
        roll_windows = self.config.get('features.rolling_windows')

        for cat in self.decompositions:
            df = self.decompositions[cat]['data'].copy()

            # Add decomposed components
            df['trend'] = self.decompositions[cat]['trend']
            df['seasonal_weekly'] = self.decompositions[cat]['seasonal_weekly']
            df['seasonal_yearly'] = self.decompositions[cat]['seasonal_yearly']

            # Temporal features
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_year'] = df['date'].dt.dayofyear
            df['month'] = df['date'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

            # Cyclic encoding
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

            # Lag features
            for lag in lag_days:
                df[f'sales_lag_{lag}'] = df['sales'].shift(lag)

            # Rolling features (shift AFTER rolling for maximum recency)
            for window in roll_windows:
                df[f'sales_roll_mean_{window}'] = df['sales'].rolling(window).mean().shift(1)
                df[f'sales_roll_std_{window}'] = df['sales'].rolling(window).std().shift(1)

            # Price features
            df['price_roll_mean_28'] = df['price'].rolling(28).mean().shift(1)
            df['price_ratio'] = df['price'] / df['price_roll_mean_28']

            # Promo features
            df['total_promo'] = df['main_promo'] + df['other_promo']
            df['promo_intensity'] = df['total_promo'] / df['assortment_count']

            # Temperature features
            df['temp_roll_mean_7'] = df['temperature'].rolling(7).mean().shift(1)
            df['temp_deviation'] = df['temperature'] - df['temp_roll_mean_7']

            all_data.append(df)

        pooled = pd.concat(all_data, ignore_index=True)
        pooled['category_encoded'] = pd.Categorical(pooled['category']).codes

        print(f"✓ {len(pooled)} records pooled")
        return pooled

    def train_seasonal_model(self, pooled_df):
        """Calculate temperature sensitivity for seasonal adjustment"""
        print("\nCalculating temperature-seasonal relationship...")

        # Calculate mean temperature and std by category for normalization
        self.temp_stats = {}

        for cat in self.categories:
            cat_data = pooled_df[pooled_df['category'] == cat]
            self.temp_stats[cat] = {
                'mean': cat_data['temperature'].mean(),
                'std': cat_data['temperature'].std()
            }

        # Get temperature sensitivity from config (default: 0.02)
        self.temp_sensitivity = self.config.get('decomposition.temp_sensitivity', 0.02)

        print(f"✓ Using temperature sensitivity: {self.temp_sensitivity}")
        print(f"✓ Temperature-based modulation (no model training needed)")

    def train_lgbm(self, pooled_df):
        """Train LightGBM for final forecast"""
        print("\nTraining LightGBM...")

        base_features = [
            'category_encoded', 'trend', 'seasonal_weekly', 'seasonal_yearly',
            'temperature', 'temp_deviation', 'main_promo', 'other_promo',
            'total_promo', 'promo_intensity', 'price', 'price_ratio',
            'holiday_gap', 'assortment_count',
            'day_of_week', 'day_of_year', 'month', 'is_weekend',
            'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos'
        ]

        # Add only available lag features
        lag_features = []
        min_pct = self.config.get('features.min_non_nan_pct')

        for col in pooled_df.columns:
            if 'lag' in col or 'roll' in col:
                if pooled_df[col].notna().sum() / len(pooled_df) > min_pct:
                    lag_features.append(col)

        self.lgbm_features = base_features + lag_features
        train = pooled_df.dropna(subset=self.lgbm_features)

        if len(train) == 0:
            self.lgbm_features = base_features
            train = pooled_df.dropna(subset=self.lgbm_features)

        X, y = train[self.lgbm_features], train['sales']

        # Check if tuned hyperparameters exist and use them
        import os
        tuned_params_file = self.config.get('output.best_params', 'best_hyperparameters.yaml')

        if os.path.exists(tuned_params_file):
            print(f"  ✓ Loading tuned hyperparameters from: {tuned_params_file}")
            from config_loader import load_config
            tuned_config = load_config(tuned_params_file)
            cfg = tuned_config['lightgbm']
        else:
            print("  ℹ Using default hyperparameters from config.yaml")
            cfg = self.config['lightgbm']

        self.lgbm_model = lgb.LGBMRegressor(
            objective=cfg['objective'],
            n_estimators=cfg['n_estimators'],
            learning_rate=cfg['learning_rate'],
            max_depth=cfg['max_depth'],
            num_leaves=cfg['num_leaves'],
            min_child_samples=cfg['min_child_samples'],
            subsample=cfg['subsample'],
            subsample_freq=cfg['subsample_freq'],
            colsample_bytree=cfg['colsample_bytree'],
            reg_alpha=cfg['reg_alpha'],
            reg_lambda=cfg['reg_lambda'],
            random_state=cfg['random_state'],
            n_jobs=-1,
            verbose=-1
        )
        self.lgbm_model.fit(X, y, eval_set=[(X, y)],
                           callbacks=[lgb.early_stopping(50, verbose=False)])

        print(f"✓ Trained on {len(X)} samples, {len(self.lgbm_features)} features")

    def generate_forecast(self, category_df, future_temps, future_promos, future_prices=None):
        """Generate forecasts for all categories"""
        print("\nGenerating forecasts...")

        all_forecasts = []

        for cat in self.decompositions:
            data = category_df[category_df['category'] == cat].sort_values('date')
            last_date = data['date'].max()
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=self.horizon)

            # Base forecast dataframe
            df = pd.DataFrame({
                'date': future_dates,
                'category': cat,
                'category_encoded': pd.Categorical([cat], categories=self.categories).codes[0]
            })

            # Trend
            df['trend'] = self.trend_forecasts[cat]

            # Weekly seasonal (repeating pattern)
            last_week = self.decompositions[cat]['seasonal_weekly'][-7:]
            df['seasonal_weekly'] = np.tile(last_week, 5)[:self.horizon]

            # Temporal features
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_year'] = df['date'].dt.dayofyear
            df['month'] = df['date'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

            # External features
            df['temperature'] = future_temps.get(cat, [data['temperature'].mean()] * self.horizon)

            # Get base yearly seasonal pattern (repeat last year)
            hist_seasonal_yearly = self.decompositions[cat]['seasonal_yearly']

            # Create mapping from day_of_year to seasonal value
            hist_data = self.decompositions[cat]['data']
            seasonal_map = {}
            for doy in range(1, 366):
                # Get all historical values for this day of year
                mask = hist_data['date'].dt.dayofyear == doy
                if mask.any():
                    seasonal_map[doy] = hist_seasonal_yearly[mask].mean()

            # Apply base seasonal pattern
            base_seasonal = df['day_of_year'].map(
                lambda doy: seasonal_map.get(doy, 0)
            ).values

            # Temperature modulation (simple bounded scaling)
            temp_mean = self.temp_stats[cat]['mean']
            temp_std = self.temp_stats[cat]['std']
            temp_deviation = (df['temperature'] - temp_mean) / temp_std

            # Bounded temperature factor: limits adjustment between 0.7 and 1.3
            temp_factor = np.clip(1 + self.temp_sensitivity * temp_deviation, 0.7, 1.3)

            # Modulated yearly seasonal
            df['seasonal_yearly'] = base_seasonal * temp_factor

            # Promos
            if cat in future_promos:
                df = df.merge(future_promos[cat], on='date', how='left')
                df['main_promo'] = df['main_promo'].fillna(0)
                df['other_promo'] = df['other_promo'].fillna(0)
            else:
                df['main_promo'] = 0
                df['other_promo'] = 0

            df['total_promo'] = df['main_promo'] + df['other_promo']

            # Price
            df['price'] = future_prices.get(cat, [data['price'].iloc[-1]] * self.horizon) if future_prices else data['price'].iloc[-1]

            # Other features
            df['holiday_gap'] = 0
            df['instock_rate'] = 100
            df['assortment_count'] = data['assortment_count'].iloc[-1]
            df['price_roll_mean_28'] = data['price'].tail(28).mean()
            df['price_ratio'] = df['price'] / df['price_roll_mean_28']
            df['promo_intensity'] = df['total_promo'] / df['assortment_count']

            # Lag features (use last known values)
            for lag in self.config.get('features.lag_days'):
                feat = f'sales_lag_{lag}'
                if feat in self.lgbm_features:
                    df[feat] = data['sales'].iloc[-lag] if len(data) >= lag else data['sales'].mean()

            for window in self.config.get('features.rolling_windows'):
                if f'sales_roll_mean_{window}' in self.lgbm_features:
                    df[f'sales_roll_mean_{window}'] = data['sales'].tail(window).mean()
                if f'sales_roll_std_{window}' in self.lgbm_features:
                    df[f'sales_roll_std_{window}'] = data['sales'].tail(window).std()

            # Ensure all required features exist
            for feat in self.lgbm_features:
                if feat not in df.columns:
                    df[feat] = 0

            # Predict
            forecast = self.lgbm_model.predict(df[self.lgbm_features])
            df['forecast'] = np.maximum(forecast, 0)
            df['forecast_trend'] = df['trend']
            df['forecast_seasonal_yearly'] = df['seasonal_yearly']
            df['forecast_seasonal_weekly'] = df['seasonal_weekly']

            all_forecasts.append(df[['date', 'category', 'forecast', 'forecast_trend',
                                     'forecast_seasonal_yearly', 'forecast_seasonal_weekly']])

            print(f"  ✓ {cat}: {forecast.min():.1f} → {forecast.max():.1f}")

        return pd.concat(all_forecasts, ignore_index=True)
