import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Any
from sklearn.feature_selection import RFECV, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
class FeatureEngineer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lag_periods = config.get('feature_engineering.lag_periods', [1, 7, 14, 28])
        self.rolling_windows = config.get('feature_engineering.rolling_windows', [3, 7, 14, 28])
        self.max_features = config.get('feature_engineering.max_features', 200)
        self.selected_features = None
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_importance_scores = None
        
    def parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        return df
        
    def generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.parse_datetime(df.copy())  
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['date'].dt.dayofweek >= 5
        logger.info(f"Generated {len([c for c in df.columns if c not in ['date']])} time features")
        return df
    
    def generate_stock_out_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=['date']).copy()
        
        df['stockout_hours_count'] = df['hours_stock_status'].astype(str).str.count('0')
        df['stockout_hours_count'] = df['stockout_hours_count'].fillna(0).astype(int)
        df['is_any_stockout'] = (df['stockout_hours_count'] > 0).astype(int)
        df['stockout_ratio'] = df['stockout_hours_count'] / 24
        
        def get_current_hour_stock(row):
            try:
                if pd.isna(row['hours_stock_status']) or len(str(row['hours_stock_status'])) != 24:
                    return 1
                hour = int(row.get('hour', 0))
                return int(str(row['hours_stock_status'])[hour])
            except (ValueError, IndexError):
                return 1
                
        df['current_hour_stock'] = df.apply(get_current_hour_stock, axis=1)
        
        #lag stockout features
        df = df.sort_values(['store_id', 'product_id', 'date'])
        df['stockout_lag_1'] = df.groupby(['store_id', 'product_id'])['is_any_stockout'].shift(1)
        df['stockout_lag_24'] = df.groupby(['store_id', 'product_id'])['is_any_stockout'].shift(24)
        df['stockout_lag_168'] = df.groupby(['store_id', 'product_id'])['is_any_stockout'].shift(168)
        
        logger.info(f"Generated {len([c for c in df.columns if c.startswith('stockout_lag')])} stockout features")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using one-hot encoding."""
        df = df.copy()
        
        # List of categorical columns to encode
        categorical_cols = [
            'temperature_category',
            'precipitation_category',
            'humidity_category',
            'wind_category'
        ]
        
        # Only encode columns that exist in the dataframe
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        if categorical_cols:
            # Perform one-hot encoding
            df = pd.get_dummies(df, columns=categorical_cols, prefix_sep='_')
            logger.info(f"Encoded {len(categorical_cols)} categorical features")
            
        return df
        
    def generate_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        features_added = []
        
        # Temperature Category
        if 'avg_temperature' in df.columns:
            def temperature_category(temp):
                if pd.isna(temp):
                    return 'unknown'
                if temp < 10:
                    return 'cold'
                elif 10 <= temp < 20:
                    return 'mild'
                elif 20 <= temp < 30:
                    return 'warm'
                else:
                    return 'hot'
            
            df['temperature_category'] = df['avg_temperature'].apply(temperature_category)
            features_added.append('temperature_category')
        
        # Precipitation Category
        if 'precpt' in df.columns:
            def precipitation_category(precpt):
                if pd.isna(precpt):
                    return 'unknown'
                if precpt == 0:
                    return 'no_rain'
                elif 0 < precpt < 5:
                    return 'light_rain'
                else:
                    return 'heavy_rain'
            
            df['precipitation_category'] = df['precpt'].apply(precipitation_category)
            features_added.append('precipitation_category')
        
        # Humidity Category
        if 'avg_humidity' in df.columns:
            def humidity_category(humidity):
                if pd.isna(humidity):
                    return 'unknown'
                if humidity < 40:
                    return 'low_humidity'
                elif 40 <= humidity < 70:
                    return 'medium_humidity'
                else:
                    return 'high_humidity'
            
            df['humidity_category'] = df['avg_humidity'].apply(humidity_category)
            features_added.append('humidity_category')
        
        # Wind Category
        if 'avg_wind_level' in df.columns:
            def wind_category(wind):
                if pd.isna(wind):
                    return 'unknown'
                if wind < 2:
                    return 'low_wind'
                elif 2 <= wind < 4:
                    return 'medium_wind'
                else:
                    return 'high_wind'
            
            df['wind_category'] = df['avg_wind_level'].apply(wind_category)
            features_added.append('wind_category')
        

        if features_added:
            logger.info(f"Created weather features: {', '.join(features_added)}")
        else:
            logger.warning("No weather features created - required columns not found in the dataset")
        categorical_columns = ['temperature_category', 'precipitation_category', 'humidity_category', 'wind_category']
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
        return df
        
    def generate_promotion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Discount features
        df['has_discount'] = (df['discount'] > 0).astype(int)
        df['discount_level'] = pd.cut(df['discount'], 
                                       bins=[0, 0.1, 0.2, 0.3, 1.0],
                                       labels=['none', 'low', 'medium', 'high'])
        
        # Activity flag features
        if 'activity_flag' in df.columns:
            df['activity_days_rolling_7d'] = df.groupby(['store_id', 'product_id'])['activity_flag'].transform(
                lambda x: x.rolling(window=168, min_periods=1).sum() 
            )
        
        # Lagged promotion effects
        if 'discount' in df.columns:
            df['discount_lag_24h'] = df.groupby(['store_id', 'product_id'])['discount'].shift(24)
            df['discount_lag_168h'] = df.groupby(['store_id', 'product_id'])['discount'].shift(168)
        
        logger.info("Created promotion features")
        return df
    
    
    def generate_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=['date']).copy()
        for period in self.lag_periods:
            df[f'sale_amount_lag_{period}'] = df['sale_amount'].shift(period)
        logger.info(f"Generated {len([c for c in df.columns if c.startswith('sale_amount_lag')])} lag features")
        return df
    
    def generate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=['date']).copy()
        for window in self.rolling_windows:
            df[f'sale_amount_rolling_{window}_mean'] = df['sale_amount'].rolling(window=window).mean()
            df[f'sale_amount_rolling_{window}_std'] = df['sale_amount'].rolling(window=window).std()
        logger.info(f"Generated {len([c for c in df.columns if c.startswith('sale_amount_rolling')])} rolling features")
        return df

    def generate_hierarchical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'sale_amount' not in df.columns:
            logger.warning("'sale_amount' column not found. Skipping hierarchical features.")
            return df
            
        df = df.copy()
        date_col = 'date' if 'date' in df.columns else 'dt' if 'dt' in df.columns else None
        if date_col is None:
            logger.warning("Neither 'date' nor 'dt' column found. Skipping hierarchical features.")
            return df
        
        # Category-level aggregations
        for cat_level in ['first_category_id', 'second_category_id', 'third_category_id']:
            if cat_level in df.columns:
                try:
                    agg_df = df.groupby([cat_level, date_col])['sale_amount'].agg(['mean', 'sum']).reset_index()
                    agg_df.columns = [cat_level, date_col, f'{cat_level}_avg_sales', f'{cat_level}_total_sales']
                    df = df.merge(agg_df, on=[cat_level, date_col], how='left')
                except Exception as e:
                    logger.warning(f"Error generating {cat_level} aggregations: {str(e)}")
        
        # Store-level aggregations
        try:
            store_agg = df.groupby(['store_id', date_col])['sale_amount'].agg(['mean', 'sum']).reset_index()
            store_agg.columns = ['store_id', date_col, 'store_avg_sales', 'store_total_sales']
            df = df.merge(store_agg, on=['store_id', date_col], how='left')
        except Exception as e:
            logger.warning(f"Error generating store-level aggregations: {str(e)}")
        
        # Log which features were added
        agg_features = [col for col in df.columns if col.endswith(('_avg_sales', '_total_sales'))]
        if agg_features:
            logger.info(f"Created {len(agg_features)} hierarchical aggregation features")
        else:
            logger.warning("No hierarchical aggregation features were created")
        
        return df
    
    def run_full_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        logger.info("FRESHRETAILNET-50K FEATURE ENGINEERING PIPELINE")
        initial_shape = df.shape
        logger.info(f"\nInput data shape: {initial_shape}")
        
        # Make a copy of the original dataframe
        df = df.copy()
        
        # Store the target variable if it exists
        target_col = 'recovered_demand' if 'recovered_demand' in df.columns else None
        if target_col:
            y = df[target_col]
        
        # Parse datetime and extract time-based features
        logger.info("\n[1/8] Time features...")
        df = self.parse_datetime(df)
        df = self.generate_time_features(df)
        
        # Generate other features
        logger.info("[2/8] Stockout features...")
        df = self.generate_stock_out_features(df)
        
        logger.info("[3/8] Weather features...")
        df = self.generate_weather_features(df)
        
        logger.info("[4/8] Promotion features...")
        df = self.generate_promotion_features(df)
        
        logger.info("[5/8] Lag features...")
        df = self.generate_lag_features(df)
        
        logger.info("[6/8] Rolling features...")
        df = self.generate_rolling_features(df)
        
        logger.info("[7/8] Hierarchical features...")
        df = self.generate_hierarchical_features(df)
        
        # Encode categorical features
        logger.info("[8/8] Encoding categorical features...")
        df = self.encode_categorical_features(df)
        
        # Add back the target variable if it existed
        if target_col is not None:
            df[target_col] = y
        
        # Handle NaN from lag/rolling features
        initial_rows = len(df)
        df = df.dropna(subset=['sale_amount'])
    
        logger.info(f"\nDropped {initial_rows - len(df)} rows with NaN")
        logger.info(f"Generated {df.shape[1] - initial_shape[1]} new features")
    
        # Define columns to exclude from features
        exclude_cols = [
            'hours_sale', 'store_id', 'product_id',
            'city_id', 'management_group_id', 'first_category_id', 
            'second_category_id', 'third_category_id', 'hours_stock_status'
        ]
        
        # Only keep numeric columns for modeling
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'uint8', 'bool']).columns.tolist()
        
        # Get final feature names (numeric columns that are not in exclude_cols)
        self.feature_names = [col for col in numeric_cols if col not in exclude_cols]
        
        # Add back target column if it exists
        if target_col in df.columns and target_col not in self.feature_names:
            self.feature_names.append(target_col)
    
        logger.info(f"\nTotal features available: {len(self.feature_names)}")
        logger.info(f"Final data shape: {df[self.feature_names].shape}")
    
        return df[self.feature_names], [f for f in self.feature_names if f != target_col]