import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Any
from sklearn.feature_selection import RFECV, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb

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
        
    def generate_time_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
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
        df['stockout_hours_count'] = df['hours_stock_status'].apply(
            lambda x: str(x).count('0') if pd.notna(x) else 0
        )
        df['is_any_stockout'] = (df['stockout_hours_count'] > 0).astype(int)
        df['stockout_ratio'] = df['stockout_hours_count'] / 24
        
        df['current_hour_stock'] = df.apply(
            lambda row: int(str(row['hours_stock_status'])[row['hour']]) 
            if pd.notna(row['hours_stock_status']) and len(str(row['hours_stock_status'])) == 24
            else 1,
            axis=1
        )
        
        #lag stockout features
        df = df.sort_values(['store_id', 'product_id', 'dt'])
        df['stockout_lag_1'] = df.groupby(['store_id', 'product_id'])['is_any_stockout'].shift(1)
        df['stockout_lag_24'] = df.groupby(['store_id', 'product_id'])['is_any_stockout'].shift(24)
        df['stockout_lag_168'] = df.groupby(['store_id', 'product_id'])['is_any_stockout'].shift(168)
        
        logger.info(f"Generated {len([c for c in df.columns if c.startswith('stockout_lag')])} stockout features")
        return df
    
    def generate_promotion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Discount features
        df['has_discount'] = (df['discount'] > 0).astype(int)
        df['discount_level'] = pd.cut(df['discount'], 
                                       bins=[0, 0.1, 0.2, 0.3, 1.0],
                                       labels=['none', 'low', 'medium', 'high'])
        
        # Activity flag features
        df['activity_days_rolling_7d'] = df.groupby(['store_id', 'product_id'])['activity_flag'].transform(
            lambda x: x.rolling(window=168, min_periods=1).sum()  # 7 days
        )
        
        # Lagged promotion effects
        df['discount_lag_24h'] = df.groupby(['store_id', 'product_id'])['discount'].shift(24)
        df['discount_lag_168h'] = df.groupby(['store_id', 'product_id'])['discount'].shift(168)
        
        logger.info("Created promotion features")
        
        return df
    
    def generate_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Weather categories
        df['has_precip'] = (df['precpt'] > 0).astype(int)
        df['precip_level'] = pd.cut(df['precpt'],
                                     bins=[0, 1, 5, 10, 100],
                                     labels=['none', 'light', 'moderate', 'heavy'])
        
        # Temperature categories
        df['temp_level'] = pd.cut(df['avg_temperature'],
                                   bins=5,
                                   labels=['very_cold', 'cold', 'mild', 'warm', 'hot'])
        
        # Weather interactions
        df['rain_weekend'] = df['has_precip'] * df['is_weekend']
        df['cold_morning'] = ((df['avg_temperature'] < 10) & (df['is_morning'])).astype(int)
        logger.info("Created weather features")
        return df
    
    def generate_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=['date']).copy()
        for period in self.lag_periods:
            df[f'sales_lag_{period}'] = df['sales'].shift(period)
        logger.info(f"Generated {len([c for c in df.columns if c.startswith('sales_lag')])} lag features")
        return df
    
    def generate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=['date']).copy()
        for window in self.rolling_windows:
            df[f'sales_rolling_{window}_mean'] = df['sales'].rolling(window=window).mean()
            df[f'sales_rolling_{window}_std'] = df['sales'].rolling(window=window).std()
        logger.info(f"Generated {len([c for c in df.columns if c.startswith('sales_rolling')])} rolling features")
        return df

    def create_hierarchical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Category-level aggregations
        for cat_level in ['first_category_id', 'second_category_id', 'third_category_id']:
            if cat_level in df.columns:
                agg_df = df.groupby([cat_level, 'dt'])['sale_amount'].agg(['mean', 'sum']).reset_index()
                agg_df.columns = [cat_level, 'dt', f'{cat_level}_avg_sales', f'{cat_level}_total_sales']
                df = df.merge(agg_df, on=[cat_level, 'dt'], how='left')
        
        # Store-level aggregations
        store_agg = df.groupby(['store_id', 'dt'])['sale_amount'].agg(['mean', 'sum']).reset_index()
        store_agg.columns = ['store_id', 'dt', 'store_avg_sales', 'store_total_sales']
        df = df.merge(store_agg, on=['store_id', 'dt'], how='left')
        
        logger.info("Created hierarchical aggregation features")
        
        return df
    
    def run_full_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        logger.info("FRESHRETAILNET-50K FEATURE ENGINEERING PIPELINE")
        initial_shape = df.shape
        logger.info(f"\nInput data shape: {initial_shape}")
        # Parse datetime
        df = self.parse_datetime(df)
        
        # Create all feature groups
        logger.info("\n[1/7] Time features...")
        df = self.create_time_features(df)
        
        logger.info("[2/7] Stockout features (UNIQUE!)...")
        df = self.create_stockout_features(df)
        
        logger.info("[3/7] Lag features...")
        df = self.create_lag_features(df)
        
        logger.info("[4/7] Rolling features...")
        df = self.create_rolling_features(df)
        
        logger.info("[5/7] Promotion features...")
        df = self.create_promotion_features(df)
        
        logger.info("[6/7] Weather features...")
        df = self.create_weather_features(df)
        
        logger.info("[7/7] Hierarchical features...")
        df = self.create_hierarchical_features(df)
        
        # Handle NaN from lag/rolling features
        initial_rows = len(df)
        df = df.dropna(subset=['sale_amount'])
    
        logger.info(f"\n✓ Dropped {initial_rows - len(df)} rows with NaN")
        logger.info(f"Final shape: {df.shape}")
        logger.info(f"Generated {df.shape[1] - initial_shape[1]} new features")
    
        # Get feature names
        exclude_cols = ['dt', 'sale_amount', 'hours_sale', 'store_id', 'product_id', 
                   'city_id', 'management_group_id', 'first_category_id', 
                   'second_category_id', 'third_category_id', 'hours_stock_status']
    
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
    
        logger.info(f"\n✓ Total features available: {len(self.feature_names)}")
    
        return df, self.feature_names