import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Tuple
from sklearn.feature_selection import RFECV, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lightbgm as lgb

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