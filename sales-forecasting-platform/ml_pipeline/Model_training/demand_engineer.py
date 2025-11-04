import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import joblib
import logging

class CensoredDemandRecovery:
    def __init__(self):
        self.recovery_model = None
        self.clean_data_stats = {}
        
    def identify_stockout_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify stockout periods in the data."""
        df = df.copy()
        df['is_any_stockout'] = (df['stockout_hours_count'] > 0).astype(int)
        stockout_rate = df['is_any_stockout'].mean()
        logger.info(f"Overall stockout rate: {stockout_rate*100:.2f}%")
        return df
    
    def _preprocess_features(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Preprocess features for the recovery model."""
        df = df.copy()
        
        # Handle datetime features
        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        for col in datetime_cols:
            for attr in ['year', 'month', 'day', 'dayofweek', 'hour']:
                df[f'{col}_{attr}'] = getattr(df[col].dt, attr)
        
        # Handle categorical features
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            df[col] = pd.Categorical(df[col]).codes
            
        # Ensure all feature columns exist
        result = pd.DataFrame()
        for col in feature_cols:
            result[col] = df.get(col, 0)  # Default to 0 if column doesn't exist
            
        return result
    
    def train_recovery_model(self, df: pd.DataFrame, feature_cols: list) -> None:
        logger.info("Training demand recovery model...")
        # Prepare data
        clean_data = df[df['is_any_stockout'] == 0].copy()
        X = self._preprocess_features(clean_data, feature_cols)
        y = clean_data['sale_amount']
        
        # Train model
        self.recovery_model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        self.recovery_model.fit(X, y)
        
    def recover_censored_demand(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Recover censored demand during stockout periods."""
        df = df.copy()
        stockout_mask = df['is_any_stockout'] == 1
        
        if stockout_mask.any():
            X_stockout = self._preprocess_features(df[stockout_mask], feature_cols)
            df.loc[stockout_mask, 'recovered_demand'] = self.recovery_model.predict(X_stockout)
        else:
            df['recovered_demand'] = df['sale_amount']
            
        return df
    
    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        joblib.dump(self.recovery_model, path)
        
    @classmethod
    def load_model(cls, path: str) -> 'CensoredDemandRecovery':
        """Load a trained model from disk."""
        instance = cls()
        instance.recovery_model = joblib.load(path)
        return instance