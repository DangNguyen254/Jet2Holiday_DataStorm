import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import joblib
import logging

logger = logging.getLogger(__name__)
class CensoredDemandRecovery:
    def __init__(self):
        self.recovery_model = None
        self.clean_data_stats = {}
        
    def identify_stockout_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'is_any_stockout' not in df.columns:
            df['is_any_stockout'] = (df['stockout_hours_count'] > 0).astype(int)
        stockout_rate = df['is_any_stockout'].mean()
        logger.info(f"Overall stockout rate: {stockout_rate*100:.2f}%")
        return df
    
    def _preprocess_features(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        df = df.copy()

        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        for col in datetime_cols:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_hour'] = df[col].dt.hour
        
        df = df.drop(columns=datetime_cols, errors='ignore')
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes
        one_hot_patterns = {
            'temperature_category': [0, 1, 2],
            'precipitation_category': [0, 1, 2],
            'humidity_category': [0, 1, 2],
            'wind_category': [0, 1]
        }
        
        for prefix, indices in one_hot_patterns.items():
            for i in indices:
                col_name = f"{prefix}_{i}"
                if col_name in feature_cols and col_name not in df.columns:
                    df[col_name] = 0
                    logger.debug(f"Added missing one-hot encoded column: {col_name}")
        
        # Only keep columns that are in feature_cols
        valid_cols = [col for col in feature_cols if col in df.columns]
        missing_cols = [col for col in feature_cols if col not in valid_cols]
        
        if missing_cols:
            logger.warning(f"The following features were not found and couldn't be imputed: {missing_cols}")
        # Ensure all columns are present and in the correct order
        result_df = pd.DataFrame()
        for col in feature_cols:
            if col in df.columns:
                result_df[col] = df[col]
            else:
                # If we get here, it's a non-one-hot column that's missing
                logger.warning(f"Could not find or impute column: {col}")
                result_df[col] = 0  # Impute with 0 as a last resort    
        return result_df
        
    def train_recovery_model(self, df: pd.DataFrame, feature_cols: list) -> lgb.LGBMRegressor:
        logger.info("STAGE 1: TRAINING DEMAND RECOVERY MODEL")
        
        # Check if required columns exist
        if 'is_any_stockout' not in df.columns:
            raise ValueError("Column 'is_any_stockout' not found in the input DataFrame")
            
        if 'sale_amount' not in df.columns:
            available_cols = ', '.join(df.columns.tolist())
            raise ValueError(f"Column 'sale_amount' not found in the input DataFrame. Available columns: {available_cols}")
        
        clean_data = df[df['is_any_stockout'] == 0].copy()
        
        logger.info(f"\nClean data (no stockouts): {len(clean_data):,} rows")
        logger.info(f"Stockout data: {len(df[df['is_any_stockout'] == 1]):,} rows")
        
        # Preprocess features
        X_clean = self._preprocess_features(clean_data, feature_cols)
        y_clean = clean_data['sale_amount']
        
        # Train-test split
        X_train, X_val, y_train, y_val = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        # Train model
        logger.info("\nTraining recovery model...")
        self.recovery_model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        self.recovery_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        # Evaluate on clean validation data
        y_pred = self.recovery_model.predict(X_val)
        mape = mean_absolute_percentage_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        logger.info(f"\nRecovery Model Performance (on clean data):")
        logger.info(f"  MAPE: {mape:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        
        # Store statistics
        self.clean_data_stats = {
            'mean_sales': y_clean.mean(),
            'std_sales': y_clean.std(),
            'median_sales': y_clean.median()
        }
        
        return self.recovery_model
    
    def recover_censored_demand(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        logger.info("RECOVERING CENSORED DEMAND")
        
        df = df.copy()
        
        # Identify stockout periods
        stockout_mask = df['is_any_stockout'] == 1
        
        if stockout_mask.sum() == 0:
            logger.info("No stockouts detected")
            df['recovered_demand'] = df['sale_amount']
            return df
        
        logger.info(f"\nStockout periods to recover: {stockout_mask.sum():,}")
        
        # Preprocess features for prediction (using the same preprocessing as training)
        X_stockout = self._preprocess_features(df[stockout_mask], feature_cols)
        
        # Predict latent demand for stockout periods
        recovered_demand = self.recovery_model.predict(X_stockout)
        
        # Create recovered demand column
        df['recovered_demand'] = df['sale_amount'].copy()
        df.loc[stockout_mask, 'recovered_demand'] = recovered_demand
        
        # Statistics
        avg_observed = df.loc[stockout_mask, 'sale_amount'].mean()
        avg_recovered = recovered_demand.mean()
        recovery_ratio = avg_recovered / (avg_observed + 1e-6)
        
        logger.info(f"\nRecovery Statistics:")
        logger.info(f"  Average observed sales during stockout: {avg_observed:.2f}")
        logger.info(f"  Average recovered demand: {avg_recovered:.2f}")
        logger.info(f"  Recovery ratio: {recovery_ratio:.2f}x")
        logger.info(f"  Total recovered demand: {recovered_demand.sum():,.0f}")
        
        # Show distribution comparison
        logger.info(f"\nDistribution Comparison:")
        logger.info(f"  Observed (stockout): min={df.loc[stockout_mask, 'sale_amount'].min():.0f}, "
                   f"max={df.loc[stockout_mask, 'sale_amount'].max():.0f}, "
                   f"mean={avg_observed:.2f}")
        logger.info(f"  Recovered: min={recovered_demand.min():.0f}, "
                   f"max={recovered_demand.max():.0f}, "
                   f"mean={avg_recovered:.2f}")
        
        return df
    
    def save_model(self, path: str):
        joblib.dump(self.recovery_model, path)
        logger.info(f" Recovery model saved to {path}")
    
    def load_model(self, path: str):
        self.recovery_model = joblib.load(path)
        logger.info(f" Recovery model loaded from {path}")