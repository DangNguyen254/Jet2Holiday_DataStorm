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
    
    def train_recovery_model(self, df: pd.DataFrame, feature_cols: list) -> lgb.LGBMRegressor:
        logger.info("STAGE 1: TRAINING DEMAND RECOVERY MODEL")
        
        clean_data = df[df['is_any_stockout'] == 0].copy()
        
        logger.info(f"\nClean data (no stockouts): {len(clean_data):,} rows")
        logger.info(f"Stockout data: {len(df[df['is_any_stockout'] == 1]):,} rows")
        
        # Prepare features
        X_clean = clean_data[feature_cols]
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
        logger.info("\n" + "="*70)
        logger.info("RECOVERING CENSORED DEMAND")
        logger.info("="*70)
        
        df = df.copy()
        
        # Identify stockout periods
        stockout_mask = df['is_any_stockout'] == 1
        
        if stockout_mask.sum() == 0:
            logger.info("No stockouts detected")
            df['recovered_demand'] = df['sale_amount']
            return df
        
        logger.info(f"\nStockout periods to recover: {stockout_mask.sum():,}")
        
        # Predict latent demand for stockout periods
        X_stockout = df.loc[stockout_mask, feature_cols]
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