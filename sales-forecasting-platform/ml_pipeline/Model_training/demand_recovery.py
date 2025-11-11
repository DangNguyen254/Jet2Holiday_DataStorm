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
        self.training_feature_columns = None
        
    
    def _preprocess_features(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        df = df.copy()
        exclude_cols = [
            'hours_sale', 'hours_stock_status', 'store_id', 'product_id',
            'city_id', 'management_group_id', 'first_category_id', 
            'second_category_id', 'third_category_id', 'sale_amount'
        ]
        df = df.drop(columns=[col for col in exclude_cols if col in df.columns], errors='ignore')
        
        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        for col in datetime_cols:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_hour'] = df[col].dt.hour
        
        df = df.drop(columns=datetime_cols, errors='ignore')
        
        # Handle categorical columns, skip columns that contain arrays/lists
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            # Skip if column is not in feature_cols
            if col not in feature_cols:
                continue
            try:
                non_null_idx = df[col].first_valid_index()
                if non_null_idx is not None:
                    sample_val = df[col].loc[non_null_idx]
                    if isinstance(sample_val, (list, np.ndarray, tuple)):
                        logger.warning(f"Dropping column {col} - contains arrays/lists")
                        df = df.drop(columns=[col], errors='ignore')
                        continue
                df[col] = df[col].astype('category').cat.codes
            except (TypeError, ValueError) as e:
                logger.warning(f"Dropping column {col} - cannot convert to categorical: {str(e)}")
                df = df.drop(columns=[col], errors='ignore')
                continue
        
        # Handle one-hot encoded categorical features
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
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'uint8', 'bool', 'int32', 'float32']).columns
        df = df[numeric_cols]
        
        # Ensure same columns as training data
        if self.training_feature_columns is not None:
            # Add missing columns with 0
            for col in self.training_feature_columns:
                if col not in df.columns:
                    df[col] = 0
            available_cols = [col for col in self.training_feature_columns if col in df.columns]
            df = df[available_cols]
        return df
        
    def train_recovery_model(self, df: pd.DataFrame, feature_cols: list) -> lgb.LGBMRegressor:
        logger.info("STAGE 1: TRAINING DEMAND RECOVERY MODEL")
        clean_data = df[df['is_any_stockout'] == 0].copy()
        logger.info(f"\nClean data (no stockouts): {len(clean_data):,} rows")
        logger.info(f"Stockout data: {len(df[df['is_any_stockout'] == 1]):,} rows")
        
        # Preprocess features
        X_clean = self._preprocess_features(clean_data, feature_cols)
        y_clean = clean_data['sale_amount']
        
        # Store the feature columns used for training
        self.training_feature_columns = X_clean.columns.tolist()
        
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
        
        # Robust MAPE calculation
        mask = (y_val > 1e-6)
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_val[mask] - y_pred[mask]) / y_val[mask])) * 100
        else:
            mape = float('inf')
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        logger.info(f"\nRecovery Model Performance (on clean data):")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  Mean actual: {y_val.mean():.4f}, Mean predicted: {y_pred.mean():.4f}")
        
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
        
        # Ensure sale_amount exists in dataframe which is needed for recovery
        if 'sale_amount' not in df.columns:
            raise ValueError("'sale_amount' column is required but not found in dataframe. "
                           "It should be preserved during feature engineering.")
        
        stockout_mask = df['is_any_stockout'] == 1
        logger.info(f"\nStockout periods to recover: {stockout_mask.sum():,}")
        
        # Initialize recovered_demand with observed sales
        df['recovered_demand'] = df['sale_amount'].copy()
        if stockout_mask.sum() > 0:
            # drops sale_amount)
            observed_sales = df.loc[stockout_mask, 'sale_amount'].values
            
            # Preprocess features
            X_stockout = self._preprocess_features(df[stockout_mask], feature_cols)
            raw_predictions = self.recovery_model.predict(X_stockout)
            below_observed_pct = (raw_predictions < observed_sales).mean() * 100
            # Ensure non-negative predictions
            recovered_demand = np.maximum(raw_predictions, 0)
            recovered_demand = np.maximum(recovered_demand, observed_sales)
            
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
            if below_observed_pct > 10:
                logger.warning(f"  WARNING: {below_observed_pct:.1f}% of raw predictions were below observed sales.")
                logger.warning(f"  This suggests the model may be using censored lag features. Consider using recovered_demand for lag features.")
            
            # Show distribution comparison
            logger.info(f"\nDistribution Comparison:")
            logger.info(f"  Observed (stockout): min={df.loc[stockout_mask, 'sale_amount'].min():.0f}, "
                       f"max={df.loc[stockout_mask, 'sale_amount'].max():.0f}, "
                       f"mean={avg_observed:.2f}")
            logger.info(f"  Recovered: min={recovered_demand.min():.0f}, "
                       f"max={recovered_demand.max():.0f}, "
                       f"mean={avg_recovered:.2f}")
        else:
            logger.info("No stockouts to recover")
        
        return df
    
    def save_model(self, path: str):
        #joblib.dump(self.recovery_model, path)
        joblib.dump({
            'model': self.recovery_model,
            'training_feature_columns': self.training_feature_columns,
            'clean_data_stats': self.clean_data_stats
        }, path)
        logger.info(f" Recovery model saved to {path}")
    
    def load_model(self, path: str):
        data = joblib.load(path)
        self.recovery_model = data['model']
        self.training_feature_columns = data.get('training_feature_columns')
        self.clean_data_stats = data.get('clean_data_stats', {})
        logger.info(f" Recovery model loaded from {path}")