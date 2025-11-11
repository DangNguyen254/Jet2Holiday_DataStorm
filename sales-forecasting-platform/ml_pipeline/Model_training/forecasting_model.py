import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import optuna
from optuna.integration import LightGBMPruningCallback
import joblib
import mlflow
import logging
import os
import matplotlib.pyplot as plt
from data_visualization import line_plot
logger = logging.getLogger(__name__)

class ForecastingModel:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_importance = None
        
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 50) -> dict:
        logger.info("HYPERPARAMETER OPTIMIZATION")
        
        # Use a single train/validation split for faster optimization
        split_idx = int(len(X_train) * 0.8)
        X_train_split = X_train.iloc[:split_idx]
        y_train_split = y_train.iloc[:split_idx]
        X_val_split = X_train.iloc[split_idx:]
        y_val_split = y_train.iloc[split_idx:]
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
                'random_state': 42
            }
            
            try:
                pruning_callback = LightGBMPruningCallback(trial, 'rmse')
                model = lgb.LGBMRegressor(
                    **params,
                    n_jobs=-1,
                    verbose=-1
                )
                model.fit(
                    X_train_split, 
                    y_train_split,
                    eval_set=[(X_val_split, y_val_split)],
                    callbacks=[pruning_callback, lgb.log_evaluation(0)]
                )
                
                y_pred = model.predict(X_val_split)
                mask = (y_val_split > 1e-6)
                if mask.sum() > 0:
                    mape = np.mean(np.abs((y_val_split[mask] - y_pred[mask]) / y_val_split[mask])) * 100
                else:
                    mape = float('inf')
                
                return mape
            except Exception as e:
                logger.warning(f"Error in trial: {str(e)}")
                return float('inf')
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.best_params = study.best_params
        logger.info(f"\n Optimization complete!")
        logger.info(f"Best MAPE: {study.best_value:.4f}")
        logger.info(f"Best parameters:")
        for key, value in self.best_params.items():
            logger.info(f"{key}: {value}")
        return self.best_params
    
    def train_final_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        logger.info("TRAINING FINAL FORECASTING MODEL")
        if self.best_params is None:
            self.best_params = {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 8,
                'num_leaves': 64,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            }
        
        self.model = lgb.LGBMRegressor(
            **self.best_params,
            objective='regression',
            metric='rmse',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)] if eval_set else None
        self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n Model training complete!")
        logger.info(f"\n Top 10 Most Important Features:")
        print(self.feature_importance.head(10))
        return self.model
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        y_pred = self.model.predict(X_test)
        mask = (y_test > 1e-6) 
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        else:
            mape = float('inf')
            logger.warning("All actual values are zero or very small, cannot calculate MAPE")
        
        # Additional diagnostics
        logger.info(f"\nTarget value statistics:")
        logger.info(f"  Min: {y_test.min():.6f}, Max: {y_test.max():.6f}")
        logger.info(f"  Mean: {y_test.mean():.6f}, Median: {y_test.median():.6f}")
        logger.info(f"  Zero count: {(y_test == 0).sum()}, Small values (<0.001): {(y_test < 0.001).sum()}")
        logger.info(f"  Prediction range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
        
        metrics = {
            'mape': mape,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': np.mean(np.abs(y_test - y_pred)),
            'bias': (y_pred - y_test).mean(),
            'bias_pct': ((y_pred - y_test).mean() / (y_test.mean() + 1e-10)) * 100,
            'r2': 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)) if y_test.std() > 0 else 0
        }
        
        logger.info("\n" + "="*70)
        logger.info("Forecast MODEL EVALUATION")
        logger.info("="*70)
        for metric, value in metrics.items():
            if metric == 'mape':
                logger.info(f"  {metric.upper()}: {value:.2f}%")
            elif metric == 'r2':
                logger.info(f"  {metric.upper()}: {value:.4f}")
            else:
                logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return metrics
    
    def save_model(self, path: str):
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        data = joblib.load(path)
        self.model = data['model']
        self.best_params = data['best_params']
        self.feature_importance = data['feature_importance']
        logger.info(f"Model loaded from {path}")
