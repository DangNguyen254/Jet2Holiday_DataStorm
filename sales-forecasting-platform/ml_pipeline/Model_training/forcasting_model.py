import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import optuna
from optuna.integration import LightGBMPruningCallback
import joblib
import mlflow
import logging

logger = logging.getLogger(__name__)

class ForecastingModel:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_importance = None
        
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 50) -> dict:
        logger.info("HYPERPARAMETER OPTIMIZATION")
        
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
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_train, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                pruning_callback = LightGBMPruningCallback(trial, 'rmse')
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[pruning_callback, lgb.log_evaluation(0)]
                )
                
                y_pred = model.predict(X_val)
                mape = mean_absolute_percentage_error(y_val, y_pred)
                scores.append(mape)
            return np.mean(scores)
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        
        logger.info(f"\n Optimization complete!")
        logger.info(f"Best MAPE: {study.best_value:.4f}")
        logger.info(f"Best parameters:")
        for key, value in self.best_params.items():
            logger.info(f"    {key}: {value}")
        
        return self.best_params
    
    def train_final_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        logger.info("TRAINING FINAL FORECASTING MODEL")
        
        if self.best_params is None:
            logger.warning("No optimized parameters found. Using defaults.")
            self.best_params = {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 8,
                'num_leaves': 64
            }
        
        # Create model with best params
        self.model = lgb.LGBMRegressor(
            **self.best_params,
            objective='regression',
            metric='rmse',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
            )
        else:
            self.model.fit(X_train, y_train)
    
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
        
        metrics = {
            'mape': mean_absolute_percentage_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': np.mean(np.abs(y_test - y_pred)),
            'bias': (y_pred - y_test).mean(),
            'bias_pct': ((y_pred - y_test).mean() / y_test.mean()) * 100
        }
        
        logger.info("\n" + "="*70)
        logger.info("MODEL EVALUATION")
        logger.info("="*70)
        for metric, value in metrics.items():
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