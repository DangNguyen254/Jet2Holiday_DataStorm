import pandas as pd
import numpy as np
from pathlib import Path
import logging
import mlflow
from feature_engineer import FeatureEngineer
from demand_recovery import CensoredDemandRecovery
from forcasting_model import ForecastingModel
import sys
from scripts.load_data import download_freshretail as load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_complete_pipeline():
    logger.info("="*80)
    logger.info("FRESHRETAILNET-50K COMPLETE TRAINING PIPELINE")
    logger.info("="*80)
    
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    
    mlflow.set_experiment("FreshRetailNet-Forecasting")
    
    with mlflow.start_run(run_name="two_stage_pipeline"):
        logger.info("\n[1/6] Loading data...")
        project_root = Path(__file__).parent.parent.parent
        train_path = project_root / 'data' / 'raw' / 'train.parquet'
        eval_path = project_root / 'data' / 'raw' / 'eval.parquet'
        
        train_df = pd.read_parquet(train_path)
        eval_df = pd.read_parquet(eval_path)
        
        logger.info(f"Training data: {train_df.shape}")
        logger.info(f"Evaluation data: {eval_df.shape}")
        
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("eval_rows", len(eval_df))
        
        logger.info("\n[2/6] Feature engineering...")
        feature_config = {
            'feature_engineering': {
                'lag_periods': [1, 7, 14, 28],
                'rolling_windows': [3, 7, 14, 28],
                'max_features': 200
            }
        }
        
        engineer = FeatureEngineer(config=feature_config)
        train_features, feature_names = engineer.run_full_pipeline(train_df)
        eval_features, _ = engineer.run_full_pipeline(eval_df)
        
        mlflow.log_param("num_features", len(feature_names))
        train_features.to_parquet('data/processed/train_features.parquet', index=False)
        eval_features.to_parquet('data/processed/eval_features.parquet', index=False)
        
        logger.info("\n[3/6] Stage 1: Censored Demand Recovery...")
        recovery = CensoredDemandRecovery()
        recovery.train_recovery_model(train_features, feature_names)
        train_features = recovery.recover_censored_demand(train_features, feature_names)
        eval_features = recovery.recover_censored_demand(eval_features, feature_names)
        recovery.save_model('models/recovery_model.pkl')
        mlflow.log_artifact('models/recovery_model.pkl')
        
        logger.info("\n[4/6] Stage 2: Training Forecasting Model...")
        X_train = train_features[feature_names]
        y_train = train_features['recovered_demand']
        X_eval = eval_features[feature_names]
        y_eval = eval_features['recovered_demand']
        
        split_idx = int(len(X_train) * 0.8)
        X_train_split = X_train.iloc[:split_idx]
        y_train_split = y_train.iloc[:split_idx]
        X_val = X_train.iloc[split_idx:]
        y_val = y_train.iloc[split_idx:]
        
        forecaster = ForecastingModel()
        
        logger.info("\nOptimizing hyperparameters...")
        best_params = forecaster.optimize_hyperparameters(
            X_train_split, y_train_split,
            n_trials=30
        )
        
        mlflow.log_params(best_params)
        forecaster.train_final_model(X_train_split, y_train_split, X_val, y_val)
        forecaster.save_model('models/forecasting_model.pkl')
        mlflow.log_artifact('models/forecasting_model.pkl')
        
        logger.info("\n[5/6] Model Evaluation...")
        metrics = forecaster.evaluate(X_eval, y_eval)
        mlflow.log_metrics(metrics)
        
        logger.info("\n[6/6] Comparing with Baseline (no demand recovery)...")
        y_train_observed = train_features['sale_amount']
        y_eval_observed = eval_features['sale_amount']
        
        baseline_model = ForecastingModel()
        baseline_model.best_params = best_params
        baseline_model.train_final_model(X_train_split, y_train_observed.iloc[:split_idx])
        
        baseline_metrics = baseline_model.evaluate(X_eval, y_eval_observed)
        
        logger.info("\n" + "="*80)
        logger.info("RESULTS COMPARISON")
        logger.info("="*80)
        logger.info("\nTwo-Stage Pipeline (with demand recovery):")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        logger.info("\nBaseline (without demand recovery):")
        for metric, value in baseline_metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        mape_improvement = (baseline_metrics['mape'] - metrics['mape']) / baseline_metrics['mape'] * 100
        bias_reduction = abs(baseline_metrics['bias_pct']) - abs(metrics['bias_pct'])
        
        logger.info("\n" + "="*80)
        logger.info("KEY IMPROVEMENTS")
        logger.info("="*80)
        logger.info(f"  MAPE Improvement: {mape_improvement:.2f}%")
        logger.info(f"  Bias Reduction: {bias_reduction:.2f} percentage points")
        logger.info("="*80)
        
        mlflow.log_metric("mape_improvement_pct", mape_improvement)
        mlflow.log_metric("bias_reduction", bias_reduction)
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        forecaster.feature_importance.head(30).plot(
            x='feature', y='importance', kind='barh'
        )
        plt.xlabel('Importance')
        plt.title('Top 30 Most Important Features')
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('outputs/feature_importance.png')
        
        logger.info("\nTraining pipeline complete!")
        logger.info("Models saved to models/")
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
if __name__ == "__main__":
    run_complete_pipeline()