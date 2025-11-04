import pandas as pd
import numpy as np
from pathlib import Path
import logging
import mlflow
from feature_engineer import FeatureEngineer
from demand_recovery import CensoredDemandRecovery
from forecasting_model import ForecastingModel
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLFLOW_TRACKING_URI = "http://localhost:5000"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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
        
        # Create processed data directory
        processed_dir = project_root / 'data' / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output file paths
        train_features_path = processed_dir / 'train_features.parquet'
        eval_features_path = processed_dir / 'eval_features.parquet'
        
        # Initialize and run feature engineering
        engineer = FeatureEngineer(config=feature_config)
        
        try:
            logger.info("\n[2/6] Running feature engineering on training data...")
            train_features, feature_names = engineer.run_full_pipeline(train_df)
            
            logger.info("\n[2.5/6] Running feature engineering on evaluation data...")
            eval_features, _ = engineer.run_full_pipeline(eval_df)
            
            #Log number of features
            mlflow.log_param("num_features", len(feature_names))
            logger.info(f"Generated {len(feature_names)} features")
            
            # Save processed data
            logger.info(f"\nSaving processed data to {processed_dir}")
            train_features.to_parquet(train_features_path, index=False)
            eval_features.to_parquet(eval_features_path, index=False)
            
        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}")
            raise
        logger.info(f"Saved processed training data to {train_features_path}")
        logger.info(f"Saved processed evaluation data to {eval_features_path}")
        
        logger.info("\n[3/6] Stage 1: Censored Demand Recovery...")
        recovery = CensoredDemandRecovery()
        recovery.train_recovery_model(train_features, feature_names)
        train_features = recovery.recover_censored_demand(train_features, feature_names)
        eval_features = recovery.recover_censored_demand(eval_features, feature_names)
        recovery.save_model('models/recovery_model.pkl')
        mlflow.log_artifact('models/recovery_model.pkl')
        
        logger.info("\n[4/6] Stage 2: Training Forecasting Model...")
        
        # Ensure both train and eval have the same columns
        all_columns = set(train_features.columns).union(set(eval_features.columns))
        for col in all_columns:
            if col not in train_features.columns:
                train_features[col] = 0
            if col not in eval_features.columns:
                eval_features[col] = 0
        
        # Make sure feature_names only contains columns that exist in the data
        valid_feature_names = [f for f in feature_names if f in train_features.columns and f in eval_features.columns]
        missing_features = set(feature_names) - set(valid_feature_names)
        
        if missing_features:
            logger.warning(f"The following features are missing from the data and will be excluded: {missing_features}")
        
        # Ensure we have the target column
        if 'recovered_demand' not in train_features.columns or 'recovered_demand' not in eval_features.columns:
            raise ValueError("'recovered_demand' column is missing from the data")
        
        # Select only the valid features and target
        X_train = train_features[valid_feature_names]
        y_train = train_features['recovered_demand']
        X_eval = eval_features[valid_feature_names]
        y_eval = eval_features['recovered_demand']
        
        split_idx = int(len(X_train) * 0.8)
        X_train_split = X_train.iloc[:split_idx]
        y_train_split = y_train.iloc[:split_idx]
        X_val = X_train.iloc[split_idx:]
        y_val = y_train.iloc[split_idx:]
        
        forecaster = ForecastingModel()
        
        logger.info("\nOptimizing hyperparameters...")
        # best_params = forecaster.optimize_hyperparameters(
        #     X_train_split, y_train_split,
        #     n_trials=30
        # )
        
        #mlflow.log_params(best_params)
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
        #baseline_model.best_params = best_params
        baseline_model.train_final_model(X_train_split, y_train_observed.iloc[:split_idx])
        
        baseline_metrics = baseline_model.evaluate(X_eval, y_eval_observed)
        
        logger.info("\nRESULTS COMPARISON")
        logger.info("\nTwo-Stage Pipeline (with demand recovery):")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        logger.info("\nBaseline (without demand recovery):")
        for metric, value in baseline_metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        mape_improvement = (baseline_metrics['mape'] - metrics['mape']) / baseline_metrics['mape'] * 100
        bias_reduction = abs(baseline_metrics['bias_pct']) - abs(metrics['bias_pct'])
        
        logger.info("\nKEY IMPROVEMENTS")
        logger.info(f"  MAPE Improvement: {mape_improvement:.2f}%")
        logger.info(f"  Bias Reduction: {bias_reduction:.2f} percentage points")
        
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