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

def run_complete_pipeline():
    logger.info("FRESHRETAILNET-50K COMPLETE TRAINING PIPELINE")
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
            
            # Align evaluation features to match training features
            logger.info("\nAligning evaluation features to training feature set...")
            for col in feature_names:
                if col not in eval_features.columns:
                    eval_features[col] = 0
            
            # CRITICAL: Preserve sale_amount even though it's not in feature_names
            # It's needed for the recovery model
            columns_to_keep = list(feature_names)
            if 'sale_amount' in eval_features.columns and 'sale_amount' not in columns_to_keep:
                columns_to_keep.append('sale_amount')
            if 'recovered_demand' in eval_features.columns and 'recovered_demand' not in columns_to_keep:
                columns_to_keep.append('recovered_demand')
            
            eval_features = eval_features[columns_to_keep]
            
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
        
        # Ensure feature alignment after demand recovery
        # Preserve sale_amount and recovered_demand columns
        columns_to_check = list(feature_names)
        if 'sale_amount' in train_features.columns:
            columns_to_check.append('sale_amount')
        if 'recovered_demand' in train_features.columns:
            columns_to_check.append('recovered_demand')
        
        for col in feature_names:
            if col not in train_features.columns:
                train_features[col] = 0
            if col not in eval_features.columns:
                eval_features[col] = 0
        
        # Select only the valid features and target
        X_train = train_features[feature_names].copy()
        y_train = train_features['recovered_demand'].copy()
        X_eval = eval_features[feature_names].copy()
        y_eval = eval_features['recovered_demand'].copy()
        
        # Ensure data is sorted by date for time-aware split
        # Note: Feature engineering should have sorted by store_id, product_id, date
        # But we need to ensure global date ordering for the split
        if 'date' in train_features.columns:
            date_sort_idx = train_features['date'].argsort()
            X_train = X_train.iloc[date_sort_idx].reset_index(drop=True)
            y_train = y_train.iloc[date_sort_idx].reset_index(drop=True)
        
        # Time-aware split: use first 80% chronologically
        # This prevents future data from leaking into training
        split_idx = int(len(X_train) * 0.8)
        X_train_split = X_train.iloc[:split_idx].copy()
        y_train_split = y_train.iloc[:split_idx].copy()
        X_val = X_train.iloc[split_idx:].copy()
        y_val = y_train.iloc[split_idx:].copy()
        
        logger.info(f"Train split: {len(X_train_split):,} samples, Validation split: {len(X_val):,} samples")
        
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
        
        logger.info("\n[6/6] Comparing with Baseline")
        y_train_observed = train_features['sale_amount']
        y_eval_observed = eval_features['sale_amount']
        
        # Baseline model should have its own hyperparameter optimization for fair comparison
        logger.info("\nOptimizing hyperparameters for baseline model...")
        baseline_model = ForecastingModel()
        baseline_best_params = baseline_model.optimize_hyperparameters(
            X_train_split, y_train_observed.iloc[:split_idx],
            n_trials=30
        )
        baseline_model.train_final_model(X_train_split, y_train_observed.iloc[:split_idx], X_val, y_train_observed.iloc[split_idx:])
        baseline_metrics = baseline_model.evaluate(X_eval, y_eval_observed)
        
        logger.info("\nRESULTS COMPARISON")
        logger.info("\nTwo-Stage Pipeline (with demand recovery):")
        for metric, value in metrics.items():
            if metric == 'mape':
                logger.info(f"  {metric.upper()}: {value:.2f}%")
            elif metric == 'r2':
                logger.info(f"  {metric.upper()}: {value:.4f}")
            else:
                logger.info(f"  {metric.upper()}: {value:.4f}")
        
        logger.info("\nBaseline (without demand recovery):")
        for metric, value in baseline_metrics.items():
            if metric == 'mape':
                logger.info(f"  {metric.upper()}: {value:.2f}%")
            elif metric == 'r2':
                logger.info(f"  {metric.upper()}: {value:.4f}")
            else:
                logger.info(f"  {metric.upper()}: {value:.4f}")
        
        # Calculate improvements (handle potential inf values)
        if np.isfinite(baseline_metrics['mape']) and np.isfinite(metrics['mape']) and baseline_metrics['mape'] > 0:
            mape_improvement = (baseline_metrics['mape'] - metrics['mape']) / baseline_metrics['mape'] * 100
        else:
            mape_improvement = 0
            logger.warning("Cannot calculate MAPE improvement due to invalid values")
        
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