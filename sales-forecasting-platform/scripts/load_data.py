from datasets import load_dataset
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_freshretail():

    logger.info("="*70)
    logger.info("Downloading FRESHRETAILNET-50K")
    logger.info("="*70)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    logger.info("\n[1/3] Downloading from Hugging Face")
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    
    logger.info("[2/3] Converting to pandas DataFrames")
    train_df = dataset['train'].to_pandas()
    eval_df = dataset['eval'].to_pandas()
    
    logger.info(f"Training data: {train_df.shape}")
    logger.info(f"Evaluation data: {eval_df.shape}")
    
    logger.info("\n[3/3] Dataset Information:")
    logger.info(f"Date range: {train_df['dt'].min()} to {train_df['dt'].max()}")
    logger.info(f"Stores: {train_df['store_id'].nunique()}")
    logger.info(f"Products: {train_df['product_id'].nunique()}")
    logger.info(f"Cities: {train_df['city_id'].nunique()}")
    
    logger.info("\nColumns:")
    for col in train_df.columns:
        logger.info(f"  - {col}")
    
    logger.info("\nSaving to CSV")
    train_df.to_csv('data/raw/train.csv', index=False)
    eval_df.to_csv('data/raw/eval.csv', index=False)

    logger.info("Saving to Parquet")
    train_df.to_parquet('data/raw/train.parquet', index=False)
    eval_df.to_parquet('data/raw/eval.parquet', index=False)
    
    logger.info("\n" + "="*70)
    logger.info("✓ DOWNLOAD COMPLETE!")
    logger.info("="*70)
    
    return train_df, eval_df

if __name__ == "__main__":
    train_df, eval_df = download_freshretail()