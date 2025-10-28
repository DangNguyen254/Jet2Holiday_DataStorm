from datasets import load_dataset
import pandas as pd
from pathlib import Path
import logging
import os
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging
log_file = log_dir / "data_loading.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_freshretail():

    logger.info("="*70)
    logger.info("Downloading FRESHRETAILNET-50K")
    logger.info("="*70)
    
    # Get the project root directory (one level up from scripts/)
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    
    # Create directories if they don't exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n[1/3] Downloading from Hugging Face")
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    
    logger.info("[2/3] Converting to pandas DataFrames")
    train_df = dataset['train'].to_pandas()
    eval_df = dataset['eval'].to_pandas()
    
    logger.info(f"Training data: {train_df.shape}")
    logger.info(f"Evaluation data: {eval_df.shape}")
    
    # Rename 'dt' column to 'date' for consistency
    train_df = train_df.rename(columns={'dt': 'date'})
    eval_df = eval_df.rename(columns={'dt': 'date'})
    
    logger.info("\n[3/3] Dataset Information:")
    logger.info(f"Date range: {train_df['date'].min()} to {train_df['date'].max()}")
    logger.info(f"Stores: {train_df['store_id'].nunique()}")
    logger.info(f"Products: {train_df['product_id'].nunique()}")
    logger.info(f"Cities: {train_df['city_id'].nunique()}")
    
    logger.info("\nColumns:")
    for col in train_df.columns:
        logger.info(f"  - {col}")
    
    logger.info("\nSaving to CSV")
    train_df.to_csv(str(raw_dir / 'train.csv'), index=False)
    eval_df.to_csv(str(raw_dir / 'eval.csv'), index=False)

    logger.info("Saving to Parquet")
    train_df.to_parquet(str(raw_dir / 'train.parquet'), index=False)
    eval_df.to_parquet(str(raw_dir / 'eval.parquet'), index=False)
    
    logger.info(f"\nData saved to: {raw_dir.absolute()}")
    
    logger.info("\n" + "="*70)
    logger.info("DOWNLOAD COMPLETE!")
    logger.info("="*70)
    
    return train_df, eval_df

if __name__ == "__main__":
    train_df, eval_df = download_freshretail()