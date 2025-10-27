from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import boto3
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseDataIngestor(ABC):
    def __init__(self, tenant_id: str, config: Dict[str, Any]):
        self.tenant_id = tenant_id
        self.config = config
        self.s3_client = boto3.client('s3')
        self.bucket = config.get('aws.s3.lakehouse_bucket')
        
    @abstractmethod
    def extract(self) -> pd.DataFrame:
        pass
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'tenant_id': self.tenant_id,
            'row_count': len(df),
            'column_count': len(df.columns),
            'null_counts': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Check for required columns
        required_cols = ['date', 'product_id', 'sales']
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            validation_report['errors'] = f"Missing required columns: {missing_cols}"
            logger.error(validation_report['errors'])
        else:
            validation_report['status'] = 'passed'
            logger.info(f"Validation passed for tenant {self.tenant_id}")
        
        return validation_report
    
    def load_to_raw_zone(self, df: pd.DataFrame, source_name: str):
        timestamp = datetime.now()
        
        # Create partition path
        s3_key = (
            f"raw/"
            f"tenant_id={self.tenant_id}/"
            f"source={source_name}/"
            f"year={timestamp.year}/"
            f"month={timestamp.month:02d}/"
            f"day={timestamp.day:02d}/"
            f"data_{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
        )
        
        # Convert to parquet in memory
        parquet_buffer = df.to_parquet(index=False, engine='pyarrow')
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=parquet_buffer,
            Metadata={
                'tenant_id': self.tenant_id,
                'source': source_name,
                'ingestion_timestamp': timestamp.isoformat(),
                'row_count': str(len(df))
            }
        )
        
        logger.info(f"Uploaded {len(df)} rows to s3://{self.bucket}/{s3_key}")
        
        return f"s3://{self.bucket}/{s3_key}"
    
    def run(self, source_name: str) -> Dict[str, Any]:
        """Execute complete ingestion pipeline"""
        logger.info(f"Starting ingestion for tenant: {self.tenant_id}, source: {source_name}")
        
        # Extract
        df = self.extract()
        logger.info(f"Extracted {len(df)} rows")
        
        # Validate
        validation_report = self.validate(df)
        
        if validation_report.get('status') != 'passed':
            raise ValueError(f"Validation failed: {validation_report.get('errors')}")
        
        # Load
        s3_path = self.load_to_raw_zone(df, source_name)
        
        return {
            'status': 'success',
            'tenant_id': self.tenant_id,
            's3_path': s3_path,
            'validation_report': validation_report
        }