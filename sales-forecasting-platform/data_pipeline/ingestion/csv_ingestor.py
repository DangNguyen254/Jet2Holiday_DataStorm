class CSVDataIngestor(BaseDataIngestor):
    def __init__(self, tenant_id: str, config: Dict[str, Any], file_path: str):
        super().__init__(tenant_id, config)
        self.file_path = file_path
    
    def extract(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.file_path,
            parse_dates=['date'],
            infer_datetime_format=True
        )
        df['_ingestion_timestamp'] = datetime.now()
        df['_tenant_id'] = self.tenant_id
        
        return df