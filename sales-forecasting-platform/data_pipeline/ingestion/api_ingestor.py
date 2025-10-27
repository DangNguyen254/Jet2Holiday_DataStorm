class APIDataIngestor(BaseDataIngestor):
    def __init__(self, tenant_id: str, config: Dict[str, Any], api_url: str, api_key: str):
        super().__init__(tenant_id, config)
        self.api_url = api_url
        self.api_key = api_key
    
    def extract(self) -> pd.DataFrame:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        response = requests.get(self.api_url, headers=headers)
        data = response.json()
        df = pd.DataFrame(data)
        df['_ingestion_timestamp'] = datetime.now()
        df['_tenant_id'] = self.tenant_id
        return df