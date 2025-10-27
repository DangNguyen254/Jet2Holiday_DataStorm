import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def init_database():
    connect = psycopg2.connect(
        host="localhost",
        user="postgres",
        password="360952",
        port="5432"
    )
    connect.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    cursor = connect.cursor()
    cursor.execute("SELECT datname FROM pg_database WHERE datname='sales_forecasting_platform'")
    cursor.execute("CREATE DATABASE freshretail_forecasting")
    cursor.close()
    connect.close()
    
    connect = psycopg2.connect(
        host="localhost",
        database="freshretail_forecasting",
        user="postgres",
        password="360952"
    )
    
    cursor = connect.cursor()
    
    # Execute schema
    with open('Database/schema.sql', 'r') as f:
        cursor.execute(f.read())
    
    connect.commit()
    cursor.close()
    connect.close()
    
    print("Database initialized successfully")

if __name__ == "__main__":
    init_database()